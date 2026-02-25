"""Run manager to orchestrate execution, persistence, and lifecycle transitions."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import threading
import uuid
from typing import Any

import numpy as np

from models.bayesian_pinns import run as run_phase_a_bayesian_pinn
from models.monte_carlo import run as run_phase_a_monte_carlo
from models.odil import run as run_phase_a_odil
from models.pift import run as run_phase_a_forward_poisson
from run_types import RunStatus
from storage import RunStorage
from utils.validators import validate_observations, validate_phase_a_config


class RunManager:
    def __init__(self, output_root: str | Path = "outputs") -> None:
        self.output_root = Path(output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.runs_root = self.output_root / "runs"
        self.runs_root.mkdir(parents=True, exist_ok=True)
        self.storage = RunStorage(self.runs_root / "run_registry.sqlite")
        self.storage.create_schema()
        self._stop_events: dict[str, threading.Event] = {}
        self._active_run_id: str | None = None
        self._lock = threading.Lock()

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _config_hash(self, config: dict[str, Any]) -> str:
        payload = json.dumps(config, sort_keys=True).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()[:16]

    def request_stop(self, run_id: str | None = None) -> bool:
        target = run_id or self._active_run_id
        if not target:
            return False
        evt = self._stop_events.get(target)
        if evt is None:
            return False
        evt.set()
        return True

    def get_active_run_id(self) -> str | None:
        return self._active_run_id

    def run_forward(
        self,
        method: str,
        config: dict[str, Any],
        device_preference: str,
        obs_csv_path: str | Path | None,
        obs_data: tuple[np.ndarray, np.ndarray] | None,
        progress_callback,
    ) -> dict[str, Any]:
        supported_methods = {
            "PIFT (Phase A Forward Poisson)": run_phase_a_forward_poisson,
            "Classical Monte Carlo": run_phase_a_monte_carlo,
            "Bayesian PINNs": run_phase_a_bayesian_pinn,
            "ODIL Baseline": run_phase_a_odil,
        }
        if method not in supported_methods:
            supported = ", ".join(supported_methods.keys())
            raise ValueError(f"Method '{method}' is not implemented. Supported: {supported}")

        cfg_errors = validate_phase_a_config(config)
        if cfg_errors:
            raise ValueError("Invalid config: " + "; ".join(cfg_errors))

        if obs_data is not None:
            obs_errors = validate_observations(obs_data[0], obs_data[1])
            if obs_errors:
                raise ValueError("Invalid observation data: " + "; ".join(obs_errors))

        run_id = str(uuid.uuid4())
        now = self._now()
        config_hash = self._config_hash(config)
        record = {
            "run_id": run_id,
            "created_at": now,
            "updated_at": now,
            "method": method,
            "status": RunStatus.QUEUED.value,
            "device_requested": device_preference,
            "device_used": "",
            "config_hash": config_hash,
            "runtime_sec": None,
            "error_code": None,
            "error_message": None,
        }
        self.storage.insert_run(record=record, config=config)

        stop_evt = threading.Event()
        self._stop_events[run_id] = stop_evt

        with self._lock:
            self._active_run_id = run_id
        event_buffer: list[tuple[str, str, float, str, str, float, str]] = []

        def _progress(percent: float, stage: str, detail: str, elapsed_sec: float) -> None:
            ts = self._now()
            event_buffer.append((run_id, ts, percent, stage, detail, elapsed_sec, "INFO"))
            if len(event_buffer) >= 20 or stage == "complete":
                self.storage.insert_events(event_buffer.copy())
                event_buffer.clear()
            if progress_callback is not None:
                progress_callback(percent, stage, detail, elapsed_sec)

        self.storage.update_run_status(
            run_id,
            RunStatus.RUNNING.value,
            fields={"updated_at": self._now()},
        )

        run_output_root = self.runs_root / run_id
        try:
            runner = supported_methods[method]
            result = runner(
                cfg=config,
                output_root=run_output_root,
                obs_csv_path=obs_csv_path,
                obs_data=obs_data,
                device_preference=device_preference,
                run_id=run_id,
                stop_signal=stop_evt.is_set,
                progress_callback=_progress,
                save_outputs=True,
            )

            status = result.get("status", RunStatus.COMPLETED.value)
            metrics = result.get("metrics", {})
            artifacts = result.get("artifacts", {})
            diagnostics = result.get("diagnostics", {})
            error = result.get("error")
            summary = result.get("summary", {})

            self.storage.upsert_metrics(run_id, metrics={**summary, **metrics, "diagnostics": diagnostics})
            self.storage.set_artifacts(run_id, artifacts=artifacts)

            if status == RunStatus.STOPPED.value:
                self.storage.update_run_status(
                    run_id,
                    RunStatus.STOPPED.value,
                    fields={
                        "updated_at": self._now(),
                        "device_used": summary.get("device_used", "cpu"),
                        "runtime_sec": metrics.get("runtime_sec"),
                    },
                )
            elif status == RunStatus.FAILED.value:
                self.storage.update_run_status(
                    run_id,
                    RunStatus.FAILED.value,
                    fields={
                        "updated_at": self._now(),
                        "device_used": summary.get("device_used", "cpu"),
                        "runtime_sec": metrics.get("runtime_sec"),
                        "error_code": (error or {}).get("code"),
                        "error_message": (error or {}).get("message"),
                    },
                )
            else:
                self.storage.update_run_status(
                    run_id,
                    RunStatus.COMPLETED.value,
                    fields={
                        "updated_at": self._now(),
                        "device_used": summary.get("device_used", "cpu"),
                        "runtime_sec": metrics.get("runtime_sec"),
                    },
                )

            result["run_id"] = run_id
            return result
        except Exception as exc:  # noqa: BLE001
            self.storage.update_run_status(
                run_id,
                RunStatus.FAILED.value,
                fields={
                    "updated_at": self._now(),
                    "error_code": "exception",
                    "error_message": str(exc),
                },
            )
            raise
        finally:
            if event_buffer:
                self.storage.insert_events(event_buffer.copy())
            with self._lock:
                self._active_run_id = None
            self._stop_events.pop(run_id, None)
