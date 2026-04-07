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
from pipelines.phase_b_beta_sweep import run_phase_b_beta_sweep
from pipelines.phase_b_model_form import run_phase_b_model_form
from pipelines.phase_c_inverse_params import run_phase_c_inverse_params
from pipelines.phase_c_inverse_source import run_phase_c_inverse_source
from pipelines.phase_d_allen_cahn import run_phase_d_allen_cahn
from run_types import RunStatus
from storage import RunStorage
from utils.validators import validate_observations, validate_phase_a_config

# Dispatch table: (example_key, method_runner_key) → pipeline function
# For single-pipeline examples, method_runner_key is None.
_PIPELINE_DISPATCH: dict[tuple[str, str | None], Any] = {
    ("phase_a_forward", "pift"): run_phase_a_forward_poisson,
    ("phase_a_forward", "mc"): run_phase_a_monte_carlo,
    ("phase_a_forward", "bpinn"): run_phase_a_bayesian_pinn,
    ("phase_a_forward", "odil"): run_phase_a_odil,
    ("phase_b_sweep", None): run_phase_b_beta_sweep,
    ("phase_b_model", None): run_phase_b_model_form,
    ("phase_c_params", None): run_phase_c_inverse_params,
    ("phase_c_source", None): run_phase_c_inverse_source,
    ("phase_d_allen_cahn", None): run_phase_d_allen_cahn,
}

# Examples that accept observation data (obs_csv_path, obs_data params)
_EXAMPLES_WITH_OBS = {"phase_a_forward"}


def _resolve_compute(config: dict[str, Any], device_preference: str) -> tuple[str, str]:
    """Return (backend, device) honoring an optional ``compute`` block in config.

    Env vars ``PIFT_FORCE_BACKEND`` / ``PIFT_FORCE_DEVICE`` (used by the Colab
    bootstrap) override everything else.
    """
    import os

    compute = config.get("compute") or {}
    backend = str(compute.get("backend", "local")).lower()
    device = str(compute.get("device", device_preference or "auto")).lower()
    backend = os.environ.get("PIFT_FORCE_BACKEND", backend).lower()
    device = os.environ.get("PIFT_FORCE_DEVICE", device).lower()
    if backend not in {"local", "modal"}:
        backend = "local"
    if device not in {"auto", "cpu", "gpu"}:
        device = "auto"
    return backend, device


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
        backend, device_resolved = _resolve_compute(config, device_preference)
        try:
            runner = supported_methods[method]
            result = runner(
                cfg=config,
                output_root=run_output_root,
                obs_csv_path=obs_csv_path,
                obs_data=obs_data,
                device_preference=device_resolved,
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

            self.storage.upsert_metrics(
                run_id,
                metrics={**summary, **metrics, "diagnostics": diagnostics, "backend_used": backend},
            )
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

    # ------------------------------------------------------------------
    # General example dispatcher (supports all Phase 1 examples)
    # ------------------------------------------------------------------

    def run_example(
        self,
        example_key: str,
        method_key: str | None,
        config: dict[str, Any],
        device_preference: str,
        obs_csv_path: str | Path | None = None,
        obs_data: tuple[np.ndarray, np.ndarray] | None = None,
        progress_callback=None,
    ) -> dict[str, Any]:
        """Dispatch any example/method combination to its pipeline runner."""
        dispatch_key = (example_key, method_key)
        runner = _PIPELINE_DISPATCH.get(dispatch_key)
        if runner is None:
            runner = _PIPELINE_DISPATCH.get((example_key, None))
        if runner is None:
            raise ValueError(
                f"No pipeline for example={example_key!r}, method={method_key!r}. "
                f"Available: {list(_PIPELINE_DISPATCH.keys())}"
            )

        # Validate Phase A config if applicable
        if example_key == "phase_a_forward":
            cfg_errors = validate_phase_a_config(config)
            if cfg_errors:
                raise ValueError("Invalid config: " + "; ".join(cfg_errors))
            if obs_data is not None:
                obs_errors = validate_observations(obs_data[0], obs_data[1])
                if obs_errors:
                    raise ValueError("Invalid observation data: " + "; ".join(obs_errors))

        method_label = method_key or example_key
        run_id = str(uuid.uuid4())
        now = self._now()
        config_hash = self._config_hash(config)
        record = {
            "run_id": run_id,
            "created_at": now,
            "updated_at": now,
            "method": method_label,
            "status": RunStatus.QUEUED.value,
            "device_requested": device_preference,
            "device_used": "",
            "config_hash": config_hash,
            "runtime_sec": None,
            "error_code": None,
            "error_message": None,
            "example_key": example_key,
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
            run_id, RunStatus.RUNNING.value, fields={"updated_at": self._now()}
        )

        run_output_root = self.runs_root / run_id
        backend, device_resolved = _resolve_compute(config, device_preference)
        try:
            if backend == "modal":
                try:
                    from backends.modal_backend import ModalBackend
                except ImportError as exc:
                    raise RuntimeError(
                        "Modal backend selected but the 'modal' package is not "
                        "installed. Install with: pip install -e '.[modal]' and "
                        "run 'modal token new' to authenticate."
                    ) from exc
                _progress(5.0, "modal", "Submitting run to Modal", 0.0)
                result = ModalBackend().run(
                    example_key=example_key,
                    method_key=method_key,
                    config=config,
                    obs_data=obs_data,
                    local_output_root=run_output_root,
                )
                _progress(95.0, "modal", "Modal run complete", 0.0)
            else:
                # Build kwargs — Phase A runners accept obs params, others don't
                kwargs: dict[str, Any] = {
                    "cfg": config,
                    "output_root": run_output_root,
                    "device_preference": device_resolved,
                    "run_id": run_id,
                    "stop_signal": stop_evt.is_set,
                    "progress_callback": _progress,
                    "save_outputs": True,
                }
                if example_key in _EXAMPLES_WITH_OBS:
                    kwargs["obs_csv_path"] = obs_csv_path
                    kwargs["obs_data"] = obs_data

                result = runner(**kwargs)

            status = result.get("status", RunStatus.COMPLETED.value)
            metrics = result.get("metrics", {})
            artifacts = result.get("artifacts", {})
            diagnostics = result.get("diagnostics", {})
            summary = result.get("summary", {})
            error = result.get("error")

            self.storage.upsert_metrics(
                run_id,
                metrics={**summary, **metrics, "diagnostics": diagnostics, "backend_used": backend},
            )
            self.storage.set_artifacts(run_id, artifacts=artifacts)

            update_fields: dict[str, Any] = {
                "updated_at": self._now(),
                "device_used": summary.get("device_used", "cpu"),
                "runtime_sec": metrics.get("runtime_sec") or result.get("runtime_sec"),
            }
            if status == RunStatus.FAILED.value:
                update_fields["error_code"] = (error or {}).get("code")
                update_fields["error_message"] = (error or {}).get("message")

            db_status = status if status in (
                RunStatus.STOPPED.value, RunStatus.FAILED.value
            ) else RunStatus.COMPLETED.value
            self.storage.update_run_status(run_id, db_status, fields=update_fields)

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
