"""SQLite persistence for run metadata, events, metrics, and artifacts."""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any


class RunStorage:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn_ref: sqlite3.Connection | None = None

    def _conn(self) -> sqlite3.Connection:
        with self._lock:
            if self._conn_ref is None:
                c = sqlite3.connect(str(self.db_path), check_same_thread=False)
                c.row_factory = sqlite3.Row
                c.execute("PRAGMA foreign_keys = ON")
                c.execute("PRAGMA journal_mode = WAL")
                c.execute("PRAGMA synchronous = NORMAL")
                self._conn_ref = c
            return self._conn_ref

    def close(self) -> None:
        with self._lock:
            if self._conn_ref is not None:
                self._conn_ref.close()
                self._conn_ref = None

    def create_schema(self) -> None:
        conn = self._conn()
        with self._lock:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    method TEXT NOT NULL,
                    status TEXT NOT NULL,
                    device_requested TEXT NOT NULL,
                    device_used TEXT,
                    config_hash TEXT NOT NULL,
                    runtime_sec REAL,
                    error_code TEXT,
                    error_message TEXT
                );

                CREATE TABLE IF NOT EXISTS run_configs (
                    run_id TEXT PRIMARY KEY,
                    config_json TEXT NOT NULL,
                    FOREIGN KEY(run_id) REFERENCES runs(run_id)
                );

                CREATE TABLE IF NOT EXISTS run_metrics (
                    run_id TEXT PRIMARY KEY,
                    metrics_json TEXT NOT NULL,
                    FOREIGN KEY(run_id) REFERENCES runs(run_id)
                );

                CREATE TABLE IF NOT EXISTS run_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    percent REAL NOT NULL,
                    stage TEXT NOT NULL,
                    detail TEXT NOT NULL,
                    elapsed_sec REAL NOT NULL,
                    level TEXT NOT NULL,
                    FOREIGN KEY(run_id) REFERENCES runs(run_id)
                );

                CREATE TABLE IF NOT EXISTS run_artifacts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    artifact_type TEXT NOT NULL,
                    path TEXT NOT NULL,
                    FOREIGN KEY(run_id) REFERENCES runs(run_id)
                );

                CREATE INDEX IF NOT EXISTS idx_runs_status_method_created
                    ON runs(status, method, created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_runs_created
                    ON runs(created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_events_run_id
                    ON run_events(run_id, id);
                CREATE INDEX IF NOT EXISTS idx_artifacts_run_id
                    ON run_artifacts(run_id);
                """
            )
            conn.commit()

    def insert_run(self, record: dict[str, Any], config: dict[str, Any]) -> None:
        conn = self._conn()
        with self._lock:
            conn.execute(
                """
                INSERT INTO runs(run_id, created_at, updated_at, method, status, device_requested, device_used,
                                 config_hash, runtime_sec, error_code, error_message)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record["run_id"],
                    record["created_at"],
                    record["updated_at"],
                    record["method"],
                    record["status"],
                    record["device_requested"],
                    record.get("device_used"),
                    record["config_hash"],
                    record.get("runtime_sec"),
                    record.get("error_code"),
                    record.get("error_message"),
                ),
            )
            conn.execute(
                "INSERT OR REPLACE INTO run_configs(run_id, config_json) VALUES (?, ?)",
                (record["run_id"], json.dumps(config, sort_keys=True)),
            )
            conn.commit()

    def update_run_status(self, run_id: str, status: str, fields: dict[str, Any] | None = None) -> None:
        fields = dict(fields or {})
        fields["status"] = status
        keys = list(fields.keys())
        sets = ", ".join([f"{k}=?" for k in keys])
        values = [fields[k] for k in keys]
        values.append(run_id)
        conn = self._conn()
        with self._lock:
            conn.execute(f"UPDATE runs SET {sets} WHERE run_id=?", values)
            conn.commit()

    def insert_event(
        self,
        run_id: str,
        timestamp: str,
        percent: float,
        stage: str,
        detail: str,
        elapsed_sec: float,
        level: str = "INFO",
    ) -> None:
        conn = self._conn()
        with self._lock:
            conn.execute(
                """
                INSERT INTO run_events(run_id, timestamp, percent, stage, detail, elapsed_sec, level)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (run_id, timestamp, percent, stage, detail, elapsed_sec, level),
            )
            conn.commit()

    def insert_events(self, events: list[tuple[str, str, float, str, str, float, str]]) -> None:
        if not events:
            return
        conn = self._conn()
        with self._lock:
            conn.executemany(
                """
                INSERT INTO run_events(run_id, timestamp, percent, stage, detail, elapsed_sec, level)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                events,
            )
            conn.commit()

    def upsert_metrics(self, run_id: str, metrics: dict[str, Any]) -> None:
        conn = self._conn()
        with self._lock:
            conn.execute(
                "INSERT OR REPLACE INTO run_metrics(run_id, metrics_json) VALUES (?, ?)",
                (run_id, json.dumps(metrics, sort_keys=True, default=str)),
            )
            conn.commit()

    def set_artifacts(self, run_id: str, artifacts: dict[str, str]) -> None:
        conn = self._conn()
        with self._lock:
            conn.execute("DELETE FROM run_artifacts WHERE run_id=?", (run_id,))
            conn.executemany(
                "INSERT INTO run_artifacts(run_id, artifact_type, path) VALUES (?, ?, ?)",
                [(run_id, k, v) for k, v in artifacts.items()],
            )
            conn.commit()

    def list_runs(self, filters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        filters = filters or {}
        where_parts: list[str] = []
        args: list[Any] = []
        if filters.get("status"):
            where_parts.append("status=?")
            args.append(filters["status"])
        if filters.get("method"):
            where_parts.append("method=?")
            args.append(filters["method"])

        where = f"WHERE {' AND '.join(where_parts)}" if where_parts else ""
        q = f"SELECT * FROM runs {where} ORDER BY created_at DESC"

        conn = self._conn()
        with self._lock:
            rows = conn.execute(q, args).fetchall()
        return [dict(r) for r in rows]

    def list_runs_with_metrics_summary(self, filters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        filters = filters or {}
        where_parts: list[str] = []
        args: list[Any] = []
        if filters.get("status"):
            where_parts.append("r.status=?")
            args.append(filters["status"])
        if filters.get("method"):
            where_parts.append("r.method=?")
            args.append(filters["method"])
        where = f"WHERE {' AND '.join(where_parts)}" if where_parts else ""
        q = f"""
            SELECT
                r.*,
                json_extract(m.metrics_json, '$.l2_error') AS l2_error,
                json_extract(m.metrics_json, '$.reference_l2_error') AS reference_l2_error,
                json_extract(m.metrics_json, '$.measurement_interval_coverage_90pct') AS coverage_90,
                json_extract(m.metrics_json, '$.diagnostics.stability_flags.nan_detected') AS nan_detected,
                json_extract(m.metrics_json, '$.diagnostics.stability_flags.inf_detected') AS inf_detected,
                json_extract(m.metrics_json, '$.diagnostics.stability_flags.divergence_detected') AS divergence_detected,
                json_extract(m.metrics_json, '$.diagnostics.stability_flags.hamiltonian_overflow') AS hamiltonian_overflow
            FROM runs r
            LEFT JOIN run_metrics m ON m.run_id = r.run_id
            {where}
            ORDER BY r.created_at DESC
        """
        conn = self._conn()
        with self._lock:
            try:
                rows = conn.execute(q, args).fetchall()
            except sqlite3.OperationalError:
                # Fallback if JSON1 extension is unavailable.
                rows = conn.execute(
                    f"SELECT r.* FROM runs r {where} ORDER BY r.created_at DESC",
                    args,
                ).fetchall()
        return [dict(r) for r in rows]

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        conn = self._conn()
        with self._lock:
            run = conn.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
            if run is None:
                return None

            cfg_row = conn.execute("SELECT config_json FROM run_configs WHERE run_id=?", (run_id,)).fetchone()
            met_row = conn.execute("SELECT metrics_json FROM run_metrics WHERE run_id=?", (run_id,)).fetchone()
            art_rows = conn.execute(
                "SELECT artifact_type, path FROM run_artifacts WHERE run_id=?", (run_id,)
            ).fetchall()
            evt_rows = conn.execute(
                "SELECT timestamp, percent, stage, detail, elapsed_sec, level FROM run_events WHERE run_id=? ORDER BY id ASC",
                (run_id,),
            ).fetchall()

        run_dict = dict(run)
        run_dict["config"] = json.loads(cfg_row["config_json"]) if cfg_row else {}
        run_dict["metrics"] = json.loads(met_row["metrics_json"]) if met_row else {}
        run_dict["artifacts"] = {r["artifact_type"]: r["path"] for r in art_rows}
        run_dict["events"] = [dict(r) for r in evt_rows]
        return run_dict

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:  # noqa: BLE001
            pass
