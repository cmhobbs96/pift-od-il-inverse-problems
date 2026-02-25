from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from storage import RunStorage


def test_storage_crud(tmp_path: Path) -> None:
    st = RunStorage(tmp_path / "runs.sqlite")
    st.create_schema()

    rec = {
        "run_id": "r1",
        "created_at": "2026-01-01T00:00:00Z",
        "updated_at": "2026-01-01T00:00:00Z",
        "method": "PIFT (Phase A Forward Poisson)",
        "status": "queued",
        "device_requested": "cpu",
        "device_used": "",
        "config_hash": "abc123",
    }
    st.insert_run(rec, config={"n_steps": 100})
    st.update_run_status("r1", "running", {"updated_at": "2026-01-01T00:01:00Z"})
    st.insert_event("r1", "2026-01-01T00:01:00Z", 10.0, "setup", "started", 1.0)
    st.upsert_metrics("r1", {"l2_error": 1.2})
    st.set_artifacts("r1", {"summary": "/tmp/summary.json"})

    rows = st.list_runs()
    assert rows and rows[0]["run_id"] == "r1"

    full = st.get_run("r1")
    assert full is not None
    assert full["config"]["n_steps"] == 100
    assert full["metrics"]["l2_error"] == 1.2
    assert full["artifacts"]["summary"] == "/tmp/summary.json"
    assert full["events"]

    summary_rows = st.list_runs_with_metrics_summary()
    assert summary_rows
    assert summary_rows[0]["run_id"] == "r1"
