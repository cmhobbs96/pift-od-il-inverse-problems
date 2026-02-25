from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pipelines.phase_a import run_phase_a_monte_carlo


def test_monte_carlo_pipeline_smoke(tmp_path: Path) -> None:
    cfg = {
        "seed": 11,
        "n_modes": 6,
        "n_obs": 20,
        "noise_std": 0.12,
        "beta": 0.8,
        "n_steps": 400,
        "burn_in": 50,
        "thin": 5,
        "n_quad": 32,
        "n_grid": 120,
        "mc_proposal_std": 1e-3,
    }
    result = run_phase_a_monte_carlo(cfg=cfg, output_root=tmp_path, save_outputs=False)

    assert result["status"] in {"completed", "failed", "stopped"}
    assert "metrics" in result
    assert "diagnostics" in result
    assert "mc_acceptance_rate" in result["metrics"]
