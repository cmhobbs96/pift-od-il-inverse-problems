from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pipelines.phase_a import run_phase_a_bayesian_pinn, run_phase_a_odil


def test_bayesian_pinn_pipeline_smoke(tmp_path: Path) -> None:
    cfg = {
        "seed": 17,
        "n_modes": 6,
        "n_obs": 20,
        "noise_std": 0.1,
        "n_steps": 300,
        "burn_in": 50,
        "thin": 5,
        "n_quad": 32,
        "n_grid": 120,
        "step_size0": 5e-4,
        "decay": 0.7,
        "bpinn_sigma_r": 0.2,
        "bpinn_prior_prec": 1e-2,
    }
    result = run_phase_a_bayesian_pinn(cfg=cfg, output_root=tmp_path, save_outputs=False)

    assert result["status"] in {"completed", "failed", "stopped"}
    assert "metrics" in result
    assert "diagnostics" in result


def test_odil_pipeline_smoke(tmp_path: Path) -> None:
    cfg = {
        "seed": 19,
        "n_modes": 6,
        "n_obs": 20,
        "noise_std": 0.1,
        "n_quad": 32,
        "n_grid": 120,
        "odil_steps": 500,
        "odil_lr": 5e-4,
        "odil_phys_weight": 6.0,
    }
    result = run_phase_a_odil(cfg=cfg, output_root=tmp_path, save_outputs=False)

    assert result["status"] in {"completed", "failed", "stopped"}
    assert "metrics" in result
    assert "diagnostics" in result
