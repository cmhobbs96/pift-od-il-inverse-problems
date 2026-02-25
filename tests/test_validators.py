from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from utils.validators import validate_observations, validate_phase_a_config


def test_validate_phase_a_config_good() -> None:
    cfg = {
        "n_steps": 100,
        "burn_in": 20,
        "thin": 2,
        "noise_std": 0.1,
        "step_size0": 1e-3,
        "beta": 1.0,
        "n_quad": 16,
        "n_modes": 8,
    }
    assert validate_phase_a_config(cfg) == []


def test_validate_phase_a_config_bad() -> None:
    cfg = {
        "n_steps": 10,
        "burn_in": 20,
        "thin": 0,
        "noise_std": -1.0,
        "step_size0": 1.0,
        "beta": 0.0,
        "n_quad": 1,
        "n_modes": 0,
    }
    errs = validate_phase_a_config(cfg)
    assert len(errs) >= 5


def test_validate_observations_bad_shape() -> None:
    x = np.array([0.1, 0.2, 0.3])
    y = np.array([0.1, 0.2])
    errs = validate_observations(x, y)
    assert errs
