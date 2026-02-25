from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pipelines.phase_a import _load_observations_csv


def test_load_observations_csv_with_header(tmp_path: Path) -> None:
    p = tmp_path / "obs.csv"
    p.write_text("x,y\n0.1,1.0\n0.2,2.0\n", encoding="utf-8")
    x, y = _load_observations_csv(p)
    assert np.allclose(x, np.array([0.1, 0.2]))
    assert np.allclose(y, np.array([1.0, 2.0]))
