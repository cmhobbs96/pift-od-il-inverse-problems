from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from utils.diagnostics_runtime import RuntimeGuard, RuntimeGuardConfig


def test_runtime_guard_detects_non_finite() -> None:
    guard = RuntimeGuard(RuntimeGuardConfig())
    should_stop, code, _ = guard.check(
        1,
        theta=np.array([0.0, np.nan]),
        grad=np.array([1.0, 2.0]),
        metrics={"hamiltonian": 1.0},
    )
    assert should_stop
    assert code == "non_finite"


def test_runtime_guard_detects_hamiltonian_overflow() -> None:
    guard = RuntimeGuard(RuntimeGuardConfig(max_abs_hamiltonian=10.0))
    should_stop, code, _ = guard.check(
        2,
        theta=np.array([0.0]),
        grad=np.array([0.1]),
        metrics={"hamiltonian": 20.0},
    )
    assert should_stop
    assert code == "hamiltonian_overflow"
