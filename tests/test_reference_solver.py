from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from core.reference_solver import solve_poisson_dirichlet_fd


def test_poisson_fd_matches_analytic_solution() -> None:
    # Ground-truth field used in Phase A setup.
    def phi_true(x: np.ndarray) -> np.ndarray:
        return np.sin(np.pi * x) + 0.35 * np.sin(2.0 * np.pi * x)

    def forcing(x: np.ndarray) -> np.ndarray:
        return (np.pi**2) * np.sin(np.pi * x) + 0.35 * (2.0 * np.pi) ** 2 * np.sin(2.0 * np.pi * x)

    x, phi_fd = solve_poisson_dirichlet_fd(
        forcing_fn=forcing,
        n_points=301,
        domain=(0.0, 1.0),
        bc=(0.0, 0.0),
    )
    phi = phi_true(x)

    l2 = float(np.sqrt(np.mean((phi_fd - phi) ** 2)))
    max_err = float(np.max(np.abs(phi_fd - phi)))

    assert l2 < 5e-5
    assert max_err < 1e-4


def test_poisson_fd_respects_boundary_conditions() -> None:
    def forcing(x: np.ndarray) -> np.ndarray:
        return np.zeros_like(x)

    x, phi_fd = solve_poisson_dirichlet_fd(forcing_fn=forcing, n_points=41, domain=(0.0, 1.0), bc=(1.0, -2.0))

    assert np.isclose(phi_fd[0], 1.0)
    assert np.isclose(phi_fd[-1], -2.0)
    assert x.shape == phi_fd.shape
