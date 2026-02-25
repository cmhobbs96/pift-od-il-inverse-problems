"""Reference finite-difference solvers for 1D validation."""

from __future__ import annotations

import numpy as np


def solve_poisson_dirichlet_fd(
    forcing_fn,
    n_points: int,
    domain: tuple[float, float] = (0.0, 1.0),
    bc: tuple[float, float] = (0.0, 0.0),
) -> tuple[np.ndarray, np.ndarray]:
    """
    Solve -phi''(x) = f(x) on [a, b] with Dirichlet boundaries.

    Uses second-order centered finite differences on a uniform grid.
    Returns (x_grid, phi_fd).
    """
    if n_points < 3:
        raise ValueError("n_points must be >= 3")

    a, b = float(domain[0]), float(domain[1])
    left_bc, right_bc = float(bc[0]), float(bc[1])

    x = np.linspace(a, b, n_points, dtype=float)
    h = x[1] - x[0]

    n_int = n_points - 2
    x_int = x[1:-1]

    # Tridiagonal system for interior points.
    main = np.full(n_int, 2.0 / (h * h), dtype=float)
    off = np.full(max(0, n_int - 1), -1.0 / (h * h), dtype=float)

    A = np.diag(main)
    if n_int > 1:
        A += np.diag(off, k=1)
        A += np.diag(off, k=-1)

    rhs = np.array(forcing_fn(x_int), dtype=float, copy=True)
    rhs[0] += left_bc / (h * h)
    rhs[-1] += right_bc / (h * h)

    phi_int = np.linalg.solve(A, rhs)

    phi = np.empty_like(x)
    phi[0] = left_bc
    phi[-1] = right_bc
    phi[1:-1] = phi_int

    return x, phi
