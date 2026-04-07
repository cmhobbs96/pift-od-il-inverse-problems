"""Parameterization helpers and reference solver for the nonlinear PDE.

Provides:
- ``make_nonlinear_field``: factory for BoundaryFourierField with defaults
  matching Examples 2, 3a, 3b (K=20, zero Dirichlet BCs).
- ``solve_nonlinear_dirichlet_fd``: Newton-iteration FD solver for
  D*phi'' - kappa*phi^3 = f used as ground-truth reference.
"""

from __future__ import annotations

import numpy as np

from .parameterizations import BoundaryFourierField


def make_nonlinear_field(K: int = 20, bc: tuple[float, float] = (0.0, 0.0)):
    """Create a BoundaryFourierField configured for the nonlinear examples.

    Parameters
    ----------
    K : int
        Number of Fourier modes (gives 2*K+1 parameters).
    bc : tuple[float, float]
        Dirichlet boundary values (phi(0), phi(1)).
    """
    return BoundaryFourierField(K=K, bc=bc)


def solve_nonlinear_dirichlet_fd(
    source_fn,
    D: float = 0.1,
    kappa: float = 1.0,
    n_points: int = 500,
    bc: tuple[float, float] = (0.0, 0.0),
    max_iter: int = 100,
    tol: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Solve D*phi'' - kappa*phi^3 = f on [0,1] with Dirichlet BCs via Newton.

    The PDE is rewritten as:
        D * phi'' = kappa * phi^3 + f

    Discretized with second-order centered finite differences.

    Parameters
    ----------
    source_fn : callable
        Source term f(x).
    D : float
        Diffusion coefficient.
    kappa : float
        Nonlinearity coefficient.
    n_points : int
        Total grid points including boundaries.
    bc : tuple[float, float]
        Dirichlet values (phi(0), phi(1)).
    max_iter : int
        Maximum Newton iterations.
    tol : float
        Convergence tolerance on residual L-inf norm.

    Returns
    -------
    x : np.ndarray, shape (n_points,)
        Grid points.
    phi : np.ndarray, shape (n_points,)
        Solution field.

    Raises
    ------
    RuntimeError
        If Newton iteration does not converge.
    """
    if n_points < 3:
        raise ValueError("n_points must be >= 3")

    x = np.linspace(0.0, 1.0, n_points, dtype=float)
    h = x[1] - x[0]
    n_int = n_points - 2
    x_int = x[1:-1]

    left_bc, right_bc = float(bc[0]), float(bc[1])

    # FD Laplacian matrix for interior points: phi'' ≈ (phi[i-1] - 2*phi[i] + phi[i+1]) / h^2
    diag_main = np.full(n_int, -2.0 / (h * h), dtype=float)
    diag_off = np.full(max(0, n_int - 1), 1.0 / (h * h), dtype=float)
    A = np.diag(diag_main) + np.diag(diag_off, k=1) + np.diag(diag_off, k=-1)

    f_int = np.asarray(source_fn(x_int), dtype=float)

    # Initialize with zeros
    phi_int = np.zeros(n_int, dtype=float)

    for it in range(max_iter):
        # Residual: D * A @ phi - kappa * phi^3 - f = 0
        residual = D * A @ phi_int - kappa * phi_int**3 - f_int
        # BC contributions to residual
        residual[0] -= D * left_bc / (h * h)
        residual[-1] -= D * right_bc / (h * h)

        if np.max(np.abs(residual)) < tol:
            break

        # Jacobian: D * A - 3 * kappa * diag(phi^2)
        J = D * A - 3.0 * kappa * np.diag(phi_int**2)

        delta = np.linalg.solve(J, -residual)
        phi_int = phi_int + delta
    else:
        raise RuntimeError(
            f"Newton iteration did not converge after {max_iter} iterations "
            f"(residual L-inf = {np.max(np.abs(residual)):.2e})"
        )

    phi = np.empty_like(x)
    phi[0] = left_bc
    phi[-1] = right_bc
    phi[1:-1] = phi_int

    return x, phi
