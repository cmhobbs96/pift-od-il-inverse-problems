"""2D field parameterizations for Allen-Cahn and other 2D examples.

Provides :class:`FourierBasis2D`, a real-valued 2D Fourier basis on
[-1, 1]^2 constructed as tensor products of 1D functions
{1, cos(pi*j*u), sin(pi*j*u)} for j = 1, ..., max_freq.

With ``max_freq=1`` the basis has (1+2*1)^2 = 9 terms, matching
Example 4 of Alberts & Bilionis (2023).
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np


class FourierBasis2D:
    """Real-valued 2D Fourier basis on [-1, 1]^2.

    The basis is the tensor product of 1D bases in x and y, where each
    1D basis contains: ``{1, cos(pi*x), sin(pi*x), cos(2*pi*x), ...}``
    up to frequency ``max_freq``.

    Parameters
    ----------
    max_freq : int
        Maximum frequency in each dimension.  ``max_freq=1`` gives
        3 functions per dimension → 9 total 2D basis functions.
    """

    def __init__(self, max_freq: int = 1) -> None:
        if max_freq < 1:
            raise ValueError("max_freq must be >= 1")
        self.max_freq = int(max_freq)
        n1d = 1 + 2 * self.max_freq  # terms per dimension
        self._n_params = n1d * n1d

    @property
    def n_params(self) -> int:
        return self._n_params

    # ------------------------------------------------------------------
    # 1D helpers
    # ------------------------------------------------------------------

    def _basis_1d(self, u):
        """Evaluate all 1D basis functions at *u*.

        Returns shape ``(len(u), 1 + 2*max_freq)``.
        Column order: [1, cos(pi*u), sin(pi*u), cos(2*pi*u), sin(2*pi*u), ...]
        """
        u = jnp.asarray(u, dtype=jnp.float64)
        cols = [jnp.ones_like(u)]
        for j in range(1, self.max_freq + 1):
            cols.append(jnp.cos(jnp.pi * j * u))
            cols.append(jnp.sin(jnp.pi * j * u))
        return jnp.stack(cols, axis=-1)  # (N, 1+2*max_freq)

    def _dbasis_1d(self, u):
        """Derivatives of 1D basis functions w.r.t. u.

        Returns shape ``(len(u), 1 + 2*max_freq)``.
        """
        u = jnp.asarray(u, dtype=jnp.float64)
        cols = [jnp.zeros_like(u)]  # d/du [1] = 0
        for j in range(1, self.max_freq + 1):
            cols.append(-jnp.pi * j * jnp.sin(jnp.pi * j * u))  # d/du cos
            cols.append(jnp.pi * j * jnp.cos(jnp.pi * j * u))   # d/du sin
        return jnp.stack(cols, axis=-1)

    # ------------------------------------------------------------------
    # 2D design matrices
    # ------------------------------------------------------------------

    def design_matrix(self, xy):
        """Evaluate 2D basis functions at points *xy*.

        Parameters
        ----------
        xy : array_like, shape (N, 2)
            2D points with columns (x, y).

        Returns
        -------
        Phi : jax.Array, shape (N, n_params)
        """
        xy = jnp.asarray(xy, dtype=jnp.float64)
        bx = self._basis_1d(xy[:, 0])   # (N, n1d)
        by = self._basis_1d(xy[:, 1])   # (N, n1d)
        # Tensor product: Phi_{i, (a*n1d+b)} = bx_{i,a} * by_{i,b}
        return (bx[:, :, None] * by[:, None, :]).reshape(xy.shape[0], -1)

    def design_matrix_dx(self, xy):
        """Derivative of 2D basis w.r.t. x: d(psi_k)/dx.

        Returns shape (N, n_params).
        """
        xy = jnp.asarray(xy, dtype=jnp.float64)
        dbx = self._dbasis_1d(xy[:, 0])
        by = self._basis_1d(xy[:, 1])
        return (dbx[:, :, None] * by[:, None, :]).reshape(xy.shape[0], -1)

    def design_matrix_dy(self, xy):
        """Derivative of 2D basis w.r.t. y: d(psi_k)/dy.

        Returns shape (N, n_params).
        """
        xy = jnp.asarray(xy, dtype=jnp.float64)
        bx = self._basis_1d(xy[:, 0])
        dby = self._dbasis_1d(xy[:, 1])
        return (bx[:, :, None] * dby[:, None, :]).reshape(xy.shape[0], -1)

    # ------------------------------------------------------------------
    # Convenience evaluation
    # ------------------------------------------------------------------

    def eval(self, xy, theta):
        """Evaluate field phi(x, y; theta) = Phi @ theta."""
        return self.design_matrix(xy) @ jnp.asarray(theta, dtype=jnp.float64)

    def grad_x(self, xy, theta):
        """Evaluate d(phi)/dx."""
        return self.design_matrix_dx(xy) @ jnp.asarray(theta, dtype=jnp.float64)

    def grad_y(self, xy, theta):
        """Evaluate d(phi)/dy."""
        return self.design_matrix_dy(xy) @ jnp.asarray(theta, dtype=jnp.float64)

    # ------------------------------------------------------------------
    # Laplacian
    # ------------------------------------------------------------------

    def _ddbasis_1d(self, u):
        """Second derivatives of the 1D basis functions w.r.t. ``u``."""
        u = jnp.asarray(u, dtype=jnp.float64)
        cols = [jnp.zeros_like(u)]  # d²/du² [1] = 0
        for j in range(1, self.max_freq + 1):
            w = jnp.pi * j
            cols.append(-(w**2) * jnp.cos(w * u))
            cols.append(-(w**2) * jnp.sin(w * u))
        return jnp.stack(cols, axis=-1)

    def design_matrix_laplacian(self, xy):
        """Laplacian Δ(psi_k) = ∂²/∂x² + ∂²/∂y² on each 2D basis function.

        Returns shape ``(N, n_params)``.
        """
        xy = jnp.asarray(xy, dtype=jnp.float64)
        bx = self._basis_1d(xy[:, 0])
        by = self._basis_1d(xy[:, 1])
        ddbx = self._ddbasis_1d(xy[:, 0])
        ddby = self._ddbasis_1d(xy[:, 1])
        # Δ(f(x) g(y)) = f''(x) g(y) + f(x) g''(y)
        term1 = ddbx[:, :, None] * by[:, None, :]
        term2 = bx[:, :, None] * ddby[:, None, :]
        return (term1 + term2).reshape(xy.shape[0], -1)

    def laplacian(self, xy, theta):
        """Evaluate Δphi(x, y; theta)."""
        return self.design_matrix_laplacian(xy) @ jnp.asarray(theta, dtype=jnp.float64)


class SineBasis2D:
    """2D sine basis on ``[0, 1]^2`` with zero Dirichlet boundaries.

    Tensor product of 1D sine bases ``{sin(pi*j*x) : j = 1, ..., n_modes}``
    in each dimension, giving ``n_modes ** 2`` total parameters.  Every basis
    function vanishes on the boundary, so any linear combination automatically
    satisfies ``phi = 0`` on ``∂[0, 1]^2``.

    The Laplacian is diagonal in this basis with eigenvalues
    ``-(pi*jx)^2 - (pi*jy)^2``, which makes the 2D Poisson problem
    ``-Δphi = f`` separable.
    """

    def __init__(self, n_modes: int = 8) -> None:
        if n_modes < 1:
            raise ValueError("n_modes must be >= 1")
        self.n_modes = int(n_modes)
        self._k = jnp.arange(1, self.n_modes + 1, dtype=jnp.float64)
        # Diagonal Laplacian eigenvalues (negative): lam_{i,j} = -(pi*i)^2 - (pi*j)^2
        kx = (jnp.pi * self._k) ** 2
        # (n_modes, n_modes), entry (i, j) → kx[i] + kx[j]
        self._neg_lap_eigs = (kx[:, None] + kx[None, :]).reshape(-1)

    @property
    def n_params(self) -> int:
        return int(self.n_modes * self.n_modes)

    # ------------------------------------------------------------------
    # Design matrices
    # ------------------------------------------------------------------

    def _sin_1d(self, u):
        u = jnp.asarray(u, dtype=jnp.float64)
        return jnp.sin(jnp.pi * u[:, None] * self._k[None, :])  # (N, n_modes)

    def design_matrix(self, xy):
        """Evaluate basis functions at points ``xy`` of shape ``(N, 2)``."""
        xy = jnp.asarray(xy, dtype=jnp.float64)
        bx = self._sin_1d(xy[:, 0])
        by = self._sin_1d(xy[:, 1])
        return (bx[:, :, None] * by[:, None, :]).reshape(xy.shape[0], -1)

    def design_matrix_neg_laplacian(self, xy):
        """Design matrix for ``-Δphi``: each column is the eigenvalue-scaled basis."""
        return self.design_matrix(xy) * self._neg_lap_eigs[None, :]

    def eval(self, xy, theta):
        return self.design_matrix(xy) @ jnp.asarray(theta, dtype=jnp.float64)

    def neg_laplacian(self, xy, theta):
        """Evaluate ``-Δphi(xy; theta)``."""
        return self.design_matrix_neg_laplacian(xy) @ jnp.asarray(theta, dtype=jnp.float64)

    def preconditioner(self) -> jnp.ndarray:
        """Diagonal preconditioner that normalizes the Hessian curvature.

        Mirrors :meth:`SineBasisField.preconditioner` for the 2D case: the
        physics Hessian is diagonal with eigenvalues
        ``0.5 * (-Δ)^2`` so the preconditioner is the inverse, normalized to
        the lowest mode.
        """
        lam = 0.5 * self._neg_lap_eigs ** 2
        return lam[0] / lam
