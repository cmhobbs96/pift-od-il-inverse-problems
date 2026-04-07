"""Karhunen-Loève Expansion via Nyström approximation.

Provides a truncated KLE for representing random source terms with a
squared-exponential covariance kernel::

    C(x, x') = exp(-(x - x')^2 / (2 * lengthscale^2))

Used by Example 3b to parameterise the unknown source ``f(x)`` as::

    f(x; z) = sum_i  z_i * sqrt(lambda_i) * phi_i(x)

where ``(lambda_i, phi_i)`` are the KLE eigenvalue/eigenfunction pairs and
``z_i ~ N(0, 1)`` are the latent coefficients to be inferred.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np


class KLE:
    """Truncated KLE for a 1D squared-exponential covariance kernel.

    Parameters
    ----------
    lengthscale : float
        Correlation length of the kernel (default 0.3).
    n_terms : int
        Number of KLE modes to retain (default 10).
    n_grid : int
        Number of Nyström quadrature nodes on [0, 1] (default 200).
    """

    def __init__(
        self,
        lengthscale: float = 0.3,
        n_terms: int = 10,
        n_grid: int = 200,
    ) -> None:
        self.lengthscale = float(lengthscale)
        self.n_terms = int(n_terms)
        self.n_grid = int(n_grid)

        # Nyström grid
        x_grid = np.linspace(0.0, 1.0, n_grid)
        h = 1.0 / (n_grid - 1)  # quadrature weight (trapezoidal)

        # Kernel matrix
        diff = x_grid[:, None] - x_grid[None, :]
        K = np.exp(-diff**2 / (2.0 * lengthscale**2))

        # Weighted eigenvalue problem: h * K @ v = lambda * v
        # Equivalent to eigendecomposing h * K
        eigvals, eigvecs = np.linalg.eigh(h * K)

        # Sort descending
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx[:n_terms]]
        eigvecs = eigvecs[:, idx[:n_terms]]

        # Store
        self._x_grid = x_grid          # (n_grid,)
        self._h = h
        self._eigvals = eigvals         # (n_terms,)  — KLE eigenvalues
        self._eigvecs = eigvecs         # (n_grid, n_terms) — discretised eigenfunctions
        self._sqrt_eigvals = np.sqrt(np.maximum(eigvals, 0.0))

        # Pre-compute normalised eigenfunctions on the Nyström grid.
        # Nyström eigenfunctions: phi_i(x_j) = v_ji / sqrt(h)
        # (so that integral of phi_i^2 ≈ h * sum phi_i(x_j)^2 = 1)
        self._phi_grid = eigvecs / np.sqrt(h)  # (n_grid, n_terms)

    @property
    def eigenvalues(self) -> np.ndarray:
        """KLE eigenvalues λ_i, shape ``(n_terms,)``."""
        return self._eigvals.copy()

    def eigenfunctions(self, x) -> jnp.ndarray:
        """Evaluate KLE eigenfunctions φ_i(x) via Nyström interpolation.

        Parameters
        ----------
        x : array_like, shape (N,)
            Evaluation points in [0, 1].

        Returns
        -------
        phi : jnp.ndarray, shape (N, n_terms)
        """
        x = jnp.asarray(x, dtype=jnp.float64)
        x_grid = jnp.asarray(self._x_grid, dtype=jnp.float64)
        ls = self.lengthscale

        # K(x, x_grid) — shape (N, n_grid)
        diff = x[:, None] - x_grid[None, :]
        Kxg = jnp.exp(-diff**2 / (2.0 * ls**2))

        # Nyström extension: phi_i(x) = (1 / lambda_i) * h * sum_j K(x, x_j) * v_ji
        # = (1 / lambda_i) * (h * Kxg) @ v_i
        # But we stored eigvals = h * raw_eigvals, and eigvecs from h*K,
        # so: phi_i(x) = (1 / eigval_i) * (h * Kxg) @ (eigvec_i)
        # Simplify: phi_i(x) = (h / eigval_i) * Kxg @ eigvec_i
        eigvecs = jnp.asarray(self._eigvecs, dtype=jnp.float64)
        eigvals = jnp.asarray(self._eigvals, dtype=jnp.float64)

        # (N, n_grid) @ (n_grid, n_terms) -> (N, n_terms)
        raw = self._h * Kxg @ eigvecs
        # Divide by eigenvalues and normalise
        phi = raw / eigvals[None, :]
        # Normalise like the grid eigenfunctions: divide by sqrt(h) already
        # accounted for in the Nyström formula
        return phi / jnp.sqrt(self._h)

    def reconstruct(self, x, z) -> jnp.ndarray:
        """Reconstruct ``f(x; z) = Σ z_i √λ_i φ_i(x)``.

        Parameters
        ----------
        x : array_like, shape (N,)
        z : array_like, shape (n_terms,)

        Returns
        -------
        f : jnp.ndarray, shape (N,)
        """
        z = jnp.asarray(z, dtype=jnp.float64)
        phi = self.eigenfunctions(x)  # (N, n_terms)
        sqrt_lam = jnp.asarray(self._sqrt_eigvals, dtype=jnp.float64)
        return phi @ (z * sqrt_lam)

    def energy_fraction(self) -> float:
        """Fraction of total covariance energy captured by retained modes."""
        # Total energy = trace of kernel = integral C(x,x) dx = 1.0
        # (since C(x,x)=1 and domain is [0,1])
        # Captured = sum of retained eigenvalues
        return float(np.sum(self._eigvals))
