"""Field parameterizations for 1D examples."""

from __future__ import annotations

import jax.numpy as jnp


class SineBasisField:
    """Sine basis field on [0, 1] satisfying zero Dirichlet boundaries."""

    def __init__(self, n_modes: int) -> None:
        if n_modes < 1:
            raise ValueError("n_modes must be >= 1")
        self.n_modes = int(n_modes)
        self._k = jnp.arange(1, self.n_modes + 1, dtype=jnp.float64)

    def design_matrix(self, x):
        x = jnp.asarray(x, dtype=jnp.float64)
        return jnp.sin(jnp.pi * x[:, None] * self._k[None, :])

    def neg_second_derivative_matrix(self, x):
        basis = self.design_matrix(x)
        weights = (jnp.pi * self._k) ** 2
        return basis * weights[None, :]

    def eval(self, x, theta):
        return self.design_matrix(x) @ jnp.asarray(theta, dtype=jnp.float64)

    def neg_second_derivative(self, x, theta):
        return self.neg_second_derivative_matrix(x) @ jnp.asarray(theta, dtype=jnp.float64)

    def preconditioner(self, beta: float = 1.0):
        """Diagonal preconditioner that normalizes curvature across modes.

        Returns a vector ``p`` of shape ``(n_modes,)`` so that the
        effective step size ``alpha_t * p_k`` is roughly equal for all
        modes.  Uses ``p_k = lambda_1 / lambda_k`` where
        ``lambda_k = (pi*k)^4 / 2`` is the k-th eigenvalue of the
        physics Hessian.
        """
        lam = 0.5 * (jnp.pi * self._k) ** 4
        return lam[0] / lam


class BoundaryFourierField:
    """Fourier basis on [0, 1] with non-zero Dirichlet boundary embedding.

    Parameterization (Alberts & Bilionis Eq. 35-36)::

        phi(x; theta) = (1-x)*phi_0 + x*phi_1 + (1-x)*x*psi(x; theta)

    where ``psi`` is a truncated Fourier series::

        psi(x; theta) = theta_0
            + sum_{j=1}^{K} [ theta_j * cos(2*pi*j*x)
                             + theta_{K+j} * sin(2*pi*j*x) ]

    giving ``2*K + 1`` free parameters.  Boundary conditions are satisfied
    exactly: ``phi(0) = phi_0``, ``phi(1) = phi_1``.
    """

    def __init__(self, K: int = 20, bc: tuple[float, float] = (1.0, 0.1)) -> None:
        if K < 1:
            raise ValueError("K must be >= 1")
        self.K = int(K)
        self.bc = (float(bc[0]), float(bc[1]))
        self._j = jnp.arange(1, self.K + 1, dtype=jnp.float64)

    @property
    def n_params(self) -> int:
        return 2 * self.K + 1

    # ------------------------------------------------------------------
    # Fourier sub-field psi and its derivative
    # ------------------------------------------------------------------

    def _psi(self, x, theta):
        """Evaluate psi(x; theta) = theta_0 + sum cos/sin terms."""
        x = jnp.asarray(x, dtype=jnp.float64)
        theta = jnp.asarray(theta, dtype=jnp.float64)
        j = self._j
        arg = 2.0 * jnp.pi * j[None, :] * x[:, None]  # (N, K)
        cos_part = jnp.cos(arg) @ theta[1 : self.K + 1]
        sin_part = jnp.sin(arg) @ theta[self.K + 1 :]
        return theta[0] + cos_part + sin_part

    def _dpsi_dx(self, x, theta):
        """Evaluate dpsi/dx analytically."""
        x = jnp.asarray(x, dtype=jnp.float64)
        theta = jnp.asarray(theta, dtype=jnp.float64)
        j = self._j
        two_pi_j = 2.0 * jnp.pi * j
        arg = two_pi_j[None, :] * x[:, None]  # (N, K)
        # d/dx cos(2*pi*j*x) = -2*pi*j*sin(2*pi*j*x)
        dcos = -jnp.sin(arg) * two_pi_j[None, :]
        # d/dx sin(2*pi*j*x) = 2*pi*j*cos(2*pi*j*x)
        dsin = jnp.cos(arg) * two_pi_j[None, :]
        return dcos @ theta[1 : self.K + 1] + dsin @ theta[self.K + 1 :]

    # ------------------------------------------------------------------
    # Full field and derivative
    # ------------------------------------------------------------------

    def eval(self, x, theta):
        """Evaluate phi(x; theta) with boundary embedding."""
        x = jnp.asarray(x, dtype=jnp.float64)
        phi0, phi1 = self.bc
        ramp = (1.0 - x) * phi0 + x * phi1
        window = (1.0 - x) * x
        return ramp + window * self._psi(x, theta)

    def deriv(self, x, theta):
        """Evaluate dphi/dx analytically via product rule."""
        x = jnp.asarray(x, dtype=jnp.float64)
        phi0, phi1 = self.bc
        dramp = -phi0 + phi1
        psi = self._psi(x, theta)
        dpsi = self._dpsi_dx(x, theta)
        dwindow = 1.0 - 2.0 * x  # d/dx[(1-x)*x]
        window = (1.0 - x) * x
        return dramp + dwindow * psi + window * dpsi

    def psi_design_matrix(self, x):
        """Design matrix for the psi sub-field (for lstsq initialization).

        Returns shape ``(len(x), 2*K+1)`` such that
        ``psi_design_matrix(x) @ theta == psi(x; theta)``.
        """
        x = jnp.asarray(x, dtype=jnp.float64)
        j = self._j
        arg = 2.0 * jnp.pi * j[None, :] * x[:, None]
        ones = jnp.ones((x.shape[0], 1), dtype=jnp.float64)
        return jnp.concatenate([ones, jnp.cos(arg), jnp.sin(arg)], axis=1)

    def preconditioner(self, beta: float = 1.0):
        """Diagonal preconditioner normalizing curvature across modes.

        The variational Hessian eigenvalues for mode *j* scale as
        ``(2*pi*j)^2``.  The constant mode (j=0) gets the same scale
        as j=1.  Returns ``lam_min / lam_k`` of shape ``(n_params,)``.
        """
        lam_j = (2.0 * jnp.pi * self._j) ** 2
        lam0 = lam_j[0]
        # constant mode, K cosine modes, K sine modes
        p_const = jnp.ones(1, dtype=jnp.float64)
        p_cos = lam0 / lam_j
        p_sin = lam0 / lam_j
        return jnp.concatenate([p_const, p_cos, p_sin])
