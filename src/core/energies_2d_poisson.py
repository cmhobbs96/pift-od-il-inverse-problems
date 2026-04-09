"""2D Poisson residual energy for the forward PIFT problem.

Mirror of :func:`core.energies.poisson_residual_energy` in 2D: the physics
prior is the squared residual of ``-Δphi - f`` averaged over the spatial
domain, evaluated by stochastic quadrature on a batch of sample points.

The energy and gradient are JIT-compatible (pure ``jax.numpy``) so the
caller can wrap them in ``jax.default_device(gpu)`` for GPU execution.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def poisson_2d_residual_energy(theta, xy_quad, field, forcing_fn):
    """Variational physics energy ``U[phi] = 0.5 * E[(-Δphi - f)^2]``.

    Parameters
    ----------
    theta : array
        Basis coefficients.
    xy_quad : array, shape ``(N, 2)``
        Stochastic quadrature points (interior of the unit square).
    field : :class:`SineBasis2D` or any object exposing
        ``design_matrix_neg_laplacian(xy)`` returning ``(N, n_params)``.
    forcing_fn : callable
        Maps ``(N, 2)`` points to ``(N,)`` forcing values.  Should be
        ``jax.numpy``-compatible to stay on device.

    Returns
    -------
    energy, grad, residual
        Scalar energy, gradient w.r.t. ``theta`` (shape ``(n_params,)``), and
        the per-point residual vector (shape ``(N,)``).  The third return
        slot keeps the signature consistent with
        :func:`core.energies.poisson_residual_energy`.
    """
    theta = jnp.asarray(theta, dtype=jnp.float64)

    op = field.design_matrix_neg_laplacian(xy_quad)  # (N, n_params)
    f_vals = jnp.asarray(forcing_fn(xy_quad), dtype=jnp.float64)
    residual = op @ theta - f_vals  # (N,)
    energy = 0.5 * jnp.mean(residual**2)

    # ∂U/∂theta = (1/N) op^T residual
    n = residual.shape[0]
    grad = (op.T @ residual) / n
    return energy, grad, residual


# Convenience JIT wrapper used by the runner.
poisson_2d_residual_energy_jit = jax.jit(
    poisson_2d_residual_energy, static_argnames=("field", "forcing_fn")
)
