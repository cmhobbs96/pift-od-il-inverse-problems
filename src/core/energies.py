"""Physics energies and gradients."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .parameterizations import SineBasisField


def poisson_residual_energy(theta, x_quad, field: SineBasisField, forcing_fn):
    """
    Integrated squared residual energy for -phi''(x) = f(x).

    Returns
    -------
    energy : float
        0.5 * mean(residual^2)
    grad : jax.Array
        Gradient of energy wrt theta.
    residual : jax.Array
        Residual values at quadrature points.
    """
    x_quad = jnp.asarray(x_quad, dtype=jnp.float64)
    theta = jnp.asarray(theta, dtype=jnp.float64)
    op = field.neg_second_derivative_matrix(x_quad)

    def _energy(theta_inner):
        residual_inner = op @ theta_inner - forcing_fn(x_quad)
        return 0.5 * jnp.mean(residual_inner**2)

    energy, grad = jax.value_and_grad(_energy)(theta)
    residual = op @ theta - forcing_fn(x_quad)
    return float(energy), grad, residual
