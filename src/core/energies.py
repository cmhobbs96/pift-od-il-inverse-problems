"""Physics energies and gradients."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .parameterizations import BoundaryFourierField, SineBasisField


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
    return energy, grad, residual


def variational_heat_energy(
    theta,
    x_quad,
    field: BoundaryFourierField,
    source_fn,
    D: float = 0.25,
):
    """
    Variational energy for the steady-state heat equation -D phi'' = q.

    U[phi] = integral_0^1 [ 1/2 D (dphi/dx)^2 - q(x) phi(x) ] dx

    approximated by Monte Carlo quadrature over *x_quad*.

    Returns
    -------
    energy : float
    grad : jax.Array
        Gradient of energy wrt theta.
    None
        Placeholder for API consistency with ``poisson_residual_energy``.
    """
    x_quad = jnp.asarray(x_quad, dtype=jnp.float64)
    theta = jnp.asarray(theta, dtype=jnp.float64)

    def _energy(theta_inner):
        phi = field.eval(x_quad, theta_inner)
        dphi = field.deriv(x_quad, theta_inner)
        q = source_fn(x_quad)
        return jnp.mean(0.5 * D * dphi**2 - q * phi)

    energy, grad = jax.value_and_grad(_energy)(theta)
    return energy, grad, None
