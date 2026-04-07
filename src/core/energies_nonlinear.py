"""Nonlinear PDE energy functionals for D*phi'' - kappa*phi^3 = f.

Implements the variational energy from Alberts & Bilionis (2023) Eq. 38:

    U[phi] = integral_0^1 [1/2 * D * (phi')^2 + 1/4 * kappa * phi^4 + phi * f] dx

The stationarity condition delta U / delta phi = 0 recovers the PDE (Appendix C).
Used by Examples 2, 3a, 3b.

Also provides the misspecified energy (Eq. 39) for Example 2 Experiment B:

    U[phi; gamma] = integral [1/2*D*(phi')^2 + gamma/4*kappa*phi^4
                              + 1/2*(1-gamma)*phi^2 + phi*f] dx
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def nonlinear_energy(theta, x_quad, field, source_fn, D=0.1, kappa=1.0):
    """
    Variational energy for the nonlinear PDE  D*phi'' - kappa*phi^3 = f.

    U[phi] = integral_0^1 [1/2*D*(dphi/dx)^2 + 1/4*kappa*phi^4 + phi*f] dx

    approximated by Monte Carlo quadrature over *x_quad*.

    Parameters
    ----------
    theta : array_like
        Field parameters.
    x_quad : array_like
        Quadrature points in [0, 1].
    field : BoundaryFourierField
        Field parameterization providing ``eval`` and ``deriv``.
    source_fn : callable
        Source term f(x).
    D : float
        Diffusion coefficient.
    kappa : float
        Nonlinearity coefficient.

    Returns
    -------
    energy : float
    grad : jax.Array
        Gradient of energy w.r.t. theta.
    None
        Placeholder for API consistency.
    """
    x_quad = jnp.asarray(x_quad, dtype=jnp.float64)
    theta = jnp.asarray(theta, dtype=jnp.float64)

    def _energy(theta_inner):
        phi = field.eval(x_quad, theta_inner)
        dphi = field.deriv(x_quad, theta_inner)
        f = source_fn(x_quad)
        return jnp.mean(0.5 * D * dphi**2 + 0.25 * kappa * phi**4 + phi * f)

    energy, grad = jax.value_and_grad(_energy)(theta)
    return energy, grad, None


def nonlinear_energy_misspecified(
    theta, x_quad, field, source_fn, D=0.1, kappa=1.0, gamma=1.0
):
    """
    Misspecified energy functional (Eq. 39) for Example 2 Experiment B.

    U[phi; gamma] = integral_0^1 [1/2*D*(dphi/dx)^2 + gamma/4*kappa*phi^4
                                  + 1/2*(1-gamma)*phi^2 + phi*f] dx

    When gamma=1 this reduces to the correct energy (Eq. 38).
    When gamma=0 the nonlinear phi^4 term is replaced by a linear phi^2 term.

    Parameters
    ----------
    theta, x_quad, field, source_fn, D, kappa
        Same as ``nonlinear_energy``.
    gamma : float
        Model correctness parameter in [0, 1].

    Returns
    -------
    energy : float
    grad : jax.Array
    None
    """
    x_quad = jnp.asarray(x_quad, dtype=jnp.float64)
    theta = jnp.asarray(theta, dtype=jnp.float64)

    def _energy(theta_inner):
        phi = field.eval(x_quad, theta_inner)
        dphi = field.deriv(x_quad, theta_inner)
        f = source_fn(x_quad)
        return jnp.mean(
            0.5 * D * dphi**2
            + gamma * 0.25 * kappa * phi**4
            + 0.5 * (1.0 - gamma) * phi**2
            + phi * f
        )

    energy, grad = jax.value_and_grad(_energy)(theta)
    return energy, grad, None
