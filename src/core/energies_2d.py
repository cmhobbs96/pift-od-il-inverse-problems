"""2D Allen-Cahn energy functional for Example 4 (Alberts & Bilionis).

PDE (Eq. 40):  epsilon * nabla^2 phi - phi(phi^2 - 1) + f = 0

Energy (Eq. 41):
    U_eps[phi] = integral_Omega [(eps/2)|nabla phi|^2
                                 + (1/4)(1 - phi^2)^2 - f phi] dx

Ground truth: phi(x,y) = 2 exp(-(x^2+y^2)) sin(pi x) sin(pi y)
on domain Omega = [-1, 1]^2 with epsilon = 0.01.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np


def allen_cahn_ground_truth(xy):
    """Ground truth field: 2 exp(-(x^2+y^2)) sin(pi x) sin(pi y).

    Parameters
    ----------
    xy : array_like, shape (N, 2)

    Returns
    -------
    phi : jax.Array, shape (N,)
    """
    xy = jnp.asarray(xy, dtype=jnp.float64)
    x, y = xy[:, 0], xy[:, 1]
    return 2.0 * jnp.exp(-(x**2 + y**2)) * jnp.sin(jnp.pi * x) * jnp.sin(jnp.pi * y)


def allen_cahn_source_term(xy, epsilon=0.01):
    """Source term f such that the ground truth satisfies the PDE.

    From Eq. 40: f = phi(phi^2 - 1) - epsilon * nabla^2 phi.
    Laplacian computed via JAX automatic differentiation.

    Parameters
    ----------
    xy : array_like, shape (N, 2)
    epsilon : float

    Returns
    -------
    f : jax.Array, shape (N,)
    """
    xy = jnp.asarray(xy, dtype=jnp.float64)

    def _phi_point(p):
        x, y = p[0], p[1]
        return 2.0 * jnp.exp(-(x**2 + y**2)) * jnp.sin(jnp.pi * x) * jnp.sin(jnp.pi * y)

    def _laplacian(p):
        H = jax.hessian(_phi_point)(p)
        return H[0, 0] + H[1, 1]

    phi = allen_cahn_ground_truth(xy)
    lap = jax.vmap(_laplacian)(xy)
    return phi * (phi**2 - 1.0) - epsilon * lap


def allen_cahn_energy(theta, xy_quad, field, source_vals, epsilon=0.01):
    """Allen-Cahn variational energy (Eq. 41).

    U[phi] = mean_over_quad [(eps/2)|nabla phi|^2
                             + (1/4)(1 - phi^2)^2 - f phi]

    Parameters
    ----------
    theta : array_like, shape (n_params,)
        Field parameters.
    xy_quad : array_like, shape (N, 2)
        Quadrature points in [-1, 1]^2.
    field : FourierBasis2D
        Field parameterization with ``eval``, ``grad_x``, ``grad_y``.
    source_vals : array_like, shape (N,)
        Precomputed source term values at quadrature points.
    epsilon : float
        Mobility parameter.

    Returns
    -------
    energy : float
    grad : jax.Array, shape (n_params,)
    None
    """
    xy_quad = jnp.asarray(xy_quad, dtype=jnp.float64)
    theta = jnp.asarray(theta, dtype=jnp.float64)
    source_vals = jnp.asarray(source_vals, dtype=jnp.float64)

    def _energy(th):
        phi = field.eval(xy_quad, th)
        dphi_dx = field.grad_x(xy_quad, th)
        dphi_dy = field.grad_y(xy_quad, th)
        grad_sq = dphi_dx**2 + dphi_dy**2
        integrand = (
            0.5 * epsilon * grad_sq
            + 0.25 * (1.0 - phi**2) ** 2
            - source_vals * phi
        )
        return jnp.mean(integrand)

    energy, grad = jax.value_and_grad(_energy)(theta)
    return energy, grad, None
