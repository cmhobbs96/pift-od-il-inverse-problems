"""Likelihood terms and gradients."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def gaussian_nll(theta, obs_matrix, y_obs, noise_std: float):
    """
    Gaussian negative log-likelihood up to additive constants.

    Omega(theta) = 0.5 / sigma^2 * ||y - R phi||^2
    """
    theta = jnp.asarray(theta, dtype=jnp.float64)
    obs_matrix = jnp.asarray(obs_matrix, dtype=jnp.float64)
    y_obs = jnp.asarray(y_obs, dtype=jnp.float64)
    inv_var = 1.0 / (noise_std**2)

    def _nll(theta_inner):
        pred_inner = obs_matrix @ theta_inner
        err_inner = pred_inner - y_obs
        return 0.5 * inv_var * jnp.sum(err_inner**2)

    nll, grad = jax.value_and_grad(_nll)(theta)
    pred = obs_matrix @ theta
    return float(nll), grad, pred
