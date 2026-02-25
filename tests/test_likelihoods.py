from __future__ import annotations

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from core.likelihoods import gaussian_nll


def test_gaussian_nll_gradient_matches_finite_diff() -> None:
    rng = np.random.default_rng(2)
    obs = jnp.asarray(rng.normal(size=(25, 5)))
    theta = jnp.asarray(rng.normal(size=5))
    y = jnp.asarray(rng.normal(size=25))
    sigma = 0.2

    _, grad, _ = gaussian_nll(theta, obs, y, sigma)

    eps = 1e-6
    theta_np = np.asarray(theta)
    fd = np.zeros_like(theta_np)
    for i in range(theta.shape[0]):
        d = np.zeros_like(theta_np)
        d[i] = eps
        e_plus, _, _ = gaussian_nll(theta_np + d, obs, y, sigma)
        e_minus, _, _ = gaussian_nll(theta_np - d, obs, y, sigma)
        fd[i] = (e_plus - e_minus) / (2.0 * eps)

    assert np.allclose(np.asarray(grad), fd, atol=1e-5, rtol=1e-5)
