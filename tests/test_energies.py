from __future__ import annotations

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from core.energies import poisson_residual_energy
from core.parameterizations import SineBasisField


def test_poisson_residual_energy_gradient_matches_finite_diff() -> None:
    rng = np.random.default_rng(1)
    field = SineBasisField(6)
    theta = jnp.asarray(rng.normal(size=6))
    x = jnp.asarray(rng.uniform(0.0, 1.0, size=120))

    def forcing(z):
        return jnp.sin(jnp.pi * z)

    _, grad, _ = poisson_residual_energy(theta, x, field, forcing)

    eps = 1e-6
    fd = np.zeros(theta.shape[0], dtype=float)
    theta_np = np.asarray(theta)
    for i in range(theta.shape[0]):
        d = np.zeros_like(theta_np)
        d[i] = eps
        e_plus, _, _ = poisson_residual_energy(theta_np + d, x, field, forcing)
        e_minus, _, _ = poisson_residual_energy(theta_np - d, x, field, forcing)
        fd[i] = (e_plus - e_minus) / (2.0 * eps)

    assert np.allclose(np.asarray(grad), fd, atol=1e-4, rtol=1e-4)
