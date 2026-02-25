from __future__ import annotations

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from utils.helpers import (
    Domain1D,
    NormalDistribution,
    ObservationModel,
    PoissonForwardProblem,
    SGLDConfig,
    denormalize_from_minus_one_plus_one,
    normalize_to_minus_one_plus_one,
    standardize,
    to_normal,
    unstandardize,
)


def test_namedtuple_configs_construct() -> None:
    problem = PoissonForwardProblem(
        domain=Domain1D(0.0, 1.0),
        beta=12.0,
        n_modes=12,
        n_quad=96,
        observation=ObservationModel(noise_std=0.08, n_obs=28),
    )
    sgld = SGLDConfig(n_steps=1000, burn_in=200, thin=5, step_size0=1e-3, decay=0.55)
    assert problem.domain.x_min == 0.0
    assert sgld.n_steps == 1000


def test_minus_one_plus_one_normalization_round_trip() -> None:
    x = jnp.array([0.0, 0.25, 0.75, 1.0])
    xi = normalize_to_minus_one_plus_one(x, 0.0, 1.0)
    x_back = denormalize_from_minus_one_plus_one(xi, 0.0, 1.0)
    assert np.allclose(np.asarray(x), np.asarray(x_back), atol=1e-12)


def test_standardize_round_trip() -> None:
    x = jnp.array([1.0, 2.0, 4.0, 8.0])
    z, mu, sigma = standardize(x)
    x_back = unstandardize(z, mu, sigma)
    assert np.allclose(np.asarray(x), np.asarray(x_back), atol=1e-12)


def test_to_normal_zero_maps_to_mean() -> None:
    dist = NormalDistribution(mu=2.5, var=0.3)
    mapped = to_normal(jnp.array([0.0]), dist)
    assert np.allclose(np.asarray(mapped), np.array([dist.mu]), atol=1e-12)
