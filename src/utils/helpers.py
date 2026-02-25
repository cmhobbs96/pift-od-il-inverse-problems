"""Project-specific lightweight containers and normalization helpers."""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
import jax.scipy.special as jspecial


class NormalDistribution(NamedTuple):
    """Univariate Gaussian parameterized by mean and variance."""

    mu: float
    var: float


class Domain1D(NamedTuple):
    """Closed 1D domain [x_min, x_max]."""

    x_min: float
    x_max: float


class ObservationModel(NamedTuple):
    """Observation setup for forward PIFT examples."""

    noise_std: float
    n_obs: int


class PoissonForwardProblem(NamedTuple):
    """Minimal 1D Poisson forward-problem specification."""

    domain: Domain1D
    beta: float
    n_modes: int
    n_quad: int
    observation: ObservationModel


class SGLDConfig(NamedTuple):
    """SGLD hyperparameters for Phase A sampling."""

    n_steps: int
    burn_in: int
    thin: int
    step_size0: float
    decay: float


def normalize_to_unit_interval(x, x_min: float, x_max: float):
    """Map values from [x_min, x_max] to [0, 1]."""
    x = jnp.asarray(x, dtype=jnp.float64)
    return (x - x_min) / (x_max - x_min)


def normalize_to_minus_one_plus_one(x, x_min: float, x_max: float):
    """Map values from [x_min, x_max] to [-1, 1]."""
    u01 = normalize_to_unit_interval(x, x_min, x_max)
    return 2.0 * u01 - 1.0


def denormalize_from_minus_one_plus_one(xi, x_min: float, x_max: float):
    """Map values from [-1, 1] back to [x_min, x_max]."""
    xi = jnp.asarray(xi, dtype=jnp.float64)
    return x_min + 0.5 * (xi + 1.0) * (x_max - x_min)


def standardize(x, mean: float | None = None, std: float | None = None):
    """Standardize x and return (z, mean, std) with numerically safe std."""
    x = jnp.asarray(x, dtype=jnp.float64)
    mu = jnp.mean(x) if mean is None else jnp.asarray(mean, dtype=jnp.float64)
    sigma = jnp.std(x) if std is None else jnp.asarray(std, dtype=jnp.float64)
    sigma = jnp.maximum(sigma, 1e-12)
    z = (x - mu) / sigma
    return z, mu, sigma


def unstandardize(z, mean, std):
    """Inverse of standardize."""
    z = jnp.asarray(z, dtype=jnp.float64)
    mu = jnp.asarray(mean, dtype=jnp.float64)
    sigma = jnp.asarray(std, dtype=jnp.float64)
    return mu + sigma * z


def to_normal(xi, dist: NormalDistribution, eps: float = 1e-7):
    """
    Map xi in [-1, 1] to N(mu, var) using inverse CDF.

    Uses ndtri for numerical stability and clips away from 0/1 to avoid inf values.
    """
    xi = jnp.asarray(xi, dtype=jnp.float64)
    u = 0.5 * (xi + 1.0)
    u = jnp.clip(u, eps, 1.0 - eps)
    z = jspecial.ndtri(u)
    return dist.mu + jnp.sqrt(dist.var) * z
