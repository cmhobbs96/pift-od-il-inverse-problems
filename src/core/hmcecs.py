"""NumPyro NUTS / HMCECS inference wrapper for PIFT posteriors.

Provides :func:`run_nuts_inference` which defines a NumPyro model
combining a physics energy prior with Gaussian observation likelihood,
then samples with NUTS (No-U-Turn Sampler).

Used by Example 4 (Allen-Cahn bimodal posterior) where SGLD is
replaced by HMC for more reliable exploration of multimodal targets.
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS


def run_nuts_inference(
    energy_fn: Callable[[jnp.ndarray], float],
    field_eval_fn: Callable[[jnp.ndarray], jnp.ndarray],
    xy_obs: jnp.ndarray,
    y_obs: jnp.ndarray,
    noise_std: float,
    beta: float,
    n_params: int,
    key: jax.Array,
    *,
    num_warmup: int = 500,
    num_samples: int = 1000,
    num_chains: int = 1,
    init_params: np.ndarray | None = None,
    prior_std: float = 10.0,
    target_accept_prob: float = 0.8,
    max_tree_depth: int = 10,
    progress_bar: bool = True,
) -> dict[str, np.ndarray]:
    """Run NumPyro NUTS on a PIFT posterior.

    Parameters
    ----------
    energy_fn : callable
        ``theta -> scalar`` physics energy (will be multiplied by -beta
        as a log-prior contribution).
    field_eval_fn : callable
        ``theta -> phi_at_obs`` evaluates the field at observation points.
    xy_obs : (n_obs, 2) observation coordinates.
    y_obs : (n_obs,) observed values.
    noise_std : observation noise standard deviation.
    beta : inverse temperature weighting the physics prior.
    n_params : number of field parameters.
    key : JAX PRNG key.
    num_warmup : NUTS warm-up steps.
    num_samples : number of posterior samples.
    num_chains : number of independent chains.
    init_params : optional initial parameter vector(s).
    prior_std : std of the (weak) Gaussian base prior on theta.
    target_accept_prob : NUTS target acceptance probability.
    max_tree_depth : NUTS maximum tree depth.
    progress_bar : show NumPyro progress bar.

    Returns
    -------
    samples : dict
        ``{"theta": np.ndarray}`` with shape
        ``(num_samples * num_chains, n_params)``.
    """
    xy_obs = jnp.asarray(xy_obs, dtype=jnp.float64)
    y_obs = jnp.asarray(y_obs, dtype=jnp.float64)

    def model():
        theta = numpyro.sample(
            "theta",
            dist.Normal(jnp.zeros(n_params), prior_std),
        )
        # Physics prior via factor
        e = energy_fn(theta)
        numpyro.factor("physics", -beta * e)
        # Gaussian likelihood
        phi_obs = field_eval_fn(theta)
        numpyro.sample("obs", dist.Normal(phi_obs, noise_std), obs=y_obs)

    kernel = NUTS(
        model,
        target_accept_prob=target_accept_prob,
        max_tree_depth=max_tree_depth,
    )

    mcmc = MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        progress_bar=progress_bar,
    )

    init_kw = {}
    if init_params is not None:
        init_kw["init_params"] = {"theta": jnp.asarray(init_params, dtype=jnp.float64)}

    mcmc.run(key, **init_kw)

    theta_samples = np.asarray(mcmc.get_samples()["theta"])
    return {"theta": theta_samples}
