"""Stochastic gradient Langevin dynamics sampler."""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np


def sgld_sample(
    theta0,
    grad_and_metrics_fn,
    n_steps: int,
    step_size0: float,
    decay: float,
    key: jax.Array,
    progress_callback: Callable[[int, int], None] | None = None,
    progress_interval: int = 100,
    stop_signal: Callable[[], bool] | None = None,
    runtime_check: Callable[[int, np.ndarray, np.ndarray, dict[str, float]], tuple[bool, str | None, str | None]]
    | None = None,
    runtime_check_interval: int = 1,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, object]]:
    """Run SGLD with polynomially decaying step size."""
    theta = jnp.asarray(theta0, dtype=jnp.float64)
    dim = int(theta.size)

    chain = np.zeros((n_steps, dim), dtype=float)
    energy_trace = np.zeros(n_steps, dtype=float)
    like_trace = np.zeros(n_steps, dtype=float)
    phys_trace = np.zeros(n_steps, dtype=float)

    run_key = key
    meta: dict[str, object] = {
        "stopped": False,
        "stop_step": None,
        "error_code": None,
        "error_detail": None,
    }

    if progress_callback is not None:
        progress_callback(0, n_steps)

    for t in range(n_steps):
        if stop_signal is not None and stop_signal():
            meta["stopped"] = True
            meta["stop_step"] = t
            break

        alpha_t = step_size0 / ((1.0 + t) ** decay)
        grad, metrics = grad_and_metrics_fn(theta)
        run_key, noise_key = jax.random.split(run_key)
        noise = jax.random.normal(noise_key, shape=theta.shape, dtype=theta.dtype)
        theta = theta - alpha_t * grad + jnp.sqrt(2.0 * alpha_t) * noise

        chain[t] = np.asarray(theta)
        energy_trace[t] = float(metrics["hamiltonian"])
        like_trace[t] = float(metrics["likelihood"])
        phys_trace[t] = float(metrics["physics"])

        if runtime_check is not None and ((t + 1) % max(1, runtime_check_interval) == 0 or (t + 1) == n_steps):
            should_stop, err_code, err_detail = runtime_check(
                t + 1, np.asarray(theta), np.asarray(grad), metrics
            )
            if should_stop:
                meta["stopped"] = True
                meta["stop_step"] = t + 1
                meta["error_code"] = err_code
                meta["error_detail"] = err_detail
                break

        if progress_callback is not None and ((t + 1) % progress_interval == 0 or (t + 1) == n_steps):
            progress_callback(t + 1, n_steps)

    return chain, {
        "hamiltonian": energy_trace,
        "likelihood": like_trace,
        "physics": phys_trace,
    }, meta
