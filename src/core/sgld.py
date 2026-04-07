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
    preconditioner: jax.Array | np.ndarray | None = None,
    max_condition_number: float = 100.0,
    chunk_size: int = 500,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, object]]:
    """Run SGLD with polynomially decaying step size.

    The inner update is JIT-compiled and executed in chunks via ``jax.lax.scan``
    so the device (CPU or GPU) sees long compiled traces instead of one Python
    iteration per step.  ``stop_signal``, ``runtime_check``, and
    ``progress_callback`` fire **between chunks**, so their effective granularity
    is at most ``chunk_size``.  ``chunk_size`` is automatically clamped down to
    ``runtime_check_interval`` and ``progress_interval`` so callers that need
    finer granularity still get it (at the cost of more host syncs).

    Parameters
    ----------
    preconditioner : array, optional
        Per-parameter diagonal preconditioner (length ``dim``).  When provided
        the update becomes ``theta -= alpha_t * precond * grad`` and the
        injected noise is scaled by ``sqrt(precond)`` so that the stationary
        distribution is preserved.
    max_condition_number : float
        Maximum allowed ratio between the largest and smallest preconditioner
        weights.  Entries are clipped from below at
        ``max(precond) / max_condition_number`` before normalising so that
        the effective per-mode step-size ratio never exceeds this value.
    chunk_size : int
        Number of SGLD steps fused into a single ``lax.scan`` call.  Larger
        values give better device utilisation; smaller values give finer
        callback / runtime-check granularity.
    """
    theta = jnp.asarray(theta0, dtype=jnp.float64)
    dim = int(theta.size)

    if preconditioner is not None:
        precond = jnp.asarray(preconditioner, dtype=jnp.float64)
        floor = jnp.max(precond) / max_condition_number
        precond = jnp.clip(precond, min=floor)
        precond = precond / jnp.max(precond)
        sqrt_precond = jnp.sqrt(precond)
        print(f"[sgld] preconditioner effective condition number: {float(jnp.max(precond) / jnp.min(precond)):.4f}")
        print(f"[sgld] precond min={float(jnp.min(precond)):.6f}  max={float(jnp.max(precond)):.6f}  mean={float(jnp.mean(precond)):.6f}")
    else:
        precond = jnp.ones_like(theta)
        sqrt_precond = jnp.ones_like(theta)

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

    # Clamp chunk size so callbacks/checks fire at the requested cadence.
    effective_chunk = max(1, int(chunk_size))
    if runtime_check is not None:
        effective_chunk = min(effective_chunk, max(1, int(runtime_check_interval)))
    effective_chunk = min(effective_chunk, max(1, int(progress_interval)))
    effective_chunk = min(effective_chunk, n_steps)

    def _call_grad(theta_c, sub_key):
        try:
            return grad_and_metrics_fn(theta_c, sub_key)
        except TypeError:
            return grad_and_metrics_fn(theta_c)

    def _step(carry, inputs):
        theta_c, key_c = carry
        alpha_t = inputs
        key_c, grad_key, noise_key = jax.random.split(key_c, 3)
        grad, metrics = _call_grad(theta_c, grad_key)
        noise = jax.random.normal(noise_key, shape=theta_c.shape, dtype=theta_c.dtype)
        theta_new = (
            theta_c
            - alpha_t * precond * grad
            + jnp.sqrt(2.0 * alpha_t) * sqrt_precond * noise
        )
        return (theta_new, key_c), (theta_new, grad, metrics)

    @jax.jit
    def _run_chunk(theta_in, key_in, alphas):
        return jax.lax.scan(_step, (theta_in, key_in), alphas)

    t = 0
    while t < n_steps:
        if stop_signal is not None and stop_signal():
            meta["stopped"] = True
            meta["stop_step"] = t
            break

        cs = min(effective_chunk, n_steps - t)
        ts = jnp.arange(t, t + cs, dtype=jnp.float64)
        alphas = step_size0 / ((1.0 + ts) ** decay)

        (theta, run_key), (theta_hist, grad_hist, metrics_hist) = _run_chunk(
            theta, run_key, alphas
        )

        chain[t : t + cs] = np.asarray(theta_hist)
        energy_trace[t : t + cs] = np.asarray(metrics_hist["hamiltonian"])
        like_trace[t : t + cs] = np.asarray(metrics_hist["likelihood"])
        phys_trace[t : t + cs] = np.asarray(metrics_hist["physics"])

        t_end = t + cs

        if runtime_check is not None and (
            t_end % max(1, runtime_check_interval) == 0 or t_end == n_steps
        ):
            last_metrics = {
                "hamiltonian": float(energy_trace[t_end - 1]),
                "likelihood": float(like_trace[t_end - 1]),
                "physics": float(phys_trace[t_end - 1]),
            }
            should_stop, err_code, err_detail = runtime_check(
                t_end,
                np.asarray(theta),
                np.asarray(grad_hist[-1]),
                last_metrics,
            )
            if should_stop:
                meta["stopped"] = True
                meta["stop_step"] = t_end
                meta["error_code"] = err_code
                meta["error_detail"] = err_detail
                t = t_end
                break

        if progress_callback is not None and (
            t_end % progress_interval == 0 or t_end == n_steps
        ):
            progress_callback(t_end, n_steps)

        t = t_end

    return chain, {
        "hamiltonian": energy_trace,
        "likelihood": like_trace,
        "physics": phys_trace,
    }, meta
