"""Nested SGLD for PIFT inverse problems (Algorithm 3, Alberts & Bilionis).

Samples from the marginal posterior p(λ|d) of PDE hyperparameters by
maintaining two inner field samplers (prior and posterior) whose gradient
estimates drive an outer SGLD loop on λ.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np


@dataclasses.dataclass(frozen=True)
class NestedSGLDResult:
    """Result container for :func:`nested_sgld`.

    Attributes
    ----------
    lambda_chain : (outer_steps, n_lambda) array of hyperparameter samples.
    phi_prior_final : last prior field state (for restart).
    phi_posterior_final : last posterior field state (for restart).
    traces : per-outer-step diagnostic arrays.
    meta : run metadata (stopped, stop_step, error_code, error_detail).
    """

    lambda_chain: np.ndarray
    phi_prior_final: np.ndarray
    phi_posterior_final: np.ndarray
    phi_prior_snapshots: np.ndarray | None
    phi_posterior_snapshots: np.ndarray | None
    snapshot_steps: np.ndarray | None
    traces: dict[str, np.ndarray]
    meta: dict[str, object]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _prepare_preconditioner(
    raw: jax.Array | np.ndarray | None,
    max_condition_number: float,
) -> tuple[jax.Array | None, jax.Array | None]:
    """Clip and normalise a diagonal preconditioner.

    Returns ``(precond, sqrt_precond)`` or ``(None, None)``.
    """
    if raw is None:
        return None, None
    p = jnp.asarray(raw, dtype=jnp.float64)
    floor = jnp.max(p) / max_condition_number
    p = jnp.clip(p, min=floor)
    p = p / jnp.max(p)
    return p, jnp.sqrt(p)


def _sgld_step(
    theta: jax.Array,
    grad: jax.Array,
    alpha_t: float,
    noise_key: jax.Array,
    precond: jax.Array | None,
    sqrt_precond: jax.Array | None,
) -> jax.Array:
    """Single SGLD update: theta <- theta - alpha*[P*]grad + sqrt(2*alpha)*[sqrtP*]noise."""
    noise = jax.random.normal(noise_key, shape=theta.shape, dtype=theta.dtype)
    if precond is not None:
        return theta - alpha_t * precond * grad + jnp.sqrt(2.0 * alpha_t) * sqrt_precond * noise
    return theta - alpha_t * grad + jnp.sqrt(2.0 * alpha_t) * noise


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def nested_sgld(
    lambda0: jax.Array,
    phi0: jax.Array,
    prior_grad_fn: Callable,
    posterior_grad_fn: Callable,
    key: jax.Array,
    *,
    lambda_prior_grad_fn: Callable | None = None,
    outer_steps: int,
    outer_step_size0: float = 0.1,
    outer_decay: float = 0.51,
    inner_T_prior: int = 10,
    inner_T_posterior: int = 1,
    inner_step_size0: float = 0.1,
    inner_decay: float = 0.51,
    warmup_steps: int = 1_000_000,
    phi_preconditioner: jax.Array | np.ndarray | None = None,
    lambda_preconditioner: jax.Array | np.ndarray | None = None,
    max_condition_number: float = 100.0,
    progress_callback: Callable[[int, int], None] | None = None,
    progress_interval: int = 1000,
    stop_signal: Callable[[], bool] | None = None,
    snapshot_interval: int = 0,
) -> NestedSGLDResult:
    """Nested SGLD for PIFT inverse problems (Algorithm 3).

    Parameters
    ----------
    lambda0 : initial hyperparameters (n_lambda,).
    phi0 : initial field parameters (n_phi,).
    prior_grad_fn :
        ``(phi, lam, key) -> (grad_phi, grad_lam_H, metrics_dict)``
        Gradient of physics-only Hamiltonian.
    posterior_grad_fn :
        ``(phi, lam, key) -> (grad_phi, grad_lam_H, metrics_dict)``
        Gradient of physics+data Hamiltonian.
    key : JAX PRNG key.
    lambda_prior_grad_fn :
        ``(lam) -> grad_lam``.  ``None`` means Jeffrey's prior (zero gradient).
    outer_steps : number of outer λ-update iterations.
    outer_step_size0, outer_decay : polynomial schedule for outer loop.
    inner_T_prior : inner prior SGLD steps per outer iteration (default 10).
    inner_T_posterior : inner posterior SGLD steps per outer iteration (default 1).
    inner_step_size0, inner_decay : polynomial schedule for inner loops.
    warmup_steps : inner SGLD steps with λ fixed before outer loop starts.
    phi_preconditioner : diagonal preconditioner for field updates.
    lambda_preconditioner : diagonal preconditioner for λ updates.
    max_condition_number : condition number cap for preconditioners.
    progress_callback : ``(current_step, total_steps) -> None``.
    progress_interval : report progress every N steps.
    stop_signal : ``() -> bool``; return True to abort.
    snapshot_interval : save inner field states every N outer steps (0 = off).

    Returns
    -------
    NestedSGLDResult
    """
    lam = jnp.asarray(lambda0, dtype=jnp.float64)
    phi_prior = jnp.asarray(phi0, dtype=jnp.float64)
    phi_posterior = jnp.asarray(phi0, dtype=jnp.float64)
    n_lambda = int(lam.size)

    phi_p, phi_sp = _prepare_preconditioner(phi_preconditioner, max_condition_number)
    lam_p, lam_sp = _prepare_preconditioner(lambda_preconditioner, max_condition_number)

    meta: dict[str, object] = {
        "stopped": False,
        "stop_step": None,
        "error_code": None,
        "error_detail": None,
    }

    total_steps = warmup_steps + outer_steps
    inner_step_prior = 0
    inner_step_posterior = 0

    # ------------------------------------------------------------------
    # Warm-up: equilibrate fields with λ fixed
    # ------------------------------------------------------------------
    print(f"[nested_sgld] Warming up: {warmup_steps} steps (λ fixed)")
    for w in range(warmup_steps):
        if stop_signal is not None and stop_signal():
            meta["stopped"] = True
            meta["stop_step"] = w
            return NestedSGLDResult(
                lambda_chain=np.zeros((0, n_lambda)),
                phi_prior_final=np.asarray(phi_prior),
                phi_posterior_final=np.asarray(phi_posterior),
                phi_prior_snapshots=None,
                phi_posterior_snapshots=None,
                snapshot_steps=None,
                traces={},
                meta=meta,
            )

        # Prior field step
        alpha_pr = inner_step_size0 / (1.0 + inner_step_prior) ** inner_decay
        key, k1, k2 = jax.random.split(key, 3)
        grad_phi_pr, _, _ = prior_grad_fn(phi_prior, lam, k1)
        phi_prior = _sgld_step(phi_prior, grad_phi_pr, alpha_pr, k2, phi_p, phi_sp)
        inner_step_prior += 1

        # Posterior field step
        alpha_po = inner_step_size0 / (1.0 + inner_step_posterior) ** inner_decay
        key, k3, k4 = jax.random.split(key, 3)
        grad_phi_po, _, _ = posterior_grad_fn(phi_posterior, lam, k3)
        phi_posterior = _sgld_step(phi_posterior, grad_phi_po, alpha_po, k4, phi_p, phi_sp)
        inner_step_posterior += 1

        if progress_callback and (w + 1) % progress_interval == 0:
            progress_callback(w + 1, total_steps)

    print("[nested_sgld] Warm-up complete, starting outer loop")

    # ------------------------------------------------------------------
    # Allocate output arrays
    # ------------------------------------------------------------------
    n_phi = int(phi_prior.size)
    lambda_chain = np.zeros((outer_steps, n_lambda), dtype=float)

    # Field snapshots for predictive plots
    _snap = snapshot_interval > 0
    snap_prior_list: list[np.ndarray] = []
    snap_posterior_list: list[np.ndarray] = []
    snap_steps_list: list[int] = []

    traces: dict[str, np.ndarray] = {
        "grad_lam_posterior": np.zeros((outer_steps, n_lambda), dtype=float),
        "grad_lam_prior": np.zeros((outer_steps, n_lambda), dtype=float),
        "grad_lam_total": np.zeros((outer_steps, n_lambda), dtype=float),
        "outer_step_size": np.zeros(outer_steps, dtype=float),
        "prior_hamiltonian": np.zeros(outer_steps, dtype=float),
        "posterior_hamiltonian": np.zeros(outer_steps, dtype=float),
    }

    # ------------------------------------------------------------------
    # Outer loop
    # ------------------------------------------------------------------
    for t in range(outer_steps):
        if stop_signal is not None and stop_signal():
            meta["stopped"] = True
            meta["stop_step"] = t
            break

        # --- Inner prior SGLD steps ---
        grad_lam_pr = jnp.zeros_like(lam)
        prior_ham = 0.0
        for _ in range(inner_T_prior):
            alpha_s = inner_step_size0 / (1.0 + inner_step_prior) ** inner_decay
            key, k1, k2 = jax.random.split(key, 3)
            g_phi, g_lam, m = prior_grad_fn(phi_prior, lam, k1)
            phi_prior = _sgld_step(phi_prior, g_phi, alpha_s, k2, phi_p, phi_sp)
            inner_step_prior += 1
            grad_lam_pr = g_lam
            prior_ham = float(m.get("hamiltonian", 0.0))

        # --- Inner posterior SGLD steps ---
        grad_lam_po = jnp.zeros_like(lam)
        post_ham = 0.0
        for _ in range(inner_T_posterior):
            alpha_s = inner_step_size0 / (1.0 + inner_step_posterior) ** inner_decay
            key, k3, k4 = jax.random.split(key, 3)
            g_phi, g_lam, m = posterior_grad_fn(phi_posterior, lam, k3)
            phi_posterior = _sgld_step(phi_posterior, g_phi, alpha_s, k4, phi_p, phi_sp)
            inner_step_posterior += 1
            grad_lam_po = g_lam
            post_ham = float(m.get("hamiltonian", 0.0))

        # --- Marginal gradient estimate ---
        if lambda_prior_grad_fn is not None:
            prior_term = jnp.asarray(lambda_prior_grad_fn(lam), dtype=jnp.float64)
        else:
            prior_term = jnp.zeros_like(lam)

        grad_lam_marginal = grad_lam_po - grad_lam_pr + prior_term

        # --- Outer SGLD step on λ ---
        outer_alpha_t = outer_step_size0 / (1.0 + t) ** outer_decay
        key, k5 = jax.random.split(key)
        lam = _sgld_step(lam, grad_lam_marginal, outer_alpha_t, k5, lam_p, lam_sp)

        # --- NaN check ---
        if not jnp.isfinite(lam).all():
            meta["stopped"] = True
            meta["stop_step"] = t
            meta["error_code"] = "nan_lambda"
            meta["error_detail"] = f"Non-finite lambda at outer step {t}"
            print(f"[nested_sgld] ERROR: non-finite lambda at step {t}")
            break

        # --- Record ---
        lambda_chain[t] = np.asarray(lam)
        traces["grad_lam_posterior"][t] = np.asarray(grad_lam_po)
        traces["grad_lam_prior"][t] = np.asarray(grad_lam_pr)
        traces["grad_lam_total"][t] = np.asarray(grad_lam_marginal)
        traces["outer_step_size"][t] = outer_alpha_t
        traces["prior_hamiltonian"][t] = prior_ham
        traces["posterior_hamiltonian"][t] = post_ham

        # --- Snapshots ---
        if _snap and (t + 1) % snapshot_interval == 0:
            snap_prior_list.append(np.asarray(phi_prior))
            snap_posterior_list.append(np.asarray(phi_posterior))
            snap_steps_list.append(t)

        if progress_callback and (t + 1) % progress_interval == 0:
            progress_callback(warmup_steps + t + 1, total_steps)

    return NestedSGLDResult(
        lambda_chain=lambda_chain,
        phi_prior_final=np.asarray(phi_prior),
        phi_posterior_final=np.asarray(phi_posterior),
        phi_prior_snapshots=np.array(snap_prior_list) if snap_prior_list else None,
        phi_posterior_snapshots=np.array(snap_posterior_list) if snap_posterior_list else None,
        snapshot_steps=np.array(snap_steps_list) if snap_steps_list else None,
        traces=traces,
        meta=meta,
    )
