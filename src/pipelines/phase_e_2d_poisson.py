"""Phase E — 2D forward Poisson PIFT (Module 2.1).

A direct 2D analogue of the working 1D Phase A pipeline.  Solves the inverse
problem of recovering ``phi(x, y)`` on ``[0, 1]^2`` from noisy observations,
using the physics prior ``U[phi] = 0.5 * E[(-Δphi - f)^2]`` and the
:class:`SineBasis2D` parameterization (zero Dirichlet BC).

Test problem: ``phi_truth(x, y) = sin(pi x) sin(pi y)`` with the matching
forcing ``f = 2 pi^2 sin(pi x) sin(pi y)``.  This admits an exact solution in
the basis (a single nonzero coefficient) so the pipeline both verifies the
energy/gradient implementation and serves as a sanity check that SGLD on the
2D problem is well-conditioned.

The runner returns the same dict shape as the 1D phase runners — ``status``,
``summary``, ``metrics``, ``diagnostics``, ``traces``, ``chain`` — so it can
be plugged into the GUI/benchmark scaffolding without further changes.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import nullcontext
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np

from core.energies_2d_poisson import poisson_2d_residual_energy
from core.likelihoods import gaussian_nll
from core.parameterizations_2d import SineBasis2D
from core.sgld import sgld_sample
from pipelines.common import emit, select_device
from run_types import RunStatus

jax.config.update("jax_enable_x64", True)


# ----------------------------------------------------------------------
# Test problem
# ----------------------------------------------------------------------


def phi_true_2d(xy):
    x = xy[..., 0]
    y = xy[..., 1]
    return jnp.sin(jnp.pi * x) * jnp.sin(jnp.pi * y)


def forcing_2d(xy):
    x = xy[..., 0]
    y = xy[..., 1]
    return 2.0 * (jnp.pi**2) * jnp.sin(jnp.pi * x) * jnp.sin(jnp.pi * y)


DEFAULT_PHASE_E_CONFIG: dict[str, float | int] = {
    "seed": 11,
    "n_modes": 6,        # 6x6 = 36 basis functions
    "n_obs": 60,
    "noise_std": 0.05,
    "beta": 5.0,
    "n_steps": 4000,
    "burn_in": 800,
    "thin": 8,
    "step_size0": 5e-4,
    "decay": 0.55,
    "n_quad": 256,
    "n_grid": 41,
}


def run_phase_e_2d_poisson(
    cfg: dict[str, float | int] | None = None,
    output_root: str | Path = "outputs",
    device_preference: str = "cpu",
    run_id: str | None = None,
    stop_signal: Callable[[], bool] | None = None,
    progress_callback: Callable[[float, str, str, float], None] | None = None,
    save_outputs: bool = False,
) -> dict[str, object]:
    """Run forward 2D Poisson PIFT with the standard runner contract."""
    started_at = time.monotonic()
    config = dict(DEFAULT_PHASE_E_CONFIG)
    if cfg:
        config.update(cfg)

    emit(progress_callback, started_at, 2.0, "setup", "Initializing 2D Poisson run")

    device, device_used = select_device(device_preference)
    device_ctx = jax.default_device(device) if hasattr(jax, "default_device") else nullcontext()

    output_root = Path(output_root)
    if save_outputs:
        (output_root / "traces").mkdir(parents=True, exist_ok=True)
        (output_root / "tables").mkdir(parents=True, exist_ok=True)

    field = SineBasis2D(int(config["n_modes"]))
    n_params = field.n_params
    key = jax.random.PRNGKey(int(config["seed"]))

    with device_ctx:
        emit(progress_callback, started_at, 8.0, "device", f"Using compute device: {device_used.upper()}")

        # ----- synthetic observations on a uniform-random interior set -----
        key, x_key, noise_key = jax.random.split(key, 3)
        xy_obs = jax.random.uniform(
            x_key, shape=(int(config["n_obs"]), 2), minval=0.05, maxval=0.95
        )
        y_clean = phi_true_2d(xy_obs)
        y_obs = y_clean + float(config["noise_std"]) * jax.random.normal(
            noise_key, shape=(int(config["n_obs"]),)
        )
        obs_matrix = field.design_matrix(xy_obs)

        emit(progress_callback, started_at, 18.0, "model", "Building physics + likelihood")

        beta = float(config["beta"])
        n_quad = int(config["n_quad"])

        def grad_and_metrics(theta, sub_key):
            quad_key = sub_key
            xy_quad = jax.random.uniform(quad_key, shape=(n_quad, 2), minval=0.0, maxval=1.0)
            phys_energy, phys_grad, _ = poisson_2d_residual_energy(theta, xy_quad, field, forcing_2d)
            like_energy, like_grad, _ = gaussian_nll(theta, obs_matrix, y_obs, float(config["noise_std"]))
            grad = like_grad + beta * phys_grad
            metrics = {
                "likelihood": like_energy,
                "physics": phys_energy,
                "hamiltonian": like_energy + beta * phys_energy,
            }
            return grad, metrics

        # Cold-start theta=0 (the truth lives in this basis as a single nonzero
        # coefficient at (0, 0); zero is just as fair as any other init).
        theta0 = jnp.zeros(n_params, dtype=jnp.float64)

        emit(progress_callback, started_at, 22.0, "sampler", f"SGLD ({int(config['n_steps'])} steps)")

        precond = field.preconditioner()
        key, sgld_key = jax.random.split(key)
        chain, traces, sgld_meta = sgld_sample(
            theta0=theta0,
            grad_and_metrics_fn=grad_and_metrics,
            n_steps=int(config["n_steps"]),
            step_size0=float(config["step_size0"]),
            decay=float(config["decay"]),
            key=sgld_key,
            preconditioner=precond,
            progress_callback=lambda step, total: emit(
                progress_callback, started_at, 22.0 + 58.0 * step / max(1, total),
                "sampler", f"SGLD step {step}/{total}",
            ),
            progress_interval=max(1, int(config["n_steps"]) // 40),
            stop_signal=stop_signal,
        )

        if sgld_meta.get("stopped"):
            status = RunStatus.STOPPED.value
            error: dict[str, object] | None = {"code": "stopped", "message": "Run stopped by user request"}
        else:
            status = RunStatus.COMPLETED.value
            error = None

        burn_in = int(config["burn_in"])
        thin = int(config["thin"])
        samples = chain[burn_in::thin] if chain.shape[0] > burn_in else np.empty((0, n_params))

        # Posterior mean field on a regular grid for L2 evaluation.
        n_grid = int(config["n_grid"])
        ax = np.linspace(0.0, 1.0, n_grid)
        xv, yv = np.meshgrid(ax, ax)
        xy_grid = np.stack([xv.ravel(), yv.ravel()], axis=-1)
        basis_grid = np.asarray(field.design_matrix(jnp.asarray(xy_grid)))

        if samples.size > 0:
            phi_samples = samples @ basis_grid.T
            phi_mean = np.mean(phi_samples, axis=0)
            phi_std = np.std(phi_samples, axis=0)
        else:
            phi_mean = np.full(xy_grid.shape[0], np.nan)
            phi_std = np.zeros_like(phi_mean)

        truth_grid = np.asarray(phi_true_2d(jnp.asarray(xy_grid)))
        l2_error = float(np.sqrt(np.mean((phi_mean - truth_grid) ** 2)))
        max_error = float(np.max(np.abs(phi_mean - truth_grid)))

        x_obs_np = np.asarray(xy_obs)
        y_obs_np = np.asarray(y_obs)

    runtime_sec = max(0.0, time.monotonic() - started_at)

    diagnostics = {
        "sgld_meta": sgld_meta,
        "n_basis_params": int(n_params),
        "n_quad_points": int(n_quad),
    }

    summary = {
        "run_id": run_id,
        "config": config,
        "device_requested": device_preference.lower(),
        "device_used": device_used,
        "n_posterior_samples": int(samples.shape[0]),
        "l2_error": l2_error,
        "max_error": max_error,
        "status": status,
    }

    metrics = {
        "runtime_sec": runtime_sec,
        "n_total_steps": int(config["n_steps"]),
        "n_effective_steps": int(chain.shape[0]),
        "l2_error": l2_error,
    }

    artifacts: dict[str, str] = {}

    if save_outputs:
        chain_path = output_root / "traces" / "phase_e_2d_poisson_chain.npz"
        np.savez(
            chain_path,
            chain=chain,
            hamiltonian=traces["hamiltonian"],
            likelihood=traces["likelihood"],
            physics=traces["physics"],
            xy_obs=x_obs_np,
            y_obs=y_obs_np,
            phi_mean=phi_mean,
            phi_truth=truth_grid,
        )
        artifacts["chain"] = str(chain_path)

    emit(progress_callback, started_at, 100.0, "complete", "Completed")

    return {
        "status": status,
        "summary": summary,
        "metrics": metrics,
        "diagnostics": diagnostics,
        "artifacts": artifacts,
        "error": error,
        "xy_grid": xy_grid,
        "phi_mean": phi_mean,
        "phi_std": phi_std,
        "phi_truth": truth_grid,
        "xy_obs": x_obs_np,
        "y_obs": y_obs_np,
        "traces": traces,
        "samples": samples,
        "chain": chain,
    }
