"""Reusable pipeline runners for PIFT experiments."""

from __future__ import annotations

import json
from collections.abc import Callable
from contextlib import nullcontext
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np

from utils.diagnostics import credible_interval, lag1_autocorr
from utils.diagnostics_runtime import RuntimeGuard, RuntimeGuardConfig
from core.energies import poisson_residual_energy
from core.likelihoods import gaussian_nll
from core.odil import odil_solve_poisson_1d
from core.parameterizations import SineBasisField
from utils.plotting import plot_diagnostics, plot_field_summary
from core.reference_solver import solve_poisson_dirichlet_fd
from run_types import RunStatus
from core.sgld import sgld_sample
from .common import emit, forcing, load_observations_csv, phi_true, prepare_observations, select_device

jax.config.update("jax_enable_x64", True)

# Backward-compatible private helper import path used by existing tests.
_load_observations_csv = load_observations_csv

DEFAULT_PHASE_A_CONFIG: dict[str, float | int] = {
    "seed": 7,
    "n_modes": 12,
    "n_obs": 28,
    "noise_std": 0.08,
    "beta": 0.5,
    "n_steps": 40000,
    "burn_in": 8000,
    "thin": 16,
    "step_size0": 2e-3,
    "decay": 0.55,
    "n_quad": 96,
    "n_grid": 300,
    "mc_proposal_std": 2e-3,
    "bpinn_sigma_r": 0.05,
    "bpinn_prior_prec": 1e-3,
    "odil_steps": 8000,
    "odil_lr": 1e-7,
    "odil_phys_weight": 12.0,
    "runtime_check_interval": 5,
    "max_condition_number": 100.0,
}

def run_phase_a_forward_poisson(
    cfg: dict[str, float | int] | None = None,
    output_root: str | Path = "outputs",
    obs_csv_path: str | Path | None = None,
    obs_data: tuple[np.ndarray, np.ndarray] | None = None,
    device_preference: str = "cpu",
    run_id: str | None = None,
    stop_signal: Callable[[], bool] | None = None,
    progress_callback: Callable[[float, str, str, float], None] | None = None,
    save_outputs: bool = True,
    init_mode: str = "fd",
) -> dict[str, object]:
    """Run Phase A forward PIFT on the 1D Poisson example.

    Parameters
    ----------
    init_mode : {"cold", "fd", "odil"}
        How to initialize the SGLD chain:
          * ``"cold"`` — start at zero theta.
          * ``"fd"`` — project the FD reference solution onto the sine basis
            (existing default; matches all paper-replication experiments).
          * ``"odil"`` — run ODIL Gauss-Newton on the same problem (incl. the
            observation data term) and project the resulting MAP grid solution
            onto the sine basis.  This is the **Module 2.7 warm start**: the
            ODIL runtime is recorded separately in
            ``metrics["odil_warmstart_sec"]`` so the benchmark can attribute it.
    """
    started_at = time.monotonic()
    config = dict(DEFAULT_PHASE_A_CONFIG)
    if cfg:
        config.update(cfg)

    print(f"[phase_a] PIFT runtime config: n_steps={config['n_steps']} beta={config['beta']} step_size0={config['step_size0']} max_condition_number={config.get('max_condition_number')}")

    emit(progress_callback, started_at, 2.0, "setup", "Initializing run configuration")
    emit(progress_callback, started_at, 4.0, "setup", "Validating hyperparameters")

    device, device_used = select_device(device_preference)
    device_ctx = jax.default_device(device) if hasattr(jax, "default_device") else nullcontext()

    key = jax.random.PRNGKey(int(config["seed"]))
    output_root = Path(output_root)
    out_fig = output_root / "figures"
    out_traces = output_root / "traces"
    out_tables = output_root / "tables"
    out_fig.mkdir(parents=True, exist_ok=True)
    out_traces.mkdir(parents=True, exist_ok=True)
    out_tables.mkdir(parents=True, exist_ok=True)

    emit(progress_callback, started_at, 6.0, "setup", "Preparing output directories")

    field = SineBasisField(int(config["n_modes"]))

    with device_ctx:
        emit(progress_callback, started_at, 8.0, "device", f"Using compute device: {device_used.upper()}")
        key, x_obs, y_obs, y_clean, observation_source = prepare_observations(
            config, key, obs_data, obs_csv_path, started_at, progress_callback
        )

        obs_matrix = field.design_matrix(x_obs)
        emit(progress_callback, started_at, 16.0, "data", "Constructed observation operator matrix")
        emit(progress_callback, started_at, 18.0, "model", "Building physics and likelihood terms")

        def grad_and_metrics(theta, sub_key=None):
            nonlocal key
            if sub_key is None:
                key, quad_key = jax.random.split(key)
            else:
                quad_key = sub_key
            x_quad = jax.random.uniform(
                quad_key,
                shape=(int(config["n_quad"]),),
                minval=0.0,
                maxval=1.0,
            )
            phys_energy, phys_grad, _ = poisson_residual_energy(theta, x_quad, field, forcing)
            like_energy, like_grad, _ = gaussian_nll(theta, obs_matrix, y_obs, float(config["noise_std"]))

            grad = like_grad + float(config["beta"]) * phys_grad
            metrics = {
                "likelihood": like_energy,
                "physics": phys_energy,
                "hamiltonian": like_energy + float(config["beta"]) * phys_energy,
            }
            return grad, metrics

        emit(progress_callback, started_at, 20.0, "sampler", f"Initializing SGLD state (init_mode={init_mode})")

        odil_warmstart_sec = 0.0
        if init_mode == "cold":
            theta0 = jnp.zeros(int(config["n_modes"]), dtype=jnp.float64)
        elif init_mode == "odil":
            # Module 2.7: warm-start from the ODIL MAP solution.  We pass the
            # observation data so ODIL solves the data-tilted MAP, not the
            # pure forward PDE.
            warm = odil_solve_poisson_1d(
                forcing_fn=forcing,
                n_grid=max(129, int(config.get("n_grid", 129))),
                domain=(0.0, 1.0),
                bc=(0.0, 0.0),
                obs=(np.asarray(x_obs), np.asarray(y_obs)),
                noise_std=float(config["noise_std"]),
                method="gauss_newton",
                max_iter=50,
                tol=1e-10,
            )
            odil_warmstart_sec = float(warm.runtime_sec)
            B_init = np.asarray(field.design_matrix(jnp.asarray(warm.x_grid)))
            theta0 = jnp.asarray(
                np.linalg.lstsq(B_init, warm.u_grid, rcond=None)[0]
            )
        else:
            # "fd" — existing default: project the FD reference solution onto
            # the sine basis so the chain starts near the physics solution.
            x_fd, phi_fd = solve_poisson_dirichlet_fd(
                forcing_fn=lambda z: np.asarray(forcing(jnp.asarray(z)), dtype=float),
                n_points=500, domain=(0.0, 1.0), bc=(0.0, 0.0),
            )
            B_init = field.design_matrix(jnp.asarray(x_fd))
            theta0 = jnp.asarray(
                np.linalg.lstsq(np.asarray(B_init), np.asarray(phi_fd), rcond=None)[0]
            )
        _init_grad, _init_metrics = grad_and_metrics(theta0)
        initial_physics_energy = float(_init_metrics["physics"])
        initial_likelihood_energy = float(_init_metrics["likelihood"])

        emit(progress_callback, started_at, 22.0, "sampler", f"Running SGLD sampler ({int(config['n_steps'])} steps)")

        guard = RuntimeGuard(RuntimeGuardConfig())

        def _sgld_progress(step: int, total: int) -> None:
            frac = 0.0 if total <= 0 else step / total
            emit(progress_callback, started_at, 22.0 + 58.0 * frac, "sampler", f"SGLD sampling step {step}/{total}")

        def _runtime_check(step: int, theta_np: np.ndarray, grad_np: np.ndarray, metrics: dict[str, float]):
            return guard.check(step, theta_np, grad_np, metrics)

        precond = field.preconditioner(beta=float(config["beta"]))
        _max_cond = float(config.get("max_condition_number", 100.0))

        key, sgld_key = jax.random.split(key)
        chain, traces, sgld_meta = sgld_sample(
            theta0=theta0,
            grad_and_metrics_fn=grad_and_metrics,
            n_steps=int(config["n_steps"]),
            step_size0=float(config["step_size0"]),
            decay=float(config["decay"]),
            key=sgld_key,
            progress_callback=_sgld_progress,
            progress_interval=max(1, int(config["n_steps"]) // 80),
            stop_signal=stop_signal,
            runtime_check=_runtime_check,
            runtime_check_interval=int(config.get("runtime_check_interval", 5)),
            preconditioner=precond,
            max_condition_number=_max_cond,
        )

        if sgld_meta.get("error_code") is not None:
            status = RunStatus.FAILED.value
            error = {
                "code": str(sgld_meta.get("error_code")),
                "message": str(sgld_meta.get("error_detail") or "Runtime stability guard triggered"),
            }
        elif bool(sgld_meta.get("stopped")):
            status = RunStatus.STOPPED.value
            error = {
                "code": "stopped",
                "message": "Run stopped by user request",
            }
        else:
            status = RunStatus.COMPLETED.value
            error = None

        emit(progress_callback, started_at, 84.0, "post", "Post-processing posterior samples")
        emit(progress_callback, started_at, 86.0, "post", "Computing posterior mean and variance")

        burn_in = int(config["burn_in"])
        thin = int(config["thin"])
        effective_chain = chain[: int(sgld_meta.get("stop_step") or chain.shape[0])]
        samples = effective_chain[burn_in::thin] if effective_chain.shape[0] > burn_in else np.empty((0, theta0.size))

        x_grid = np.linspace(0.0, 1.0, int(config["n_grid"]))
        basis_grid = np.asarray(field.design_matrix(jnp.asarray(x_grid)))

        if samples.size > 0:
            phi_samples = samples @ basis_grid.T
            phi_mean = np.mean(phi_samples, axis=0)
            phi_std = np.std(phi_samples, axis=0)
            pred_obs_samples = samples @ np.asarray(obs_matrix).T
            pred_obs_mean = np.mean(pred_obs_samples, axis=0)
            pred_low, pred_high = credible_interval(pred_obs_samples, low=0.05, high=0.95)
        else:
            phi_samples = np.zeros((0, x_grid.size), dtype=float)
            phi_mean = np.full(x_grid.shape, np.nan)
            phi_std = np.full(x_grid.shape, np.nan)
            pred_obs_mean = np.full((int(np.asarray(x_obs).shape[0]),), np.nan)
            pred_low = np.full((int(np.asarray(x_obs).shape[0]),), np.nan)
            pred_high = np.full((int(np.asarray(x_obs).shape[0]),), np.nan)

        phi_truth = np.asarray(phi_true(jnp.asarray(x_grid)))
        x_ref, phi_ref = solve_poisson_dirichlet_fd(
            forcing_fn=lambda z: np.asarray(forcing(jnp.asarray(z)), dtype=float),
            n_points=int(config["n_grid"]),
            domain=(0.0, 1.0),
            bc=(0.0, 0.0),
        )
        emit(progress_callback, started_at, 90.0, "post", "Computing predictive intervals and diagnostics")

        x_obs_np = np.asarray(x_obs)
        y_obs_np = np.asarray(y_obs)
        y_clean_np = np.asarray(y_clean)

    diagnostics = guard.diagnostics_payload()
    diagnostics["sgld_meta"] = sgld_meta
    diagnostics["initial_physics_energy"] = initial_physics_energy
    diagnostics["initial_likelihood_energy"] = initial_likelihood_energy

    l2_error = float(np.sqrt(np.mean((phi_mean - phi_truth) ** 2))) if np.isfinite(phi_mean).all() else float("nan")
    max_error = float(np.max(np.abs(phi_mean - phi_truth))) if np.isfinite(phi_mean).all() else float("nan")
    reference_l2_error = (
        float(np.sqrt(np.mean((phi_mean - phi_ref) ** 2))) if np.isfinite(phi_mean).all() else float("nan")
    )
    reference_max_error = (
        float(np.max(np.abs(phi_mean - phi_ref))) if np.isfinite(phi_mean).all() else float("nan")
    )
    reference_solver_truth_l2 = float(np.sqrt(np.mean((phi_ref - phi_truth) ** 2)))
    lag1 = lag1_autocorr(traces["hamiltonian"][burn_in:]) if burn_in < traces["hamiltonian"].size else float("nan")

    coverage = float(np.mean((y_clean_np >= pred_low) & (y_clean_np <= pred_high))) if np.isfinite(pred_low).all() else float("nan")

    runtime_sec = max(0.0, time.monotonic() - started_at)

    summary = {
        "run_id": run_id,
        "config": config,
        "source_observations": observation_source,
        "device_requested": device_preference.lower(),
        "device_used": device_used,
        "n_posterior_samples": int(samples.shape[0]),
        "l2_error": l2_error,
        "max_error": max_error,
        "reference_l2_error": reference_l2_error,
        "reference_max_error": reference_max_error,
        "reference_solver_truth_l2": reference_solver_truth_l2,
        "hamiltonian_lag1_autocorr": lag1,
        "measurement_interval_coverage_90pct": coverage,
        "status": status,
    }

    artifacts = {
        "field_plot": str(out_fig / "phase_a_forward_poisson_field.png"),
        "trace_plot": str(out_fig / "phase_a_forward_poisson_trace.png"),
        "chain": str(out_traces / "phase_a_forward_poisson_chain.npz"),
        "summary": str(out_tables / "phase_a_forward_poisson_summary.json"),
    }

    if save_outputs:
        emit(progress_callback, started_at, 92.0, "artifacts", "Writing field and trace plots")
        plot_field_summary(
            x_grid=x_grid,
            truth=phi_truth,
            mean=phi_mean,
            std=phi_std,
            x_obs=x_obs_np,
            y_obs=y_obs_np,
            out_path=artifacts["field_plot"],
        )
        plot_diagnostics(traces["hamiltonian"], out_path=artifacts["trace_plot"])

        np.savez(
            artifacts["chain"],
            chain=chain,
            hamiltonian=traces["hamiltonian"],
            likelihood=traces["likelihood"],
            physics=traces["physics"],
            x_obs=x_obs_np,
            y_obs=y_obs_np,
            pred_obs_mean=pred_obs_mean,
            pred_obs_low=pred_low,
            pred_obs_high=pred_high,
            status=status,
            runtime_sec=runtime_sec,
        )
        emit(progress_callback, started_at, 96.0, "artifacts", "Saving chain and trace artifacts")

        payload = {
            "summary": summary,
            "metrics": {
                "runtime_sec": runtime_sec,
                "n_total_steps": int(config["n_steps"]),
                "n_effective_steps": int(effective_chain.shape[0]),
                "reference_solver_truth_l2": reference_solver_truth_l2,
                "init_mode": init_mode,
                "odil_warmstart_sec": float(odil_warmstart_sec),
            },
            "diagnostics": diagnostics,
            "error": error,
            "artifacts": artifacts,
        }
        with open(artifacts["summary"], "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=str)
        emit(progress_callback, started_at, 99.0, "artifacts", "Saving summary report")

    final_detail = "Completed"
    if status == RunStatus.FAILED.value:
        final_detail = "Failed"
    elif status == RunStatus.STOPPED.value:
        final_detail = "Stopped"
    emit(progress_callback, started_at, 100.0, "complete", final_detail)

    return {
        "status": status,
        "summary": summary,
        "metrics": {
            "runtime_sec": runtime_sec,
            "n_total_steps": int(config["n_steps"]),
            "n_effective_steps": int(effective_chain.shape[0]),
            "reference_solver_truth_l2": reference_solver_truth_l2,
            "init_mode": init_mode,
            "odil_warmstart_sec": float(odil_warmstart_sec),
        },
        "diagnostics": diagnostics,
        "artifacts": artifacts,
        "error": error,
        "x_grid": x_grid,
        "phi_truth": phi_truth,
        "x_reference": x_ref,
        "phi_reference": phi_ref,
        "phi_mean": phi_mean,
        "phi_std": phi_std,
        "x_obs": x_obs_np,
        "y_obs": y_obs_np,
        "traces": traces,
        "samples": samples,
        "chain": effective_chain,
        "output_paths": artifacts,
    }


def run_phase_a_monte_carlo(
    cfg: dict[str, float | int] | None = None,
    output_root: str | Path = "outputs",
    obs_csv_path: str | Path | None = None,
    obs_data: tuple[np.ndarray, np.ndarray] | None = None,
    device_preference: str = "cpu",
    run_id: str | None = None,
    stop_signal: Callable[[], bool] | None = None,
    progress_callback: Callable[[float, str, str, float], None] | None = None,
    save_outputs: bool = True,
) -> dict[str, object]:
    """Run Phase A with a simple Random-Walk Metropolis Monte Carlo sampler."""
    started_at = time.monotonic()
    config = dict(DEFAULT_PHASE_A_CONFIG)
    if cfg:
        config.update(cfg)

    proposal_std = float(config.get("mc_proposal_std", 2e-3))
    key = jax.random.PRNGKey(int(config["seed"]))

    emit(progress_callback, started_at, 2.0, "setup", "Initializing Monte Carlo configuration")
    emit(progress_callback, started_at, 4.0, "setup", "Validating hyperparameters")

    device, device_used = select_device(device_preference)
    device_ctx = jax.default_device(device) if hasattr(jax, "default_device") else nullcontext()

    output_root = Path(output_root)
    out_fig = output_root / "figures"
    out_traces = output_root / "traces"
    out_tables = output_root / "tables"
    out_fig.mkdir(parents=True, exist_ok=True)
    out_traces.mkdir(parents=True, exist_ok=True)
    out_tables.mkdir(parents=True, exist_ok=True)
    emit(progress_callback, started_at, 6.0, "setup", "Preparing output directories")

    rng = np.random.default_rng(int(config["seed"]))
    field = SineBasisField(int(config["n_modes"]))

    with device_ctx:
        emit(progress_callback, started_at, 8.0, "device", f"Using compute device: {device_used.upper()}")
        key, x_obs, y_obs, y_clean, observation_source = prepare_observations(
            config, key, obs_data, obs_csv_path, started_at, progress_callback
        )

        x_obs_np = np.asarray(x_obs)
        y_obs_np = np.asarray(y_obs)
        y_clean_np = np.asarray(y_clean)

        obs_matrix = np.asarray(field.design_matrix(x_obs))
        emit(progress_callback, started_at, 16.0, "data", "Constructed observation operator matrix")
        emit(progress_callback, started_at, 18.0, "model", "Building physics and likelihood terms")

        n_quad_det = max(128, int(config["n_quad"]) * 4)
        x_quad = np.linspace(0.0, 1.0, n_quad_det, dtype=float)
        op_quad = np.asarray(field.neg_second_derivative_matrix(jnp.asarray(x_quad)))
        forcing_quad = np.asarray(forcing(jnp.asarray(x_quad)), dtype=float)
        inv_var = 1.0 / float(config["noise_std"]) ** 2
        beta = float(config["beta"])

        def _energy(theta_np: np.ndarray) -> tuple[float, float, float]:
            residual = op_quad @ theta_np - forcing_quad
            phys = 0.5 * float(np.mean(residual**2))
            err = obs_matrix @ theta_np - y_obs_np
            like = 0.5 * inv_var * float(np.sum(err**2))
            return like + beta * phys, like, phys

        n_steps = int(config["n_steps"])
        dim = int(config["n_modes"])
        chain = np.zeros((n_steps, dim), dtype=float)
        ham_trace = np.zeros(n_steps, dtype=float)
        like_trace = np.zeros(n_steps, dtype=float)
        phys_trace = np.zeros(n_steps, dtype=float)

        # Initialize at the FD solution projected onto the sine basis.
        x_fd, phi_fd = solve_poisson_dirichlet_fd(
            forcing_fn=lambda z: np.asarray(forcing(jnp.asarray(z)), dtype=float),
            n_points=500, domain=(0.0, 1.0), bc=(0.0, 0.0),
        )
        B_init = np.asarray(field.design_matrix(jnp.asarray(x_fd)))
        theta = np.linalg.lstsq(B_init, np.asarray(phi_fd), rcond=None)[0]
        curr_h, curr_like, curr_phys = _energy(theta)
        initial_physics_energy = float(curr_phys)
        initial_likelihood_energy = float(curr_like)
        accepted = 0

        emit(progress_callback, started_at, 20.0, "sampler", "Initializing Random-Walk Metropolis state")
        emit(progress_callback, started_at, 22.0, "sampler", f"Running Monte Carlo sampler ({n_steps} steps)")

        status = RunStatus.COMPLETED.value
        error = None
        stop_step: int | None = None
        for t in range(n_steps):
            if stop_signal is not None and stop_signal():
                status = RunStatus.STOPPED.value
                error = {"code": "stopped", "message": "Run stopped by user request"}
                stop_step = t
                break

            prop = theta + proposal_std * rng.normal(size=dim)
            prop_h, prop_like, prop_phys = _energy(prop)

            if not np.isfinite(prop_h):
                status = RunStatus.FAILED.value
                error = {"code": "non_finite", "message": "Non-finite energy in Monte Carlo proposal"}
                stop_step = t + 1
                break

            log_alpha = -(prop_h - curr_h)
            if log_alpha >= 0.0 or np.log(rng.uniform()) < log_alpha:
                theta = prop
                curr_h, curr_like, curr_phys = prop_h, prop_like, prop_phys
                accepted += 1

            chain[t] = theta
            ham_trace[t] = curr_h
            like_trace[t] = curr_like
            phys_trace[t] = curr_phys

            interval = max(1, n_steps // 80)
            if t % interval == 0 or (t + 1) == n_steps:
                frac = (t + 1) / n_steps
                emit(progress_callback, started_at, 22.0 + 58.0 * frac, "sampler", f"MC sampling step {t+1}/{n_steps}")

        effective_steps = stop_step if stop_step is not None else n_steps
        effective_chain = chain[:effective_steps]
        traces = {
            "hamiltonian": ham_trace[:effective_steps] if effective_steps > 0 else ham_trace[:0],
            "likelihood": like_trace[:effective_steps] if effective_steps > 0 else like_trace[:0],
            "physics": phys_trace[:effective_steps] if effective_steps > 0 else phys_trace[:0],
        }

        emit(progress_callback, started_at, 84.0, "post", "Post-processing posterior samples")
        emit(progress_callback, started_at, 86.0, "post", "Computing posterior mean and variance")

        burn_in = int(config["burn_in"])
        thin = int(config["thin"])
        samples = effective_chain[burn_in::thin] if effective_chain.shape[0] > burn_in else np.empty((0, dim))

        x_grid = np.linspace(0.0, 1.0, int(config["n_grid"]))
        basis_grid = np.asarray(field.design_matrix(jnp.asarray(x_grid)))

        if samples.size > 0:
            phi_samples = samples @ basis_grid.T
            phi_mean = np.mean(phi_samples, axis=0)
            phi_std = np.std(phi_samples, axis=0)
            pred_obs_samples = samples @ obs_matrix.T
            pred_obs_mean = np.mean(pred_obs_samples, axis=0)
            pred_low, pred_high = credible_interval(pred_obs_samples, low=0.05, high=0.95)
        else:
            phi_mean = np.full(x_grid.shape, np.nan)
            phi_std = np.full(x_grid.shape, np.nan)
            pred_obs_mean = np.full((x_obs_np.shape[0],), np.nan)
            pred_low = np.full((x_obs_np.shape[0],), np.nan)
            pred_high = np.full((x_obs_np.shape[0],), np.nan)

        phi_truth = np.asarray(phi_true(jnp.asarray(x_grid)))
        x_ref, phi_ref = solve_poisson_dirichlet_fd(
            forcing_fn=lambda z: np.asarray(forcing(jnp.asarray(z)), dtype=float),
            n_points=int(config["n_grid"]),
            domain=(0.0, 1.0),
            bc=(0.0, 0.0),
        )
        emit(progress_callback, started_at, 90.0, "post", "Computing predictive intervals and diagnostics")

    l2_error = float(np.sqrt(np.mean((phi_mean - phi_truth) ** 2))) if np.isfinite(phi_mean).all() else float("nan")
    max_error = float(np.max(np.abs(phi_mean - phi_truth))) if np.isfinite(phi_mean).all() else float("nan")
    reference_l2_error = (
        float(np.sqrt(np.mean((phi_mean - phi_ref) ** 2))) if np.isfinite(phi_mean).all() else float("nan")
    )
    reference_max_error = (
        float(np.max(np.abs(phi_mean - phi_ref))) if np.isfinite(phi_mean).all() else float("nan")
    )
    reference_solver_truth_l2 = float(np.sqrt(np.mean((phi_ref - phi_truth) ** 2)))
    lag1 = lag1_autocorr(traces["hamiltonian"][burn_in:]) if burn_in < traces["hamiltonian"].size else float("nan")
    coverage = float(np.mean((y_clean_np >= pred_low) & (y_clean_np <= pred_high))) if np.isfinite(pred_low).all() else float("nan")
    runtime_sec = max(0.0, time.monotonic() - started_at)
    acceptance_rate = float(accepted / max(1, effective_steps))

    diagnostics = {
        "stability_flags": {
            "nan_detected": bool(np.isnan(traces["hamiltonian"]).any()),
            "inf_detected": bool(np.isinf(traces["hamiltonian"]).any()),
            "divergence_detected": False,
            "hamiltonian_overflow": False,
        },
        "first_bad_step": stop_step,
        "max_abs_hamiltonian": float(np.max(np.abs(traces["hamiltonian"]))) if traces["hamiltonian"].size else 0.0,
        "nan_detected": bool(np.isnan(traces["hamiltonian"]).any()),
        "suggested_actions": ["No critical issues detected"] if status == RunStatus.COMPLETED.value else ["Reduce mc_proposal_std"],
        "mc_acceptance_rate": acceptance_rate,
        "initial_physics_energy": initial_physics_energy,
        "initial_likelihood_energy": initial_likelihood_energy,
    }

    summary = {
        "run_id": run_id,
        "config": config,
        "source_observations": observation_source,
        "device_requested": device_preference.lower(),
        "device_used": device_used,
        "n_posterior_samples": int(samples.shape[0]),
        "l2_error": l2_error,
        "max_error": max_error,
        "reference_l2_error": reference_l2_error,
        "reference_max_error": reference_max_error,
        "reference_solver_truth_l2": reference_solver_truth_l2,
        "hamiltonian_lag1_autocorr": lag1,
        "measurement_interval_coverage_90pct": coverage,
        "status": status,
    }

    artifacts = {
        "field_plot": str(out_fig / "phase_a_monte_carlo_field.png"),
        "trace_plot": str(out_fig / "phase_a_monte_carlo_trace.png"),
        "chain": str(out_traces / "phase_a_monte_carlo_chain.npz"),
        "summary": str(out_tables / "phase_a_monte_carlo_summary.json"),
    }

    if save_outputs:
        emit(progress_callback, started_at, 92.0, "artifacts", "Writing field and trace plots")
        plot_field_summary(
            x_grid=x_grid,
            truth=phi_truth,
            mean=phi_mean,
            std=phi_std,
            x_obs=x_obs_np,
            y_obs=y_obs_np,
            out_path=artifacts["field_plot"],
        )
        plot_diagnostics(traces["hamiltonian"], out_path=artifacts["trace_plot"])

        np.savez(
            artifacts["chain"],
            chain=effective_chain,
            hamiltonian=traces["hamiltonian"],
            likelihood=traces["likelihood"],
            physics=traces["physics"],
            x_obs=x_obs_np,
            y_obs=y_obs_np,
            pred_obs_mean=pred_obs_mean,
            pred_obs_low=pred_low,
            pred_obs_high=pred_high,
            status=status,
            runtime_sec=runtime_sec,
            mc_acceptance_rate=acceptance_rate,
        )
        emit(progress_callback, started_at, 96.0, "artifacts", "Saving chain and trace artifacts")

        payload = {
            "summary": summary,
            "metrics": {
                "runtime_sec": runtime_sec,
                "n_total_steps": int(config["n_steps"]),
                "n_effective_steps": int(effective_steps),
                "reference_solver_truth_l2": reference_solver_truth_l2,
                "mc_acceptance_rate": acceptance_rate,
            },
            "diagnostics": diagnostics,
            "error": error,
            "artifacts": artifacts,
        }
        with open(artifacts["summary"], "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=str)
        emit(progress_callback, started_at, 99.0, "artifacts", "Saving summary report")

    final_detail = "Completed"
    if status == RunStatus.FAILED.value:
        final_detail = "Failed"
    elif status == RunStatus.STOPPED.value:
        final_detail = "Stopped"
    emit(progress_callback, started_at, 100.0, "complete", final_detail)

    return {
        "status": status,
        "summary": summary,
        "metrics": {
            "runtime_sec": runtime_sec,
            "n_total_steps": int(config["n_steps"]),
            "n_effective_steps": int(effective_steps),
            "reference_solver_truth_l2": reference_solver_truth_l2,
            "mc_acceptance_rate": acceptance_rate,
        },
        "diagnostics": diagnostics,
        "artifacts": artifacts,
        "error": error,
        "x_grid": x_grid,
        "phi_truth": phi_truth,
        "x_reference": x_ref,
        "phi_reference": phi_ref,
        "phi_mean": phi_mean,
        "phi_std": phi_std,
        "x_obs": x_obs_np,
        "y_obs": y_obs_np,
        "traces": traces,
        "samples": samples,
        "chain": effective_chain,
        "output_paths": artifacts,
    }



def run_phase_a_bayesian_pinn(
    cfg: dict[str, float | int] | None = None,
    output_root: str | Path = "outputs",
    obs_csv_path: str | Path | None = None,
    obs_data: tuple[np.ndarray, np.ndarray] | None = None,
    device_preference: str = "cpu",
    run_id: str | None = None,
    stop_signal: Callable[[], bool] | None = None,
    progress_callback: Callable[[float, str, str, float], None] | None = None,
    save_outputs: bool = True,
) -> dict[str, object]:
    """Bayesian PINNs-style sampler using data likelihood + residual likelihood + Gaussian prior."""
    started_at = time.monotonic()
    config = dict(DEFAULT_PHASE_A_CONFIG)
    if cfg:
        config.update(cfg)

    sigma_r = float(config.get("bpinn_sigma_r", 0.05))
    prior_prec = float(config.get("bpinn_prior_prec", 1e-3))

    emit(progress_callback, started_at, 2.0, "setup", "Initializing Bayesian PINNs configuration")
    emit(progress_callback, started_at, 4.0, "setup", "Validating hyperparameters")

    device, device_used = select_device(device_preference)
    device_ctx = jax.default_device(device) if hasattr(jax, "default_device") else nullcontext()

    key = jax.random.PRNGKey(int(config["seed"]))
    output_root = Path(output_root)
    out_fig = output_root / "figures"
    out_traces = output_root / "traces"
    out_tables = output_root / "tables"
    out_fig.mkdir(parents=True, exist_ok=True)
    out_traces.mkdir(parents=True, exist_ok=True)
    out_tables.mkdir(parents=True, exist_ok=True)
    emit(progress_callback, started_at, 6.0, "setup", "Preparing output directories")

    field = SineBasisField(int(config["n_modes"]))

    with device_ctx:
        emit(progress_callback, started_at, 8.0, "device", f"Using compute device: {device_used.upper()}")
        key, x_obs, y_obs, y_clean, observation_source = prepare_observations(
            config, key, obs_data, obs_csv_path, started_at, progress_callback
        )

        obs_matrix = field.design_matrix(x_obs)
        emit(progress_callback, started_at, 16.0, "data", "Constructed observation operator matrix")
        emit(progress_callback, started_at, 18.0, "model", "Building data/residual/prior terms")

        inv_var_d = 1.0 / float(config["noise_std"]) ** 2
        inv_var_r = 1.0 / (sigma_r**2)

        def grad_and_metrics(theta, sub_key=None):
            nonlocal key
            if sub_key is None:
                key, quad_key = jax.random.split(key)
            else:
                quad_key = sub_key
            x_quad = jax.random.uniform(
                quad_key,
                shape=(int(config["n_quad"]),),
                minval=0.0,
                maxval=1.0,
            )
            op = field.neg_second_derivative_matrix(x_quad)
            residual = op @ theta - forcing(x_quad)
            residual_nll = 0.5 * inv_var_r * jnp.mean(residual**2)

            pred = obs_matrix @ theta
            err = pred - y_obs
            data_nll = 0.5 * inv_var_d * jnp.sum(err**2)

            prior_term = 0.5 * prior_prec * jnp.sum(theta**2)
            h = data_nll + residual_nll + prior_term

            def _h(theta_inner):
                pred_i = obs_matrix @ theta_inner
                err_i = pred_i - y_obs
                data_i = 0.5 * inv_var_d * jnp.sum(err_i**2)

                res_i = op @ theta_inner - forcing(x_quad)
                res_nll_i = 0.5 * inv_var_r * jnp.mean(res_i**2)
                prior_i = 0.5 * prior_prec * jnp.sum(theta_inner**2)
                return data_i + res_nll_i + prior_i

            grad = jax.grad(_h)(theta)
            return grad, {
                "likelihood": data_nll + residual_nll,
                "physics": residual_nll,
                "hamiltonian": h,
            }

        emit(progress_callback, started_at, 20.0, "sampler", "Initializing Bayesian sampler")

        # Project the FD reference solution onto the sine basis.
        x_fd, phi_fd = solve_poisson_dirichlet_fd(
            forcing_fn=lambda z: np.asarray(forcing(jnp.asarray(z)), dtype=float),
            n_points=500, domain=(0.0, 1.0), bc=(0.0, 0.0),
        )
        B_init = field.design_matrix(jnp.asarray(x_fd))
        theta0 = jnp.asarray(
            np.linalg.lstsq(np.asarray(B_init), np.asarray(phi_fd), rcond=None)[0]
        )
        _init_grad, _init_metrics = grad_and_metrics(theta0)
        initial_physics_energy = float(_init_metrics["physics"])
        initial_likelihood_energy = float(_init_metrics["likelihood"])

        emit(progress_callback, started_at, 22.0, "sampler", f"Running SGLD sampler ({int(config['n_steps'])} steps)")

        guard = RuntimeGuard(RuntimeGuardConfig(max_abs_hamiltonian=1e300, grad_ratio_limit=1e9))

        def _sgld_progress(step: int, total: int) -> None:
            frac = 0.0 if total <= 0 else step / total
            emit(progress_callback, started_at, 22.0 + 58.0 * frac, "sampler", f"B-PINN SGLD step {step}/{total}")

        def _runtime_check(step: int, theta_np: np.ndarray, grad_np: np.ndarray, metrics: dict[str, float]):
            return guard.check(step, theta_np, grad_np, metrics)

        # B-PINN's residual_nll has curvature inv_var_r*(πk)^4 per mode.
        # The preconditioner p_k=1/k^4 exactly cancels this, but only if
        # we don't clip it.  Use the full raw condition number (20736 for
        # 12 modes) so all modes see equal effective curvature.
        _bpinn_max_cond = 25000.0
        _bpinn_step = float(config["step_size0"]) * 0.5
        precond = field.preconditioner(beta=float(config.get("beta", 0.5)))

        key, sgld_key = jax.random.split(key)
        chain, traces, sgld_meta = sgld_sample(
            theta0=theta0,
            grad_and_metrics_fn=grad_and_metrics,
            n_steps=int(config["n_steps"]),
            step_size0=_bpinn_step,
            decay=float(config["decay"]),
            key=sgld_key,
            progress_callback=_sgld_progress,
            progress_interval=max(1, int(config["n_steps"]) // 80),
            stop_signal=stop_signal,
            runtime_check=_runtime_check,
            runtime_check_interval=int(config.get("runtime_check_interval", 5)),
            preconditioner=precond,
            max_condition_number=_bpinn_max_cond,
        )

        if sgld_meta.get("error_code") is not None:
            status = RunStatus.FAILED.value
            error = {
                "code": str(sgld_meta.get("error_code")),
                "message": str(sgld_meta.get("error_detail") or "Runtime stability guard triggered"),
            }
        elif bool(sgld_meta.get("stopped")):
            status = RunStatus.STOPPED.value
            error = {"code": "stopped", "message": "Run stopped by user request"}
        else:
            status = RunStatus.COMPLETED.value
            error = None

        emit(progress_callback, started_at, 84.0, "post", "Post-processing posterior samples")

        burn_in = int(config["burn_in"])
        thin = int(config["thin"])
        effective_chain = chain[: int(sgld_meta.get("stop_step") or chain.shape[0])]
        samples = effective_chain[burn_in::thin] if effective_chain.shape[0] > burn_in else np.empty((0, theta0.size))

        x_grid = np.linspace(0.0, 1.0, int(config["n_grid"]))
        basis_grid = np.asarray(field.design_matrix(jnp.asarray(x_grid)))

        if samples.size > 0:
            phi_samples = samples @ basis_grid.T
            phi_mean = np.mean(phi_samples, axis=0)
            phi_std = np.std(phi_samples, axis=0)
            pred_obs_samples = samples @ np.asarray(obs_matrix).T
            pred_obs_mean = np.mean(pred_obs_samples, axis=0)
            pred_low, pred_high = credible_interval(pred_obs_samples, low=0.05, high=0.95)
        else:
            phi_mean = np.full(x_grid.shape, np.nan)
            phi_std = np.full(x_grid.shape, np.nan)
            pred_obs_mean = np.full((int(np.asarray(x_obs).shape[0]),), np.nan)
            pred_low = np.full((int(np.asarray(x_obs).shape[0]),), np.nan)
            pred_high = np.full((int(np.asarray(x_obs).shape[0]),), np.nan)

        phi_truth = np.asarray(phi_true(jnp.asarray(x_grid)))
        x_ref, phi_ref = solve_poisson_dirichlet_fd(
            forcing_fn=lambda z: np.asarray(forcing(jnp.asarray(z)), dtype=float),
            n_points=int(config["n_grid"]),
            domain=(0.0, 1.0),
            bc=(0.0, 0.0),
        )

        x_obs_np = np.asarray(x_obs)
        y_obs_np = np.asarray(y_obs)
        y_clean_np = np.asarray(y_clean)

    diagnostics = guard.diagnostics_payload()
    diagnostics["sgld_meta"] = sgld_meta
    diagnostics["bpinn_sigma_r"] = sigma_r
    diagnostics["initial_physics_energy"] = initial_physics_energy
    diagnostics["initial_likelihood_energy"] = initial_likelihood_energy

    l2_error = float(np.sqrt(np.mean((phi_mean - phi_truth) ** 2))) if np.isfinite(phi_mean).all() else float("nan")
    max_error = float(np.max(np.abs(phi_mean - phi_truth))) if np.isfinite(phi_mean).all() else float("nan")
    reference_l2_error = float(np.sqrt(np.mean((phi_mean - phi_ref) ** 2))) if np.isfinite(phi_mean).all() else float("nan")
    reference_max_error = float(np.max(np.abs(phi_mean - phi_ref))) if np.isfinite(phi_mean).all() else float("nan")
    reference_solver_truth_l2 = float(np.sqrt(np.mean((phi_ref - phi_truth) ** 2)))
    lag1 = lag1_autocorr(traces["hamiltonian"][burn_in:]) if burn_in < traces["hamiltonian"].size else float("nan")
    coverage = float(np.mean((y_clean_np >= pred_low) & (y_clean_np <= pred_high))) if np.isfinite(pred_low).all() else float("nan")
    runtime_sec = max(0.0, time.monotonic() - started_at)

    summary = {
        "run_id": run_id,
        "config": config,
        "source_observations": observation_source,
        "device_requested": device_preference.lower(),
        "device_used": device_used,
        "n_posterior_samples": int(samples.shape[0]),
        "l2_error": l2_error,
        "max_error": max_error,
        "reference_l2_error": reference_l2_error,
        "reference_max_error": reference_max_error,
        "reference_solver_truth_l2": reference_solver_truth_l2,
        "hamiltonian_lag1_autocorr": lag1,
        "measurement_interval_coverage_90pct": coverage,
        "status": status,
    }

    artifacts = {
        "field_plot": str(out_fig / "phase_a_bpinn_field.png"),
        "trace_plot": str(out_fig / "phase_a_bpinn_trace.png"),
        "chain": str(out_traces / "phase_a_bpinn_chain.npz"),
        "summary": str(out_tables / "phase_a_bpinn_summary.json"),
    }

    if save_outputs:
        emit(progress_callback, started_at, 92.0, "artifacts", "Writing field and trace plots")
        plot_field_summary(
            x_grid=x_grid,
            truth=phi_truth,
            mean=phi_mean,
            std=phi_std,
            x_obs=x_obs_np,
            y_obs=y_obs_np,
            out_path=artifacts["field_plot"],
        )
        plot_diagnostics(traces["hamiltonian"], out_path=artifacts["trace_plot"])

        np.savez(
            artifacts["chain"],
            chain=chain,
            hamiltonian=traces["hamiltonian"],
            likelihood=traces["likelihood"],
            physics=traces["physics"],
            x_obs=x_obs_np,
            y_obs=y_obs_np,
            pred_obs_mean=pred_obs_mean,
            pred_obs_low=pred_low,
            pred_obs_high=pred_high,
            status=status,
            runtime_sec=runtime_sec,
        )

        payload = {
            "summary": summary,
            "metrics": {
                "runtime_sec": runtime_sec,
                "n_total_steps": int(config["n_steps"]),
                "n_effective_steps": int(effective_chain.shape[0]),
                "reference_solver_truth_l2": reference_solver_truth_l2,
                "init_mode": init_mode,
                "odil_warmstart_sec": float(odil_warmstart_sec),
            },
            "diagnostics": diagnostics,
            "error": error,
            "artifacts": artifacts,
        }
        with open(artifacts["summary"], "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=str)

    final_detail = "Completed"
    if status == RunStatus.FAILED.value:
        final_detail = "Failed"
    elif status == RunStatus.STOPPED.value:
        final_detail = "Stopped"
    emit(progress_callback, started_at, 100.0, "complete", final_detail)

    return {
        "status": status,
        "summary": summary,
        "metrics": {
            "runtime_sec": runtime_sec,
            "n_total_steps": int(config["n_steps"]),
            "n_effective_steps": int(effective_chain.shape[0]),
            "reference_solver_truth_l2": reference_solver_truth_l2,
        },
        "diagnostics": diagnostics,
        "artifacts": artifacts,
        "error": error,
        "x_grid": x_grid,
        "phi_truth": phi_truth,
        "x_reference": x_ref,
        "phi_reference": phi_ref,
        "phi_mean": phi_mean,
        "phi_std": phi_std,
        "x_obs": x_obs_np,
        "y_obs": y_obs_np,
        "traces": traces,
        "samples": samples,
        "chain": effective_chain,
        "output_paths": artifacts,
    }


def run_phase_a_odil(
    cfg: dict[str, float | int] | None = None,
    output_root: str | Path = "outputs",
    obs_csv_path: str | Path | None = None,
    obs_data: tuple[np.ndarray, np.ndarray] | None = None,
    device_preference: str = "cpu",
    run_id: str | None = None,
    stop_signal: Callable[[], bool] | None = None,
    progress_callback: Callable[[float, str, str, float], None] | None = None,
    save_outputs: bool = True,
) -> dict[str, object]:
    """ODIL-style deterministic baseline: optimize discrete residual/data loss."""
    started_at = time.monotonic()
    config = dict(DEFAULT_PHASE_A_CONFIG)
    if cfg:
        config.update(cfg)

    # NB: ``odil_steps``/``odil_lr``/``odil_phys_weight`` are kept in the config
    # for backward compatibility with saved presets, but the proper Karnakov-style
    # ODIL solver in ``core.odil`` does not need them.  ``odil_steps`` is reused
    # as the iteration cap for Gauss-Newton.
    odil_max_iter = int(config.get("odil_max_iter", min(50, int(config.get("odil_steps", 50)))))
    odil_method = str(config.get("odil_method", "gauss_newton"))
    odil_n_grid = int(config.get("odil_n_grid", max(129, int(config.get("n_grid", 129)))))
    odil_bc_weight = float(config.get("odil_bc_weight", 1.0e6))

    emit(progress_callback, started_at, 2.0, "setup", "Initializing ODIL configuration")
    emit(progress_callback, started_at, 4.0, "setup", "Validating hyperparameters")

    device, device_used = select_device(device_preference)
    device_ctx = jax.default_device(device) if hasattr(jax, "default_device") else nullcontext()

    key = jax.random.PRNGKey(int(config["seed"]))
    output_root = Path(output_root)
    out_fig = output_root / "figures"
    out_traces = output_root / "traces"
    out_tables = output_root / "tables"
    out_fig.mkdir(parents=True, exist_ok=True)
    out_traces.mkdir(parents=True, exist_ok=True)
    out_tables.mkdir(parents=True, exist_ok=True)
    emit(progress_callback, started_at, 6.0, "setup", "Preparing output directories")

    field = SineBasisField(int(config["n_modes"]))

    with device_ctx:
        emit(progress_callback, started_at, 8.0, "device", f"Using compute device: {device_used.upper()}")
        key, x_obs, y_obs, y_clean, observation_source = prepare_observations(
            config, key, obs_data, obs_csv_path, started_at, progress_callback
        )

        obs_matrix = field.design_matrix(x_obs)
        emit(progress_callback, started_at, 16.0, "data", "Constructed observation operator matrix")
        emit(progress_callback, started_at, 18.0, "model", "Building ODIL discrete loss")

        status = RunStatus.COMPLETED.value
        error = None
        stop_step: int | None = None

        emit(progress_callback, started_at, 20.0, "sampler", "Initializing ODIL Gauss-Newton solver")
        emit(
            progress_callback,
            started_at,
            22.0,
            "sampler",
            f"Running ODIL ({odil_method}, max_iter={odil_max_iter}, n_grid={odil_n_grid})",
        )

        # Solve the discrete FD residual loss with optional Gaussian data term.
        odil_result = odil_solve_poisson_1d(
            forcing_fn=forcing,
            n_grid=odil_n_grid,
            domain=(0.0, 1.0),
            bc=(0.0, 0.0),
            obs=(np.asarray(x_obs), np.asarray(y_obs)),
            noise_std=float(config["noise_std"]),
            method=odil_method,
            max_iter=odil_max_iter,
            tol=1e-10,
            bc_weight=odil_bc_weight,
        )

        if not np.all(np.isfinite(odil_result.u_grid)):
            status = RunStatus.FAILED.value
            error = {"code": "non_finite", "message": "Non-finite ODIL solution"}
            stop_step = odil_result.n_iterations

        emit(
            progress_callback,
            started_at,
            70.0,
            "sampler",
            f"ODIL converged in {odil_result.n_iterations} iterations",
        )

        # Project the grid solution onto the sine basis so the rest of the
        # pipeline (chain, plots, predictives) keeps the same shape as PIFT/MC.
        basis_at_solver_grid = np.asarray(field.design_matrix(jnp.asarray(odil_result.x_grid)))
        theta_hat_np, *_ = np.linalg.lstsq(basis_at_solver_grid, odil_result.u_grid, rcond=None)
        theta_hat = theta_hat_np.astype(float)

        # Build a "chain" of length n_iter+1 by tiling the final theta.  This
        # preserves the (steps, n_modes) shape that the GUI/diagnostics expect
        # without pretending we have intermediate basis-coefficient samples.
        n_iter_total = int(odil_result.n_iterations) + 1
        chain = np.tile(theta_hat[None, :], (n_iter_total, 1))

        # Loss history goes into the hamiltonian trace; likelihood/physics are
        # not separately tracked by the solver, so we record zeros except for
        # the final step which holds the corresponding decomposed energies.
        ham_trace = odil_result.loss_history.astype(float)
        if ham_trace.size != n_iter_total:
            # Pad/truncate to match chain length so downstream slicing is safe.
            if ham_trace.size < n_iter_total:
                ham_trace = np.concatenate(
                    [ham_trace, np.full(n_iter_total - ham_trace.size, ham_trace[-1])]
                )
            else:
                ham_trace = ham_trace[:n_iter_total]

        # Compute final-iteration likelihood / physics components for diagnostics.
        theta_hat_j = jnp.asarray(theta_hat)
        x_quad = jnp.linspace(0.0, 1.0, max(128, int(config["n_quad"]) * 4))
        phys_energy_final, _, _ = poisson_residual_energy(theta_hat_j, x_quad, field, forcing)
        like_energy_final, _, _ = gaussian_nll(theta_hat_j, obs_matrix, y_obs, float(config["noise_std"]))
        like_trace = np.zeros(n_iter_total, dtype=float)
        phys_trace = np.zeros(n_iter_total, dtype=float)
        like_trace[-1] = float(like_energy_final)
        phys_trace[-1] = float(phys_energy_final)
        initial_physics_energy = float(phys_energy_final)
        initial_likelihood_energy = float(like_energy_final)

        effective_steps = n_iter_total
        effective_chain = chain
        traces = {
            "hamiltonian": ham_trace,
            "likelihood": like_trace,
            "physics": phys_trace,
        }

        emit(progress_callback, started_at, 84.0, "post", "Post-processing deterministic solution")

        x_grid = np.linspace(0.0, 1.0, int(config["n_grid"]))
        basis_grid = np.asarray(field.design_matrix(jnp.asarray(x_grid)))
        phi_mean = theta_hat @ basis_grid.T
        phi_std = np.zeros_like(phi_mean)
        phi_truth = np.asarray(phi_true(jnp.asarray(x_grid)))

        pred_obs_mean = np.asarray(obs_matrix) @ theta_hat
        pred_low = pred_obs_mean.copy()
        pred_high = pred_obs_mean.copy()

        x_ref, phi_ref = solve_poisson_dirichlet_fd(
            forcing_fn=lambda z: np.asarray(forcing(jnp.asarray(z)), dtype=float),
            n_points=int(config["n_grid"]),
            domain=(0.0, 1.0),
            bc=(0.0, 0.0),
        )

        x_obs_np = np.asarray(x_obs)
        y_obs_np = np.asarray(y_obs)
        y_clean_np = np.asarray(y_clean)

    l2_error = float(np.sqrt(np.mean((phi_mean - phi_truth) ** 2)))
    max_error = float(np.max(np.abs(phi_mean - phi_truth)))
    reference_l2_error = float(np.sqrt(np.mean((phi_mean - phi_ref) ** 2)))
    reference_max_error = float(np.max(np.abs(phi_mean - phi_ref)))
    reference_solver_truth_l2 = float(np.sqrt(np.mean((phi_ref - phi_truth) ** 2)))
    lag1 = lag1_autocorr(traces["hamiltonian"]) if traces["hamiltonian"].size > 2 else float("nan")
    coverage = float(np.mean((y_clean_np >= pred_low) & (y_clean_np <= pred_high)))
    runtime_sec = max(0.0, time.monotonic() - started_at)

    diagnostics = {
        "stability_flags": {
            "nan_detected": bool(np.isnan(traces["hamiltonian"]).any()),
            "inf_detected": bool(np.isinf(traces["hamiltonian"]).any()),
            "divergence_detected": False,
            "hamiltonian_overflow": False,
        },
        "first_bad_step": stop_step,
        "max_abs_hamiltonian": float(np.max(np.abs(traces["hamiltonian"]))) if traces["hamiltonian"].size else 0.0,
        "nan_detected": bool(np.isnan(traces["hamiltonian"]).any()),
        "suggested_actions": ["No critical issues detected"] if status == RunStatus.COMPLETED.value else ["Reduce odil_lr"],
        "initial_physics_energy": initial_physics_energy,
        "initial_likelihood_energy": initial_likelihood_energy,
    }

    summary = {
        "run_id": run_id,
        "config": config,
        "source_observations": observation_source,
        "device_requested": device_preference.lower(),
        "device_used": device_used,
        "n_posterior_samples": 1,
        "l2_error": l2_error,
        "max_error": max_error,
        "reference_l2_error": reference_l2_error,
        "reference_max_error": reference_max_error,
        "reference_solver_truth_l2": reference_solver_truth_l2,
        "hamiltonian_lag1_autocorr": lag1,
        "measurement_interval_coverage_90pct": coverage,
        "status": status,
    }

    artifacts = {
        "field_plot": str(out_fig / "phase_a_odil_field.png"),
        "trace_plot": str(out_fig / "phase_a_odil_trace.png"),
        "chain": str(out_traces / "phase_a_odil_chain.npz"),
        "summary": str(out_tables / "phase_a_odil_summary.json"),
    }

    if save_outputs:
        emit(progress_callback, started_at, 92.0, "artifacts", "Writing field and trace plots")
        plot_field_summary(
            x_grid=x_grid,
            truth=phi_truth,
            mean=phi_mean,
            std=phi_std,
            x_obs=x_obs_np,
            y_obs=y_obs_np,
            out_path=artifacts["field_plot"],
        )
        plot_diagnostics(traces["hamiltonian"], out_path=artifacts["trace_plot"])

        np.savez(
            artifacts["chain"],
            chain=effective_chain,
            hamiltonian=traces["hamiltonian"],
            likelihood=traces["likelihood"],
            physics=traces["physics"],
            x_obs=x_obs_np,
            y_obs=y_obs_np,
            pred_obs_mean=pred_obs_mean,
            pred_obs_low=pred_low,
            pred_obs_high=pred_high,
            status=status,
            runtime_sec=runtime_sec,
        )

        payload = {
            "summary": summary,
            "metrics": {
                "runtime_sec": runtime_sec,
                "n_total_steps": int(n_iter_total),
                "n_effective_steps": int(effective_steps),
                "reference_solver_truth_l2": reference_solver_truth_l2,
                "odil_iterations": int(odil_result.n_iterations),
                "odil_converged": bool(odil_result.converged),
                "odil_method": str(odil_result.method_used),
                "odil_runtime_sec": float(odil_result.runtime_sec),
            },
            "diagnostics": diagnostics,
            "error": error,
            "artifacts": artifacts,
        }
        with open(artifacts["summary"], "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=str)

    final_detail = "Completed"
    if status == RunStatus.FAILED.value:
        final_detail = "Failed"
    elif status == RunStatus.STOPPED.value:
        final_detail = "Stopped"
    emit(progress_callback, started_at, 100.0, "complete", final_detail)

    return {
        "status": status,
        "summary": summary,
        "metrics": {
            "runtime_sec": runtime_sec,
            "n_total_steps": int(n_iter_total),
            "n_effective_steps": int(effective_steps),
            "reference_solver_truth_l2": reference_solver_truth_l2,
            "odil_iterations": int(odil_result.n_iterations),
            "odil_converged": bool(odil_result.converged),
            "odil_method": str(odil_result.method_used),
            "odil_runtime_sec": float(odil_result.runtime_sec),
        },
        "diagnostics": diagnostics,
        "artifacts": artifacts,
        "error": error,
        "x_grid": x_grid,
        "phi_truth": phi_truth,
        "x_reference": x_ref,
        "phi_reference": phi_ref,
        "phi_mean": phi_mean,
        "phi_std": phi_std,
        "x_obs": x_obs_np,
        "y_obs": y_obs_np,
        "traces": traces,
        "output_paths": artifacts,
    }
