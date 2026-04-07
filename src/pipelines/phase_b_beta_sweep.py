"""Phase B: Beta sensitivity sweep reproducing Figure 1 from Alberts & Bilionis."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from contextlib import nullcontext
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from core.energies import variational_heat_energy
from core.parameterizations import BoundaryFourierField
from core.reference_solver import solve_poisson_dirichlet_fd
from core.sgld import sgld_sample
from utils.diagnostics_runtime import RuntimeGuard, RuntimeGuardConfig
from utils.plotting import plot_beta_sweep_panels, plot_variance_scaling
from .common import emit, select_device

jax.config.update("jax_enable_x64", True)

DEFAULT_PHASE_B_CONFIG: dict[str, float | int | list] = {
    "seed": 42,
    "K": 20,
    "D": 0.25,
    "bc_left": 1.0,
    "bc_right": 0.1,
    "beta_values": [1, 10, 100, 1000],
    "n_steps": 50000,
    "burn_in": 10000,
    "thin": 20,
    "step_size0": 0.1,
    "decay": 0.51,
    "n_quad": 128,
    "n_grid": 300,
    "max_condition_number": 100.0,
    "runtime_check_interval": 5,
}


def _source_fn(x):
    """Source term q(x) = exp(-x) for the heat equation."""
    return jnp.exp(-x)


def run_phase_b_beta_sweep(
    cfg: dict | None = None,
    output_root: str | Path = "outputs",
    device_preference: str = "cpu",
    run_id: str | None = None,
    stop_signal: Callable[[], bool] | None = None,
    progress_callback: Callable[[float, str, str, float], None] | None = None,
    save_outputs: bool = True,
) -> dict[str, object]:
    """Run beta sensitivity sweep for 1D steady-state heat equation.

    Reproduces Figure 1 from Alberts & Bilionis: posterior variance
    collapses as beta -> infinity.

    Returns dict with keys: ``status``, ``beta_results``,
    ``variance_scaling``, ``x_grid``, ``phi_truth``, ``artifacts``.
    """
    started_at = time.monotonic()
    config = dict(DEFAULT_PHASE_B_CONFIG)
    if cfg:
        config.update(cfg)

    beta_values = list(config["beta_values"])
    K = int(config["K"])
    D = float(config["D"])
    bc_left = float(config["bc_left"])
    bc_right = float(config["bc_right"])
    n_steps = int(config["n_steps"])
    burn_in = int(config["burn_in"])
    thin = int(config["thin"])
    step_size0 = float(config["step_size0"])
    decay = float(config["decay"])
    n_quad = int(config["n_quad"])
    n_grid = int(config["n_grid"])
    _max_cond = float(config.get("max_condition_number", 100.0))
    _check_interval = int(config.get("runtime_check_interval", 5))

    print(
        f"[phase_b] Beta sweep: betas={beta_values} K={K} D={D} "
        f"n_steps={n_steps} step_size0={step_size0}"
    )

    emit(progress_callback, started_at, 2.0, "setup", "Initializing beta sweep")

    device, device_used = select_device(device_preference)
    device_ctx = (
        jax.default_device(device) if hasattr(jax, "default_device") else nullcontext()
    )

    output_root = Path(output_root)
    out_fig = output_root / "figures"
    out_tables = output_root / "tables"
    out_fig.mkdir(parents=True, exist_ok=True)
    out_tables.mkdir(parents=True, exist_ok=True)

    # ----- Field and reference solution -----
    field = BoundaryFourierField(K=K, bc=(bc_left, bc_right))

    # FD solver solves -phi'' = f; our PDE is -D*phi'' = q, so f = q/D.
    def fd_forcing(x):
        return np.exp(-x) / D

    x_ref, phi_ref = solve_poisson_dirichlet_fd(
        fd_forcing, n_points=500, bc=(bc_left, bc_right),
    )
    emit(progress_callback, started_at, 5.0, "setup", "Computed FD reference solution")

    # ----- Initialize theta0 via lstsq projection -----
    # Subtract BC ramp, divide by window, fit psi on interior points.
    interior = (x_ref > 0.02) & (x_ref < 0.98)
    x_int = x_ref[interior]
    phi_int = phi_ref[interior]
    ramp_int = (1.0 - x_int) * bc_left + x_int * bc_right
    window_int = (1.0 - x_int) * x_int
    psi_target = (phi_int - ramp_int) / window_int

    psi_dm = np.asarray(field.psi_design_matrix(jnp.asarray(x_int)))
    theta0, *_ = np.linalg.lstsq(psi_dm, psi_target, rcond=None)
    theta0 = jnp.asarray(theta0, dtype=jnp.float64)

    emit(progress_callback, started_at, 8.0, "setup", "Projected FD solution onto Fourier basis")

    # ----- Evaluation grid -----
    x_grid = np.linspace(0.0, 1.0, n_grid)
    phi_truth = np.asarray(phi_ref)
    x_grid_ref = x_ref  # keep for truth interpolation
    # Interpolate truth onto evaluation grid
    phi_truth_grid = np.interp(x_grid, x_grid_ref, phi_truth)

    # ----- Beta sweep -----
    key = jax.random.PRNGKey(int(config["seed"]))
    beta_results = []
    n_betas = len(beta_values)

    with device_ctx:
        for bi, beta_val in enumerate(beta_values):
            beta_val = float(beta_val)
            pct_base = 10.0 + 80.0 * bi / n_betas
            pct_end = 10.0 + 80.0 * (bi + 1) / n_betas
            emit(
                progress_callback, started_at, pct_base,
                "sampling", f"Beta={beta_val:.0f} — starting SGLD",
            )
            print(f"[phase_b] Running beta={beta_val:.0f} ...")

            key, sgld_key, quad_base_key = jax.random.split(key, 3)
            guard = RuntimeGuard(RuntimeGuardConfig())

            def _guard_check(step, theta, grad, metrics, _g=guard):
                return _g.check(step, theta, grad, metrics)

            # Closure over beta_val for grad_and_metrics
            _beta = beta_val
            _quad_key_holder = [quad_base_key]

            def grad_and_metrics(theta, sub_key=None, _beta=_beta, _holder=_quad_key_holder):
                if sub_key is None:
                    _holder[0], qk = jax.random.split(_holder[0])
                else:
                    qk = sub_key
                x_q = jax.random.uniform(qk, shape=(n_quad,), minval=0.0, maxval=1.0)
                phys_energy, phys_grad, _ = variational_heat_energy(
                    theta, x_q, field, _source_fn, D,
                )
                grad = _beta * phys_grad
                metrics = {
                    "likelihood": 0.0,
                    "physics": phys_energy,
                    "hamiltonian": _beta * phys_energy,
                }
                return grad, metrics

            precond = field.preconditioner(beta=beta_val)

            def _sgld_progress(step, total, _base=pct_base, _end=pct_end, _b=beta_val):
                frac = step / max(1, total)
                pct = _base + (_end - _base) * frac
                emit(
                    progress_callback, started_at, pct,
                    "sampling",
                    f"Beta={_b:.0f} — SGLD step {step}/{total}",
                )

            chain, traces, sgld_meta = sgld_sample(
                theta0=theta0,
                grad_and_metrics_fn=grad_and_metrics,
                n_steps=n_steps,
                step_size0=step_size0,
                decay=decay,
                key=sgld_key,
                progress_callback=_sgld_progress,
                progress_interval=max(1, n_steps // 50),
                stop_signal=stop_signal,
                runtime_check=_guard_check,
                runtime_check_interval=_check_interval,
                preconditioner=precond,
                max_condition_number=_max_cond,
            )

            if sgld_meta.get("stopped"):
                print(
                    f"[phase_b] WARNING: beta={beta_val:.0f} stopped early: "
                    f"{sgld_meta.get('error_code')} — {sgld_meta.get('error_detail')}"
                )

            # Post-process
            samples = chain[burn_in::thin]
            x_grid_jnp = jnp.asarray(x_grid)
            phi_samples = np.array(
                [np.asarray(field.eval(x_grid_jnp, s)) for s in samples]
            )
            phi_mean = np.mean(phi_samples, axis=0)
            phi_std = np.std(phi_samples, axis=0)

            # Variance at midpoint for Klein-Gordon check
            mid_idx = n_grid // 2
            var_mid = float(phi_std[mid_idx] ** 2)

            beta_results.append({
                "beta": beta_val,
                "phi_mean": phi_mean,
                "phi_std": phi_std,
                "phi_samples": phi_samples,
                "variance_at_midpoint": var_mid,
                "n_samples": len(samples),
                "stopped_early": bool(sgld_meta.get("stopped")),
                "traces": {k: np.asarray(v) for k, v in traces.items()},
            })

            emit(
                progress_callback, started_at, pct_end,
                "sampling",
                f"Beta={beta_val:.0f} done — var(x=0.5)={var_mid:.6f}, n_samples={len(samples)}",
            )

    # ----- Variance scaling diagnostic -----
    betas_arr = [r["beta"] for r in beta_results]
    vars_arr = [r["variance_at_midpoint"] for r in beta_results]
    var_times_beta = [v * b for v, b in zip(vars_arr, betas_arr)]
    variance_scaling = {
        "betas": betas_arr,
        "variances": vars_arr,
        "var_times_beta": var_times_beta,
    }
    print(f"[phase_b] Variance scaling check (var*beta): {var_times_beta}")

    # ----- Save outputs -----
    artifacts = {}
    if save_outputs:
        panel_path = str(out_fig / "phase_b_beta_sweep_panels.png")
        plot_beta_sweep_panels(x_grid, phi_truth_grid, beta_results, out_path=panel_path)
        artifacts["panel_figure"] = panel_path

        scaling_path = str(out_fig / "phase_b_variance_scaling.png")
        plot_variance_scaling(betas_arr, vars_arr, out_path=scaling_path)
        artifacts["scaling_figure"] = scaling_path

        summary = {
            "betas": betas_arr,
            "variances_at_midpoint": vars_arr,
            "var_times_beta": var_times_beta,
            "n_samples_per_beta": [r["n_samples"] for r in beta_results],
            "config": {k: v for k, v in config.items()},
            "runtime_sec": time.monotonic() - started_at,
        }
        summary_path = str(out_tables / "phase_b_beta_sweep_summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        artifacts["summary"] = summary_path

        emit(progress_callback, started_at, 95.0, "output", "Saved figures and summary")

    emit(progress_callback, started_at, 100.0, "done", "Beta sweep complete")

    return {
        "status": "completed",
        "x_grid": x_grid,
        "phi_truth": phi_truth_grid,
        "beta_results": beta_results,
        "variance_scaling": variance_scaling,
        "artifacts": artifacts,
        "config": config,
        "runtime_sec": time.monotonic() - started_at,
    }
