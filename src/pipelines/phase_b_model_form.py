"""Phase B: Model-form uncertainty — Example 2 from Alberts & Bilionis.

Reproduces Figure 2: inferred beta increases as physics correctness gamma
increases. Two experiments:

  (a) Source term error: f_gamma(x) = gamma*cos(4x) + (1-gamma)*exp(-x)
  (b) Energy functional error: U_gamma mixes phi^4 (correct) with phi^2

For each gamma, nested SGLD infers log(beta) with Jeffrey's prior.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from contextlib import nullcontext
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from core.energies_nonlinear import nonlinear_energy, nonlinear_energy_misspecified
from core.nested_sgld import nested_sgld
from core.parameterizations_nonlinear import make_nonlinear_field, solve_nonlinear_dirichlet_fd
from .common import emit, select_device

jax.config.update("jax_enable_x64", True)

# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------

DEFAULT_PHASE_B_MODEL_FORM_CONFIG: dict[str, object] = {
    "seed": 42,
    # PDE ground truth
    "D": 0.1,
    "kappa": 1.0,
    "bc_left": 0.0,
    "bc_right": 0.0,
    # Field parameterization
    "K": 20,
    # Observations
    "n_obs": 40,
    "noise_std": 0.01,
    # Model correctness sweep
    "gamma_values": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    # Nested SGLD
    "outer_steps": 5000,
    "outer_step_size0": 0.1,
    "outer_decay": 0.51,
    "inner_T_prior": 10,
    "inner_T_posterior": 1,
    "inner_step_size0": 0.1,
    "inner_decay": 0.51,
    "warmup_steps": 5000,
    "n_quad": 128,
    "max_condition_number": 100.0,
    # Post-processing
    "burn_in_frac": 0.2,
    "n_grid": 300,
}


# ---------------------------------------------------------------------------
# Source term constructors
# ---------------------------------------------------------------------------

def _true_source(x):
    """Ground truth source f(x) = cos(4x)."""
    return jnp.cos(4.0 * x)


def _true_source_np(x):
    return np.cos(4.0 * x)


def _make_misspecified_source(gamma: float):
    """Experiment A source: f_gamma(x) = gamma*cos(4x) + (1-gamma)*exp(-x)."""
    def source(x):
        return gamma * jnp.cos(4.0 * x) + (1.0 - gamma) * jnp.exp(-x)
    return source


# ---------------------------------------------------------------------------
# Gradient function builders
# ---------------------------------------------------------------------------

def _build_grad_fns(
    field,
    source_fn,
    x_obs,
    y_obs,
    noise_std: float,
    n_quad: int,
    D: float,
    kappa: float,
    experiment: str,
    gamma: float,
):
    """Build prior_grad_fn and posterior_grad_fn for nested SGLD.

    Parameters
    ----------
    experiment : "source_error" or "energy_error"
    gamma : model correctness in [0, 1]
    """
    inv_var = 1.0 / (noise_std ** 2)

    def _physics_energy_and_grad(phi, x_q):
        """Compute energy and gradient for the (possibly misspecified) physics."""
        if experiment == "source_error":
            return nonlinear_energy(phi, x_q, field, source_fn, D, kappa)
        else:  # energy_error
            return nonlinear_energy_misspecified(
                phi, x_q, field, _true_source, D, kappa, gamma
            )

    def _likelihood_energy_and_grad(phi):
        """Gaussian NLL: 0.5/sigma^2 * ||y - phi(x_obs)||^2."""
        phi_arr = jnp.asarray(phi, dtype=jnp.float64)

        def _nll(phi_inner):
            pred = field.eval(x_obs, phi_inner)
            return 0.5 * inv_var * jnp.sum((pred - y_obs) ** 2)

        nll_val, nll_grad = jax.value_and_grad(_nll)(phi_arr)
        return float(nll_val), nll_grad

    def prior_grad_fn(phi, lam, key):
        beta = jnp.exp(lam[0])
        key, subkey = jax.random.split(key)
        x_q = jax.random.uniform(subkey, (n_quad,), dtype=jnp.float64)

        e_phys, g_phys, _ = _physics_energy_and_grad(phi, x_q)

        grad_phi = beta * g_phys
        # d/d(log beta) [beta * U] = beta * U
        grad_lam = jnp.array([beta * e_phys])
        ham = float(beta * e_phys)

        return grad_phi, grad_lam, {"hamiltonian": ham, "physics": e_phys}

    def posterior_grad_fn(phi, lam, key):
        beta = jnp.exp(lam[0])
        key, subkey = jax.random.split(key)
        x_q = jax.random.uniform(subkey, (n_quad,), dtype=jnp.float64)

        e_phys, g_phys, _ = _physics_energy_and_grad(phi, x_q)
        e_like, g_like = _likelihood_energy_and_grad(phi)

        grad_phi = beta * g_phys + g_like
        # Likelihood doesn't depend on beta, so grad_lam same as prior
        grad_lam = jnp.array([beta * e_phys])
        ham = float(beta * e_phys + e_like)

        return grad_phi, grad_lam, {
            "hamiltonian": ham,
            "physics": e_phys,
            "likelihood": e_like,
        }

    return prior_grad_fn, posterior_grad_fn


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_phase_b_model_form(
    cfg: dict | None = None,
    output_root: str | Path = "outputs",
    device_preference: str = "cpu",
    run_id: str | None = None,
    stop_signal: Callable[[], bool] | None = None,
    progress_callback: Callable[[float, str, str, float], None] | None = None,
    save_outputs: bool = True,
) -> dict[str, object]:
    """Run model-form uncertainty experiment (Example 2).

    For each experiment type (source_error, energy_error) and each gamma,
    runs nested SGLD to infer log(beta) with Jeffrey's prior.

    Returns dict with ``status``, ``experiments``, ``artifacts``, ``config``.
    """
    started_at = time.monotonic()
    config = dict(DEFAULT_PHASE_B_MODEL_FORM_CONFIG)
    if cfg:
        config.update(cfg)

    D = float(config["D"])
    kappa = float(config["kappa"])
    K = int(config["K"])
    bc = (float(config["bc_left"]), float(config["bc_right"]))
    n_obs = int(config["n_obs"])
    noise_std = float(config["noise_std"])
    gamma_values = list(config["gamma_values"])
    outer_steps = int(config["outer_steps"])
    outer_step_size0 = float(config["outer_step_size0"])
    outer_decay = float(config["outer_decay"])
    inner_T_prior = int(config["inner_T_prior"])
    inner_T_posterior = int(config["inner_T_posterior"])
    inner_step_size0 = float(config["inner_step_size0"])
    inner_decay = float(config["inner_decay"])
    warmup_steps = int(config["warmup_steps"])
    n_quad = int(config["n_quad"])
    max_cond = float(config["max_condition_number"])
    burn_in_frac = float(config["burn_in_frac"])
    n_grid = int(config["n_grid"])

    print(
        f"[model_form] Example 2: D={D}, kappa={kappa}, "
        f"gammas={gamma_values}, outer_steps={outer_steps}"
    )
    emit(progress_callback, started_at, 1.0, "setup", "Initializing model-form experiment")

    device, _ = select_device(device_preference)
    device_ctx = (
        jax.default_device(device) if hasattr(jax, "default_device") else nullcontext()
    )

    output_root = Path(output_root)
    out_fig = output_root / "figures"
    out_tables = output_root / "tables"
    out_fig.mkdir(parents=True, exist_ok=True)
    out_tables.mkdir(parents=True, exist_ok=True)

    # ----- Ground truth and observations -----
    field = make_nonlinear_field(K=K, bc=bc)
    x_ref, phi_ref = solve_nonlinear_dirichlet_fd(
        _true_source_np, D=D, kappa=kappa, n_points=500, bc=bc,
    )

    key = jax.random.PRNGKey(int(config["seed"]))
    key, obs_key = jax.random.split(key)

    # Equidistant observations excluding boundaries
    x_obs = jnp.linspace(0.0, 1.0, n_obs + 2)[1:-1]
    phi_at_obs = jnp.interp(x_obs, jnp.asarray(x_ref), jnp.asarray(phi_ref))
    y_obs = phi_at_obs + noise_std * jax.random.normal(obs_key, shape=(n_obs,))

    emit(progress_callback, started_at, 5.0, "setup", f"Generated {n_obs} observations")

    # ----- Initial theta via lstsq projection of FD solution -----
    interior = (x_ref > 0.02) & (x_ref < 0.98)
    x_int = x_ref[interior]
    phi_int = phi_ref[interior]
    ramp_int = (1.0 - x_int) * bc[0] + x_int * bc[1]
    window_int = (1.0 - x_int) * x_int
    psi_target = (phi_int - ramp_int) / (window_int + 1e-12)

    psi_dm = np.asarray(field.psi_design_matrix(jnp.asarray(x_int)))
    theta0, *_ = np.linalg.lstsq(psi_dm, psi_target, rcond=None)
    phi0 = jnp.asarray(theta0, dtype=jnp.float64)

    # Initial log(beta) — start at moderate value
    lambda0 = jnp.array([np.log(1e4)], dtype=jnp.float64)

    precond = field.preconditioner()

    emit(progress_callback, started_at, 8.0, "setup", "Setup complete")

    # ----- Run experiments -----
    experiments = {}
    experiment_types = ["source_error", "energy_error"]
    total_runs = len(experiment_types) * len(gamma_values)
    run_idx = 0

    with device_ctx:
        for exp_type in experiment_types:
            exp_results = []
            for gamma in gamma_values:
                pct = 10.0 + 85.0 * run_idx / total_runs
                emit(
                    progress_callback, started_at, pct,
                    "sampling",
                    f"{exp_type} gamma={gamma:.2f}",
                )
                print(f"[model_form] {exp_type} gamma={gamma:.2f} ...")

                # Build source function for this gamma
                if exp_type == "source_error":
                    source_fn = _make_misspecified_source(gamma)
                else:
                    source_fn = _true_source  # energy error uses correct source

                prior_grad, posterior_grad = _build_grad_fns(
                    field=field,
                    source_fn=source_fn,
                    x_obs=x_obs,
                    y_obs=y_obs,
                    noise_std=noise_std,
                    n_quad=n_quad,
                    D=D,
                    kappa=kappa,
                    experiment=exp_type,
                    gamma=gamma,
                )

                key, sgld_key = jax.random.split(key)

                result = nested_sgld(
                    lambda0=lambda0,
                    phi0=phi0,
                    prior_grad_fn=prior_grad,
                    posterior_grad_fn=posterior_grad,
                    key=sgld_key,
                    lambda_prior_grad_fn=None,  # Jeffrey's prior
                    outer_steps=outer_steps,
                    outer_step_size0=outer_step_size0,
                    outer_decay=outer_decay,
                    inner_T_prior=inner_T_prior,
                    inner_T_posterior=inner_T_posterior,
                    inner_step_size0=inner_step_size0,
                    inner_decay=inner_decay,
                    warmup_steps=warmup_steps,
                    phi_preconditioner=precond,
                    max_condition_number=max_cond,
                    stop_signal=stop_signal,
                )

                # Extract beta samples (discard burn-in)
                burn_in = int(burn_in_frac * outer_steps)
                log_beta_chain = result.lambda_chain[burn_in:, 0]
                beta_samples = np.exp(log_beta_chain)

                # Compute summary statistics
                beta_median = float(np.median(beta_samples))
                beta_q05 = float(np.percentile(beta_samples, 5))
                beta_q25 = float(np.percentile(beta_samples, 25))
                beta_q75 = float(np.percentile(beta_samples, 75))
                beta_q95 = float(np.percentile(beta_samples, 95))

                stopped = bool(result.meta.get("stopped", False))
                if stopped:
                    print(
                        f"[model_form] WARNING: {exp_type} gamma={gamma:.2f} "
                        f"stopped early: {result.meta.get('error_detail')}"
                    )

                exp_results.append({
                    "gamma": gamma,
                    "beta_median": beta_median,
                    "beta_q05": beta_q05,
                    "beta_q25": beta_q25,
                    "beta_q75": beta_q75,
                    "beta_q95": beta_q95,
                    "beta_samples": beta_samples,
                    "log_beta_chain": log_beta_chain,
                    "stopped_early": stopped,
                })

                print(
                    f"[model_form]   beta median={beta_median:.1f} "
                    f"[{beta_q05:.1f}, {beta_q95:.1f}]"
                )
                run_idx += 1

            experiments[exp_type] = exp_results

    # ----- Save outputs -----
    artifacts = {}
    if save_outputs:
        from utils.plotting import plot_model_form_beta_vs_gamma

        fig_path = str(out_fig / "phase_b_model_form.png")
        plot_model_form_beta_vs_gamma(experiments, out_path=fig_path)
        artifacts["figure"] = fig_path

        # Summary JSON (without large arrays)
        summary = {
            "config": {k: v for k, v in config.items()},
            "runtime_sec": time.monotonic() - started_at,
        }
        for exp_type in experiment_types:
            summary[exp_type] = [
                {
                    "gamma": r["gamma"],
                    "beta_median": r["beta_median"],
                    "beta_q05": r["beta_q05"],
                    "beta_q25": r["beta_q25"],
                    "beta_q75": r["beta_q75"],
                    "beta_q95": r["beta_q95"],
                    "stopped_early": r["stopped_early"],
                }
                for r in experiments[exp_type]
            ]
        summary_path = str(out_tables / "phase_b_model_form_summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        artifacts["summary"] = summary_path

    emit(progress_callback, started_at, 100.0, "done", "Model-form experiment complete")

    return {
        "status": "completed",
        "experiments": experiments,
        "artifacts": artifacts,
        "config": config,
        "runtime_sec": time.monotonic() - started_at,
    }
