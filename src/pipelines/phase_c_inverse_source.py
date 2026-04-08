"""Phase C: Source term identification (Example 3b, Fig. 4).

Jointly infer D, κ, and the source term f(x) of the nonlinear PDE
D·φ'' − κ·φ³ = f via nested SGLD.  The source is parameterised with a
truncated Karhunen-Loève expansion:

    f(x; z) = Σᵢ zᵢ √λᵢ φᵢ(x)

where the zᵢ have iid N(0,1) priors and (λᵢ, φᵢ) come from a
squared-exponential covariance kernel.
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

from core.kle import KLE
from core.nested_sgld import nested_sgld
from core.parameterizations_nonlinear import make_nonlinear_field, solve_nonlinear_dirichlet_fd
from utils.plotting import plot_figure4
from .common import emit, select_device

jax.config.update("jax_enable_x64", True)

DEFAULT_PHASE_C_SOURCE_CONFIG: dict[str, float | int] = {
    "seed": 42,
    "K": 20,
    "D_true": 0.1,
    "kappa_true": 1.0,
    "beta": 1e5,
    "n_obs": 40,
    "noise_std": 0.01,
    # KLE
    "kle_lengthscale": 0.3,
    "kle_n_terms": 10,
    "kle_n_grid": 200,
    # Initial guess (log-space for D, κ; zero for z)
    "lambda0_log_D": 0.0,
    "lambda0_log_kappa": 0.5,
    # Nested SGLD
    "warmup_steps": 50_000,
    "outer_steps": 10_000,
    "outer_step_size0": 0.1,
    "outer_decay": 0.51,
    "inner_T_prior": 10,
    "inner_T_posterior": 1,
    "inner_step_size0": 0.1,
    "inner_decay": 0.51,
    # Quadrature / grid
    "n_quad": 128,
    "n_grid": 300,
    "max_condition_number": 100.0,
    # Post-processing
    "burn_in_frac": 0.2,
    "snapshot_interval": 100,
}


def _source_fn_true(x):
    """Ground-truth source f(x) = cos(4x)."""
    return jnp.cos(4.0 * x)


def _source_fn_true_np(x):
    return np.cos(4.0 * x)


def _build_grad_fns(field, kle, beta, n_quad, x_obs, y_obs, noise_std):
    """Build prior/posterior gradient closures for source identification.

    λ = [log(D), log(κ), z₁, ..., z_{n_kle}]
    """

    def _physics_energy(phi, lam, x_quad):
        """β · U[φ; D, κ, f(z)] differentiable in phi and lam."""
        D = jnp.exp(lam[0])
        kappa = jnp.exp(lam[1])
        z = lam[2:]
        phi_vals = field.eval(x_quad, phi)
        dphi_vals = field.deriv(x_quad, phi)
        f_vals = kle.reconstruct(x_quad, z)
        return beta * jnp.mean(
            0.5 * D * dphi_vals**2 + 0.25 * kappa * phi_vals**4 + phi_vals * f_vals
        )

    def _data_nll(phi):
        pred = field.eval(x_obs, phi)
        return 0.5 / (noise_std**2) * jnp.sum((pred - y_obs) ** 2)

    def prior_grad_fn(phi, lam, key):
        key, qk = jax.random.split(key)
        x_quad = jax.random.uniform(qk, shape=(n_quad,), minval=0.0, maxval=1.0)

        energy_val = _physics_energy(phi, lam, x_quad)
        grad_phi, grad_lam = jax.grad(_physics_energy, argnums=(0, 1))(phi, lam, x_quad)
        # Keep traced (no float cast) so this works inside lax.scan / jit.
        return grad_phi, grad_lam, {"hamiltonian": energy_val, "physics": energy_val, "likelihood": jnp.zeros(())}

    def posterior_grad_fn(phi, lam, key):
        key, qk = jax.random.split(key)
        x_quad = jax.random.uniform(qk, shape=(n_quad,), minval=0.0, maxval=1.0)

        phys_energy_val = _physics_energy(phi, lam, x_quad)
        grad_phi_phys, grad_lam = jax.grad(_physics_energy, argnums=(0, 1))(phi, lam, x_quad)

        nll_val = _data_nll(phi)
        grad_phi_data = jax.grad(_data_nll)(phi)

        total_grad_phi = grad_phi_phys + grad_phi_data
        return (
            total_grad_phi,
            grad_lam,
            {"hamiltonian": phys_energy_val + nll_val, "physics": phys_energy_val, "likelihood": nll_val},
        )

    return prior_grad_fn, posterior_grad_fn


def _lambda_prior_grad(lam):
    """Combined prior gradient: Jeffrey's on D,κ (zero) + N(0,1) on z (grad = z)."""
    grad = jnp.zeros_like(lam)
    return grad.at[2:].set(lam[2:])


def run_phase_c_inverse_source(
    cfg: dict | None = None,
    output_root: str | Path = "outputs",
    device_preference: str = "cpu",
    run_id: str | None = None,
    stop_signal: Callable[[], bool] | None = None,
    progress_callback: Callable[[float, str, str, float], None] | None = None,
    save_outputs: bool = True,
) -> dict[str, object]:
    """Run Example 3b: jointly infer D, κ, and f(x) via nested SGLD + KLE."""
    started_at = time.monotonic()
    config = dict(DEFAULT_PHASE_C_SOURCE_CONFIG)
    if cfg:
        config.update(cfg)

    K = int(config["K"])
    D_true = float(config["D_true"])
    kappa_true = float(config["kappa_true"])
    beta = float(config["beta"])
    n_obs = int(config["n_obs"])
    noise_std = float(config["noise_std"])
    n_quad = int(config["n_quad"])
    n_grid = int(config["n_grid"])
    warmup_steps = int(config["warmup_steps"])
    outer_steps = int(config["outer_steps"])
    snapshot_interval = int(config.get("snapshot_interval", 100))
    burn_in_frac = float(config.get("burn_in_frac", 0.2))
    kle_n_terms = int(config.get("kle_n_terms", 10))

    print(
        f"[phase_c_source] Inverse source: D_true={D_true} kappa_true={kappa_true} "
        f"beta={beta:.0e} KLE_terms={kle_n_terms} outer={outer_steps} warmup={warmup_steps}"
    )

    emit(progress_callback, started_at, 2.0, "setup", "Initializing source identification")

    device, device_used = select_device(device_preference)
    device_ctx = jax.default_device(device) if hasattr(jax, "default_device") else nullcontext()

    output_root = Path(output_root)
    out_fig = output_root / "figures"
    out_tables = output_root / "tables"
    out_traces = output_root / "traces"
    out_fig.mkdir(parents=True, exist_ok=True)
    out_tables.mkdir(parents=True, exist_ok=True)
    out_traces.mkdir(parents=True, exist_ok=True)

    # ----- KLE setup -----
    kle = KLE(
        lengthscale=float(config.get("kle_lengthscale", 0.3)),
        n_terms=kle_n_terms,
        n_grid=int(config.get("kle_n_grid", 200)),
    )
    print(f"[phase_c_source] KLE energy fraction: {kle.energy_fraction():.4f}")
    emit(progress_callback, started_at, 4.0, "setup", f"KLE: {kle_n_terms} terms, energy fraction={kle.energy_fraction():.3f}")

    # ----- Field and reference solution -----
    field = make_nonlinear_field(K=K, bc=(0.0, 0.0))
    x_ref, phi_ref = solve_nonlinear_dirichlet_fd(
        _source_fn_true_np, D=D_true, kappa=kappa_true, n_points=500, bc=(0.0, 0.0),
    )
    emit(progress_callback, started_at, 6.0, "setup", "Computed FD reference solution")

    # ----- Generate observations -----
    key = jax.random.PRNGKey(int(config["seed"]))
    x_obs_np = np.linspace(0.0, 1.0, n_obs + 2)[1:-1]
    phi_obs_true = np.interp(x_obs_np, x_ref, phi_ref)
    key, noise_key = jax.random.split(key)
    noise = float(noise_std) * np.asarray(jax.random.normal(noise_key, shape=(n_obs,)))
    y_obs_np = phi_obs_true + noise

    x_obs_jnp = jnp.asarray(x_obs_np)
    y_obs_jnp = jnp.asarray(y_obs_np)
    emit(progress_callback, started_at, 8.0, "data", f"Generated {n_obs} observations (σ={noise_std})")

    # ----- Initialize phi0 via lstsq -----
    interior = (x_ref > 0.02) & (x_ref < 0.98)
    x_int = x_ref[interior]
    phi_int = phi_ref[interior]
    window_int = (1.0 - x_int) * x_int
    psi_target = phi_int / window_int

    psi_dm = np.asarray(field.psi_design_matrix(jnp.asarray(x_int)))
    phi0, *_ = np.linalg.lstsq(psi_dm, psi_target, rcond=None)
    phi0 = jnp.asarray(phi0, dtype=jnp.float64)

    # ----- Initialize lambda0 -----
    # [log(D), log(κ), z₁, ..., z_n]
    n_lam = 2 + kle_n_terms
    lambda0 = jnp.zeros(n_lam, dtype=jnp.float64)
    lambda0 = lambda0.at[0].set(float(config["lambda0_log_D"]))
    lambda0 = lambda0.at[1].set(float(config["lambda0_log_kappa"]))
    # z_i start at zero (prior mean)
    emit(progress_callback, started_at, 10.0, "setup", f"λ₀: log(D)={float(lambda0[0]):.2f}, log(κ)={float(lambda0[1]):.2f}, z=zeros({kle_n_terms})")

    # ----- Build gradient closures -----
    prior_grad_fn, posterior_grad_fn = _build_grad_fns(
        field, kle, beta, n_quad, x_obs_jnp, y_obs_jnp, noise_std,
    )

    # ----- Preconditioner -----
    precond = field.preconditioner(beta=beta)

    # ----- Run nested SGLD -----
    emit(progress_callback, started_at, 12.0, "sampling", "Starting nested SGLD")

    with device_ctx:
        result = nested_sgld(
            lambda0=lambda0,
            phi0=phi0,
            prior_grad_fn=prior_grad_fn,
            posterior_grad_fn=posterior_grad_fn,
            key=key,
            lambda_prior_grad_fn=_lambda_prior_grad,
            outer_steps=outer_steps,
            outer_step_size0=float(config["outer_step_size0"]),
            outer_decay=float(config["outer_decay"]),
            inner_T_prior=int(config["inner_T_prior"]),
            inner_T_posterior=int(config["inner_T_posterior"]),
            inner_step_size0=float(config["inner_step_size0"]),
            inner_decay=float(config["inner_decay"]),
            warmup_steps=warmup_steps,
            phi_preconditioner=precond,
            max_condition_number=float(config["max_condition_number"]),
            snapshot_interval=snapshot_interval,
            stop_signal=stop_signal,
        )

    emit(progress_callback, started_at, 85.0, "sampling", "Nested SGLD complete")

    if result.meta.get("stopped"):
        print(
            f"[phase_c_source] WARNING: stopped early at step {result.meta['stop_step']}: "
            f"{result.meta.get('error_code')}"
        )

    # ----- Post-processing -----
    lambda_chain = result.lambda_chain
    burn_in = int(outer_steps * burn_in_frac)
    post_chain = lambda_chain[burn_in:]

    D_post = np.exp(post_chain[:, 0])
    kappa_post = np.exp(post_chain[:, 1])
    z_post = post_chain[:, 2:]

    D_mean, D_std = float(np.mean(D_post)), float(np.std(D_post))
    kappa_mean, kappa_std = float(np.mean(kappa_post)), float(np.std(kappa_post))

    print(f"[phase_c_source] Posterior D: {D_mean:.4f} ± {D_std:.4f} (true={D_true})")
    print(f"[phase_c_source] Posterior κ: {kappa_mean:.4f} ± {kappa_std:.4f} (true={kappa_true})")

    # ----- Source reconstruction -----
    x_grid = np.linspace(0.0, 1.0, n_grid)
    x_grid_jnp = jnp.asarray(x_grid)
    phi_truth_grid = np.interp(x_grid, x_ref, phi_ref)
    f_truth_grid = np.asarray(_source_fn_true(x_grid_jnp))

    # Reconstruct source for post-burn-in z samples
    f_samples = np.array(
        [np.asarray(kle.reconstruct(x_grid_jnp, jnp.asarray(z))) for z in z_post]
    )
    f_mean = np.mean(f_samples, axis=0)
    f_std = np.std(f_samples, axis=0)

    print(f"[phase_c_source] Source L2 error (mean): {np.sqrt(np.mean((f_mean - f_truth_grid)**2)):.4f}")

    # ----- Field samples for predictive plots -----
    phi_prior_samples = None
    phi_posterior_samples = None
    if result.phi_prior_snapshots is not None:
        snap_steps = result.snapshot_steps
        post_mask = snap_steps >= burn_in
        if np.any(post_mask):
            prior_snaps = result.phi_prior_snapshots[post_mask]
            posterior_snaps = result.phi_posterior_snapshots[post_mask]
            phi_prior_samples = np.array(
                [np.asarray(field.eval(x_grid_jnp, jnp.asarray(s))) for s in prior_snaps]
            )
            phi_posterior_samples = np.array(
                [np.asarray(field.eval(x_grid_jnp, jnp.asarray(s))) for s in posterior_snaps]
            )

    emit(progress_callback, started_at, 90.0, "output", "Computing predictive fields")

    # ----- Save outputs -----
    artifacts = {}
    summary = {}
    if save_outputs:
        fig4_path = str(out_fig / "phase_c_figure4.png")
        plot_figure4(
            lambda_chain=lambda_chain,
            phi_prior_samples=phi_prior_samples,
            phi_posterior_samples=phi_posterior_samples,
            f_samples=f_samples,
            x_grid=x_grid,
            phi_truth=phi_truth_grid,
            f_truth=f_truth_grid,
            x_obs=x_obs_np,
            y_obs=y_obs_np,
            true_params={"D": D_true, "kappa": kappa_true},
            burn_in=burn_in,
            out_path=fig4_path,
        )
        artifacts["figure4"] = fig4_path

        chain_path = str(out_traces / "phase_c_source_lambda_chain.npz")
        np.savez(chain_path, lambda_chain=lambda_chain, **{k: v for k, v in result.traces.items()})
        artifacts["chain"] = chain_path

        summary = {
            "D_true": D_true,
            "kappa_true": kappa_true,
            "D_posterior_mean": D_mean,
            "D_posterior_std": D_std,
            "kappa_posterior_mean": kappa_mean,
            "kappa_posterior_std": kappa_std,
            "source_l2_error": float(np.sqrt(np.mean((f_mean - f_truth_grid) ** 2))),
            "kle_n_terms": kle_n_terms,
            "kle_energy_fraction": kle.energy_fraction(),
            "beta": beta,
            "outer_steps": outer_steps,
            "warmup_steps": warmup_steps,
            "burn_in": burn_in,
            "stopped_early": bool(result.meta.get("stopped")),
            "runtime_sec": time.monotonic() - started_at,
            "config": {k: v for k, v in config.items()},
        }
        summary_path = str(out_tables / "phase_c_inverse_source_summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        artifacts["summary"] = summary_path

        emit(progress_callback, started_at, 95.0, "output", "Saved figures and summary")

    emit(progress_callback, started_at, 100.0, "done", "Source identification complete")

    return {
        "status": "completed",
        "lambda_chain": lambda_chain,
        "D_posterior": {"mean": D_mean, "std": D_std},
        "kappa_posterior": {"mean": kappa_mean, "std": kappa_std},
        "f_samples": f_samples,
        "f_mean": f_mean,
        "f_std": f_std,
        "x_grid": x_grid,
        "phi_truth": phi_truth_grid,
        "f_truth": f_truth_grid,
        "x_obs": x_obs_np,
        "y_obs": y_obs_np,
        "phi_prior_samples": phi_prior_samples,
        "phi_posterior_samples": phi_posterior_samples,
        "kle": kle,
        "nested_result": result,
        "artifacts": artifacts,
        "config": config,
        "summary": summary,
        "runtime_sec": time.monotonic() - started_at,
    }
