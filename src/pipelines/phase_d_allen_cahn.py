"""Phase D: 2D Allen-Cahn bimodal posterior — Example 4, Alberts & Bilionis.

Reproduces Figures 5-8: bimodal posterior from the Allen-Cahn energy
on [-1,1]^2, with observations on 3 boundaries only.

Uses NumPyro NUTS (the paper's HMCECS) instead of SGLD, because HMC
explores multimodal targets more reliably.  A Gaussian mixture model
separates the two posterior modes for per-mode predictions.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from core.energies_2d import allen_cahn_energy, allen_cahn_ground_truth, allen_cahn_source_term
from core.hmcecs import run_nuts_inference
from core.parameterizations_2d import FourierBasis2D
from .common import emit, select_device

# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------

DEFAULT_PHASE_D_CONFIG: dict[str, object] = {
    "seed": 42,
    # PDE
    "epsilon": 0.01,
    "beta": 100.0,
    # Field
    "max_freq": 1,  # 9 Fourier basis terms
    # Observations
    "n_obs_per_boundary": 15,
    "noise_std": 0.01,
    "observed_boundaries": ["left", "right", "bottom"],  # 3 of 4 sides
    # Quadrature
    "n_quad_per_dim": 40,  # 40x40 = 1600 quadrature points
    # NumPyro NUTS
    "num_warmup": 500,
    "num_samples": 1000,
    "num_chains": 1,
    "target_accept_prob": 0.8,
    "max_tree_depth": 10,
    "prior_std": 10.0,
    # GMM
    "n_gmm_components": 2,
    # Evaluation
    "n_grid_per_dim": 50,
}


# ---------------------------------------------------------------------------
# Observation helpers
# ---------------------------------------------------------------------------

def _boundary_points(boundary: str, n_points: int):
    """Generate equidistant points on a boundary of [-1, 1]^2.

    Returns (n_points, 2) array.
    """
    t = np.linspace(-1.0, 1.0, n_points + 2)[1:-1]  # exclude corners
    if boundary == "left":
        return np.column_stack([np.full(len(t), -1.0), t])
    elif boundary == "right":
        return np.column_stack([np.full(len(t), 1.0), t])
    elif boundary == "bottom":
        return np.column_stack([t, np.full(len(t), -1.0)])
    elif boundary == "top":
        return np.column_stack([t, np.full(len(t), 1.0)])
    else:
        raise ValueError(f"Unknown boundary: {boundary}")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_phase_d_allen_cahn(
    cfg: dict | None = None,
    output_root: str | Path = "outputs",
    device_preference: str = "cpu",
    run_id: str | None = None,
    stop_signal: Callable[[], bool] | None = None,
    progress_callback: Callable[[float, str, str, float], None] | None = None,
    save_outputs: bool = True,
) -> dict[str, object]:
    """Run 2D Allen-Cahn bimodal posterior experiment (Example 4).

    Samples the posterior over 9 Fourier coefficients using NumPyro
    NUTS, then separates the two posterior modes with a GMM.

    Returns dict with ``status``, ``theta_samples``, ``mode_labels``,
    ``mode_fields``, ``artifacts``, ``config``.
    """
    started_at = time.monotonic()
    config = dict(DEFAULT_PHASE_D_CONFIG)
    if cfg:
        config.update(cfg)

    epsilon = float(config["epsilon"])
    beta = float(config["beta"])
    max_freq = int(config["max_freq"])
    n_obs_per_bdy = int(config["n_obs_per_boundary"])
    noise_std = float(config["noise_std"])
    obs_bdys = list(config["observed_boundaries"])
    n_quad = int(config["n_quad_per_dim"])
    num_warmup = int(config["num_warmup"])
    num_samples = int(config["num_samples"])
    num_chains = int(config["num_chains"])
    target_accept = float(config["target_accept_prob"])
    max_tree = int(config["max_tree_depth"])
    prior_std = float(config["prior_std"])
    n_gmm = int(config["n_gmm_components"])
    n_grid = int(config["n_grid_per_dim"])

    print(
        f"[allen_cahn] Example 4: eps={epsilon}, beta={beta}, "
        f"max_freq={max_freq} ({(1+2*max_freq)**2} params), "
        f"NUTS warmup={num_warmup} samples={num_samples}"
    )
    emit(progress_callback, started_at, 1.0, "setup", "Initializing Allen-Cahn")

    # ----- Field and quadrature -----
    field = FourierBasis2D(max_freq=max_freq)

    # Dense grid quadrature on [-1, 1]^2
    qx = np.linspace(-1.0, 1.0, n_quad + 2)[1:-1]
    qy = np.linspace(-1.0, 1.0, n_quad + 2)[1:-1]
    qxx, qyy = np.meshgrid(qx, qy)
    xy_quad = jnp.asarray(
        np.column_stack([qxx.ravel(), qyy.ravel()]), dtype=jnp.float64
    )

    # Precompute source term on quadrature grid
    print("[allen_cahn] Precomputing source term on quadrature grid ...")
    source_vals = allen_cahn_source_term(xy_quad, epsilon=epsilon)
    emit(progress_callback, started_at, 10.0, "setup", "Source term computed")

    # ----- Observations on 3 boundaries -----
    key = jax.random.PRNGKey(int(config["seed"]))
    obs_parts = []
    for bdy in obs_bdys:
        obs_parts.append(_boundary_points(bdy, n_obs_per_bdy))
    xy_obs = jnp.asarray(np.vstack(obs_parts), dtype=jnp.float64)
    phi_obs_true = allen_cahn_ground_truth(xy_obs)
    key, noise_key = jax.random.split(key)
    y_obs = phi_obs_true + noise_std * jax.random.normal(
        noise_key, shape=phi_obs_true.shape
    )
    n_obs_total = xy_obs.shape[0]
    emit(
        progress_callback, started_at, 12.0, "setup",
        f"{n_obs_total} observations on {len(obs_bdys)} boundaries",
    )

    # ----- Energy and likelihood closures -----
    def energy_fn(theta):
        phi = field.eval(xy_quad, theta)
        dphi_dx = field.grad_x(xy_quad, theta)
        dphi_dy = field.grad_y(xy_quad, theta)
        grad_sq = dphi_dx**2 + dphi_dy**2
        integrand = (
            0.5 * epsilon * grad_sq
            + 0.25 * (1.0 - phi**2) ** 2
            - source_vals * phi
        )
        return jnp.mean(integrand)

    def field_eval_at_obs(theta):
        return field.eval(xy_obs, theta)

    # ----- Initial theta via lstsq projection of ground truth -----
    phi_gt_quad = allen_cahn_ground_truth(xy_quad)
    dm = field.design_matrix(xy_quad)
    theta_init, *_ = np.linalg.lstsq(np.asarray(dm), np.asarray(phi_gt_quad), rcond=None)

    emit(progress_callback, started_at, 15.0, "sampling", "Starting NUTS")

    device, device_label = select_device(device_preference)
    if device_label in ("gpu", "metal"):
        try:
            import numpyro
            numpyro.set_platform("gpu")
        except Exception as exc:  # noqa: BLE001
            print(f"[allen_cahn] WARNING: numpyro.set_platform('gpu') failed: {exc}")
    print(f"[allen_cahn] device: {device_label}")

    # ----- Run NumPyro NUTS -----
    key, nuts_key = jax.random.split(key)
    print("[allen_cahn] Running NumPyro NUTS ...")
    samples_dict = run_nuts_inference(
        energy_fn=energy_fn,
        field_eval_fn=field_eval_at_obs,
        xy_obs=xy_obs,
        y_obs=y_obs,
        noise_std=noise_std,
        beta=beta,
        n_params=field.n_params,
        key=nuts_key,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        init_params=theta_init,
        prior_std=prior_std,
        target_accept_prob=target_accept,
        max_tree_depth=max_tree,
        progress_bar=True,
    )
    theta_samples = samples_dict["theta"]  # (num_samples, n_params)
    print(f"[allen_cahn] NUTS complete: {theta_samples.shape[0]} samples")
    emit(progress_callback, started_at, 70.0, "analysis", "Sampling complete")

    # ----- GMM mode separation -----
    from sklearn.mixture import GaussianMixture

    gmm = GaussianMixture(n_components=n_gmm, random_state=int(config["seed"]))
    labels = gmm.fit_predict(theta_samples)
    mode_indices = {
        k: np.where(labels == k)[0] for k in range(n_gmm)
    }
    print(
        f"[allen_cahn] GMM: "
        + ", ".join(f"mode {k}: {len(idx)} samples" for k, idx in mode_indices.items())
    )
    emit(progress_callback, started_at, 75.0, "analysis", "GMM mode separation done")

    # ----- Evaluate fields on grid -----
    gx = np.linspace(-1.0, 1.0, n_grid)
    gy = np.linspace(-1.0, 1.0, n_grid)
    gxx, gyy = np.meshgrid(gx, gy)
    xy_grid = jnp.asarray(
        np.column_stack([gxx.ravel(), gyy.ravel()]), dtype=jnp.float64
    )

    phi_truth_grid = np.asarray(allen_cahn_ground_truth(xy_grid)).reshape(n_grid, n_grid)

    # All-sample statistics
    phi_all = np.array([np.asarray(field.eval(xy_grid, s)) for s in theta_samples])
    phi_mean = phi_all.mean(axis=0).reshape(n_grid, n_grid)
    phi_median = np.median(phi_all, axis=0).reshape(n_grid, n_grid)

    # Per-mode fields
    mode_fields = {}
    for k, idx in mode_indices.items():
        if len(idx) == 0:
            mode_fields[k] = np.zeros((n_grid, n_grid))
            continue
        phi_mode = np.array([np.asarray(field.eval(xy_grid, theta_samples[i])) for i in idx])
        mode_fields[k] = phi_mode.mean(axis=0).reshape(n_grid, n_grid)

    emit(progress_callback, started_at, 85.0, "output", "Fields evaluated")

    # ----- Save outputs -----
    output_root = Path(output_root)
    out_fig = output_root / "figures"
    out_tables = output_root / "tables"
    out_fig.mkdir(parents=True, exist_ok=True)
    out_tables.mkdir(parents=True, exist_ok=True)

    artifacts = {}
    if save_outputs:
        from utils.plotting import (
            plot_allen_cahn_prior,
            plot_allen_cahn_marginals,
            plot_allen_cahn_predictions,
            plot_allen_cahn_errors,
        )

        grid_coords = (gx, gy)
        xy_obs_np = np.asarray(xy_obs)

        # Fig 5: prior log-probability
        fig5_path = str(out_fig / "phase_d_fig5_prior.png")
        plot_allen_cahn_prior(
            field, xy_quad, source_vals, epsilon, beta, out_path=fig5_path,
        )
        artifacts["fig5_prior"] = fig5_path

        # Fig 6: bimodal marginal posteriors
        fig6_path = str(out_fig / "phase_d_fig6_marginals.png")
        plot_allen_cahn_marginals(theta_samples, out_path=fig6_path)
        artifacts["fig6_marginals"] = fig6_path

        # Fig 7: predictions from each mode
        fig7_path = str(out_fig / "phase_d_fig7_predictions.png")
        plot_allen_cahn_predictions(
            grid_coords, phi_truth_grid, phi_median,
            mode_fields, xy_obs_np, out_path=fig7_path,
        )
        artifacts["fig7_predictions"] = fig7_path

        # Fig 8: absolute errors
        fig8_path = str(out_fig / "phase_d_fig8_errors.png")
        plot_allen_cahn_errors(
            grid_coords, phi_truth_grid, phi_mean, phi_median,
            mode_fields, out_path=fig8_path,
        )
        artifacts["fig8_errors"] = fig8_path

        # Summary JSON
        summary = {
            "config": {k: v for k, v in config.items()},
            "n_params": field.n_params,
            "n_samples": int(theta_samples.shape[0]),
            "n_obs": int(n_obs_total),
            "mode_sizes": {str(k): int(len(idx)) for k, idx in mode_indices.items()},
            "runtime_sec": time.monotonic() - started_at,
        }
        summary_path = str(out_tables / "phase_d_allen_cahn_summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        artifacts["summary"] = summary_path

    emit(progress_callback, started_at, 100.0, "done", "Allen-Cahn experiment complete")

    return {
        "status": "completed",
        "theta_samples": theta_samples,
        "mode_labels": labels,
        "mode_indices": mode_indices,
        "grid_coords": (gx, gy),
        "phi_truth": phi_truth_grid,
        "phi_mean": phi_mean,
        "phi_median": phi_median,
        "mode_fields": mode_fields,
        "artifacts": artifacts,
        "config": config,
        "runtime_sec": time.monotonic() - started_at,
    }
