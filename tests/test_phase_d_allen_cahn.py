"""Tests for Phase D: 2D Allen-Cahn bimodal posterior (Example 4).

Covers:
1. FourierBasis2D: n_params, eval, design matrix, derivative correctness
2. Allen-Cahn energy: gradient FD verification
3. Ground truth satisfies PDE (source term consistency)
4. NumPyro NUTS wrapper smoke test
5. Pipeline smoke test
"""

from __future__ import annotations

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from core.parameterizations_2d import FourierBasis2D
from core.energies_2d import allen_cahn_energy, allen_cahn_ground_truth, allen_cahn_source_term


# ---------------------------------------------------------------------------
# 1. FourierBasis2D
# ---------------------------------------------------------------------------


def test_fourier_2d_n_params():
    """max_freq=1 gives 9 params, max_freq=2 gives 25."""
    assert FourierBasis2D(max_freq=1).n_params == 9
    assert FourierBasis2D(max_freq=2).n_params == 25


def test_fourier_2d_eval_shape():
    field = FourierBasis2D(max_freq=1)
    rng = np.random.default_rng(0)
    xy = rng.uniform(-1, 1, (50, 2))
    theta = rng.normal(size=field.n_params)
    phi = field.eval(jnp.asarray(xy), jnp.asarray(theta))
    assert phi.shape == (50,)


def test_fourier_2d_design_matrix_shape():
    field = FourierBasis2D(max_freq=1)
    rng = np.random.default_rng(1)
    xy = rng.uniform(-1, 1, (30, 2))
    dm = field.design_matrix(jnp.asarray(xy))
    assert dm.shape == (30, 9)


def test_fourier_2d_grad_fd():
    """Spatial gradients match finite differences."""
    rng = np.random.default_rng(2)
    field = FourierBasis2D(max_freq=1)
    theta = jnp.asarray(rng.normal(size=field.n_params) * 0.5)
    xy = jnp.asarray(rng.uniform(-0.8, 0.8, (20, 2)))

    # Analytical gradients
    gx_analytical = np.asarray(field.grad_x(xy, theta))
    gy_analytical = np.asarray(field.grad_y(xy, theta))

    # FD gradients
    eps = 1e-7
    xy_np = np.asarray(xy)
    gx_fd = np.zeros(len(xy_np))
    gy_fd = np.zeros(len(xy_np))
    for i in range(len(xy_np)):
        xy_px = xy_np.copy(); xy_px[i, 0] += eps
        xy_mx = xy_np.copy(); xy_mx[i, 0] -= eps
        gx_fd[i] = (float(field.eval(jnp.asarray(xy_px), theta)[i])
                     - float(field.eval(jnp.asarray(xy_mx), theta)[i])) / (2 * eps)

        xy_py = xy_np.copy(); xy_py[i, 1] += eps
        xy_my = xy_np.copy(); xy_my[i, 1] -= eps
        gy_fd[i] = (float(field.eval(jnp.asarray(xy_py), theta)[i])
                     - float(field.eval(jnp.asarray(xy_my), theta)[i])) / (2 * eps)

    assert np.allclose(gx_analytical, gx_fd, atol=1e-5), (
        f"grad_x max diff: {np.max(np.abs(gx_analytical - gx_fd))}"
    )
    assert np.allclose(gy_analytical, gy_fd, atol=1e-5), (
        f"grad_y max diff: {np.max(np.abs(gy_analytical - gy_fd))}"
    )


# ---------------------------------------------------------------------------
# 2. Allen-Cahn energy gradient FD
# ---------------------------------------------------------------------------


def test_allen_cahn_energy_grad_fd():
    """Energy gradient w.r.t. theta matches central finite differences."""
    rng = np.random.default_rng(10)
    field = FourierBasis2D(max_freq=1)
    theta = jnp.asarray(rng.normal(size=field.n_params) * 0.3)
    xy_quad = jnp.asarray(rng.uniform(-1, 1, (100, 2)))
    source_vals = allen_cahn_source_term(xy_quad, epsilon=0.01)

    _, grad, _ = allen_cahn_energy(theta, xy_quad, field, source_vals, epsilon=0.01)
    grad = np.asarray(grad)

    eps = 1e-6
    fd = np.zeros(field.n_params, dtype=float)
    theta_np = np.asarray(theta)
    for i in range(field.n_params):
        d = np.zeros_like(theta_np)
        d[i] = eps
        e_plus, _, _ = allen_cahn_energy(theta_np + d, xy_quad, field, source_vals)
        e_minus, _, _ = allen_cahn_energy(theta_np - d, xy_quad, field, source_vals)
        fd[i] = (e_plus - e_minus) / (2.0 * eps)

    assert np.allclose(grad, fd, atol=1e-4, rtol=1e-4), (
        f"Max diff: {np.max(np.abs(grad - fd))}"
    )


# ---------------------------------------------------------------------------
# 3. Source term consistency
# ---------------------------------------------------------------------------


def test_source_term_pde_consistency():
    """Ground truth + source term satisfy the Allen-Cahn PDE."""
    rng = np.random.default_rng(20)
    epsilon = 0.01
    # Use interior points (avoid boundary where gradients may be large)
    xy = jnp.asarray(rng.uniform(-0.7, 0.7, (50, 2)))

    phi = allen_cahn_ground_truth(xy)
    f = allen_cahn_source_term(xy, epsilon=epsilon)

    # Compute Laplacian of ground truth via autodiff
    def _phi_point(p):
        x, y = p[0], p[1]
        return 2.0 * jnp.exp(-(x**2 + y**2)) * jnp.sin(jnp.pi * x) * jnp.sin(jnp.pi * y)

    def _laplacian(p):
        H = jax.hessian(_phi_point)(p)
        return H[0, 0] + H[1, 1]

    lap = jax.vmap(_laplacian)(xy)

    # PDE residual: epsilon * lap - phi(phi^2 - 1) + f should be ~0
    residual = epsilon * lap - np.asarray(phi) * (np.asarray(phi)**2 - 1) + np.asarray(f)
    assert np.max(np.abs(np.asarray(residual))) < 1e-8, (
        f"PDE residual too large: {np.max(np.abs(np.asarray(residual))):.2e}"
    )


# ---------------------------------------------------------------------------
# 4. NumPyro NUTS smoke test
# ---------------------------------------------------------------------------


def test_nuts_smoke():
    """NumPyro NUTS wrapper runs and returns correct shape."""
    from core.hmcecs import run_nuts_inference

    def simple_energy(theta):
        return 0.5 * jnp.sum(theta**2)

    xy_obs = jnp.array([[0.0, 0.0], [1.0, 0.0]])
    y_obs = jnp.array([0.0, 0.0])

    def field_eval(theta):
        return theta[0] * jnp.ones(2)

    key = jax.random.PRNGKey(0)
    result = run_nuts_inference(
        energy_fn=simple_energy,
        field_eval_fn=field_eval,
        xy_obs=xy_obs,
        y_obs=y_obs,
        noise_std=0.1,
        beta=1.0,
        n_params=3,
        key=key,
        num_warmup=50,
        num_samples=100,
        progress_bar=False,
    )

    assert "theta" in result
    assert result["theta"].shape == (100, 3)
    assert np.all(np.isfinite(result["theta"]))


# ---------------------------------------------------------------------------
# 5. Pipeline smoke test
# ---------------------------------------------------------------------------


def test_phase_d_smoke(tmp_path):
    """Small Allen-Cahn run completes with expected structure."""
    from pipelines.phase_d_allen_cahn import run_phase_d_allen_cahn

    result = run_phase_d_allen_cahn(
        cfg={
            "max_freq": 1,
            "n_obs_per_boundary": 5,
            "n_quad_per_dim": 10,
            "num_warmup": 30,
            "num_samples": 50,
            "n_grid_per_dim": 10,
            "beta": 10.0,
        },
        output_root=str(tmp_path),
        save_outputs=True,
    )

    assert result["status"] == "completed"
    assert result["theta_samples"].shape == (50, 9)
    assert result["phi_truth"].shape == (10, 10)
    assert result["phi_mean"].shape == (10, 10)
    assert result["phi_median"].shape == (10, 10)
    assert len(result["mode_labels"]) == 50
    assert len(result["mode_fields"]) == 2

    # Check output files
    assert (tmp_path / "figures" / "phase_d_fig5_prior.png").exists()
    assert (tmp_path / "figures" / "phase_d_fig6_marginals.png").exists()
    assert (tmp_path / "figures" / "phase_d_fig7_predictions.png").exists()
    assert (tmp_path / "figures" / "phase_d_fig8_errors.png").exists()
    assert (tmp_path / "tables" / "phase_d_allen_cahn_summary.json").exists()
