"""Tests for Phase B beta sweep: BoundaryFourierField, variational energy, pipeline."""

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

from core.parameterizations import BoundaryFourierField
from core.energies import variational_heat_energy
from core.reference_solver import solve_poisson_dirichlet_fd


# ---------------------------------------------------------------------------
# BoundaryFourierField tests
# ---------------------------------------------------------------------------


def test_boundary_fourier_bcs():
    """phi(0) == bc_left, phi(1) == bc_right for arbitrary theta."""
    rng = np.random.default_rng(0)
    for bc in [(1.0, 0.1), (0.0, 0.0), (2.5, -1.3)]:
        field = BoundaryFourierField(K=10, bc=bc)
        theta = jnp.asarray(rng.normal(size=field.n_params))
        x = jnp.array([0.0, 1.0])
        phi = field.eval(x, theta)
        assert np.isclose(float(phi[0]), bc[0], atol=1e-12), f"phi(0)={phi[0]}, expected {bc[0]}"
        assert np.isclose(float(phi[1]), bc[1], atol=1e-12), f"phi(1)={phi[1]}, expected {bc[1]}"


def test_boundary_fourier_n_params():
    field = BoundaryFourierField(K=20)
    assert field.n_params == 41


def test_boundary_fourier_deriv_fd():
    """Analytical deriv matches finite-difference of eval."""
    rng = np.random.default_rng(1)
    field = BoundaryFourierField(K=8, bc=(1.0, 0.1))
    theta = jnp.asarray(rng.normal(size=field.n_params) * 0.5)
    x = jnp.asarray(rng.uniform(0.05, 0.95, size=50))

    dphi_analytical = np.asarray(field.deriv(x, theta))

    eps = 1e-7
    dphi_fd = np.zeros_like(dphi_analytical)
    x_np = np.asarray(x)
    for i in range(len(x_np)):
        x_plus = jnp.asarray(x_np.copy())
        x_minus = jnp.asarray(x_np.copy())
        x_plus = x_plus.at[i].set(x_np[i] + eps)
        x_minus = x_minus.at[i].set(x_np[i] - eps)
        phi_plus = float(field.eval(x_plus, theta)[i])
        phi_minus = float(field.eval(x_minus, theta)[i])
        dphi_fd[i] = (phi_plus - phi_minus) / (2.0 * eps)

    assert np.allclose(dphi_analytical, dphi_fd, atol=1e-5, rtol=1e-5), (
        f"Max diff: {np.max(np.abs(dphi_analytical - dphi_fd))}"
    )


# ---------------------------------------------------------------------------
# Variational heat energy tests
# ---------------------------------------------------------------------------


def test_variational_energy_grad_fd():
    """Energy gradient matches finite-difference (central differences)."""
    rng = np.random.default_rng(2)
    field = BoundaryFourierField(K=6, bc=(1.0, 0.1))
    theta = jnp.asarray(rng.normal(size=field.n_params) * 0.3)
    x_quad = jnp.asarray(rng.uniform(0.0, 1.0, size=200))
    source_fn = lambda x: jnp.exp(-x)
    D = 0.25

    _, grad, _ = variational_heat_energy(theta, x_quad, field, source_fn, D)
    grad = np.asarray(grad)

    eps = 1e-6
    fd = np.zeros(field.n_params, dtype=float)
    theta_np = np.asarray(theta)
    for i in range(field.n_params):
        d = np.zeros_like(theta_np)
        d[i] = eps
        e_plus, _, _ = variational_heat_energy(theta_np + d, x_quad, field, source_fn, D)
        e_minus, _, _ = variational_heat_energy(theta_np - d, x_quad, field, source_fn, D)
        fd[i] = (e_plus - e_minus) / (2.0 * eps)

    assert np.allclose(grad, fd, atol=1e-4, rtol=1e-4), (
        f"Max diff: {np.max(np.abs(grad - fd))}"
    )


def test_energy_minimizer_matches_fd():
    """Minimizing U[phi] recovers the FD reference solution."""
    from scipy.optimize import minimize

    D = 0.25
    bc = (1.0, 0.1)
    field = BoundaryFourierField(K=15, bc=bc)

    # FD reference: solve -phi'' = q/D with BCs
    def fd_forcing(x):
        return np.exp(-x) / D

    x_ref, phi_ref = solve_poisson_dirichlet_fd(fd_forcing, n_points=500, bc=bc)

    # Deterministic quadrature for optimization (no MC noise)
    x_quad = jnp.linspace(0.01, 0.99, 500)
    source_fn = lambda x: jnp.exp(-x)

    def objective(theta_np):
        theta = jnp.asarray(theta_np)
        e, g, _ = variational_heat_energy(theta, x_quad, field, source_fn, D)
        return float(e), np.asarray(g, dtype=np.float64)

    theta0 = np.zeros(field.n_params)
    result = minimize(objective, theta0, jac=True, method="L-BFGS-B", options={"maxiter": 2000})
    assert result.success, f"Optimization failed: {result.message}"

    # Compare on evaluation grid
    x_eval = jnp.linspace(0.0, 1.0, 200)
    phi_opt = np.asarray(field.eval(x_eval, jnp.asarray(result.x)))
    phi_truth = np.interp(np.asarray(x_eval), x_ref, phi_ref)

    l2 = np.sqrt(np.mean((phi_opt - phi_truth) ** 2))
    assert l2 < 0.05, f"L2 error too large: {l2:.6f}"


# ---------------------------------------------------------------------------
# Pipeline smoke test
# ---------------------------------------------------------------------------


def test_phase_b_smoke(tmp_path):
    """Small run completes and returns expected keys."""
    from pipelines.phase_b_beta_sweep import run_phase_b_beta_sweep

    result = run_phase_b_beta_sweep(
        cfg={
            "beta_values": [1, 100],
            "K": 5,
            "n_steps": 300,
            "burn_in": 50,
            "thin": 5,
            "n_quad": 32,
            "n_grid": 50,
            "step_size0": 0.01,
        },
        output_root=str(tmp_path),
        save_outputs=True,
    )

    assert result["status"] == "completed"
    assert len(result["beta_results"]) == 2
    assert "variance_scaling" in result
    assert "x_grid" in result
    assert "phi_truth" in result

    # Check that higher beta has lower variance
    var_low_beta = result["beta_results"][0]["variance_at_midpoint"]
    var_high_beta = result["beta_results"][1]["variance_at_midpoint"]
    assert var_high_beta < var_low_beta, (
        f"Expected var(beta=100) < var(beta=1), got {var_high_beta} >= {var_low_beta}"
    )

    # Check output files exist
    assert (tmp_path / "figures" / "phase_b_beta_sweep_panels.png").exists()
    assert (tmp_path / "figures" / "phase_b_variance_scaling.png").exists()
    assert (tmp_path / "tables" / "phase_b_beta_sweep_summary.json").exists()
