"""Tests for Phase C: inverse parameter identification (Example 3a)."""

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

from core.parameterizations_nonlinear import make_nonlinear_field, solve_nonlinear_dirichlet_fd


# Paper parameters
D_TRUE = 0.1
KAPPA_TRUE = 1.0
BETA = 1e3  # smaller for tests
SOURCE_FN = lambda x: jnp.cos(4.0 * x)
SOURCE_FN_NP = lambda x: np.cos(4.0 * x)
BC = (0.0, 0.0)


def _make_test_setup(K=5, n_obs=10, noise_std=0.01):
    """Shared setup for gradient function tests."""
    field = make_nonlinear_field(K=K, bc=BC)
    x_ref, phi_ref = solve_nonlinear_dirichlet_fd(SOURCE_FN_NP, D=D_TRUE, kappa=KAPPA_TRUE, bc=BC)

    rng = np.random.default_rng(42)
    x_obs = np.linspace(0.0, 1.0, n_obs + 2)[1:-1]
    phi_obs_true = np.interp(x_obs, x_ref, phi_ref)
    y_obs = phi_obs_true + noise_std * rng.normal(size=n_obs)

    x_obs_jnp = jnp.asarray(x_obs)
    y_obs_jnp = jnp.asarray(y_obs)

    # Initialize phi from FD solution
    interior = (x_ref > 0.02) & (x_ref < 0.98)
    x_int = x_ref[interior]
    phi_int = phi_ref[interior]
    window_int = (1.0 - x_int) * x_int
    psi_target = phi_int / window_int
    psi_dm = np.asarray(field.psi_design_matrix(jnp.asarray(x_int)))
    phi0, *_ = np.linalg.lstsq(psi_dm, psi_target, rcond=None)
    phi0 = jnp.asarray(phi0, dtype=jnp.float64)

    return field, phi0, x_obs_jnp, y_obs_jnp, noise_std


def _make_grad_fns(field, n_quad, x_obs, y_obs, noise_std, beta):
    """Build gradient closures matching pipeline implementation."""
    def _physics_energy(phi, lam, x_quad):
        D = jnp.exp(lam[0])
        kappa = jnp.exp(lam[1])
        phi_vals = field.eval(x_quad, phi)
        dphi_vals = field.deriv(x_quad, phi)
        f_vals = SOURCE_FN(x_quad)
        return beta * jnp.mean(0.5 * D * dphi_vals**2 + 0.25 * kappa * phi_vals**4 + phi_vals * f_vals)

    def _data_nll(phi):
        pred = field.eval(x_obs, phi)
        return 0.5 / (noise_std**2) * jnp.sum((pred - y_obs)**2)

    def prior_grad_fn(phi, lam, key):
        key, qk = jax.random.split(key)
        x_quad = jax.random.uniform(qk, shape=(n_quad,))
        energy_val = _physics_energy(phi, lam, x_quad)
        grad_phi, grad_lam = jax.grad(_physics_energy, argnums=(0, 1))(phi, lam, x_quad)
        e = float(energy_val)
        return grad_phi, grad_lam, {"hamiltonian": e, "physics": e, "likelihood": 0.0}

    def posterior_grad_fn(phi, lam, key):
        key, qk = jax.random.split(key)
        x_quad = jax.random.uniform(qk, shape=(n_quad,))
        phys_e = _physics_energy(phi, lam, x_quad)
        grad_phi_phys, grad_lam = jax.grad(_physics_energy, argnums=(0, 1))(phi, lam, x_quad)
        nll_val = _data_nll(phi)
        grad_phi_data = jax.grad(_data_nll)(phi)
        return (
            grad_phi_phys + grad_phi_data,
            grad_lam,
            {"hamiltonian": float(phys_e + nll_val), "physics": float(phys_e), "likelihood": float(nll_val)},
        )

    return prior_grad_fn, posterior_grad_fn


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_prior_grad_fn_shapes():
    """Prior grad returns correct shapes and finite values."""
    field, phi0, x_obs, y_obs, noise_std = _make_test_setup()
    prior_fn, _ = _make_grad_fns(field, n_quad=64, x_obs=x_obs, y_obs=y_obs, noise_std=noise_std, beta=BETA)

    lam = jnp.array([np.log(D_TRUE), np.log(KAPPA_TRUE)])
    key = jax.random.PRNGKey(0)

    grad_phi, grad_lam, metrics = prior_fn(phi0, lam, key)

    assert grad_phi.shape == phi0.shape
    assert grad_lam.shape == lam.shape
    assert jnp.isfinite(grad_phi).all()
    assert jnp.isfinite(grad_lam).all()
    assert "hamiltonian" in metrics


def test_posterior_grad_fn_includes_data():
    """Posterior grad_phi differs from prior (data term present)."""
    field, phi0, x_obs, y_obs, noise_std = _make_test_setup()
    prior_fn, post_fn = _make_grad_fns(field, n_quad=64, x_obs=x_obs, y_obs=y_obs, noise_std=noise_std, beta=BETA)

    lam = jnp.array([np.log(D_TRUE), np.log(KAPPA_TRUE)])
    key = jax.random.PRNGKey(1)

    # Use same key so quadrature points match
    grad_phi_prior, _, _ = prior_fn(phi0, lam, key)
    grad_phi_post, _, m_post = post_fn(phi0, lam, key)

    # They should differ because posterior includes data likelihood
    diff = jnp.max(jnp.abs(grad_phi_post - grad_phi_prior))
    assert float(diff) > 1e-6, "Posterior and prior phi gradients should differ (data term)"
    assert m_post["likelihood"] > 0.0


def test_grad_lam_fd():
    """Gradient of energy w.r.t. lambda matches finite differences."""
    field, phi0, x_obs, y_obs, noise_std = _make_test_setup(K=5)

    n_quad = 200
    # Use fixed quadrature for reproducibility
    x_quad = jnp.linspace(0.01, 0.99, n_quad)

    def energy_fn(lam):
        D = jnp.exp(lam[0])
        kappa = jnp.exp(lam[1])
        phi_vals = field.eval(x_quad, phi0)
        dphi_vals = field.deriv(x_quad, phi0)
        f_vals = SOURCE_FN(x_quad)
        return BETA * jnp.mean(0.5 * D * dphi_vals**2 + 0.25 * kappa * phi_vals**4 + phi_vals * f_vals)

    lam = jnp.array([np.log(D_TRUE), np.log(KAPPA_TRUE)])
    grad_lam = jax.grad(energy_fn)(lam)
    grad_lam_np = np.asarray(grad_lam)

    eps = 1e-5
    fd = np.zeros(2)
    lam_np = np.asarray(lam)
    for i in range(2):
        d = np.zeros(2)
        d[i] = eps
        e_plus = float(energy_fn(jnp.asarray(lam_np + d)))
        e_minus = float(energy_fn(jnp.asarray(lam_np - d)))
        fd[i] = (e_plus - e_minus) / (2.0 * eps)

    assert np.allclose(grad_lam_np, fd, atol=1e-3, rtol=1e-3), (
        f"grad_lam={grad_lam_np}, fd={fd}, diff={np.abs(grad_lam_np - fd)}"
    )


def test_phase_c_smoke(tmp_path):
    """Small run completes with expected output keys."""
    from pipelines.phase_c_inverse_params import run_phase_c_inverse_params

    result = run_phase_c_inverse_params(
        cfg={
            "K": 5,
            "beta": 1e3,
            "n_obs": 10,
            "warmup_steps": 50,
            "outer_steps": 100,
            "outer_step_size0": 0.01,
            "inner_step_size0": 0.01,
            "n_quad": 32,
            "n_grid": 50,
            "snapshot_interval": 20,
            "burn_in_frac": 0.2,
        },
        output_root=str(tmp_path),
        save_outputs=True,
    )

    assert result["status"] == "completed"
    assert result["lambda_chain"].shape == (100, 2)
    assert "D_posterior" in result
    assert "kappa_posterior" in result
    assert result["D_posterior"]["mean"] > 0
    assert result["kappa_posterior"]["mean"] > 0

    # Check output files
    assert (tmp_path / "figures" / "phase_c_figure3.png").exists()
    assert (tmp_path / "tables" / "phase_c_inverse_params_summary.json").exists()
    assert (tmp_path / "traces" / "phase_c_lambda_chain.npz").exists()


def test_lambda_chain_has_variation(tmp_path):
    """Post-warmup chain shows the sampler is exploring (non-trivial variance)."""
    from pipelines.phase_c_inverse_params import run_phase_c_inverse_params

    result = run_phase_c_inverse_params(
        cfg={
            "K": 5,
            "beta": 1e2,
            "n_obs": 10,
            "warmup_steps": 200,
            "outer_steps": 300,
            "outer_step_size0": 1e-3,
            "inner_step_size0": 1e-3,
            "n_quad": 64,
            "n_grid": 30,
            "snapshot_interval": 0,
        },
        output_root=str(tmp_path),
        save_outputs=False,
    )

    assert not result["nested_result"].meta["stopped"], (
        f"Sampler stopped early: {result['nested_result'].meta}"
    )

    chain = result["lambda_chain"]
    post = chain[60:]  # after 20% burn-in
    std_log_D = np.std(post[:, 0])
    std_log_kappa = np.std(post[:, 1])
    assert std_log_D > 1e-6, f"log(D) chain has no variation: std={std_log_D}"
    assert std_log_kappa > 1e-6, f"log(κ) chain has no variation: std={std_log_kappa}"
