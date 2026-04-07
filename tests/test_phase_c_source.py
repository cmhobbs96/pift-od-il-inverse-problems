"""Tests for Phase C: source term identification (Example 3b)."""

from __future__ import annotations

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from core.kle import KLE
from core.parameterizations_nonlinear import make_nonlinear_field, solve_nonlinear_dirichlet_fd


D_TRUE = 0.1
KAPPA_TRUE = 1.0
BETA = 1e3
SOURCE_FN_NP = lambda x: np.cos(4.0 * x)
BC = (0.0, 0.0)


def _make_test_setup(K=5, kle_n_terms=3, n_obs=10, noise_std=0.01):
    """Shared setup for gradient function tests."""
    field = make_nonlinear_field(K=K, bc=BC)
    kle = KLE(lengthscale=0.3, n_terms=kle_n_terms, n_grid=100)

    x_ref, phi_ref = solve_nonlinear_dirichlet_fd(SOURCE_FN_NP, D=D_TRUE, kappa=KAPPA_TRUE, bc=BC)

    rng = np.random.default_rng(42)
    x_obs = np.linspace(0.0, 1.0, n_obs + 2)[1:-1]
    phi_obs_true = np.interp(x_obs, x_ref, phi_ref)
    y_obs = phi_obs_true + noise_std * rng.normal(size=n_obs)

    x_obs_jnp = jnp.asarray(x_obs)
    y_obs_jnp = jnp.asarray(y_obs)

    interior = (x_ref > 0.02) & (x_ref < 0.98)
    x_int = x_ref[interior]
    phi_int = phi_ref[interior]
    window_int = (1.0 - x_int) * x_int
    psi_target = phi_int / window_int
    psi_dm = np.asarray(field.psi_design_matrix(jnp.asarray(x_int)))
    phi0, *_ = np.linalg.lstsq(psi_dm, psi_target, rcond=None)
    phi0 = jnp.asarray(phi0, dtype=jnp.float64)

    return field, kle, phi0, x_obs_jnp, y_obs_jnp, noise_std


def _make_grad_fns(field, kle, n_quad, x_obs, y_obs, noise_std, beta):
    """Build gradient closures matching pipeline implementation."""

    def _physics_energy(phi, lam, x_quad):
        D = jnp.exp(lam[0])
        kappa = jnp.exp(lam[1])
        z = lam[2:]
        phi_vals = field.eval(x_quad, phi)
        dphi_vals = field.deriv(x_quad, phi)
        f_vals = kle.reconstruct(x_quad, z)
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


def test_grad_fn_shapes():
    """Gradient closures return correct shapes including z components."""
    field, kle, phi0, x_obs, y_obs, noise_std = _make_test_setup(K=5, kle_n_terms=3)
    prior_fn, post_fn = _make_grad_fns(field, kle, n_quad=64, x_obs=x_obs, y_obs=y_obs, noise_std=noise_std, beta=BETA)

    n_lam = 2 + 3  # log(D), log(kappa), z1, z2, z3
    lam = jnp.zeros(n_lam)
    lam = lam.at[0].set(jnp.log(D_TRUE))
    lam = lam.at[1].set(jnp.log(KAPPA_TRUE))
    key = jax.random.PRNGKey(0)

    grad_phi, grad_lam, metrics = prior_fn(phi0, lam, key)
    assert grad_phi.shape == phi0.shape
    assert grad_lam.shape == (n_lam,)
    assert jnp.isfinite(grad_phi).all()
    assert jnp.isfinite(grad_lam).all()

    grad_phi_p, grad_lam_p, m_p = post_fn(phi0, lam, key)
    assert grad_phi_p.shape == phi0.shape
    assert m_p["likelihood"] > 0


def test_prior_grad_includes_z_component():
    """Prior gradient w.r.t. z should be non-zero when z != 0."""
    field, kle, phi0, x_obs, y_obs, noise_std = _make_test_setup(K=5, kle_n_terms=3)
    prior_fn, _ = _make_grad_fns(field, kle, n_quad=128, x_obs=x_obs, y_obs=y_obs, noise_std=noise_std, beta=BETA)

    lam = jnp.array([jnp.log(D_TRUE), jnp.log(KAPPA_TRUE), 1.0, 0.5, -0.3])
    key = jax.random.PRNGKey(1)

    _, grad_lam, _ = prior_fn(phi0, lam, key)
    # z components of gradient should be non-zero (source term depends on z)
    z_grad = grad_lam[2:]
    assert jnp.any(jnp.abs(z_grad) > 1e-8), f"z gradient should be non-zero: {z_grad}"


def test_lambda_prior_grad():
    """Combined prior: Jeffrey's on D,κ (zero) + N(0,1) on z (grad=z)."""
    from pipelines.phase_c_inverse_source import _lambda_prior_grad

    lam = jnp.array([0.5, -0.3, 1.0, 2.0, -1.5])
    grad = _lambda_prior_grad(lam)

    # First 2 components (log D, log κ) should be zero (Jeffrey's)
    np.testing.assert_allclose(np.asarray(grad[:2]), 0.0, atol=1e-15)
    # z components should equal z (standard normal prior gradient)
    np.testing.assert_allclose(np.asarray(grad[2:]), np.asarray(lam[2:]), atol=1e-15)


def test_grad_z_fd():
    """Gradient of energy w.r.t. z matches finite differences."""
    field, kle, phi0, x_obs, y_obs, noise_std = _make_test_setup(K=5, kle_n_terms=3)

    x_quad = jnp.linspace(0.01, 0.99, 200)

    def energy_fn(lam):
        D = jnp.exp(lam[0])
        kappa = jnp.exp(lam[1])
        z = lam[2:]
        phi_vals = field.eval(x_quad, phi0)
        dphi_vals = field.deriv(x_quad, phi0)
        f_vals = kle.reconstruct(x_quad, z)
        return BETA * jnp.mean(0.5 * D * dphi_vals**2 + 0.25 * kappa * phi_vals**4 + phi_vals * f_vals)

    lam = jnp.array([jnp.log(D_TRUE), jnp.log(KAPPA_TRUE), 0.5, -0.3, 1.0])
    grad_lam = np.asarray(jax.grad(energy_fn)(lam))

    eps = 1e-5
    fd = np.zeros(5)
    lam_np = np.asarray(lam)
    for i in range(5):
        d = np.zeros(5)
        d[i] = eps
        e_plus = float(energy_fn(jnp.asarray(lam_np + d)))
        e_minus = float(energy_fn(jnp.asarray(lam_np - d)))
        fd[i] = (e_plus - e_minus) / (2.0 * eps)

    np.testing.assert_allclose(grad_lam, fd, atol=1e-3, rtol=1e-3)


def test_phase_c_source_smoke(tmp_path):
    """Small run completes with expected output keys."""
    from pipelines.phase_c_inverse_source import run_phase_c_inverse_source

    result = run_phase_c_inverse_source(
        cfg={
            "K": 5,
            "beta": 1e3,
            "n_obs": 10,
            "kle_n_terms": 3,
            "kle_n_grid": 50,
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
    assert result["lambda_chain"].shape == (100, 2 + 3)  # log(D), log(κ), z1-z3
    assert "D_posterior" in result
    assert "kappa_posterior" in result
    assert result["f_samples"] is not None
    assert result["f_mean"].shape == (50,)  # n_grid

    assert (tmp_path / "figures" / "phase_c_figure4.png").exists()
    assert (tmp_path / "tables" / "phase_c_inverse_source_summary.json").exists()
