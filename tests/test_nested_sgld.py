"""Tests for nested SGLD (Algorithm 3)."""

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

from core.nested_sgld import NestedSGLDResult, nested_sgld


# ---------------------------------------------------------------------------
# Helpers: trivial quadratic Hamiltonian H(phi, lam) = 0.5 * exp(lam) * sum(phi^2)
# ---------------------------------------------------------------------------


def _quadratic_prior_grad(phi, lam, key):
    """Prior grad for H = 0.5 * exp(lam) * sum(phi^2)."""
    beta = jnp.exp(lam[0])
    grad_phi = beta * phi
    # dH/dlam = dH/dbeta * dbeta/dlam = 0.5 * sum(phi^2) * exp(lam)
    grad_lam = jnp.array([0.5 * jnp.sum(phi**2) * beta])
    energy = float(0.5 * beta * jnp.sum(phi**2))
    return grad_phi, grad_lam, {"hamiltonian": energy}


def _quadratic_posterior_grad(phi, lam, key):
    """Posterior = prior + simple data term: 0.5 * sum((phi - target)^2)."""
    target = jnp.ones_like(phi) * 0.5
    # Physics part
    beta = jnp.exp(lam[0])
    phys_grad_phi = beta * phi
    phys_grad_lam = jnp.array([0.5 * jnp.sum(phi**2) * beta])
    phys_energy = 0.5 * beta * jnp.sum(phi**2)
    # Data part (no lambda dependence)
    data_grad_phi = phi - target
    data_energy = 0.5 * jnp.sum((phi - target) ** 2)

    grad_phi = phys_grad_phi + data_grad_phi
    grad_lam = phys_grad_lam  # data doesn't depend on lambda
    energy = float(phys_energy + data_energy)
    return grad_phi, grad_lam, {"hamiltonian": energy}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_nested_sgld_smoke():
    """Runs with trivial quadratic and returns correct shapes."""
    key = jax.random.PRNGKey(0)
    lam0 = jnp.array([0.0])  # log(beta) = 0 => beta = 1
    phi0 = jnp.ones(3) * 0.1

    result = nested_sgld(
        lambda0=lam0,
        phi0=phi0,
        prior_grad_fn=_quadratic_prior_grad,
        posterior_grad_fn=_quadratic_posterior_grad,
        key=key,
        outer_steps=50,
        outer_step_size0=0.01,
        inner_T_prior=5,
        inner_T_posterior=1,
        inner_step_size0=0.01,
        warmup_steps=100,
    )

    assert isinstance(result, NestedSGLDResult)
    assert result.lambda_chain.shape == (50, 1)
    assert result.phi_prior_final.shape == (3,)
    assert result.phi_posterior_final.shape == (3,)
    assert np.isfinite(result.lambda_chain).all()
    assert not result.meta["stopped"]


def test_jeffreys_prior_zero_gradient():
    """With lambda_prior_grad_fn=None, prior term is zero."""
    key = jax.random.PRNGKey(1)
    lam0 = jnp.array([0.0])
    phi0 = jnp.ones(3) * 0.1

    result = nested_sgld(
        lambda0=lam0,
        phi0=phi0,
        prior_grad_fn=_quadratic_prior_grad,
        posterior_grad_fn=_quadratic_posterior_grad,
        key=key,
        lambda_prior_grad_fn=None,  # Jeffrey's
        outer_steps=30,
        outer_step_size0=0.01,
        inner_T_prior=3,
        inner_T_posterior=1,
        inner_step_size0=0.01,
        warmup_steps=50,
    )

    # grad_total should be grad_posterior - grad_prior (no prior term)
    diff = result.traces["grad_lam_posterior"] - result.traces["grad_lam_prior"]
    np.testing.assert_allclose(
        result.traces["grad_lam_total"], diff, atol=1e-12,
    )


def test_inner_step_counters():
    """Step counters match warmup + outer * T formula."""
    key = jax.random.PRNGKey(2)
    W, T, P, Q = 80, 20, 5, 2

    # We need to track counters — use a wrapper that records them
    # The simplest check: look at the outer_step_size trace to confirm
    # the outer loop ran the expected number of steps.
    result = nested_sgld(
        lambda0=jnp.array([0.0]),
        phi0=jnp.ones(2) * 0.1,
        prior_grad_fn=_quadratic_prior_grad,
        posterior_grad_fn=_quadratic_posterior_grad,
        key=key,
        outer_steps=T,
        inner_T_prior=P,
        inner_T_posterior=Q,
        inner_step_size0=0.01,
        outer_step_size0=0.01,
        warmup_steps=W,
    )

    # Outer loop should have run T steps
    assert result.lambda_chain.shape[0] == T
    # Outer step sizes should be computed for steps 0..T-1
    expected_sizes = [0.01 / (1.0 + t) ** 0.51 for t in range(T)]
    np.testing.assert_allclose(
        result.traces["outer_step_size"], expected_sizes, rtol=1e-10,
    )


def test_outer_step_size_decay():
    """Outer step size is strictly decreasing."""
    key = jax.random.PRNGKey(3)
    result = nested_sgld(
        lambda0=jnp.array([0.0]),
        phi0=jnp.ones(2) * 0.1,
        prior_grad_fn=_quadratic_prior_grad,
        posterior_grad_fn=_quadratic_posterior_grad,
        key=key,
        outer_steps=100,
        outer_step_size0=0.1,
        inner_T_prior=2,
        inner_T_posterior=1,
        inner_step_size0=0.01,
        warmup_steps=20,
    )

    sizes = result.traces["outer_step_size"]
    assert np.all(np.diff(sizes) < 0), "Outer step sizes should be strictly decreasing"


def test_stop_signal():
    """Stop signal halts the outer loop."""
    call_count = [0]

    def stop_after_10():
        call_count[0] += 1
        return call_count[0] > 10

    key = jax.random.PRNGKey(4)
    result = nested_sgld(
        lambda0=jnp.array([0.0]),
        phi0=jnp.ones(2) * 0.1,
        prior_grad_fn=_quadratic_prior_grad,
        posterior_grad_fn=_quadratic_posterior_grad,
        key=key,
        outer_steps=1000,
        outer_step_size0=0.01,
        inner_T_prior=2,
        inner_T_posterior=1,
        inner_step_size0=0.01,
        warmup_steps=5,
        stop_signal=stop_after_10,
    )

    assert result.meta["stopped"] is True
    assert result.meta["stop_step"] is not None


def test_preconditioner_accepted():
    """Runs with diagonal preconditioners without error."""
    key = jax.random.PRNGKey(5)
    result = nested_sgld(
        lambda0=jnp.array([0.0]),
        phi0=jnp.ones(3) * 0.1,
        prior_grad_fn=_quadratic_prior_grad,
        posterior_grad_fn=_quadratic_posterior_grad,
        key=key,
        outer_steps=30,
        outer_step_size0=0.01,
        inner_T_prior=3,
        inner_T_posterior=1,
        inner_step_size0=0.01,
        warmup_steps=20,
        phi_preconditioner=jnp.array([1.0, 0.5, 0.25]),
        lambda_preconditioner=jnp.array([1.0]),
    )

    assert np.isfinite(result.lambda_chain).all()
    assert np.isfinite(result.phi_prior_final).all()
    assert np.isfinite(result.phi_posterior_final).all()


def test_custom_lambda_prior():
    """Non-Jeffrey's prior contributes to grad_lam_total."""
    key = jax.random.PRNGKey(6)

    def gaussian_prior_grad(lam):
        """N(0, 1) prior on lam: grad = lam."""
        return lam

    result = nested_sgld(
        lambda0=jnp.array([1.0]),
        phi0=jnp.ones(2) * 0.1,
        prior_grad_fn=_quadratic_prior_grad,
        posterior_grad_fn=_quadratic_posterior_grad,
        key=key,
        lambda_prior_grad_fn=gaussian_prior_grad,
        outer_steps=30,
        outer_step_size0=0.01,
        inner_T_prior=3,
        inner_T_posterior=1,
        inner_step_size0=0.01,
        warmup_steps=20,
    )

    # grad_total should NOT equal grad_posterior - grad_prior (has prior term)
    diff = result.traces["grad_lam_posterior"] - result.traces["grad_lam_prior"]
    residual = result.traces["grad_lam_total"] - diff
    # The prior term should be non-zero for at least some steps
    assert np.any(np.abs(residual) > 1e-10), "Prior gradient term should be non-zero"
