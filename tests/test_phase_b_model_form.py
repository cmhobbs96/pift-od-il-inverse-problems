"""Tests for Phase B model-form uncertainty (Example 2).

Covers:
1. Misspecified energy gradient finite-difference verification
2. Misspecified energy reduces to correct energy at gamma=1
3. Source term interpolation correctness
4. Pipeline smoke test (small config)
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

from core.energies_nonlinear import nonlinear_energy, nonlinear_energy_misspecified
from core.parameterizations_nonlinear import make_nonlinear_field

SOURCE_FN = lambda x: jnp.cos(4.0 * x)
D = 0.1
KAPPA = 1.0
BC = (0.0, 0.0)


# ---------------------------------------------------------------------------
# 1. Misspecified energy gradient FD
# ---------------------------------------------------------------------------


def test_misspecified_energy_grad_fd():
    """Gradient of misspecified energy matches central finite differences."""
    rng = np.random.default_rng(42)
    field = make_nonlinear_field(K=6, bc=BC)
    theta = jnp.asarray(rng.normal(size=field.n_params) * 0.3)
    x_quad = jnp.asarray(rng.uniform(0.0, 1.0, size=200))

    for gamma in [0.0, 0.5, 1.0]:
        _, grad, _ = nonlinear_energy_misspecified(
            theta, x_quad, field, SOURCE_FN, D, KAPPA, gamma
        )
        grad = np.asarray(grad)

        eps = 1e-6
        fd = np.zeros(field.n_params, dtype=float)
        theta_np = np.asarray(theta)
        for i in range(field.n_params):
            d = np.zeros_like(theta_np)
            d[i] = eps
            e_plus, _, _ = nonlinear_energy_misspecified(
                theta_np + d, x_quad, field, SOURCE_FN, D, KAPPA, gamma
            )
            e_minus, _, _ = nonlinear_energy_misspecified(
                theta_np - d, x_quad, field, SOURCE_FN, D, KAPPA, gamma
            )
            fd[i] = (e_plus - e_minus) / (2.0 * eps)

        assert np.allclose(grad, fd, atol=1e-4, rtol=1e-4), (
            f"gamma={gamma}: max diff = {np.max(np.abs(grad - fd))}"
        )


# ---------------------------------------------------------------------------
# 2. Misspecified energy at gamma=1 equals correct energy
# ---------------------------------------------------------------------------


def test_misspecified_equals_correct_at_gamma_1():
    """At gamma=1, misspecified energy equals the standard nonlinear energy."""
    rng = np.random.default_rng(7)
    field = make_nonlinear_field(K=8, bc=BC)
    theta = jnp.asarray(rng.normal(size=field.n_params) * 0.3)
    x_quad = jnp.asarray(rng.uniform(0.0, 1.0, size=200))

    e_correct, g_correct, _ = nonlinear_energy(
        theta, x_quad, field, SOURCE_FN, D, KAPPA
    )
    e_misspec, g_misspec, _ = nonlinear_energy_misspecified(
        theta, x_quad, field, SOURCE_FN, D, KAPPA, gamma=1.0
    )

    assert np.isclose(e_correct, e_misspec, atol=1e-12), (
        f"Energy mismatch: {e_correct} vs {e_misspec}"
    )
    assert np.allclose(np.asarray(g_correct), np.asarray(g_misspec), atol=1e-10), (
        "Gradient mismatch at gamma=1"
    )


# ---------------------------------------------------------------------------
# 3. Source term interpolation
# ---------------------------------------------------------------------------


def test_source_term_interpolation():
    """Misspecified source f_gamma mixes correctly between cos(4x) and exp(-x)."""
    from pipelines.phase_b_model_form import _make_misspecified_source, _true_source

    x = jnp.linspace(0.0, 1.0, 50)

    # gamma=1: should be cos(4x)
    f1 = _make_misspecified_source(1.0)(x)
    assert np.allclose(np.asarray(f1), np.asarray(_true_source(x)), atol=1e-12)

    # gamma=0: should be exp(-x)
    f0 = _make_misspecified_source(0.0)(x)
    assert np.allclose(np.asarray(f0), np.asarray(jnp.exp(-x)), atol=1e-12)

    # gamma=0.5: should be 0.5*cos(4x) + 0.5*exp(-x)
    f05 = _make_misspecified_source(0.5)(x)
    expected = 0.5 * jnp.cos(4.0 * x) + 0.5 * jnp.exp(-x)
    assert np.allclose(np.asarray(f05), np.asarray(expected), atol=1e-12)


# ---------------------------------------------------------------------------
# 4. Pipeline smoke test
# ---------------------------------------------------------------------------


def test_phase_b_model_form_smoke(tmp_path):
    """Small model-form run completes and returns expected structure."""
    from pipelines.phase_b_model_form import run_phase_b_model_form

    result = run_phase_b_model_form(
        cfg={
            "gamma_values": [0.0, 1.0],
            "K": 5,
            "n_obs": 10,
            "outer_steps": 100,
            "warmup_steps": 50,
            "n_quad": 32,
            "n_grid": 50,
            "burn_in_frac": 0.2,
        },
        output_root=str(tmp_path),
        save_outputs=True,
    )

    assert result["status"] == "completed"
    assert "experiments" in result
    assert "source_error" in result["experiments"]
    assert "energy_error" in result["experiments"]

    # Each experiment should have results for 2 gamma values
    for exp_type in ["source_error", "energy_error"]:
        exp = result["experiments"][exp_type]
        assert len(exp) == 2
        for r in exp:
            assert "gamma" in r
            assert "beta_median" in r
            assert "beta_samples" in r
            assert np.isfinite(r["beta_median"])
            assert len(r["beta_samples"]) > 0

    # Check output files exist
    assert (tmp_path / "figures" / "phase_b_model_form.png").exists()
    assert (tmp_path / "tables" / "phase_b_model_form_summary.json").exists()
