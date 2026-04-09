"""Tests for the 2D Poisson forward PIFT runner (Module 2.1)."""

from __future__ import annotations

import jax
import numpy as np
import pytest

from core.energies_2d_poisson import poisson_2d_residual_energy
from core.parameterizations_2d import SineBasis2D
from pipelines.phase_e_2d_poisson import (
    DEFAULT_PHASE_E_CONFIG,
    forcing_2d,
    phi_true_2d,
    run_phase_e_2d_poisson,
)


def _gpu_available() -> bool:
    try:
        return len(jax.devices("gpu")) > 0
    except Exception:  # noqa: BLE001
        return False


DEVICES = ["cpu", "gpu"]


def _maybe_skip(device: str):
    if device == "gpu" and not _gpu_available():
        pytest.skip("no JAX GPU device available")


def test_sine_basis_2d_laplacian_eigenvalues():
    """The 2D sine basis is an exact eigenbasis of -Δ; check the spectrum."""
    field = SineBasis2D(n_modes=3)
    # Single mode (i=2, j=3): coefficient at row 1, col 2 ⇒ flat index 1*3+2 = 5
    theta = np.zeros(field.n_params)
    theta[1 * 3 + 2] = 1.0

    xy = np.array([[0.3, 0.7], [0.5, 0.5], [0.1, 0.9]])
    import jax.numpy as jnp  # noqa: PLC0415

    phi = np.asarray(field.eval(jnp.asarray(xy), jnp.asarray(theta)))
    neg_lap = np.asarray(field.neg_laplacian(jnp.asarray(xy), jnp.asarray(theta)))

    # Eigenvalue: (pi*2)^2 + (pi*3)^2
    expected_eig = (np.pi * 2) ** 2 + (np.pi * 3) ** 2
    np.testing.assert_allclose(neg_lap, expected_eig * phi, rtol=1e-10, atol=1e-12)


def test_poisson_2d_energy_zero_at_truth():
    """At the analytical solution the residual energy must vanish (in the limit)."""
    import jax.numpy as jnp  # noqa: PLC0415

    field = SineBasis2D(n_modes=4)
    # Truth = sin(pi x) sin(pi y) ⇒ coefficient (i=1, j=1), flat index 0*4+0 = 0.
    theta = np.zeros(field.n_params)
    theta[0] = 1.0

    rng = np.random.default_rng(0)
    xy = rng.uniform(0.0, 1.0, size=(512, 2))
    energy, grad, _ = poisson_2d_residual_energy(jnp.asarray(theta), jnp.asarray(xy), field, forcing_2d)

    assert float(energy) < 1e-20
    assert float(np.max(np.abs(np.asarray(grad)))) < 1e-10


@pytest.mark.parametrize("device_preference", DEVICES)
def test_phase_e_runner_recovers_solution(device_preference):
    _maybe_skip(device_preference)

    cfg = dict(DEFAULT_PHASE_E_CONFIG)
    cfg.update({
        "n_steps": 1500,
        "burn_in": 300,
        "thin": 8,
        "n_quad": 192,
        "n_modes": 4,  # 16 params — small for the test
    })
    r = run_phase_e_2d_poisson(cfg=cfg, device_preference=device_preference, save_outputs=False)

    assert r["status"] == "completed"
    # Sanity: returned dict shape.
    for key in ("summary", "metrics", "diagnostics", "traces", "chain", "phi_mean", "phi_truth"):
        assert key in r

    l2 = r["metrics"]["l2_error"]
    assert np.isfinite(l2)
    # The truth lives exactly in this basis; with 1500 SGLD steps the
    # posterior mean should be well within 0.1 RMSE.
    assert l2 < 0.15
