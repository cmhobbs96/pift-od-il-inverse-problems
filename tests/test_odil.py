"""Tests for the ODIL 1D Poisson solver (``core.odil``).

These tests are parametrized over CPU/GPU.  The GPU case is skipped when no
GPU device is visible to JAX, so the suite passes on CPU-only CI machines but
exercises both code paths whenever a GPU is available.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from core.odil import ODILResult, odil_solve_poisson_1d


def _gpu_available() -> bool:
    try:
        return len(jax.devices("gpu")) > 0
    except Exception:  # noqa: BLE001
        return False


def _device_context(preference: str):
    if preference == "gpu":
        if not _gpu_available():
            pytest.skip("no JAX GPU device available")
        device = jax.devices("gpu")[0]
    else:
        device = jax.devices("cpu")[0]
    return jax.default_device(device)


DEVICES = ["cpu", "gpu"]


def _forcing_sin_pi(x):
    # f(x) = pi^2 sin(pi x)  ⇒  -u'' = f  ⇒  u(x) = sin(pi x).
    return (jnp.pi**2) * jnp.sin(jnp.pi * jnp.asarray(x))


def _truth_sin_pi(x):
    return np.sin(np.pi * np.asarray(x, dtype=float))


@pytest.mark.parametrize("device_preference", DEVICES)
def test_gauss_newton_recovers_analytical(device_preference):
    with _device_context(device_preference):
        result = odil_solve_poisson_1d(
            forcing_fn=_forcing_sin_pi,
            n_grid=129,
            domain=(0.0, 1.0),
            bc=(0.0, 0.0),
            method="gauss_newton",
            max_iter=50,
            tol=1e-10,
        )
    assert isinstance(result, ODILResult)
    assert result.converged
    # The discrete FD scheme is 2nd order, so on N=129 we expect L2 ≲ 1e-4.
    truth = _truth_sin_pi(result.x_grid)
    l2 = float(np.sqrt(np.mean((result.u_grid - truth) ** 2)))
    assert l2 < 5e-4
    # Linear problem ⇒ Gauss-Newton converges in O(1) iterations.
    assert result.n_iterations <= 25
    # Loss decreased monotonically (LM accepted every step).
    assert np.all(np.diff(result.loss_history) <= 1e-12)


@pytest.mark.parametrize("device_preference", DEVICES)
def test_lbfgs_recovers_analytical(device_preference):
    with _device_context(device_preference):
        result = odil_solve_poisson_1d(
            forcing_fn=_forcing_sin_pi,
            n_grid=65,
            domain=(0.0, 1.0),
            bc=(0.0, 0.0),
            method="lbfgs",
            max_iter=500,
            tol=1e-10,
        )
    assert result.method_used == "lbfgs"
    truth = _truth_sin_pi(result.x_grid)
    l2 = float(np.sqrt(np.mean((result.u_grid - truth) ** 2)))
    # L-BFGS on a stiff quadratic doesn't reach FD-truncation accuracy as
    # fast as GN; relax the bound but still demand a meaningful solve.
    assert l2 < 5e-3


@pytest.mark.parametrize("device_preference", DEVICES)
def test_data_term_tilts_solution(device_preference):
    """With observations far from the analytical solution, the MAP shifts."""
    rng = np.random.default_rng(0)
    x_obs = np.linspace(0.1, 0.9, 9)
    # Bias the observations upward by 0.5 to force a tilt.
    y_obs = _truth_sin_pi(x_obs) + 0.5 + 0.01 * rng.standard_normal(x_obs.size)

    with _device_context(device_preference):
        plain = odil_solve_poisson_1d(
            forcing_fn=_forcing_sin_pi, n_grid=65, bc=(0.0, 0.0),
            method="gauss_newton", max_iter=50, tol=1e-10,
        )
        with_data = odil_solve_poisson_1d(
            forcing_fn=_forcing_sin_pi, n_grid=65, bc=(0.0, 0.0),
            obs=(x_obs, y_obs), noise_std=0.05,
            method="gauss_newton", max_iter=50, tol=1e-10,
        )

    # The data-tilted solution should sit higher than the pure-physics one
    # in the interior (where the data biases it up).
    interior = slice(5, 60)
    assert np.mean(with_data.u_grid[interior]) > np.mean(plain.u_grid[interior]) + 0.05


def test_jacobian_structure_is_tridiagonal_plus_bc():
    """Sanity check on the residual Jacobian: PDE rows are tridiagonal."""
    from core.odil import _make_residual_fn  # noqa: PLC0415

    n_grid = 16
    h = 1.0 / (n_grid - 1)
    x_grid_j = jnp.linspace(0.0, 1.0, n_grid)
    f_grid_j = jnp.zeros(n_grid)

    residual_fn, _ = _make_residual_fn(
        n_grid=n_grid, h=h, x_grid_j=x_grid_j, f_grid_j=f_grid_j,
        bc_left=0.0, bc_right=0.0, w_bc=1.0,
        x_obs_j=None, y_obs_j=None, inv_sigma=0.0, domain_a=0.0,
    )
    j = np.asarray(jax.jacfwd(residual_fn)(jnp.zeros(n_grid)))
    # Interior PDE rows occupy the first n_grid - 2 rows; only columns
    # i, i+1, i+2 (relative to row index) are nonzero.
    n_int = n_grid - 2
    pde_rows = j[:n_int]
    for i in range(n_int):
        nonzero_cols = np.flatnonzero(np.abs(pde_rows[i]) > 1e-12)
        assert set(nonzero_cols.tolist()).issubset({i, i + 1, i + 2})
