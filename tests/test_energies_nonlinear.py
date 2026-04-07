"""Tests for nonlinear PDE energy and reference solver.

Covers:
1. Gradient finite-difference verification for nonlinear_energy
2. Stationarity condition: energy minimizer satisfies D*phi'' - kappa*phi^3 = f
3. Energy minimizer matches nonlinear FD solver (L2 < 0.05)
4. Nonlinear FD solver convergence and PDE residual
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

from core.energies_nonlinear import nonlinear_energy
from core.parameterizations_nonlinear import make_nonlinear_field, solve_nonlinear_dirichlet_fd

# Paper parameters: D=0.1, kappa=1, f(x)=cos(4x), zero Dirichlet BCs
D = 0.1
KAPPA = 1.0
SOURCE_FN = lambda x: jnp.cos(4.0 * x)
SOURCE_FN_NP = lambda x: np.cos(4.0 * x)
BC = (0.0, 0.0)


# ---------------------------------------------------------------------------
# 1. Gradient finite-difference verification
# ---------------------------------------------------------------------------


def test_nonlinear_energy_grad_fd():
    """Energy gradient matches central finite differences."""
    rng = np.random.default_rng(42)
    field = make_nonlinear_field(K=6, bc=BC)
    theta = jnp.asarray(rng.normal(size=field.n_params) * 0.3)
    x_quad = jnp.asarray(rng.uniform(0.0, 1.0, size=200))

    _, grad, _ = nonlinear_energy(theta, x_quad, field, SOURCE_FN, D, KAPPA)
    grad = np.asarray(grad)

    eps = 1e-6
    fd = np.zeros(field.n_params, dtype=float)
    theta_np = np.asarray(theta)
    for i in range(field.n_params):
        d = np.zeros_like(theta_np)
        d[i] = eps
        e_plus, _, _ = nonlinear_energy(theta_np + d, x_quad, field, SOURCE_FN, D, KAPPA)
        e_minus, _, _ = nonlinear_energy(theta_np - d, x_quad, field, SOURCE_FN, D, KAPPA)
        fd[i] = (e_plus - e_minus) / (2.0 * eps)

    assert np.allclose(grad, fd, atol=1e-4, rtol=1e-4), (
        f"Max diff: {np.max(np.abs(grad - fd))}"
    )


# ---------------------------------------------------------------------------
# 2. Stationarity: minimizer satisfies the PDE
# ---------------------------------------------------------------------------


def test_stationarity_recovers_pde():
    """At the energy minimum, D*phi'' - kappa*phi^3 ≈ f (Appendix C).

    Verified by comparing the optimized field against the FD reference
    solver, which directly enforces the PDE. Agreement implies the
    energy stationarity condition recovers the correct PDE.
    """
    from scipy.optimize import minimize

    field = make_nonlinear_field(K=20, bc=BC)
    x_quad = jnp.linspace(0.01, 0.99, 500)

    def objective(theta_np):
        theta = jnp.asarray(theta_np)
        e, g, _ = nonlinear_energy(theta, x_quad, field, SOURCE_FN, D, KAPPA)
        return float(e), np.asarray(g, dtype=np.float64)

    theta0 = np.zeros(field.n_params)
    result = minimize(objective, theta0, jac=True, method="L-BFGS-B", options={"maxiter": 3000})
    assert result.success, f"Optimization failed: {result.message}"

    # Compare energy minimizer against FD solver (which enforces D*phi'' - kappa*phi^3 = f)
    x_ref, phi_ref = solve_nonlinear_dirichlet_fd(
        SOURCE_FN_NP, D=D, kappa=KAPPA, n_points=500, bc=BC
    )

    # Evaluate on interior points (avoid boundary artifacts from window function)
    x_eval = np.linspace(0.05, 0.95, 200)
    theta_opt = jnp.asarray(result.x)
    phi_opt = np.asarray(field.eval(jnp.asarray(x_eval), theta_opt))
    phi_truth = np.interp(x_eval, x_ref, phi_ref)

    # Pointwise agreement implies the stationarity condition recovers the PDE
    max_err = np.max(np.abs(phi_opt - phi_truth))
    assert max_err < 0.05, (
        f"Energy minimizer doesn't match FD PDE solver: max |err| = {max_err:.4f}"
    )


# ---------------------------------------------------------------------------
# 3. Energy minimizer matches FD reference solver
# ---------------------------------------------------------------------------


def test_energy_minimizer_matches_fd():
    """Minimizing U[phi] recovers the nonlinear FD reference solution."""
    from scipy.optimize import minimize

    field = make_nonlinear_field(K=20, bc=BC)

    # FD reference
    x_ref, phi_ref = solve_nonlinear_dirichlet_fd(
        SOURCE_FN_NP, D=D, kappa=KAPPA, n_points=500, bc=BC
    )

    # Energy minimization with deterministic quadrature
    x_quad = jnp.linspace(0.01, 0.99, 500)

    def objective(theta_np):
        theta = jnp.asarray(theta_np)
        e, g, _ = nonlinear_energy(theta, x_quad, field, SOURCE_FN, D, KAPPA)
        return float(e), np.asarray(g, dtype=np.float64)

    theta0 = np.zeros(field.n_params)
    result = minimize(objective, theta0, jac=True, method="L-BFGS-B", options={"maxiter": 3000})
    assert result.success, f"Optimization failed: {result.message}"

    # Compare on evaluation grid
    x_eval = jnp.linspace(0.0, 1.0, 200)
    phi_opt = np.asarray(field.eval(x_eval, jnp.asarray(result.x)))
    phi_truth = np.interp(np.asarray(x_eval), x_ref, phi_ref)

    l2 = np.sqrt(np.mean((phi_opt - phi_truth) ** 2))
    assert l2 < 0.05, f"L2 error too large: {l2:.6f}"


# ---------------------------------------------------------------------------
# 4. Nonlinear FD solver convergence
# ---------------------------------------------------------------------------


def test_nonlinear_fd_solver_convergence():
    """FD solver converges and satisfies the PDE residual."""
    x, phi = solve_nonlinear_dirichlet_fd(
        SOURCE_FN_NP, D=D, kappa=KAPPA, n_points=500, bc=BC
    )

    # Check boundary conditions
    assert np.isclose(phi[0], BC[0], atol=1e-12)
    assert np.isclose(phi[-1], BC[1], atol=1e-12)

    # Check PDE residual at interior points
    h = x[1] - x[0]
    phi_pp = (phi[2:] - 2.0 * phi[1:-1] + phi[:-2]) / (h * h)
    phi_mid = phi[1:-1]
    f_mid = SOURCE_FN_NP(x[1:-1])

    # D*phi'' - kappa*phi^3 - f should be ≈ 0
    pde_residual = D * phi_pp - KAPPA * phi_mid**3 - f_mid
    assert np.max(np.abs(pde_residual)) < 1e-4, (
        f"PDE residual too large: max |r| = {np.max(np.abs(pde_residual)):.6f}"
    )

    # Solution should be finite and smooth
    assert np.all(np.isfinite(phi))
    assert np.max(np.abs(phi)) < 10.0, "Solution magnitude unreasonable"
