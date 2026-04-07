"""Tests for Karhunen-Loève Expansion via Nyström approximation."""

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


def test_eigenvalues_positive_decreasing():
    """KLE eigenvalues should be positive and sorted descending."""
    kle = KLE(lengthscale=0.3, n_terms=10, n_grid=200)
    eigs = kle.eigenvalues
    assert len(eigs) == 10
    assert np.all(eigs > 0), "Eigenvalues should be positive"
    assert np.all(np.diff(eigs) <= 0), "Eigenvalues should be non-increasing"


def test_eigenfunctions_orthonormal():
    """Eigenfunctions should be approximately orthonormal under L2 inner product."""
    kle = KLE(lengthscale=0.3, n_terms=10, n_grid=300)
    x = np.linspace(0.0, 1.0, 500)
    h = 1.0 / (len(x) - 1)
    phi = np.asarray(kle.eigenfunctions(jnp.asarray(x)))  # (500, 10)

    # Gram matrix: G[i,j] = h * sum phi_i(x) * phi_j(x) ≈ delta_ij
    G = h * phi.T @ phi
    np.testing.assert_allclose(G, np.eye(10), atol=0.15)


def test_reconstruct_known_coefficients():
    """Reconstruction with known z recovers expected function shape."""
    kle = KLE(lengthscale=0.3, n_terms=10, n_grid=200)
    x = jnp.linspace(0.0, 1.0, 100)

    # Zero coefficients → zero function
    z_zero = jnp.zeros(10)
    f_zero = kle.reconstruct(x, z_zero)
    np.testing.assert_allclose(np.asarray(f_zero), 0.0, atol=1e-12)

    # Non-zero coefficients → non-zero function
    z = jnp.ones(10)
    f = kle.reconstruct(x, z)
    assert np.max(np.abs(np.asarray(f))) > 0.01, "Non-zero z should produce non-zero f"


def test_covariance_reproduced():
    """Truncated KLE should approximately reproduce the kernel at grid points."""
    ls = 0.3
    kle = KLE(lengthscale=ls, n_terms=10, n_grid=200)
    x = jnp.linspace(0.0, 1.0, 50)

    phi = np.asarray(kle.eigenfunctions(x))  # (50, 10)
    eigs = kle.eigenvalues  # (10,)

    # Reconstructed covariance: C_hat[i,j] = sum_k lambda_k phi_k(x_i) phi_k(x_j)
    C_hat = phi @ np.diag(eigs) @ phi.T

    # True kernel
    x_np = np.asarray(x)
    diff = x_np[:, None] - x_np[None, :]
    C_true = np.exp(-diff**2 / (2.0 * ls**2))

    # Should be close (not exact due to truncation)
    rel_err = np.linalg.norm(C_hat - C_true) / np.linalg.norm(C_true)
    assert rel_err < 0.15, f"Covariance relative error too large: {rel_err:.4f}"


def test_energy_fraction():
    """Energy fraction should be between 0 and 1 and close to 1 for smooth kernels."""
    kle = KLE(lengthscale=0.3, n_terms=10, n_grid=200)
    frac = kle.energy_fraction()
    assert 0.0 < frac <= 1.05, f"Energy fraction out of range: {frac}"
    assert frac > 0.8, f"Energy fraction too low for smooth kernel: {frac}"


def test_reconstruct_differentiable():
    """Reconstruction should be differentiable w.r.t. z for JAX autodiff."""
    kle = KLE(lengthscale=0.3, n_terms=5, n_grid=100)
    x = jnp.linspace(0.0, 1.0, 30)
    z = jnp.ones(5) * 0.5

    def loss(z_inner):
        f = kle.reconstruct(x, z_inner)
        return jnp.sum(f**2)

    grad = jax.grad(loss)(z)
    assert jnp.isfinite(grad).all(), "Gradient should be finite"
    assert jnp.any(grad != 0), "Gradient should be non-zero"
