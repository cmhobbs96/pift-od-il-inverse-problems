"""Tests for extended diagnostics: ACF, ESS, R-hat, coverage."""

from __future__ import annotations

import numpy as np
import pytest

from src.utils.diagnostics import (
    autocorrelation,
    credible_interval_coverage,
    effective_sample_size,
    effective_sample_size_bulk,
    gelman_rubin_rhat,
)


# ---------------------------------------------------------------------------
# autocorrelation
# ---------------------------------------------------------------------------

def test_autocorrelation_white_noise():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(5000)
    acf = autocorrelation(x, max_lag=50)
    assert acf[0] == pytest.approx(1.0)
    # Lags > 0 should be near zero for white noise
    assert np.all(np.abs(acf[1:]) < 0.1)


def test_autocorrelation_ar1():
    """ACF of AR(1) with rho=0.9 should decay as rho^k."""
    rng = np.random.default_rng(0)
    rho = 0.9
    n = 20000
    x = np.empty(n)
    x[0] = 0.0
    for i in range(1, n):
        x[i] = rho * x[i - 1] + rng.standard_normal()
    acf = autocorrelation(x, max_lag=10)
    for k in range(1, 6):
        expected = rho ** k
        assert acf[k] == pytest.approx(expected, abs=0.05), f"lag {k}: {acf[k]} vs {expected}"


def test_autocorrelation_short_series():
    acf = autocorrelation(np.array([1.0]))
    assert len(acf) == 1
    assert acf[0] == 1.0


def test_autocorrelation_constant_series():
    acf = autocorrelation(np.ones(100), max_lag=10)
    assert acf[0] == 1.0


# ---------------------------------------------------------------------------
# effective_sample_size
# ---------------------------------------------------------------------------

def test_ess_iid():
    """ESS of iid samples should be close to N."""
    rng = np.random.default_rng(123)
    x = rng.standard_normal(2000)
    ess = effective_sample_size(x)
    assert ess > 1500  # should be ~2000 but allow some slack


def test_ess_correlated():
    """ESS of highly correlated samples should be << N."""
    rng = np.random.default_rng(0)
    n = 5000
    x = np.empty(n)
    x[0] = 0.0
    for i in range(1, n):
        x[i] = 0.99 * x[i - 1] + rng.standard_normal()
    ess = effective_sample_size(x)
    assert ess < n / 5  # ESS should be much less than N


def test_ess_short():
    ess = effective_sample_size(np.array([1.0, 2.0]))
    assert ess > 0


# ---------------------------------------------------------------------------
# effective_sample_size_bulk
# ---------------------------------------------------------------------------

def test_ess_bulk_shape():
    rng = np.random.default_rng(7)
    chain = rng.standard_normal((500, 4))
    ess = effective_sample_size_bulk(chain)
    assert ess.shape == (4,)
    assert np.all(ess > 0)


def test_ess_bulk_1d():
    rng = np.random.default_rng(7)
    x = rng.standard_normal(500)
    ess = effective_sample_size_bulk(x)
    assert ess.shape == (1,)


# ---------------------------------------------------------------------------
# credible_interval_coverage
# ---------------------------------------------------------------------------

def test_coverage_well_calibrated():
    """Draw true values from the same distribution as samples; coverage ~ level."""
    rng = np.random.default_rng(42)
    n_points = 500
    n_samples = 2000
    # True values drawn from N(0, 1), samples also from N(0, 1).
    # Each true value is an *independent* draw, not the mean, so some will
    # fall outside the 95% quantile band => coverage should be ~95%.
    samples = rng.standard_normal((n_samples, n_points))
    true_values = rng.standard_normal(n_points)
    cov = credible_interval_coverage(samples, true_values, level=0.95)
    assert cov == pytest.approx(0.95, abs=0.05)


def test_coverage_narrow_interval():
    """Very narrow CI should have low coverage."""
    rng = np.random.default_rng(0)
    true_values = np.array([0.0, 1.0, 2.0])
    # Samples very tightly clustered at 10.0
    samples = 10.0 + rng.standard_normal((1000, 3)) * 0.001
    cov = credible_interval_coverage(samples, true_values, level=0.95)
    assert cov == pytest.approx(0.0, abs=0.01)


# ---------------------------------------------------------------------------
# gelman_rubin_rhat
# ---------------------------------------------------------------------------

def test_rhat_same_distribution():
    """R-hat of chains from the same distribution should be close to 1."""
    rng = np.random.default_rng(99)
    chains = [rng.standard_normal((1000, 3)) for _ in range(4)]
    rhat = gelman_rubin_rhat(chains)
    assert rhat.shape == (3,)
    assert np.all(rhat < 1.05)


def test_rhat_different_means():
    """R-hat of chains with different means should be >> 1."""
    rng = np.random.default_rng(0)
    chains = [
        rng.standard_normal((1000, 2)) + i * 5.0
        for i in range(3)
    ]
    rhat = gelman_rubin_rhat(chains)
    assert np.all(rhat > 1.5)
