"""Simple diagnostics for Markov chains."""

from __future__ import annotations

import numpy as np


def lag1_autocorr(series: np.ndarray) -> float:
    series = np.asarray(series, dtype=float)
    if series.size < 2:
        return float("nan")
    x = series[:-1]
    y = series[1:]
    x_std = float(np.std(x))
    y_std = float(np.std(y))
    if x_std == 0.0 or y_std == 0.0 or not np.isfinite(x_std) or not np.isfinite(y_std):
        return 0.0
    xz = (x - np.mean(x)) / x_std
    yz = (y - np.mean(y)) / y_std
    val = float(np.mean(xz * yz))
    if not np.isfinite(val):
        return float("nan")
    return float(np.clip(val, -1.0, 1.0))


def credible_interval(samples: np.ndarray, low: float = 0.05, high: float = 0.95) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(samples, dtype=float)
    return np.quantile(arr, low, axis=0), np.quantile(arr, high, axis=0)


def autocorrelation(series: np.ndarray, max_lag: int | None = None) -> np.ndarray:
    """FFT-based normalized autocorrelation function.

    Parameters
    ----------
    series : 1-D array of length *N*.
    max_lag : largest lag to return.  Defaults to ``N // 2``.

    Returns
    -------
    acf : 1-D array of shape ``(max_lag + 1,)`` with ``acf[0] = 1``.
    """
    x = np.asarray(series, dtype=float).ravel()
    n = x.size
    if n < 2:
        return np.array([1.0])
    if max_lag is None:
        max_lag = n // 2
    max_lag = min(max_lag, n - 1)

    x = x - x.mean()
    var = float(np.dot(x, x))
    if var == 0.0:
        return np.ones(max_lag + 1)

    # zero-pad to next power-of-two for FFT efficiency
    nfft = 1
    while nfft < 2 * n:
        nfft *= 2
    fwd = np.fft.rfft(x, n=nfft)
    acf_full = np.fft.irfft(fwd * np.conj(fwd), n=nfft)[:n]
    acf_full /= var
    return acf_full[: max_lag + 1]


def effective_sample_size(series: np.ndarray, max_lag: int | None = None) -> float:
    """Effective sample size via Geyer's initial positive sequence estimator.

    Sums consecutive *pairs* of autocorrelations and stops when a pair sum
    is negative, preventing noisy high-lag estimates from inflating ESS.
    """
    x = np.asarray(series, dtype=float).ravel()
    n = x.size
    if n < 4:
        return float(n)

    acf = autocorrelation(x, max_lag=max_lag)

    # Sum consecutive pairs: (acf[1]+acf[2]), (acf[3]+acf[4]), ...
    tau = 0.0
    k = 1
    while k + 1 < len(acf):
        pair = float(acf[k] + acf[k + 1])
        if pair < 0.0:
            break
        tau += pair
        k += 2
    # If odd lag remaining and positive, include it
    if k < len(acf) and float(acf[k]) > 0.0:
        tau += float(acf[k])

    ess = n / (1.0 + 2.0 * tau)
    return max(1.0, ess)


def effective_sample_size_bulk(chain: np.ndarray, max_lag: int | None = None) -> np.ndarray:
    """Per-dimension ESS for a chain of shape ``(n_samples, dim)``."""
    chain = np.asarray(chain, dtype=float)
    if chain.ndim == 1:
        return np.array([effective_sample_size(chain, max_lag=max_lag)])
    return np.array([effective_sample_size(chain[:, j], max_lag=max_lag) for j in range(chain.shape[1])])


def credible_interval_coverage(
    samples: np.ndarray,
    true_values: np.ndarray,
    level: float = 0.95,
) -> float:
    """Fraction of *true_values* falling inside the credible interval.

    Parameters
    ----------
    samples : array of shape ``(n_samples, n_points)``
    true_values : array of shape ``(n_points,)``
    level : coverage level, e.g. 0.95 for a 95 % CI.
    """
    alpha = (1.0 - level) / 2.0
    lo, hi = credible_interval(samples, low=alpha, high=1.0 - alpha)
    true_values = np.asarray(true_values)
    inside = (true_values >= lo) & (true_values <= hi)
    return float(np.mean(inside))


def gelman_rubin_rhat(chains: list[np.ndarray]) -> np.ndarray:
    """Split R-hat diagnostic from *m* chains.

    Parameters
    ----------
    chains : list of arrays each with shape ``(n_samples, dim)``.
             All must share the same ``n_samples`` and ``dim``.

    Returns
    -------
    rhat : array of shape ``(dim,)``
    """
    chains = [np.asarray(c, dtype=float) for c in chains]
    m = len(chains)
    n, d = chains[0].shape

    # Split each chain in half for split-R-hat
    split = []
    for c in chains:
        half = n // 2
        split.append(c[:half])
        split.append(c[half : 2 * half])
    m_split = len(split)
    n_split = split[0].shape[0]

    means = np.array([s.mean(axis=0) for s in split])       # (m_split, d)
    variances = np.array([s.var(axis=0, ddof=1) for s in split])  # (m_split, d)

    grand_mean = means.mean(axis=0)                           # (d,)
    B = n_split * np.var(means, axis=0, ddof=1)               # between-chain variance
    W = variances.mean(axis=0)                                # within-chain variance

    var_hat = (1.0 - 1.0 / n_split) * W + B / n_split
    rhat = np.sqrt(var_hat / np.maximum(W, 1e-30))
    return rhat
