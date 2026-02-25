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
