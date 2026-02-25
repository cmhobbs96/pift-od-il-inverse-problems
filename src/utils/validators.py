"""Validation helpers for configs and observation data."""

from __future__ import annotations

from typing import Any

import numpy as np


class ValidationError(ValueError):
    """Validation failure with actionable message."""


def validate_phase_a_config(cfg: dict[str, Any]) -> list[str]:
    errors: list[str] = []

    n_steps = int(cfg.get("n_steps", 0))
    burn_in = int(cfg.get("burn_in", 0))
    thin = int(cfg.get("thin", 0))
    noise_std = float(cfg.get("noise_std", 0.0))
    step_size0 = float(cfg.get("step_size0", 0.0))
    beta = float(cfg.get("beta", 0.0))
    n_quad = int(cfg.get("n_quad", 0))
    n_modes = int(cfg.get("n_modes", 0))
    runtime_check_interval = int(cfg.get("runtime_check_interval", 1))

    if n_steps <= 0:
        errors.append("n_steps must be > 0")
    if burn_in < 0:
        errors.append("burn_in must be >= 0")
    if n_steps <= burn_in:
        errors.append("n_steps must be > burn_in")
    if thin < 1:
        errors.append("thin must be >= 1")
    if noise_std <= 0:
        errors.append("noise_std must be > 0")
    if not (1e-7 <= step_size0 <= 1e-1):
        errors.append("step_size0 must be in [1e-7, 1e-1]")
    if beta <= 0:
        errors.append("beta must be > 0")
    if n_quad < 4:
        errors.append("n_quad must be >= 4")
    if n_modes < 1:
        errors.append("n_modes must be >= 1")
    if runtime_check_interval < 1:
        errors.append("runtime_check_interval must be >= 1")

    return errors


def validate_observations(x: np.ndarray, y: np.ndarray) -> list[str]:
    errors: list[str] = []
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if x.shape != y.shape:
        errors.append("Observation x and y must have the same shape")
        return errors

    if x.ndim != 1:
        errors.append("Observation x and y must be 1D arrays")
    if x.size < 4:
        errors.append("At least 4 observation points are required")
    if np.any(~np.isfinite(x)) or np.any(~np.isfinite(y)):
        errors.append("Observation arrays must be finite")
    if np.any((x < 0.0) | (x > 1.0)):
        errors.append("Observation x must be within [0, 1]")

    return errors
