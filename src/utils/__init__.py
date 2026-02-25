"""Shared utility modules (diagnostics, helpers, plotting, validation)."""

from .diagnostics import credible_interval, lag1_autocorr
from .diagnostics_runtime import RuntimeGuard, RuntimeGuardConfig
from .helpers import (
    Domain1D,
    NormalDistribution,
    ObservationModel,
    PoissonForwardProblem,
    SGLDConfig,
    denormalize_from_minus_one_plus_one,
    normalize_to_minus_one_plus_one,
    normalize_to_unit_interval,
    standardize,
    to_normal,
    unstandardize,
)
from .validators import ValidationError, validate_observations, validate_phase_a_config

__all__ = [
    "RuntimeGuard",
    "RuntimeGuardConfig",
    "ValidationError",
    "validate_observations",
    "validate_phase_a_config",
    "credible_interval",
    "lag1_autocorr",
    "NormalDistribution",
    "Domain1D",
    "ObservationModel",
    "PoissonForwardProblem",
    "SGLDConfig",
    "normalize_to_unit_interval",
    "normalize_to_minus_one_plus_one",
    "denormalize_from_minus_one_plus_one",
    "standardize",
    "unstandardize",
    "to_normal",
]
