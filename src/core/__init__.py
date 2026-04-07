"""Shared numerical core used by model pipelines."""

from .energies import poisson_residual_energy, variational_heat_energy
from .energies_2d import allen_cahn_energy, allen_cahn_ground_truth, allen_cahn_source_term
from .energies_nonlinear import nonlinear_energy, nonlinear_energy_misspecified
from .hmcecs import run_nuts_inference
from .likelihoods import gaussian_nll
from .nested_sgld import NestedSGLDResult, nested_sgld
from .parameterizations import BoundaryFourierField, SineBasisField
from .parameterizations_2d import FourierBasis2D
from .reference_solver import solve_poisson_dirichlet_fd
from .sgld import sgld_sample

__all__ = [
    "poisson_residual_energy",
    "variational_heat_energy",
    "nonlinear_energy",
    "nonlinear_energy_misspecified",
    "allen_cahn_energy",
    "allen_cahn_ground_truth",
    "allen_cahn_source_term",
    "gaussian_nll",
    "NestedSGLDResult",
    "nested_sgld",
    "run_nuts_inference",
    "BoundaryFourierField",
    "SineBasisField",
    "FourierBasis2D",
    "solve_poisson_dirichlet_fd",
    "sgld_sample",
]
