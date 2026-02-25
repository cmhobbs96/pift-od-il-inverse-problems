"""Bayesian PINNs pipeline entry points."""

from pipelines.phase_a import run_phase_a_bayesian_pinn

run = run_phase_a_bayesian_pinn

__all__ = ["run", "run_phase_a_bayesian_pinn"]
