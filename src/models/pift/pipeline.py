"""PIFT pipeline entry points."""

from pipelines.phase_a import run_phase_a_forward_poisson

run = run_phase_a_forward_poisson

__all__ = ["run", "run_phase_a_forward_poisson"]
