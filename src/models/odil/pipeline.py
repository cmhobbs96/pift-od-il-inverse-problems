"""ODIL baseline pipeline entry points."""

from pipelines.phase_a import run_phase_a_odil

run = run_phase_a_odil

__all__ = ["run", "run_phase_a_odil"]
