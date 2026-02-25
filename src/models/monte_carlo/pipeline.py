"""Classical Monte Carlo pipeline entry points."""

from pipelines.phase_a import run_phase_a_monte_carlo

run = run_phase_a_monte_carlo

__all__ = ["run", "run_phase_a_monte_carlo"]
