"""Runtime stability diagnostics and fail-fast policies."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(slots=True)
class RuntimeGuardConfig:
    max_abs_hamiltonian: float = 1e300
    grad_window: int = 8
    grad_ratio_limit: float = 1e8


@dataclass(slots=True)
class RuntimeGuardState:
    first_bad_step: int | None = None
    max_abs_hamiltonian: float = 0.0
    nan_detected: bool = False
    inf_detected: bool = False
    divergence_detected: bool = False


class RuntimeGuard:
    """Detect unstable numerical behavior during iterative sampling."""

    def __init__(self, cfg: RuntimeGuardConfig | None = None) -> None:
        self.cfg = cfg or RuntimeGuardConfig()
        self.state = RuntimeGuardState()
        self._grad_norm_hist: deque[float] = deque(maxlen=self.cfg.grad_window)

    def check(self, step: int, theta: np.ndarray, grad: np.ndarray, metrics: dict[str, Any]) -> tuple[bool, str | None, str | None]:
        h = float(metrics.get("hamiltonian", 0.0))
        self.state.max_abs_hamiltonian = max(self.state.max_abs_hamiltonian, abs(h))

        grad_np = np.asarray(grad, dtype=float)
        theta_np = np.asarray(theta, dtype=float)

        if not np.all(np.isfinite(grad_np)) or not np.all(np.isfinite(theta_np)) or not np.isfinite(h):
            self.state.first_bad_step = step if self.state.first_bad_step is None else self.state.first_bad_step
            self.state.nan_detected = bool(np.isnan(grad_np).any() or np.isnan(theta_np).any() or np.isnan(h))
            self.state.inf_detected = bool(np.isinf(grad_np).any() or np.isinf(theta_np).any() or np.isinf(h))
            return True, "non_finite", "Detected NaN/Inf in gradient, theta, or Hamiltonian"

        if abs(h) > self.cfg.max_abs_hamiltonian:
            self.state.first_bad_step = step if self.state.first_bad_step is None else self.state.first_bad_step
            return True, "hamiltonian_overflow", (
                f"|Hamiltonian|={abs(h):.3e} exceeded threshold {self.cfg.max_abs_hamiltonian:.3e}"
            )

        gnorm = float(np.linalg.norm(grad_np))
        self._grad_norm_hist.append(gnorm)
        if len(self._grad_norm_hist) == self.cfg.grad_window:
            hist = np.asarray(self._grad_norm_hist, dtype=float)
            baseline = np.median(hist[:-1]) if hist.size > 1 else hist[0]
            baseline = max(baseline, 1e-12)
            if hist[-1] > self.cfg.grad_ratio_limit * baseline:
                self.state.first_bad_step = step if self.state.first_bad_step is None else self.state.first_bad_step
                self.state.divergence_detected = True
                return True, "gradient_divergence", (
                    f"Gradient norm diverged: {hist[-1]:.3e} vs baseline {baseline:.3e}"
                )

        return False, None, None

    def diagnostics_payload(self) -> dict[str, Any]:
        flags = {
            "nan_detected": self.state.nan_detected,
            "inf_detected": self.state.inf_detected,
            "divergence_detected": self.state.divergence_detected,
            "hamiltonian_overflow": self.state.max_abs_hamiltonian > self.cfg.max_abs_hamiltonian,
        }
        return {
            "stability_flags": flags,
            "first_bad_step": self.state.first_bad_step,
            "max_abs_hamiltonian": self.state.max_abs_hamiltonian,
            "nan_detected": self.state.nan_detected,
            "suggested_actions": suggest_actions(flags),
        }


def suggest_actions(flags: dict[str, bool]) -> list[str]:
    actions: list[str] = []
    if flags.get("nan_detected") or flags.get("inf_detected"):
        actions.append("Reduce step_size0 (e.g., 10x smaller)")
        actions.append("Increase n_quad to reduce stochastic gradient variance")
        actions.append("Lower beta to reduce physics stiffness")
    if flags.get("divergence_detected"):
        actions.append("Reduce step_size0 and consider higher decay")
        actions.append("Increase noise_std floor for noisy observations")
    if flags.get("hamiltonian_overflow"):
        actions.append("Lower beta and step_size0")
        actions.append("Check observation scale and normalize inputs")
    if not actions:
        actions.append("No critical issues detected")
    return actions
