"""Plot helpers for Phase A outputs."""

from __future__ import annotations

import matplotlib
import numpy as np

# Always use non-interactive backend to keep plotting safe in worker threads.
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt


def plot_field_summary(
    x_grid: np.ndarray,
    truth: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    x_obs: np.ndarray,
    y_obs: np.ndarray,
    out_path: str,
) -> None:
    plt.figure(figsize=(9, 5))
    plt.plot(x_grid, truth, label="ground truth", linewidth=2.0)
    plt.plot(x_grid, mean, label="posterior mean", linewidth=2.0)
    plt.fill_between(x_grid, mean - 2.0 * std, mean + 2.0 * std, alpha=0.25, label="mean ± 2 std")
    plt.scatter(x_obs, y_obs, s=24, marker="x", label="measurements")
    plt.xlabel("x")
    plt.ylabel("phi(x)")
    plt.title("PIFT Forward Poisson: posterior field summary")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_diagnostics(ham: np.ndarray, out_path: str) -> None:
    plt.figure(figsize=(9, 4))
    plt.plot(ham)
    plt.xlabel("SGLD step")
    plt.ylabel("Hamiltonian proxy")
    plt.title("SGLD Diagnostic: Hamiltonian trace")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()
