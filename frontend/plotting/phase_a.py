"""Plotting adapter for Phase A: Forward 1D Poisson."""

from __future__ import annotations

from typing import Any

import numpy as np
from matplotlib.figure import Figure

from frontend.plotting.common import apply_pub_style


def plot_posterior_overlay(fig: Figure, result: dict[str, Any]) -> None:
    """TL: Multi-method posterior overlay with credible bands."""
    ax = fig.add_subplot(111)
    apply_pub_style(ax)

    # If batch result, iterate sub-results
    results_list = result.get("results", [result])
    colors = ["#185fa5", "#ba7517", "#3b6d11", "#a32d2d"]

    for i, res in enumerate(results_list):
        x = res.get("x_grid")
        mean = res.get("phi_mean")
        std = res.get("phi_std")
        if x is None or mean is None:
            continue
        label = (
            res.get("summary", {}).get("method")
            or res.get("method")
            or f"Method {i}"
        )[:20]
        c = colors[i % len(colors)]
        if isinstance(x, np.ndarray) and isinstance(mean, np.ndarray):
            ax.plot(x, mean, color=c, linewidth=1.5, label=label)
            if std is not None and isinstance(std, np.ndarray) and np.any(std > 0):
                ax.fill_between(x, mean - 1.96 * std, mean + 1.96 * std,
                                alpha=0.15, color=c)

    # Ground truth
    x = results_list[0].get("x_grid") if results_list else None
    truth = results_list[0].get("phi_truth") if results_list else None
    if x is not None and truth is not None:
        ax.plot(x, truth, "--", color="#888780", linewidth=1.5, label="ground truth")

    # Observations
    x_obs = results_list[0].get("x_obs") if results_list else None
    y_obs = results_list[0].get("y_obs") if results_list else None
    if x_obs is not None and y_obs is not None:
        ax.scatter(x_obs, y_obs, c="#e24b4a", s=12, zorder=5, label="observations")

    ax.legend(fontsize=7, loc="upper right")
    ax.set_xlabel("x", fontsize=8)
    ax.set_ylabel("φ(x)", fontsize=8)


def plot_hamiltonian_trace(fig: Figure, result: dict[str, Any]) -> None:
    """TR: Hamiltonian trace with burn-in shading."""
    ax = fig.add_subplot(111)
    apply_pub_style(ax)

    res = result.get("results", [result])[0] if "results" in result else result
    traces = res.get("traces", {})
    ham = traces.get("hamiltonian")
    config = res.get("config", result.get("config", {}))
    burn_in = config.get("burn_in", 0)

    if ham is not None:
        ham = np.asarray(ham, dtype=float)
        steps = np.arange(len(ham))

        # Clip y-axis to the 99th percentile of post-burn-in values so
        # overflow spikes don't flatten the entire trace to zero.
        post_burn = ham[burn_in:] if burn_in > 0 and burn_in < len(ham) else ham
        finite = post_burn[np.isfinite(post_burn)]
        if len(finite) > 0:
            y_top = float(np.percentile(finite, 99)) * 1.5
            y_bot = float(np.percentile(finite, 1)) - abs(float(np.percentile(finite, 1))) * 0.1
            ax.set_ylim(y_bot, y_top)

        # Raw trace
        ax.plot(steps, ham, linewidth=0.4, alpha=0.5, color="#185fa5")

        # Burn-in region: light red fill
        if burn_in > 0 and burn_in < len(ham):
            ax.axvspan(0, burn_in, alpha=0.12, color="#e24b4a", zorder=0)

            # Running mean (contrasting orange, thicker)
            post = ham[burn_in:]
            window = max(len(post) // 50, 10)
            if len(post) > window:
                cumsum = np.cumsum(np.nan_to_num(post, nan=0.0))
                running = (cumsum[window:] - cumsum[:-window]) / window
                ax.plot(np.arange(burn_in + window, burn_in + window + len(running)),
                        running, color="#ba7517", linewidth=2, label="running mean")
                ax.legend(fontsize=7, loc="upper right")

        ax.set_xlabel("Step", fontsize=8)
        ax.set_ylabel("H", fontsize=8)

        # Annotation
        chain = res.get("chain")
        if chain is not None and hasattr(chain, "shape") and chain.ndim == 2:
            try:
                from utils.diagnostics import lag1_autocorr, effective_sample_size
                ac = lag1_autocorr(chain[:, 0])
                ess = effective_sample_size(chain[:, 0])
                ax.text(0.98, 0.95,
                        f"lag-1 AC: {ac:.2f}\nESS: {ess:.0f}/{chain.shape[0]}",
                        transform=ax.transAxes, fontsize=7,
                        va="top", ha="right",
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
            except Exception:
                pass


def plot_preconditioner(fig: Figure, result: dict[str, Any]) -> None:
    """BL: Preconditioner mode weights bar chart (log scale)."""
    ax = fig.add_subplot(111)
    apply_pub_style(ax)

    res = result.get("results", [result])[0] if "results" in result else result
    config = res.get("config", result.get("config", {}))
    diagnostics = res.get("diagnostics", {})
    n_modes = config.get("n_modes", 12)
    max_cond = config.get("max_condition_number", 100)
    # Prefer the actual clipped condition number from the run diagnostics
    actual_cond = diagnostics.get("condition_number") or max_cond

    modes = np.arange(1, int(n_modes) + 1)
    weights = 1.0 / (modes * np.pi) ** 4
    weights /= weights.max()

    ax.bar(modes, weights, color="#185fa5", alpha=0.7, width=0.8)
    ax.set_yscale("log")
    if actual_cond and float(actual_cond) > 0:
        floor = 1.0 / float(actual_cond)
        ax.axhline(floor, color="#e24b4a", linewidth=0.8, linestyle="--",
                    label=f"floor (cond={int(actual_cond)})")
        ax.legend(fontsize=7)
    ax.set_xlabel("mode k", fontsize=8)
    ax.set_ylabel("weight (log)", fontsize=8)


def plot_coverage(fig: Figure, result: dict[str, Any]) -> None:
    """BR: Credible interval calibration curve with miscalibration shading."""
    ax = fig.add_subplot(111)
    apply_pub_style(ax)

    results_list = result.get("results", [result])
    colors = ["#185fa5", "#ba7517", "#3b6d11"]
    nominals = np.linspace(0.05, 0.99, 20)

    all_curves: list[np.ndarray] = []
    all_labels: list[str] = []

    for i, res in enumerate(results_list):
        samples = res.get("samples")
        x_obs = res.get("x_obs")
        y_obs = res.get("y_obs")
        if samples is None or x_obs is None or y_obs is None:
            continue
        if not hasattr(samples, "shape") or samples.ndim != 2:
            continue

        label = res.get("summary", {}).get("method") or res.get("method") or f"Method {i}"
        try:
            from core.parameterizations import SineBasisField
            n_modes = samples.shape[1]
            basis = SineBasisField(n_modes)
            obs_matrix = basis.design_matrix(x_obs)
            phi_at_obs = samples @ obs_matrix.T  # (n_samples, n_obs)

            empirical = []
            for nom in nominals:
                lo = np.percentile(phi_at_obs, (1 - nom) / 2 * 100, axis=0)
                hi = np.percentile(phi_at_obs, (1 + nom) / 2 * 100, axis=0)
                frac = np.mean((y_obs >= lo) & (y_obs <= hi))
                empirical.append(frac)

            emp_arr = np.array(empirical) * 100
            ax.plot(nominals * 100, emp_arr,
                    color=colors[i % len(colors)], linewidth=1.5, label=label)
            all_curves.append(emp_arr)
            all_labels.append(label)
        except Exception:
            continue

    # Shade between the two method curves to highlight miscalibration gap
    if len(all_curves) >= 2:
        ax.fill_between(
            nominals * 100, all_curves[0], all_curves[1],
            alpha=0.10, color="#854f0b", label="miscalibration gap",
        )

    # Ideal diagonal
    ax.plot([0, 100], [0, 100], "--", color="#888780", linewidth=1, label="ideal")

    # Annotate the 95% nominal point
    ax.axvline(95, color="#888780", linewidth=0.8, linestyle=":")
    for i, (curve, label) in enumerate(zip(all_curves, all_labels)):
        # Interpolate empirical coverage at 95% nominal
        emp_at_95 = float(np.interp(95, nominals * 100, curve))
        ax.plot(95, emp_at_95, "o", color=colors[i % len(colors)], markersize=5, zorder=5)
        offset = 8 if i == 0 else -12
        ax.annotate(
            f"{emp_at_95:.0f}%",
            xy=(95, emp_at_95), xytext=(95 + offset, emp_at_95 + 4),
            fontsize=7, color=colors[i % len(colors)],
            arrowprops=dict(arrowstyle="-", color=colors[i % len(colors)], lw=0.5),
        )

    ax.set_xlabel("nominal coverage (%)", fontsize=8)
    ax.set_ylabel("empirical coverage (%)", fontsize=8)
    ax.legend(fontsize=6, loc="upper left")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 105)


PLOT_FUNCTIONS = [
    plot_posterior_overlay,
    plot_hamiltonian_trace,
    plot_preconditioner,
    plot_coverage,
]
