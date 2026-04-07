"""Plotting adapters for Phase B: Beta sweep (Ex. 1) and Model-form (Ex. 2)."""

from __future__ import annotations

from typing import Any

import numpy as np
from matplotlib.figure import Figure

from frontend.plotting.common import apply_pub_style


# ---------------------------------------------------------------------------
# Phase B sweep (Example 1)
# ---------------------------------------------------------------------------

def plot_beta_posteriors(fig: Figure, result: dict[str, Any]) -> None:
    """TL: Multi-panel field posteriors for each β."""
    beta_results = result.get("beta_results", [])
    x_grid = result.get("x_grid")
    phi_truth = result.get("phi_truth")

    n = len(beta_results)
    if n == 0:
        return

    cols = min(n, 2)
    rows = (n + cols - 1) // cols
    for i, br in enumerate(beta_results):
        ax = fig.add_subplot(rows, cols, i + 1)
        apply_pub_style(ax)
        beta = br.get("beta", "?")
        mean = br.get("phi_mean")
        std = br.get("phi_std")
        if x_grid is not None and mean is not None:
            ax.plot(x_grid, mean, color="#185fa5", linewidth=1.2)
            if std is not None:
                ax.fill_between(x_grid, mean - 2 * std, mean + 2 * std,
                                alpha=0.2, color="#185fa5")
        if x_grid is not None and phi_truth is not None:
            ax.plot(x_grid, phi_truth, "--", color="#888780", linewidth=1)
        ax.set_title(f"β = {beta}", fontsize=8)
        ax.tick_params(labelsize=6)


def plot_variance_scaling(fig: Figure, result: dict[str, Any]) -> None:
    """TR: Variance vs β on log-log axes."""
    ax = fig.add_subplot(111)
    apply_pub_style(ax)

    vs = result.get("variance_scaling", {})
    betas = vs.get("betas", [])
    variances = vs.get("variances", [])
    if not betas or not variances:
        return

    betas_arr = np.array(betas, dtype=float)
    var_arr = np.array(variances, dtype=float)
    ax.loglog(betas_arr, var_arr, "o-", color="#185fa5", linewidth=1.5, label="measured")
    # 1/β reference line
    ref = var_arr[0] * betas_arr[0] / betas_arr
    ax.loglog(betas_arr, ref, "--", color="#888780", linewidth=1, label="∝ 1/β")
    ax.set_xlabel("β", fontsize=8)
    ax.set_ylabel("Var(φ) at x=0.5", fontsize=8)
    ax.legend(fontsize=7)


def plot_beta_traces(fig: Figure, result: dict[str, Any]) -> None:
    """BL: Hamiltonian traces per β."""
    beta_results = result.get("beta_results", [])
    if not beta_results:
        return

    ax = fig.add_subplot(111)
    apply_pub_style(ax)
    colors = ["#185fa5", "#ba7517", "#3b6d11", "#a32d2d"]
    for i, br in enumerate(beta_results):
        traces = br.get("traces", {})
        ham = traces.get("hamiltonian")
        if ham is not None:
            ax.plot(ham[::10], linewidth=0.5, alpha=0.7,
                    color=colors[i % len(colors)],
                    label=f"β={br.get('beta', '?')}")
    ax.set_xlabel("Step (÷10)", fontsize=8)
    ax.set_ylabel("H", fontsize=8)
    ax.legend(fontsize=7)


def plot_ess_summary(fig: Figure, result: dict[str, Any]) -> None:
    """BR: ESS summary per β."""
    beta_results = result.get("beta_results", [])
    if not beta_results:
        return

    ax = fig.add_subplot(111)
    apply_pub_style(ax)

    betas = []
    n_samples_list = []
    for br in beta_results:
        betas.append(br.get("beta", 0))
        n_samples_list.append(br.get("n_samples", 0))

    ax.bar(range(len(betas)), n_samples_list, color="#185fa5", alpha=0.7)
    ax.set_xticks(range(len(betas)))
    ax.set_xticklabels([str(b) for b in betas], fontsize=7)
    ax.set_xlabel("β", fontsize=8)
    ax.set_ylabel("Samples", fontsize=8)


# ---------------------------------------------------------------------------
# Phase B model-form (Example 2) — uses same module, different PLOT_FUNCTIONS
# ---------------------------------------------------------------------------

def plot_model_form_source(fig: Figure, result: dict[str, Any]) -> None:
    """TL: Violin plots — β vs γ for source term error."""
    _plot_violins(fig, result, "source_error", "(a) Source term error")


def plot_model_form_energy(fig: Figure, result: dict[str, Any]) -> None:
    """TR: Violin plots — β vs γ for energy error."""
    _plot_violins(fig, result, "energy_error", "(b) Energy functional error")


def _plot_violins(fig: Figure, result: dict[str, Any], exp_key: str, title: str) -> None:
    experiments = result.get("experiments", {})
    data = experiments.get(exp_key, [])
    if not data:
        return

    ax = fig.add_subplot(111)
    apply_pub_style(ax)

    gammas = [r["gamma"] for r in data]
    samples_list = [np.asarray(r["beta_samples"]) for r in data if "beta_samples" in r]
    if not samples_list:
        return

    parts = ax.violinplot(samples_list, positions=gammas, widths=0.12,
                          showmedians=True, showextrema=False)
    for pc in parts["bodies"]:
        pc.set_facecolor("#185fa5")
        pc.set_alpha(0.45)
    parts["cmedians"].set_color("#185fa5")
    parts["cmedians"].set_linewidth(2)

    ax.set_yscale("log")
    ax.set_xlabel("γ (model correctness)", fontsize=8)
    ax.set_ylabel("β", fontsize=8)
    ax.set_title(title, fontsize=9)


def plot_nested_trace(fig: Figure, result: dict[str, Any]) -> None:
    """BL: Nested SGLD outer chain trace (log β)."""
    experiments = result.get("experiments", {})
    ax = fig.add_subplot(111)
    apply_pub_style(ax)

    colors = ["#185fa5", "#ba7517"]
    for i, (name, data) in enumerate(experiments.items()):
        if data and "log_beta_chain" in data[-1]:
            chain = np.asarray(data[-1]["log_beta_chain"])
            ax.plot(chain, linewidth=0.5, alpha=0.7,
                    color=colors[i % len(colors)], label=f"{name} (γ=1)")
    ax.set_xlabel("Outer step", fontsize=8)
    ax.set_ylabel("log(β)", fontsize=8)
    ax.legend(fontsize=7)


def plot_model_summary(fig: Figure, result: dict[str, Any]) -> None:
    """BR: Summary text."""
    ax = fig.add_subplot(111)
    ax.axis("off")
    experiments = result.get("experiments", {})
    lines = ["Model-form uncertainty summary\n"]
    for name, data in experiments.items():
        lines.append(f"{name}: {len(data)} γ values")
        if data:
            lines.append(f"  β median range: {data[0].get('beta_median', '?'):.0f} – "
                         f"{data[-1].get('beta_median', '?'):.0f}")
    ax.text(0.05, 0.95, "\n".join(lines), transform=ax.transAxes,
            fontsize=8, va="top", family="monospace")


# Exported: the plot grid picks PLOT_FUNCTIONS based on example_key.
# For phase_b_sweep vs phase_b_model, the app.py will set the right list.
# We export both sets; the plot_grid dispatches based on example_key.

PLOT_FUNCTIONS_SWEEP = [
    plot_beta_posteriors,
    plot_variance_scaling,
    plot_beta_traces,
    plot_ess_summary,
]

PLOT_FUNCTIONS_MODEL = [
    plot_model_form_source,
    plot_model_form_energy,
    plot_nested_trace,
    plot_model_summary,
]

# Default — will be overridden by plot_grid based on example key
PLOT_FUNCTIONS = PLOT_FUNCTIONS_SWEEP
