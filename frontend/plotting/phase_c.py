"""Plotting adapters for Phase C: Inverse problems (Examples 3a, 3b)."""

from __future__ import annotations

from typing import Any

import numpy as np
from matplotlib.figure import Figure

from frontend.plotting.common import apply_pub_style


def plot_lambda_trace(fig: Figure, result: dict[str, Any]) -> None:
    """TL: Outer SGLD trace of log-parameters."""
    ax = fig.add_subplot(111)
    apply_pub_style(ax)

    chain = result.get("lambda_chain")
    if chain is None:
        return

    config = result.get("config", {})
    burn_in_frac = config.get("burn_in_frac", 0.2)
    burn_in = int(chain.shape[0] * burn_in_frac)

    ax.plot(chain[:, 0], linewidth=0.5, alpha=0.7, color="#185fa5", label="log(D)")
    if chain.shape[1] > 1:
        ax.plot(chain[:, 1], linewidth=0.5, alpha=0.7, color="#ba7517", label="log(κ)")

    # True values
    D_true = config.get("D_true")
    kappa_true = config.get("kappa_true")
    if D_true:
        ax.axhline(np.log(D_true), color="#185fa5", linestyle="--", linewidth=0.8, alpha=0.6)
    if kappa_true:
        ax.axhline(np.log(kappa_true), color="#ba7517", linestyle="--", linewidth=0.8, alpha=0.6)

    if burn_in > 0:
        ax.axvline(burn_in, color="gray", linestyle=":", linewidth=0.8)

    ax.set_xlabel("Outer iteration", fontsize=8)
    ax.set_ylabel("Log-parameter", fontsize=8)
    ax.legend(fontsize=7)


def plot_joint_posterior(fig: Figure, result: dict[str, Any]) -> None:
    """TR: Joint D-κ scatter with KDE contours."""
    ax = fig.add_subplot(111)
    apply_pub_style(ax)

    chain = result.get("lambda_chain")
    config = result.get("config", {})
    if chain is None or chain.shape[1] < 2:
        return

    burn_in_frac = config.get("burn_in_frac", 0.2)
    burn_in = int(chain.shape[0] * burn_in_frac)
    post = chain[burn_in:]
    D_samples = np.exp(post[:, 0])
    kappa_samples = np.exp(post[:, 1])

    ax.scatter(D_samples, kappa_samples, s=2, alpha=0.3, color="#185fa5")

    # KDE contours
    try:
        from scipy.stats import gaussian_kde
        xy = np.vstack([D_samples, kappa_samples])
        kde = gaussian_kde(xy)
        xg = np.linspace(D_samples.min(), D_samples.max(), 80)
        yg = np.linspace(kappa_samples.min(), kappa_samples.max(), 80)
        X, Y = np.meshgrid(xg, yg)
        Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)
        ax.contour(X, Y, Z, levels=5, colors="#185fa5", linewidths=0.8, alpha=0.7)
    except Exception:
        pass

    # True values
    D_true = config.get("D_true")
    kappa_true = config.get("kappa_true")
    if D_true and kappa_true:
        ax.axvline(D_true, color="#e24b4a", linestyle="--", linewidth=0.8)
        ax.axhline(kappa_true, color="#e24b4a", linestyle="--", linewidth=0.8)
        ax.plot(D_true, kappa_true, "x", color="#e24b4a", markersize=8, markeredgewidth=2)

    ax.set_xlabel("D", fontsize=8)
    ax.set_ylabel("κ", fontsize=8)


def plot_prior_predictive(fig: Figure, result: dict[str, Any]) -> None:
    """BL: Prior predictive field ± credible band."""
    ax = fig.add_subplot(111)
    apply_pub_style(ax)

    x_grid = result.get("x_grid")
    phi_truth = result.get("phi_truth")
    prior_samples = result.get("phi_prior_samples")

    if x_grid is not None and phi_truth is not None:
        ax.plot(x_grid, phi_truth, "--", color="#888780", linewidth=1.2, label="truth")

    if prior_samples is not None and x_grid is not None and prior_samples.ndim == 2:
        mean = np.mean(prior_samples, axis=0)
        std = np.std(prior_samples, axis=0)
        ax.plot(x_grid[:len(mean)], mean, color="#3b6d11", linewidth=1.2, label="prior mean")
        ax.fill_between(x_grid[:len(mean)], mean - 2 * std, mean + 2 * std,
                        alpha=0.15, color="#3b6d11")

    ax.set_xlabel("x", fontsize=8)
    ax.set_ylabel("φ(x)", fontsize=8)
    ax.legend(fontsize=7)
    ax.set_title("Prior predictive", fontsize=9)


def plot_posterior_predictive(fig: Figure, result: dict[str, Any]) -> None:
    """BR: Posterior predictive field ± credible band with observations."""
    ax = fig.add_subplot(111)
    apply_pub_style(ax)

    x_grid = result.get("x_grid")
    phi_truth = result.get("phi_truth")
    posterior_samples = result.get("phi_posterior_samples")

    if x_grid is not None and phi_truth is not None:
        ax.plot(x_grid, phi_truth, "--", color="#888780", linewidth=1.2, label="truth")

    if posterior_samples is not None and x_grid is not None and posterior_samples.ndim == 2:
        mean = np.mean(posterior_samples, axis=0)
        std = np.std(posterior_samples, axis=0)
        ax.plot(x_grid[:len(mean)], mean, color="#185fa5", linewidth=1.2, label="posterior mean")
        ax.fill_between(x_grid[:len(mean)], mean - 2 * std, mean + 2 * std,
                        alpha=0.15, color="#185fa5")

    # Observations
    x_obs = result.get("x_obs")
    y_obs = result.get("y_obs")
    if x_obs is not None and y_obs is not None:
        ax.scatter(x_obs, y_obs, c="#e24b4a", s=12, zorder=5, label="observations")

    ax.set_xlabel("x", fontsize=8)
    ax.set_ylabel("φ(x)", fontsize=8)
    ax.legend(fontsize=7)
    ax.set_title("Posterior predictive", fontsize=9)


# For phase_c_source, replace prior_predictive with source reconstruction
def plot_source_reconstruction(fig: Figure, result: dict[str, Any]) -> None:
    """BL (source variant): Recovered source f(x) ± credible band."""
    ax = fig.add_subplot(111)
    apply_pub_style(ax)

    x_grid = result.get("x_grid")
    f_mean = result.get("f_mean")
    f_std = result.get("f_std")
    f_truth = result.get("f_truth")

    if x_grid is not None and f_truth is not None:
        ax.plot(x_grid[:len(f_truth)], f_truth, "--", color="#888780",
                linewidth=1.2, label="true f(x)")
    if f_mean is not None and x_grid is not None:
        ax.plot(x_grid[:len(f_mean)], f_mean, color="#3b6d11",
                linewidth=1.2, label="recovered mean")
        if f_std is not None:
            ax.fill_between(x_grid[:len(f_mean)],
                            f_mean - 2 * f_std, f_mean + 2 * f_std,
                            alpha=0.15, color="#3b6d11")

    ax.set_xlabel("x", fontsize=8)
    ax.set_ylabel("f(x)", fontsize=8)
    ax.legend(fontsize=7)
    ax.set_title("Source reconstruction", fontsize=9)


PLOT_FUNCTIONS = [
    plot_lambda_trace,
    plot_joint_posterior,
    plot_prior_predictive,
    plot_posterior_predictive,
]

PLOT_FUNCTIONS_SOURCE = [
    plot_lambda_trace,
    plot_joint_posterior,
    plot_source_reconstruction,
    plot_posterior_predictive,
]
