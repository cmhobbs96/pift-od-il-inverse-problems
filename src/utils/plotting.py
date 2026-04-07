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


# ---------------------------------------------------------------------------
# Extended diagnostics plots
# ---------------------------------------------------------------------------


def plot_trace_per_coefficient(
    chain: np.ndarray,
    burn_in: int = 0,
    max_coeffs: int = 6,
    out_path: str | None = None,
) -> "plt.Figure":
    """Trace plots for individual basis coefficients.

    Shows the first ``max_coeffs`` coefficients in a vertical subplot grid,
    with a vertical line at ``burn_in``.
    """
    n_steps, dim = chain.shape
    n_show = min(dim, max_coeffs)
    fig, axes = plt.subplots(n_show, 1, figsize=(10, 2.2 * n_show), sharex=True)
    if n_show == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        ax.plot(chain[:, i], linewidth=0.4)
        if burn_in > 0:
            ax.axvline(burn_in, color="red", linestyle="--", linewidth=0.8, label="burn-in")
        ax.set_ylabel(f"k={i + 1}")
    axes[-1].set_xlabel("SGLD step")
    axes[0].set_title("Per-coefficient trace plots")
    if burn_in > 0:
        axes[0].legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=160)
    return fig


def plot_autocorrelation(
    acf_dict: dict[str, np.ndarray],
    out_path: str | None = None,
) -> "plt.Figure":
    """Overlay ACF curves for several series."""
    fig, ax = plt.subplots(figsize=(8, 4))
    for label, acf in acf_dict.items():
        ax.plot(acf, label=label, linewidth=1.2)
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.set_xlabel("Lag")
    ax.set_ylabel("Autocorrelation")
    ax.set_title("Autocorrelation function")
    ax.legend(fontsize=8)
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=160)
    return fig


def plot_ess_bar(
    ess_per_dim: np.ndarray,
    labels: list[str] | None = None,
    out_path: str | None = None,
) -> "plt.Figure":
    """Bar chart of ESS per coefficient."""
    n = len(ess_per_dim)
    if labels is None:
        labels = [f"k={i + 1}" for i in range(n)]
    fig, ax = plt.subplots(figsize=(max(6, 0.6 * n), 4))
    ax.bar(range(n), ess_per_dim, tick_label=labels)
    ax.set_ylabel("Effective sample size")
    ax.set_title("ESS per basis coefficient")
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=160)
    return fig


def plot_sensitivity_sweep(
    sweep_results: dict,
    param_name: str,
    x_grid: np.ndarray,
    phi_truth: np.ndarray | None = None,
    out_path: str | None = None,
) -> "plt.Figure":
    """Overlay posterior means and CI bands for different parameter values."""
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(sweep_results["param_values"])))
    for idx, val in enumerate(sweep_results["param_values"]):
        mean = sweep_results["phi_means"][idx]
        std = sweep_results["phi_stds"][idx]
        c = colors[idx]
        ax.plot(x_grid, mean, color=c, linewidth=1.5, label=f"{param_name}={val}")
        ax.fill_between(x_grid, mean - 2 * std, mean + 2 * std, color=c, alpha=0.12)
    if phi_truth is not None:
        ax.plot(x_grid, phi_truth, "k--", linewidth=1.5, label="ground truth")
    ax.set_xlabel("x")
    ax.set_ylabel("phi(x)")
    ax.set_title(f"Sensitivity to {param_name}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=160)
    return fig


def plot_rhat_bar(
    rhat: np.ndarray,
    labels: list[str] | None = None,
    out_path: str | None = None,
) -> "plt.Figure":
    """Bar chart of R-hat per coefficient with a 1.1 threshold line."""
    n = len(rhat)
    if labels is None:
        labels = [f"k={i + 1}" for i in range(n)]
    fig, ax = plt.subplots(figsize=(max(6, 0.6 * n), 4))
    ax.bar(range(n), rhat, tick_label=labels)
    ax.axhline(1.1, color="red", linestyle="--", linewidth=1.0, label="R-hat = 1.1")
    ax.set_ylabel("R-hat")
    ax.set_title("Gelman-Rubin R-hat per coefficient")
    ax.legend(fontsize=8)
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=160)
    return fig


def plot_beta_sweep_panels(
    x_grid: np.ndarray,
    phi_truth: np.ndarray,
    beta_results: list[dict],
    out_path: str | None = None,
) -> "plt.Figure":
    """2x2 multi-panel figure showing posterior collapse as beta increases."""
    n = len(beta_results)
    ncols = min(n, 2)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)

    # Shared y-limits across panels
    all_upper = [r["phi_mean"] + 2 * r["phi_std"] for r in beta_results]
    all_lower = [r["phi_mean"] - 2 * r["phi_std"] for r in beta_results]
    ymin = min(np.min(a) for a in all_lower + [phi_truth])
    ymax = max(np.max(a) for a in all_upper + [phi_truth])
    margin = 0.05 * (ymax - ymin)

    for idx, res in enumerate(beta_results):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]
        mean = res["phi_mean"]
        std = res["phi_std"]
        ax.plot(x_grid, phi_truth, "k--", linewidth=1.5, label="ground truth")
        ax.plot(x_grid, mean, linewidth=1.5, label="posterior mean")
        ax.fill_between(
            x_grid, mean - 2 * std, mean + 2 * std, alpha=0.3, label="mean +/- 2 std",
        )
        ax.set_title(f"$\\beta = {res['beta']:.0f}$")
        ax.set_xlabel("x")
        ax.set_ylabel("$\\phi(x)$")
        ax.set_ylim(ymin - margin, ymax + margin)
        if idx == 0:
            ax.legend(fontsize=7)

    # Hide unused axes
    for idx in range(n, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    fig.suptitle("Beta sensitivity: posterior variance collapse", fontsize=13)
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=160)
    return fig


def plot_variance_scaling(
    betas: list[float],
    variances: list[float],
    out_path: str | None = None,
) -> "plt.Figure":
    """Log-log plot of posterior variance vs beta with 1/beta reference."""
    betas = np.asarray(betas, dtype=float)
    variances = np.asarray(variances, dtype=float)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.loglog(betas, variances, "o-", linewidth=1.5, markersize=6, label="measured var")
    # 1/beta reference line anchored at first point
    ref = variances[0] * betas[0] / betas
    ax.loglog(betas, ref, "k--", linewidth=1.0, alpha=0.6, label="$\\propto 1/\\beta$")
    ax.set_xlabel("$\\beta$")
    ax.set_ylabel("Posterior variance at $x=0.5$")
    ax.set_title("Variance scaling (Klein-Gordon prediction)")
    ax.legend()
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=160)
    return fig


def plot_held_out_validation(
    x_grid: np.ndarray,
    phi_mean: np.ndarray,
    phi_std: np.ndarray,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    pred_mean_test: np.ndarray,
    pred_low_test: np.ndarray,
    pred_high_test: np.ndarray,
    phi_truth: np.ndarray | None = None,
    out_path: str | None = None,
) -> "plt.Figure":
    """Field plot with held-out points, coloured by inside/outside CI."""
    fig, ax = plt.subplots(figsize=(10, 5))
    if phi_truth is not None:
        ax.plot(x_grid, phi_truth, "k--", linewidth=1.5, label="ground truth")
    ax.plot(x_grid, phi_mean, linewidth=1.5, label="posterior mean")
    ax.fill_between(x_grid, phi_mean - 2 * phi_std, phi_mean + 2 * phi_std, alpha=0.2, label="mean +/- 2 std")
    ax.scatter(x_train, y_train, s=20, marker="x", color="gray", label="train obs")
    inside = (y_test >= pred_low_test) & (y_test <= pred_high_test)
    ax.scatter(x_test[inside], y_test[inside], s=60, marker="o", color="green", zorder=5, label="held-out (inside CI)")
    if (~inside).any():
        ax.scatter(x_test[~inside], y_test[~inside], s=60, marker="o", color="red", zorder=5, label="held-out (outside CI)")
    ax.set_xlabel("x")
    ax.set_ylabel("phi(x)")
    ax.set_title("Held-out posterior predictive check")
    ax.legend(fontsize=8)
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=160)
    return fig


def plot_model_form_beta_vs_gamma(
    experiments: dict[str, list[dict]],
    out_path: str | None = None,
) -> plt.Figure:
    """Reproduce Figure 2: violin plots of posterior β vs model correctness γ.

    Parameters
    ----------
    experiments : dict mapping experiment name to list of per-gamma result dicts.
        Each result dict must have keys: ``gamma``, ``beta_samples``
        (array of posterior β values).  Falls back to quantile-based
        rendering if ``beta_samples`` is not available.
    out_path : optional file path to save the figure.
    """
    exp_names = list(experiments.keys())
    n_panels = len(exp_names)
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5), squeeze=False)

    titles = {
        "source_error": "(a) Source term error",
        "energy_error": "(b) Energy functional error",
    }

    for ax, exp_name in zip(axes[0], exp_names):
        results = experiments[exp_name]
        gammas = [r["gamma"] for r in results]
        samples_list = [np.asarray(r["beta_samples"]) for r in results]

        parts = ax.violinplot(
            samples_list,
            positions=gammas,
            widths=0.12,
            showmedians=True,
            showextrema=False,
        )
        for pc in parts["bodies"]:
            pc.set_facecolor("C0")
            pc.set_alpha(0.45)
        parts["cmedians"].set_color("C0")
        parts["cmedians"].set_linewidth(2)

        ax.set_yscale("log")
        ax.set_xlabel(r"$\gamma$ (model correctness)")
        ax.set_ylabel(r"$\beta$ (inverse temperature)")
        ax.set_title(titles.get(exp_name, exp_name))
        ax.grid(True, alpha=0.3)

    fig.suptitle("Example 2: Sensitivity of β to model correctness", fontsize=13)
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=160)
    return fig


# ---------------------------------------------------------------------------
# Figure 3: Inverse parameter identification (Example 3a)
# ---------------------------------------------------------------------------


def plot_inverse_trace(
    ax: "plt.Axes",
    lambda_chain: np.ndarray,
    true_values: dict[str, float],
    burn_in: int,
    thin: int = 1,
) -> None:
    """Panel (a): SGLD trace of log-parameters."""
    steps = np.arange(lambda_chain.shape[0])
    chain_thin = lambda_chain[::thin]
    steps_thin = steps[::thin]

    ax.plot(steps_thin, chain_thin[:, 0], linewidth=0.5, alpha=0.8, label="$\\log(D)$")
    ax.plot(steps_thin, chain_thin[:, 1], linewidth=0.5, alpha=0.8, label="$\\log(\\kappa)$")
    ax.axhline(np.log(true_values["D"]), color="C0", linestyle="--", linewidth=1, alpha=0.6)
    ax.axhline(np.log(true_values["kappa"]), color="C1", linestyle="--", linewidth=1, alpha=0.6)
    if burn_in > 0:
        ax.axvline(burn_in, color="gray", linestyle=":", linewidth=1, label="burn-in")
    ax.set_xlabel("Outer iteration")
    ax.set_ylabel("Log-parameter value")
    ax.set_title("(a) SGLD trace")
    ax.legend(fontsize=7, loc="upper right")


def plot_joint_posterior(
    ax: "plt.Axes",
    lambda_chain: np.ndarray,
    true_values: dict[str, float],
    burn_in: int,
) -> None:
    """Panel (b): Joint posterior of D and κ (original space)."""
    post = lambda_chain[burn_in:]
    D_samples = np.exp(post[:, 0])
    kappa_samples = np.exp(post[:, 1])

    ax.scatter(D_samples, kappa_samples, s=2, alpha=0.15, c="C1", rasterized=True)

    # KDE contours if scipy available
    try:
        from scipy.stats import gaussian_kde
        xy = np.vstack([D_samples, kappa_samples])
        kde = gaussian_kde(xy)
        D_grid = np.linspace(D_samples.min(), D_samples.max(), 80)
        k_grid = np.linspace(kappa_samples.min(), kappa_samples.max(), 80)
        Dg, Kg = np.meshgrid(D_grid, k_grid)
        Z = kde(np.vstack([Dg.ravel(), Kg.ravel()])).reshape(Dg.shape)
        ax.contour(Dg, Kg, Z, levels=5, colors="C1", linewidths=0.8, alpha=0.7)
    except Exception:
        pass

    ax.axvline(true_values["D"], color="k", linestyle="--", linewidth=1, alpha=0.5)
    ax.axhline(true_values["kappa"], color="k", linestyle="--", linewidth=1, alpha=0.5)
    ax.plot(true_values["D"], true_values["kappa"], "k+", markersize=10, markeredgewidth=2)
    ax.set_xlabel("$D$")
    ax.set_ylabel("$\\kappa$")
    ax.set_title("(b) Joint posterior")


def plot_predictive_field(
    ax: "plt.Axes",
    x_grid: np.ndarray,
    phi_samples: np.ndarray | None,
    phi_truth: np.ndarray,
    x_obs: np.ndarray | None = None,
    y_obs: np.ndarray | None = None,
    title: str = "Predictive",
) -> None:
    """Panel (c) or (d): field samples with credible band."""
    ax.plot(x_grid, phi_truth, "k--", linewidth=1.5, label="ground truth")

    if phi_samples is not None and len(phi_samples) > 0:
        mean = np.mean(phi_samples, axis=0)
        std = np.std(phi_samples, axis=0)
        ax.plot(x_grid, mean, linewidth=1.5, label="predictive mean")
        ax.fill_between(x_grid, mean - 2 * std, mean + 2 * std, alpha=0.25, label="mean ± 2σ")
        # Plot a few individual samples
        n_show = min(5, len(phi_samples))
        for i in range(0, len(phi_samples), max(1, len(phi_samples) // n_show)):
            ax.plot(x_grid, phi_samples[i], linewidth=0.3, alpha=0.4, color="C1")

    if x_obs is not None and y_obs is not None:
        ax.scatter(x_obs, y_obs, s=15, marker="o", color="C1", zorder=5, alpha=0.7, label="observations")

    ax.set_xlabel("$x$")
    ax.set_ylabel("$\\phi(x)$")
    ax.set_title(title)
    ax.legend(fontsize=7)


def plot_figure3(
    lambda_chain: np.ndarray,
    phi_prior_samples: np.ndarray | None,
    phi_posterior_samples: np.ndarray | None,
    x_grid: np.ndarray,
    phi_truth: np.ndarray,
    x_obs: np.ndarray,
    y_obs: np.ndarray,
    true_params: dict[str, float],
    burn_in: int,
    out_path: str | None = None,
) -> "plt.Figure":
    """Composite 2×2 Figure 3 from Alberts & Bilionis Example 3a."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # (a) Trace plot — thin for readability
    thin = max(1, lambda_chain.shape[0] // 2000)
    plot_inverse_trace(axes[0, 0], lambda_chain, true_params, burn_in, thin=thin)

    # (b) Joint posterior contour
    plot_joint_posterior(axes[0, 1], lambda_chain, true_params, burn_in)

    # (c) Fitted prior predictive
    plot_predictive_field(
        axes[1, 0], x_grid, phi_prior_samples, phi_truth,
        title="(c) Prior predictive",
    )

    # (d) Posterior predictive
    plot_predictive_field(
        axes[1, 1], x_grid, phi_posterior_samples, phi_truth,
        x_obs=x_obs, y_obs=y_obs,
        title="(d) Posterior predictive",
    )

    fig.suptitle("Example 3a: Inverse parameter identification", fontsize=14)
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=160)
    return fig


# ---------------------------------------------------------------------------
# Figure 4: Source term identification (Example 3b)
# ---------------------------------------------------------------------------


def plot_source_posterior(
    ax: "plt.Axes",
    x_grid: np.ndarray,
    f_samples: np.ndarray,
    f_truth: np.ndarray,
) -> None:
    """Panel (b): Recovered source term f(x) with credible band."""
    f_mean = np.mean(f_samples, axis=0)
    f_std = np.std(f_samples, axis=0)
    ax.plot(x_grid, f_truth, "k--", linewidth=1.5, label="true $f(x)$")
    ax.plot(x_grid, f_mean, linewidth=1.5, label="posterior mean")
    ax.fill_between(x_grid, f_mean - 2 * f_std, f_mean + 2 * f_std, alpha=0.25, label="mean $\\pm$ 2$\\sigma$")
    n_show = min(5, len(f_samples))
    step = max(1, len(f_samples) // n_show)
    for i in range(0, len(f_samples), step):
        ax.plot(x_grid, f_samples[i], linewidth=0.3, alpha=0.4, color="C1")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$f(x)$")
    ax.set_title("(b) Source posterior")
    ax.legend(fontsize=7)


def plot_figure4(
    lambda_chain: np.ndarray,
    phi_prior_samples: np.ndarray | None,
    phi_posterior_samples: np.ndarray | None,
    f_samples: np.ndarray,
    x_grid: np.ndarray,
    phi_truth: np.ndarray,
    f_truth: np.ndarray,
    x_obs: np.ndarray,
    y_obs: np.ndarray,
    true_params: dict[str, float],
    burn_in: int,
    out_path: str | None = None,
) -> "plt.Figure":
    """Composite 2x2 Figure 4 from Alberts & Bilionis Example 3b."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # (a) Joint D-κ posterior
    plot_joint_posterior(axes[0, 0], lambda_chain, true_params, burn_in)
    axes[0, 0].set_title("(a) Joint posterior ($D$, $\\kappa$)")

    # (b) Source posterior
    plot_source_posterior(axes[0, 1], x_grid, f_samples, f_truth)

    # (c) Prior predictive
    plot_predictive_field(
        axes[1, 0], x_grid, phi_prior_samples, phi_truth,
        title="(c) Prior predictive",
    )

    # (d) Posterior predictive
    plot_predictive_field(
        axes[1, 1], x_grid, phi_posterior_samples, phi_truth,
        x_obs=x_obs, y_obs=y_obs,
        title="(d) Posterior predictive",
    )

    fig.suptitle("Example 3b: Source term identification", fontsize=14)
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=160)
    return fig


# ---------------------------------------------------------------------------
# Figures 5-8: Allen-Cahn bimodal posterior (Example 4)
# ---------------------------------------------------------------------------


def plot_allen_cahn_prior(
    field,
    xy_quad,
    source_vals,
    epsilon: float,
    beta: float,
    out_path: str | None = None,
) -> plt.Figure:
    """Figure 5: physics-informed prior log p(theta) for the well-informed basis.

    Shows 1D slice along the dominant Fourier coefficient, with all
    other coefficients set to zero.
    """
    import jax.numpy as jnp

    # Find which basis function is closest to the ground truth
    # (the sin(pi x)*sin(pi y) term, index 8 for max_freq=1)
    n_params = field.n_params
    theta_range = np.linspace(-3.5, 3.5, 200)
    log_priors = np.zeros_like(theta_range)

    xy_q = jnp.asarray(xy_quad, dtype=jnp.float64)
    sv = jnp.asarray(source_vals, dtype=jnp.float64)

    for i, val in enumerate(theta_range):
        theta = jnp.zeros(n_params, dtype=jnp.float64).at[n_params - 1].set(val)
        phi = field.eval(xy_q, theta)
        dphi_dx = field.grad_x(xy_q, theta)
        dphi_dy = field.grad_y(xy_q, theta)
        grad_sq = dphi_dx**2 + dphi_dy**2
        integrand = 0.5 * epsilon * grad_sq + 0.25 * (1 - phi**2)**2 - sv * phi
        energy = float(jnp.mean(integrand))
        log_priors[i] = -beta * energy

    log_priors -= log_priors.max()

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(theta_range, log_priors, linewidth=2, color="C0")
    ax.axvline(2.0, color="k", linestyle="--", linewidth=1, alpha=0.6, label=r"$\theta^* = 2$")
    ax.set_xlabel(r"$\theta_{sin(\pi x)sin(\pi y)}$")
    ax.set_ylabel(r"$\log\, p(\theta)$ (unnormalized)")
    ax.set_title(rf"Figure 5: Physics-informed prior ($\beta = {beta:.0f}$)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=160)
    return fig


def plot_allen_cahn_marginals(
    theta_samples: np.ndarray,
    out_path: str | None = None,
) -> plt.Figure:
    """Figure 6: marginal posteriors showing bimodality.

    Shows histograms of the two most variable parameters.
    """
    n_params = theta_samples.shape[1]
    variances = np.var(theta_samples, axis=0)
    top2 = np.argsort(variances)[-2:][::-1]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, idx in zip(axes, top2):
        ax.hist(theta_samples[:, idx], bins=40, density=True, alpha=0.7, color="C0")
        ax.set_xlabel(rf"$\theta_{{{idx+1}}}$")
        ax.set_ylabel("Density")
        ax.set_title(f"Marginal posterior of θ{idx+1}")
        ax.grid(True, alpha=0.3)

    fig.suptitle("Figure 6: Bimodal marginal posteriors", fontsize=13)
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=160)
    return fig


def plot_allen_cahn_predictions(
    grid_coords: tuple[np.ndarray, np.ndarray],
    phi_truth: np.ndarray,
    phi_median: np.ndarray,
    mode_fields: dict[int, np.ndarray],
    xy_obs: np.ndarray | None = None,
    out_path: str | None = None,
) -> plt.Figure:
    """Figure 7: ground truth, median, and per-mode predictions (2x2 heatmaps)."""
    gx, gy = grid_coords
    panels = [
        ("(a) Ground truth", phi_truth),
        ("(b) Median", phi_median),
    ]
    for k in sorted(mode_fields.keys()):
        panels.append((f"(c) Mode {k+1}" if k == 0 else f"(d) Mode {k+1}", mode_fields[k]))
    panels = panels[:4]

    vmin = min(p[1].min() for p in panels)
    vmax = max(p[1].max() for p in panels)

    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    for ax, (title, field_data) in zip(axes.ravel(), panels):
        im = ax.pcolormesh(gx, gy, field_data, shading="auto", cmap="RdBu_r", vmin=vmin, vmax=vmax)
        if xy_obs is not None:
            ax.scatter(xy_obs[:, 0], xy_obs[:, 1], s=8, c="k", marker="x", linewidths=0.5)
        ax.set_xlabel("$x$")
        ax.set_ylabel("$y$")
        ax.set_title(title)
        ax.set_aspect("equal")

    fig.colorbar(im, ax=axes, shrink=0.6, label=r"$\phi(x,y)$")
    fig.suptitle("Figure 7: Predictions from posterior modes", fontsize=13)
    if out_path:
        fig.savefig(out_path, dpi=160)
    return fig


def plot_allen_cahn_errors(
    grid_coords: tuple[np.ndarray, np.ndarray],
    phi_truth: np.ndarray,
    phi_mean: np.ndarray,
    phi_median: np.ndarray,
    mode_fields: dict[int, np.ndarray],
    out_path: str | None = None,
) -> plt.Figure:
    """Figure 8: absolute errors for mean, median, and per-mode predictions."""
    gx, gy = grid_coords
    panels = [
        ("(a) Mean error", np.abs(phi_mean - phi_truth)),
        ("(b) Median error", np.abs(phi_median - phi_truth)),
    ]
    for k in sorted(mode_fields.keys()):
        label = f"(c) Mode {k+1} error" if k == 0 else f"(d) Mode {k+1} error"
        panels.append((label, np.abs(mode_fields[k] - phi_truth)))
    panels = panels[:4]

    vmax = max(p[1].max() for p in panels)

    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    for ax, (title, err) in zip(axes.ravel(), panels):
        im = ax.pcolormesh(gx, gy, err, shading="auto", cmap="viridis", vmin=0, vmax=vmax)
        ax.set_xlabel("$x$")
        ax.set_ylabel("$y$")
        ax.set_title(title)
        ax.set_aspect("equal")

    fig.colorbar(im, ax=axes, shrink=0.6, label="Absolute error")
    fig.suptitle("Figure 8: Absolute prediction errors", fontsize=13)
    if out_path:
        fig.savefig(out_path, dpi=160)
    return fig
