"""Declarative example registry — single source of truth for GUI behavior.

When the user selects an example from the sidebar dropdown, the registry
determines which methods are available, which config keys to display,
which plotting adapter to use, and how to dispatch the pipeline runner.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class MethodSpec:
    """Describes one selectable method within an example."""

    display_name: str
    runner_key: str | None  # key for RunManager dispatch; None = not yet implemented
    badge: str  # "ok" | "ready" | "unstable" | "disabled"
    enabled: bool = True  # whether the checkbox is interactive
    checked: bool = False  # default check state


@dataclass(frozen=True)
class ExampleSpec:
    """Describes one selectable example (problem + method set)."""

    key: str  # internal identifier, e.g. "phase_a_forward"
    display_name: str  # shown in the dropdown
    phase: str  # "A", "B", "C", "D"
    methods: tuple[MethodSpec, ...]
    config_display_keys: tuple[str, ...]  # which keys to show in sidebar
    plot_adapter: str  # module name under frontend.plotting
    has_observations: bool = True  # show data-source section?
    multi_method: bool = False  # can the user select >1 method?
    default_config: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Method specs reused across examples
# ---------------------------------------------------------------------------

_PIFT = MethodSpec("PIFT (SGLD)", "pift", "ok", enabled=True, checked=True)
_MC = MethodSpec("Monte Carlo", "mc", "ok", enabled=True, checked=True)
_ODIL = MethodSpec("ODIL (Newton)", "odil", "ready", enabled=True)
_ODIL_WARM = MethodSpec("ODIL warm-start PIFT", None, "ready", enabled=False)
_BPINN = MethodSpec("Bayesian PINNs", "bpinn", "unstable", enabled=True)
_HMCECS = MethodSpec("HMCECS (NumPyro)", "hmcecs", "disabled", enabled=False)

_NESTED_SGLD = MethodSpec("Nested SGLD", None, "ok", enabled=True, checked=True)
_HMCECS_ONLY = MethodSpec("HMCECS (NumPyro)", "hmcecs", "ok", enabled=True, checked=True)

# ---------------------------------------------------------------------------
# Example definitions
# ---------------------------------------------------------------------------

EXAMPLES: dict[str, ExampleSpec] = {}


def _register(spec: ExampleSpec) -> ExampleSpec:
    EXAMPLES[spec.key] = spec
    return spec


# ---- Phase A: Forward 1D Poisson ----
_register(ExampleSpec(
    key="phase_a_forward",
    display_name="Forward 1D Poisson",
    phase="A",
    methods=(_PIFT, _MC, _ODIL, _ODIL_WARM, _BPINN, _HMCECS),
    config_display_keys=(
        "beta", "n_steps", "burn_in", "thin",
        "step_size0", "decay", "n_quad", "n_modes",
        "max_condition_number", "noise_std", "n_obs",
    ),
    plot_adapter="phase_a",
    has_observations=True,
    multi_method=True,
    default_config={
        "seed": 7, "n_modes": 12, "n_obs": 28, "noise_std": 0.08,
        "beta": 0.5, "n_steps": 40000, "burn_in": 8000, "thin": 16,
        "step_size0": 2e-3, "decay": 0.55, "n_quad": 96, "n_grid": 300,
        "max_condition_number": 100.0, "runtime_check_interval": 5,
    },
))

# ---- Phase B: Beta sensitivity sweep (Example 1) ----
_register(ExampleSpec(
    key="phase_b_sweep",
    display_name="\u03b2 sensitivity sweep (Ex. 1)",
    phase="B",
    methods=(MethodSpec("PIFT (SGLD)", None, "ok", enabled=True, checked=True),),
    config_display_keys=(
        "beta_values", "K", "D", "bc_left", "bc_right",
        "n_steps", "burn_in", "thin", "step_size0", "decay",
        "n_quad", "max_condition_number",
    ),
    plot_adapter="phase_b",
    has_observations=False,
    default_config={
        "seed": 42, "K": 20, "D": 0.25, "bc_left": 1.0, "bc_right": 0.1,
        "beta_values": [1, 10, 100, 1000],
        "n_steps": 50000, "burn_in": 10000, "thin": 20,
        "step_size0": 0.1, "decay": 0.51, "n_quad": 128, "n_grid": 300,
        "max_condition_number": 100.0, "runtime_check_interval": 5,
    },
))

# ---- Phase B: Model-form uncertainty (Example 2) ----
_register(ExampleSpec(
    key="phase_b_model",
    display_name="Model-form uncertainty (Ex. 2)",
    phase="B",
    methods=(_NESTED_SGLD,),
    config_display_keys=(
        "gamma_values", "D", "kappa", "n_obs", "noise_std",
        "outer_steps", "outer_step_size0", "outer_decay",
        "inner_T_prior", "inner_T_posterior",
        "warmup_steps", "n_quad",
    ),
    plot_adapter="phase_b",
    has_observations=False,
    default_config={
        "seed": 42, "D": 0.1, "kappa": 1.0,
        "bc_left": 0.0, "bc_right": 0.0, "K": 20,
        "n_obs": 40, "noise_std": 0.01,
        "gamma_values": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        "outer_steps": 5000, "outer_step_size0": 0.1, "outer_decay": 0.51,
        "inner_T_prior": 10, "inner_T_posterior": 1,
        "inner_step_size0": 0.1, "inner_decay": 0.51,
        "warmup_steps": 5000, "n_quad": 128,
        "max_condition_number": 100.0, "burn_in_frac": 0.2, "n_grid": 300,
    },
))

# ---- Phase C: Inverse parameter identification (Example 3a) ----
_register(ExampleSpec(
    key="phase_c_params",
    display_name="Inverse: D and \u03ba (Ex. 3a)",
    phase="C",
    methods=(_NESTED_SGLD,),
    config_display_keys=(
        "beta", "D_true", "kappa_true", "n_obs", "noise_std",
        "outer_steps", "outer_step_size0", "outer_decay",
        "inner_T_prior", "inner_T_posterior",
        "warmup_steps", "n_quad", "snapshot_interval",
    ),
    plot_adapter="phase_c",
    has_observations=False,
    default_config={
        "seed": 42, "K": 20, "D_true": 0.1, "kappa_true": 1.0,
        "beta": 1e5, "n_obs": 40, "noise_std": 0.01,
        "lambda0_log_D": 0.0, "lambda0_log_kappa": 0.5,
        "warmup_steps": 50000, "outer_steps": 10000,
        "outer_step_size0": 0.1, "outer_decay": 0.51,
        "inner_T_prior": 10, "inner_T_posterior": 1,
        "inner_step_size0": 0.1, "inner_decay": 0.51,
        "n_quad": 128, "n_grid": 300, "max_condition_number": 100.0,
        "burn_in_frac": 0.2, "snapshot_interval": 100,
    },
))

# ---- Phase C: Source term identification (Example 3b) ----
_register(ExampleSpec(
    key="phase_c_source",
    display_name="Inverse: D, \u03ba and f (Ex. 3b)",
    phase="C",
    methods=(_NESTED_SGLD,),
    config_display_keys=(
        "beta", "D_true", "kappa_true", "n_obs", "noise_std",
        "kle_lengthscale", "kle_n_terms",
        "outer_steps", "outer_step_size0", "outer_decay",
        "inner_T_prior", "inner_T_posterior",
        "warmup_steps", "n_quad", "snapshot_interval",
    ),
    plot_adapter="phase_c",
    has_observations=False,
    default_config={
        "seed": 42, "K": 20, "D_true": 0.1, "kappa_true": 1.0,
        "beta": 1e5, "n_obs": 40, "noise_std": 0.01,
        "kle_lengthscale": 0.3, "kle_n_terms": 10, "kle_n_grid": 200,
        "lambda0_log_D": 0.0, "lambda0_log_kappa": 0.5,
        "warmup_steps": 50000, "outer_steps": 10000,
        "outer_step_size0": 0.1, "outer_decay": 0.51,
        "inner_T_prior": 10, "inner_T_posterior": 1,
        "inner_step_size0": 0.1, "inner_decay": 0.51,
        "n_quad": 128, "n_grid": 300, "max_condition_number": 100.0,
        "burn_in_frac": 0.2, "snapshot_interval": 100,
    },
))

# ---- Phase D: 2D Allen-Cahn bimodal posterior (Example 4) ----
_register(ExampleSpec(
    key="phase_d_allen_cahn",
    display_name="2D Allen-Cahn bimodal (Ex. 4)",
    phase="D",
    methods=(_HMCECS_ONLY,),
    config_display_keys=(
        "epsilon", "beta", "max_freq",
        "n_obs_per_boundary", "noise_std",
        "num_warmup", "num_samples", "num_chains",
        "target_accept_prob", "max_tree_depth",
        "n_gmm_components",
    ),
    plot_adapter="phase_d",
    has_observations=False,
    default_config={
        "seed": 42, "epsilon": 0.01, "beta": 100.0,
        "max_freq": 1, "n_obs_per_boundary": 15, "noise_std": 0.01,
        "observed_boundaries": ["left", "right", "bottom"],
        "n_quad_per_dim": 40,
        "num_warmup": 500, "num_samples": 1000, "num_chains": 1,
        "target_accept_prob": 0.8, "max_tree_depth": 10, "prior_std": 10.0,
        "n_gmm_components": 2, "n_grid_per_dim": 50,
    },
))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

EXAMPLE_KEYS = list(EXAMPLES.keys())
EXAMPLE_DISPLAY_NAMES = [EXAMPLES[k].display_name for k in EXAMPLE_KEYS]


def get_example(key: str) -> ExampleSpec:
    return EXAMPLES[key]


def get_example_by_index(idx: int) -> ExampleSpec:
    return EXAMPLES[EXAMPLE_KEYS[idx]]
