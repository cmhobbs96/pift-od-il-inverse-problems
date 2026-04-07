"""Sensitivity sweep and multi-chain utilities."""

from __future__ import annotations

import copy
from typing import Any


def run_sensitivity_sweep(
    param_name: str,
    param_values: list[float | int],
    base_config: dict[str, Any] | None = None,
    pipeline_fn: Any | None = None,
    pipeline_kwargs: dict[str, Any] | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """Run a pipeline repeatedly, varying one parameter.

    Parameters
    ----------
    param_name : config key to sweep (e.g. ``"beta"``, ``"n_modes"``).
    param_values : values to try.
    base_config : starting config dict; ``None`` uses pipeline defaults.
    pipeline_fn : callable, defaults to ``run_phase_a_forward_poisson``.
    pipeline_kwargs : extra keyword arguments forwarded to *pipeline_fn*
        (e.g. ``{"output_root": "/tmp"}``, ``{"obs_data": (x, y)}``).
    verbose : print progress.

    Returns
    -------
    dict with keys ``param_name``, ``param_values``, ``results``,
    ``phi_means``, ``phi_stds``, ``summaries``.
    """
    if pipeline_fn is None:
        from src.pipelines.phase_a import run_phase_a_forward_poisson
        pipeline_fn = run_phase_a_forward_poisson
    if pipeline_kwargs is None:
        pipeline_kwargs = {}

    results = []
    phi_means = []
    phi_stds = []
    summaries = []

    for i, val in enumerate(param_values):
        cfg = copy.deepcopy(base_config) if base_config else {}
        cfg[param_name] = val
        if verbose:
            print(f"[sweep {i + 1}/{len(param_values)}] {param_name}={val}")
        r = pipeline_fn(cfg=cfg, **pipeline_kwargs)
        results.append(r)
        phi_means.append(r["phi_mean"])
        phi_stds.append(r["phi_std"])
        summaries.append(r.get("summary"))

    return {
        "param_name": param_name,
        "param_values": param_values,
        "results": results,
        "phi_means": phi_means,
        "phi_stds": phi_stds,
        "summaries": summaries,
    }


def run_multi_chain(
    n_chains: int = 3,
    base_config: dict[str, Any] | None = None,
    base_seed: int = 7,
    pipeline_fn: Any | None = None,
    pipeline_kwargs: dict[str, Any] | None = None,
    verbose: bool = True,
) -> list[dict[str, Any]]:
    """Run the pipeline *n_chains* times with different seeds.

    Seeds are ``base_seed``, ``base_seed + 1000``, ``base_seed + 2000``, etc.

    Returns
    -------
    list of pipeline result dicts (one per chain).
    """
    if pipeline_fn is None:
        from src.pipelines.phase_a import run_phase_a_forward_poisson
        pipeline_fn = run_phase_a_forward_poisson
    if pipeline_kwargs is None:
        pipeline_kwargs = {}

    results = []
    for i in range(n_chains):
        seed = base_seed + i * 1000
        cfg = copy.deepcopy(base_config) if base_config else {}
        cfg["seed"] = seed
        if verbose:
            print(f"[chain {i + 1}/{n_chains}] seed={seed}")
        r = pipeline_fn(cfg=cfg, **pipeline_kwargs)
        results.append(r)

    return results
