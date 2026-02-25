"""Shared helpers for Phase A pipelines."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np


def phi_true(x):
    return jnp.sin(jnp.pi * x) + 0.35 * jnp.sin(2.0 * jnp.pi * x)


def forcing(x):
    return (jnp.pi**2) * jnp.sin(jnp.pi * x) + 0.35 * (2.0 * jnp.pi) ** 2 * jnp.sin(2.0 * jnp.pi * x)


def load_observations_csv(csv_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    arr = np.genfromtxt(csv_path, delimiter=",", names=True)
    if getattr(arr, "dtype", None) is not None and arr.dtype.names:
        names_lower = [n.lower() for n in arr.dtype.names]
        if "x" in names_lower and "y" in names_lower:
            x_col = arr[arr.dtype.names[names_lower.index("x")]]
            y_col = arr[arr.dtype.names[names_lower.index("y")]]
            return np.asarray(x_col, dtype=float), np.asarray(y_col, dtype=float)

    raw = np.genfromtxt(csv_path, delimiter=",", dtype=float)
    raw = np.atleast_2d(raw)
    if raw.shape[1] < 2:
        raise ValueError("CSV must contain at least two columns: x,y")
    return raw[:, 0], raw[:, 1]


def select_device(preference: str):
    pref = preference.lower().strip()
    if pref not in {"cpu", "gpu"}:
        pref = "cpu"

    if pref == "gpu":
        try:
            gpus = jax.devices("gpu")
        except Exception:  # noqa: BLE001
            gpus = []
        if gpus:
            return gpus[0], "gpu"

    cpus = jax.devices("cpu")
    return cpus[0], "cpu"


def emit(
    progress_callback: Callable[[float, str, str, float], None] | None,
    started_at: float,
    percent: float,
    stage: str,
    detail: str,
) -> None:
    if progress_callback is None:
        return
    progress_callback(percent, stage, detail, max(0.0, time.monotonic() - started_at))


def prepare_observations(
    config: dict[str, float | int],
    key: jax.Array,
    obs_data: tuple[np.ndarray, np.ndarray] | None,
    obs_csv_path: str | Path | None,
    started_at: float,
    progress_callback: Callable[[float, str, str, float], None] | None,
):
    emit(progress_callback, started_at, 12.0, "data", "Preparing observations")
    if obs_data is not None:
        x_obs_np, y_obs_np = obs_data
        x_obs_np = np.asarray(x_obs_np, dtype=float)
        y_obs_np = np.asarray(y_obs_np, dtype=float)
        order = np.argsort(x_obs_np)
        x_obs_np = x_obs_np[order]
        y_obs_np = y_obs_np[order]
        x_obs = jnp.asarray(x_obs_np)
        y_obs = jnp.asarray(y_obs_np)
        y_clean = phi_true(x_obs)
        observation_source = "seed_data"
        emit(progress_callback, started_at, 14.0, "data", f"Loaded seed observations ({x_obs_np.size} points)")
    elif obs_csv_path is not None:
        x_obs_np, y_obs_np = load_observations_csv(obs_csv_path)
        order = np.argsort(x_obs_np)
        x_obs_np = x_obs_np[order]
        y_obs_np = y_obs_np[order]
        x_obs = jnp.asarray(x_obs_np)
        y_obs = jnp.asarray(y_obs_np)
        y_clean = phi_true(x_obs)
        observation_source = "csv"
        emit(progress_callback, started_at, 14.0, "data", f"Loaded CSV observations ({x_obs_np.size} points)")
    else:
        key, x_key, noise_key = jax.random.split(key, 3)
        x_obs = jnp.sort(
            jax.random.uniform(
                x_key,
                shape=(int(config["n_obs"]),),
                minval=0.03,
                maxval=0.97,
            )
        )
        y_clean = phi_true(x_obs)
        y_obs = y_clean + float(config["noise_std"]) * jax.random.normal(
            noise_key,
            shape=(int(config["n_obs"]),),
        )
        observation_source = "synthetic"
        emit(
            progress_callback,
            started_at,
            14.0,
            "data",
            f"Generated synthetic observations ({int(config['n_obs'])} points)",
        )
    return key, x_obs, y_obs, y_clean, observation_source

