"""Plotting adapter for Phase D: 2D Allen-Cahn bimodal posterior (Example 4)."""

from __future__ import annotations

from typing import Any

import numpy as np
from matplotlib.figure import Figure

from frontend.plotting.common import apply_pub_style


def _plot_2d_field(ax, field_2d, grid_coords, title: str, cmap: str = "viridis",
                   obs_xy=None):
    """Helper to plot a 2D field as heatmap."""
    apply_pub_style(ax)
    if field_2d is None or grid_coords is None:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=9,
                transform=ax.transAxes)
        return

    gx, gy = grid_coords
    im = ax.pcolormesh(gx, gy, field_2d, shading="auto", cmap=cmap)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    if obs_xy is not None:
        ax.scatter(obs_xy[:, 0], obs_xy[:, 1], c="red", s=6, marker="x",
                   linewidths=0.5, zorder=5)

    ax.set_title(title, fontsize=9)
    ax.set_xlabel("x", fontsize=7)
    ax.set_ylabel("y", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.set_aspect("equal")


def plot_2d_truth(fig: Figure, result: dict[str, Any]) -> None:
    """TL: Ground truth 2D field."""
    ax = fig.add_subplot(111)
    grid_coords = result.get("grid_coords")
    phi_truth = result.get("phi_truth")
    _plot_2d_field(ax, phi_truth, grid_coords, "Ground truth")


def plot_2d_median(fig: Figure, result: dict[str, Any]) -> None:
    """TR: Posterior median field."""
    ax = fig.add_subplot(111)
    grid_coords = result.get("grid_coords")
    phi_median = result.get("phi_median")
    _plot_2d_field(ax, phi_median, grid_coords, "Posterior median")


def plot_2d_mode1(fig: Figure, result: dict[str, Any]) -> None:
    """BL: Mode 1 mean field."""
    ax = fig.add_subplot(111)
    grid_coords = result.get("grid_coords")
    mode_fields = result.get("mode_fields", {})
    field = mode_fields.get(0)
    _plot_2d_field(ax, field, grid_coords, "Mode 1 mean")


def plot_2d_mode2(fig: Figure, result: dict[str, Any]) -> None:
    """BR: Mode 2 mean field."""
    ax = fig.add_subplot(111)
    grid_coords = result.get("grid_coords")
    mode_fields = result.get("mode_fields", {})
    field = mode_fields.get(1)
    _plot_2d_field(ax, field, grid_coords, "Mode 2 mean")


PLOT_FUNCTIONS = [
    plot_2d_truth,
    plot_2d_median,
    plot_2d_mode1,
    plot_2d_mode2,
]
