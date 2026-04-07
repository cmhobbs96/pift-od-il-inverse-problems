"""Shared matplotlib utilities for embedded plot canvases."""

from __future__ import annotations

from pathlib import Path

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure


def make_canvas(parent=None, figsize=(5, 3.5), dpi=100):
    """Create a Figure + FigureCanvasQTAgg pair for embedding in Qt."""
    fig = Figure(figsize=figsize, dpi=dpi, tight_layout=True)
    canvas = FigureCanvasQTAgg(fig)
    if parent is not None:
        canvas.setParent(parent)
    return fig, canvas


def apply_pub_style(ax):
    """Apply publication-quality styling to an Axes."""
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.tick_params(labelsize=8)
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)


def clear_figure(fig: Figure):
    """Clear all axes from a figure."""
    fig.clear()


def export_figure(fig: Figure, path: str | Path, fmt: str = "pdf", dpi: int = 300):
    """Export a figure to PDF or PNG."""
    fig.savefig(str(path), format=fmt, dpi=dpi, bbox_inches="tight")
