"""2x2 plot grid with embedded matplotlib FigureCanvas widgets."""

from __future__ import annotations

import importlib
from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QGridLayout,
    QLabel,
    QSizePolicy,
    QStackedLayout,
    QVBoxLayout,
    QWidget,
)

from frontend.plotting.common import clear_figure, make_canvas

# Plot cell titles per example
_PLOT_TITLES: dict[str, list[tuple[str, str]]] = {
    "phase_a_forward": [
        ("Posterior field overlay", "PIFT vs MC \u00b7 5\u201395% bands"),
        ("Hamiltonian trace", "Burn-in shading \u00b7 mixing diagnostics"),
        ("Preconditioner mode weights", "Normalized \u00b7 condition number clipping"),
        ("Credible interval calibration", "Empirical vs nominal coverage"),
    ],
    "phase_b_sweep": [
        ("Multi-\u03b2 posterior fields", "Mean \u00b1 credible band per \u03b2"),
        ("Variance scaling", "Variance vs \u03b2 (log-log) \u00b7 1/\u03b2 reference"),
        ("SGLD traces per \u03b2", "Hamiltonian convergence"),
        ("ESS summary", "Effective sample size per \u03b2"),
    ],
    "phase_b_model": [
        ("\u03b2 vs \u03b3 \u2014 source error", "Violin plots of posterior \u03b2"),
        ("\u03b2 vs \u03b3 \u2014 energy error", "Violin plots of posterior \u03b2"),
        ("Nested SGLD trace", "Outer chain log(\u03b2)"),
        ("Summary", "Model-form uncertainty metrics"),
    ],
    "phase_c_params": [
        ("Parameter trace", "log(D), log(\u03ba) outer chain"),
        ("Joint posterior", "D-\u03ba scatter + KDE contours"),
        ("Prior predictive", "Field \u00b1 credible band (no data)"),
        ("Posterior predictive", "Field \u00b1 credible band (with obs)"),
    ],
    "phase_c_source": [
        ("Parameter trace", "log(D), log(\u03ba), z coefficients"),
        ("Joint D-\u03ba posterior", "Scatter + KDE contours"),
        ("Recovered source f(x)", "Mean \u00b1 2\u03c3 vs true"),
        ("Posterior predictive", "Field \u00b1 credible band (with obs)"),
    ],
    "phase_d_allen_cahn": [
        ("Ground truth", "2D Allen-Cahn field"),
        ("Posterior median", "All-sample median field"),
        ("Mode 1 prediction", "GMM cluster 0 mean field"),
        ("Mode 2 prediction", "GMM cluster 1 mean field"),
    ],
}


class PlotCell(QWidget):
    """Single plot cell: title + subtitle + FigureCanvas with placeholder."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 4)
        layout.setSpacing(2)

        self.title_label = QLabel()
        self.title_label.setStyleSheet(
            "font-size: 11px; font-weight: 500; color: #5f5e5a;"
        )
        self.subtitle_label = QLabel()
        self.subtitle_label.setStyleSheet("font-size: 10px; color: #888780;")

        layout.addWidget(self.title_label)
        layout.addWidget(self.subtitle_label)

        # Stacked layout: placeholder (index 0) / canvas (index 1)
        self._stack = QStackedLayout()
        self._placeholder = QLabel("run a method to populate")
        self._placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._placeholder.setStyleSheet(
            "color: #b4b2a9; font-size: 12px; font-style: italic;"
        )
        self._stack.addWidget(self._placeholder)

        self.figure, self.canvas = make_canvas(self)
        if isinstance(self.canvas, QWidget):
            self.canvas.setSizePolicy(
                QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
            )
            self._stack.addWidget(self.canvas)
            self._has_canvas = True
        else:
            self._has_canvas = False

        stack_container = QWidget()
        stack_container.setLayout(self._stack)
        layout.addWidget(stack_container, 1)

        self._showing_plot = False

    def set_titles(self, title: str, subtitle: str):
        self.title_label.setText(title)
        self.subtitle_label.setText(subtitle)

    def show_placeholder(self):
        self._stack.setCurrentIndex(0)
        self._showing_plot = False

    def show_canvas(self):
        if self._has_canvas:
            self._stack.setCurrentIndex(1)
        self._showing_plot = True

    def clear(self):
        clear_figure(self.figure)
        self._safe_draw()
        self.show_placeholder()

    def _safe_draw(self):
        if self._has_canvas:
            self.canvas.draw_idle()


class PlotGridWidget(QWidget):
    """2x2 grid of embedded matplotlib plots that adapts per example."""

    def __init__(self, parent=None):
        super().__init__(parent)
        grid = QGridLayout(self)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(1)

        self.cells: list[PlotCell] = []
        for i in range(4):
            cell = PlotCell(self)
            row, col = divmod(i, 2)
            grid.addWidget(cell, row, col)
            self.cells.append(cell)

    def set_example(self, example_key: str):
        """Update titles/subtitles and show placeholders for a new example."""
        titles = _PLOT_TITLES.get(example_key, _PLOT_TITLES["phase_a_forward"])
        for cell, (title, subtitle) in zip(self.cells, titles):
            cell.set_titles(title, subtitle)
            cell.clear()

    def update_plots(self, example_key: str, result: dict[str, Any]):
        """Render plots using the appropriate phase adapter."""
        from frontend.registry import get_example
        spec = get_example(example_key)
        module_name = f"frontend.plotting.{spec.plot_adapter}"
        try:
            mod = importlib.import_module(module_name)
        except ImportError:
            for cell in self.cells:
                cell.clear()
            return

        # Select the right PLOT_FUNCTIONS variant
        if example_key == "phase_b_model":
            plot_fns = getattr(mod, "PLOT_FUNCTIONS_MODEL", [])
        elif example_key == "phase_c_source":
            plot_fns = getattr(mod, "PLOT_FUNCTIONS_SOURCE", [])
        elif example_key == "phase_b_sweep":
            plot_fns = getattr(mod, "PLOT_FUNCTIONS_SWEEP", getattr(mod, "PLOT_FUNCTIONS", []))
        else:
            plot_fns = getattr(mod, "PLOT_FUNCTIONS", [])

        for i, cell in enumerate(self.cells):
            clear_figure(cell.figure)
            if i < len(plot_fns) and plot_fns[i] is not None:
                try:
                    plot_fns[i](cell.figure, result)
                    cell.show_canvas()
                except Exception as e:
                    ax = cell.figure.add_subplot(111)
                    ax.text(
                        0.5, 0.5, f"Plot error:\n{e}",
                        ha="center", va="center", fontsize=9, color="red",
                        transform=ax.transAxes,
                    )
                    cell.show_canvas()
            else:
                cell.show_placeholder()
            cell._safe_draw()

    def get_figures(self):
        """Return list of Figure objects for export."""
        return [cell.figure for cell in self.cells]
