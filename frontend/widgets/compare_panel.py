"""Comparison panel: pin baseline, delta table, overlay plot, export."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from frontend.plotting.common import clear_figure, make_canvas
from frontend.style import COLORS


class ComparePanel(QWidget):
    """Module 4.4: multi-run comparison with baseline pinning."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)

        # Top: baseline selector
        top = QHBoxLayout()
        top.addWidget(QLabel("Baseline:"))
        self._baseline_combo = QComboBox()
        self._baseline_combo.setMinimumWidth(200)
        top.addWidget(self._baseline_combo, 1)
        self._btn_pin = QPushButton("Pin")
        self._btn_pin.setFixedHeight(28)
        self._btn_pin.clicked.connect(self._pin_baseline)
        top.addWidget(self._btn_pin)
        self._baseline_label = QLabel("No baseline pinned")
        self._baseline_label.setStyleSheet(f"font-size: 11px; color: {COLORS['text3']};")
        top.addWidget(self._baseline_label)
        top.addStretch()
        layout.addLayout(top)

        # Comparison table
        self._table = QTableWidget()
        self._table.setColumnCount(7)
        self._table.setHorizontalHeaderLabels([
            "Run ID", "Method", "L2", "Max Error", "Coverage", "ESS", "Runtime"
        ])
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self._table.setAlternatingRowColors(True)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        layout.addWidget(self._table)

        # Overlay canvas
        self._overlay_fig, self._overlay_canvas = make_canvas(self, figsize=(8, 3))
        if isinstance(self._overlay_canvas, QWidget):
            layout.addWidget(self._overlay_canvas)

        # Export buttons
        btn_row = QHBoxLayout()
        btn_csv = QPushButton("Export CSV")
        btn_csv.clicked.connect(self._export_csv)
        btn_latex = QPushButton("Export LaTeX")
        btn_latex.clicked.connect(self._export_latex)
        for b in (btn_csv, btn_latex):
            b.setFixedHeight(28)
            btn_row.addWidget(b)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        self._baseline_id: str | None = None
        self._runs: list[dict[str, Any]] = []

    def set_runs(self, runs: list[dict[str, Any]]):
        """Populate the comparison from a list of run records with metrics."""
        self._runs = runs
        self._baseline_combo.clear()
        for r in runs:
            rid = r.get("run_id", "?")[:8]
            method = r.get("method", "?")
            self._baseline_combo.addItem(f"{rid} ({method})", r.get("run_id"))

        self._table.setRowCount(len(runs))
        for i, r in enumerate(runs):
            metrics = r.get("metrics", {})
            items = [
                r.get("run_id", "?")[:8],
                r.get("method", "?"),
                self._fmt(metrics.get("l2_error") or metrics.get("reference_l2_error")),
                self._fmt(metrics.get("max_error")),
                self._fmt(metrics.get("measurement_interval_coverage_90pct")),
                self._fmt(metrics.get("ess_min")),
                self._fmt(r.get("runtime_sec")),
            ]
            for j, val in enumerate(items):
                item = QTableWidgetItem(val)
                # Color deltas vs baseline
                if self._baseline_id and j in (2, 3) and i > 0:
                    try:
                        base_row = next(
                            rr for rr in runs if rr.get("run_id") == self._baseline_id
                        )
                        base_val = base_row.get("metrics", {}).get(
                            "l2_error" if j == 2 else "max_error"
                        )
                        cur_val = metrics.get("l2_error" if j == 2 else "max_error")
                        if base_val and cur_val:
                            if float(cur_val) < float(base_val):
                                item.setForeground(Qt.GlobalColor.darkGreen)
                            elif float(cur_val) > float(base_val):
                                item.setForeground(Qt.GlobalColor.red)
                    except (StopIteration, ValueError, TypeError):
                        pass
                self._table.setItem(i, j, item)

    def _pin_baseline(self):
        idx = self._baseline_combo.currentIndex()
        if idx >= 0:
            self._baseline_id = self._baseline_combo.currentData()
            self._baseline_label.setText(f"Baseline: {self._baseline_id[:8]}")
            # Re-render table with deltas
            if self._runs:
                self.set_runs(self._runs)

    def update_overlay(self, results: list[dict[str, Any]]):
        """Draw multi-run posterior overlay."""
        clear_figure(self._overlay_fig)
        ax = self._overlay_fig.add_subplot(111)

        colors = ["#185fa5", "#ba7517", "#3b6d11", "#a32d2d", "#854f0b"]
        for i, res in enumerate(results):
            x = res.get("x_grid")
            mean = res.get("phi_mean")
            std = res.get("phi_std")
            label = res.get("summary", {}).get("method", f"Run {i}")
            c = colors[i % len(colors)]
            if x is not None and mean is not None:
                ax.plot(x, mean, color=c, linewidth=1.5, label=label)
                if std is not None:
                    lo = mean - 1.96 * std
                    hi = mean + 1.96 * std
                    ax.fill_between(x, lo, hi, alpha=0.15, color=c)

        truth = results[0].get("phi_truth") if results else None
        x = results[0].get("x_grid") if results else None
        if truth is not None and x is not None:
            ax.plot(x, truth, "--", color="#888780", linewidth=1.5, label="truth")

        ax.legend(fontsize=8)
        ax.set_xlabel("x")
        ax.set_ylabel("φ(x)")
        ax.set_title("Multi-run posterior overlay", fontsize=10)
        ax.grid(True, alpha=0.3)
        self._overlay_canvas.draw_idle()

    def _export_csv(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save CSV", "", "CSV (*.csv)")
        if not path:
            return
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["run_id", "method", "l2", "max_error", "coverage", "runtime"])
            for r in self._runs:
                m = r.get("metrics", {})
                w.writerow([
                    r.get("run_id", ""), r.get("method", ""),
                    m.get("l2_error", ""), m.get("max_error", ""),
                    m.get("measurement_interval_coverage_90pct", ""),
                    r.get("runtime_sec", ""),
                ])

    def _export_latex(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save LaTeX", "", "TeX (*.tex)")
        if not path:
            return
        lines = [
            r"\begin{tabular}{lccccc}",
            r"\toprule",
            r"Method & L2 & Max Error & Coverage & ESS & Runtime \\",
            r"\midrule",
        ]
        for r in self._runs:
            m = r.get("metrics", {})
            lines.append(
                f"{r.get('method', '')} & "
                f"{self._fmt(m.get('l2_error'))} & "
                f"{self._fmt(m.get('max_error'))} & "
                f"{self._fmt(m.get('measurement_interval_coverage_90pct'))} & "
                f"{self._fmt(m.get('ess_min'))} & "
                f"{self._fmt(r.get('runtime_sec'))} \\\\"
            )
        lines += [r"\bottomrule", r"\end{tabular}"]
        with open(path, "w") as f:
            f.write("\n".join(lines))

    @staticmethod
    def _fmt(val) -> str:
        if val is None:
            return "—"
        try:
            return f"{float(val):.4f}"
        except (ValueError, TypeError):
            return str(val)
