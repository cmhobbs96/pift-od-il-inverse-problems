"""Diagnostics panel: ESS, R-hat, trace viewer, preconditioner, export."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from frontend.plotting.common import clear_figure, export_figure, make_canvas
from frontend.style import COLORS


class DiagnosticsPanel(QWidget):
    """Module 4.3: diagnostics dashboard with trace plots and export."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)

        # Top: summary metric cards
        cards = QHBoxLayout()
        self._ess_label = self._make_card("ESS", cards)
        self._rhat_label = self._make_card("R-hat", cards)
        self._coverage_label = self._make_card("Coverage 90%", cards)
        self._autocorr_label = self._make_card("Lag-1 AC", cards)
        layout.addLayout(cards)

        # Middle: two canvases side by side (trace + preconditioner)
        plot_row = QHBoxLayout()
        self._trace_fig, self._trace_canvas = make_canvas(self, figsize=(5, 2.5))
        self._precond_fig, self._precond_canvas = make_canvas(self, figsize=(5, 2.5))
        if isinstance(self._trace_canvas, QWidget):
            plot_row.addWidget(self._trace_canvas, 1)
        if isinstance(self._precond_canvas, QWidget):
            plot_row.addWidget(self._precond_canvas, 1)
        layout.addLayout(plot_row, 1)

        # Bottom: JSON viewer + export buttons
        btn_row = QHBoxLayout()
        btn_copy = QPushButton("Copy JSON")
        btn_copy.clicked.connect(self._copy_json)
        btn_latex = QPushButton("Export LaTeX table")
        btn_latex.clicked.connect(self._export_latex)
        btn_csv = QPushButton("Export CSV")
        btn_csv.clicked.connect(self._export_csv)
        for b in (btn_copy, btn_latex, btn_csv):
            b.setFixedHeight(28)
            btn_row.addWidget(b)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        self._json_view = QPlainTextEdit()
        self._json_view.setReadOnly(True)
        self._json_view.setMaximumHeight(200)
        layout.addWidget(self._json_view)

        self._last_result: dict[str, Any] = {}
        self._last_diag_json: str = ""

    def _make_card(self, label: str, layout: QHBoxLayout) -> QLabel:
        w = QWidget()
        vl = QVBoxLayout(w)
        vl.setContentsMargins(8, 6, 8, 6)
        lbl = QLabel(label.upper())
        lbl.setStyleSheet(f"font-size: 9px; color: {COLORS['text3']}; letter-spacing: 0.4px;")
        val = QLabel("—")
        val.setStyleSheet("font-size: 15px; font-weight: 500;")
        vl.addWidget(lbl)
        vl.addWidget(val)
        w.setStyleSheet(f"background-color: {COLORS['bg3']}; border-radius: 6px;")
        layout.addWidget(w)
        return val

    def update_diagnostics(self, result: dict[str, Any]):
        self._last_result = result
        diagnostics = result.get("diagnostics", {})
        metrics = result.get("metrics", {})
        summary = result.get("summary", {})
        traces = result.get("traces", {})
        config = result.get("config", {})

        # ESS
        chain = result.get("chain")
        if chain is not None and hasattr(chain, "shape") and chain.ndim == 2:
            try:
                from utils.diagnostics import effective_sample_size_bulk
                ess_arr = effective_sample_size_bulk(chain)
                ess_min = float(np.min(ess_arr))
                self._ess_label.setText(f"{ess_min:.0f}")
            except Exception:
                self._ess_label.setText("—")
        else:
            self._ess_label.setText("—")

        # R-hat (single chain → N/A)
        self._rhat_label.setText("N/A")

        # Coverage
        cov = summary.get("measurement_interval_coverage_90pct")
        if cov is not None:
            self._coverage_label.setText(f"{cov*100:.0f}%")
        else:
            self._coverage_label.setText("—")

        # Lag-1 autocorrelation
        if chain is not None and hasattr(chain, "shape") and chain.ndim == 2:
            try:
                from utils.diagnostics import lag1_autocorr
                ac = lag1_autocorr(chain[:, 0])
                self._autocorr_label.setText(f"{ac:.2f}")
            except Exception:
                self._autocorr_label.setText("—")
        else:
            self._autocorr_label.setText("—")

        # Trace plot
        clear_figure(self._trace_fig)
        ham = traces.get("hamiltonian")
        if ham is not None:
            ax = self._trace_fig.add_subplot(111)
            ax.plot(ham, linewidth=0.5, alpha=0.7, color="#185fa5")
            burn_in = config.get("burn_in", 0)
            if burn_in > 0:
                ax.axvspan(0, burn_in, alpha=0.1, color="#f1efe8")
                ax.axvline(burn_in, color="#e24b4a", linewidth=0.8, linestyle="--")
            ax.set_xlabel("Step", fontsize=8)
            ax.set_ylabel("H", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.set_title("Hamiltonian trace", fontsize=9)
        self._trace_canvas.draw_idle()

        # Preconditioner bar chart
        clear_figure(self._precond_fig)
        n_modes = config.get("n_modes") or config.get("K")
        if n_modes:
            ax = self._precond_fig.add_subplot(111)
            modes = np.arange(1, int(n_modes) + 1)
            weights = 1.0 / (modes * np.pi) ** 4
            weights /= weights.max()
            max_cond = config.get("max_condition_number", 100)
            floor = 1.0 / max_cond if max_cond else 0
            ax.bar(modes, weights, color="#185fa5", alpha=0.7)
            if floor > 0:
                ax.axhline(floor, color="#e24b4a", linewidth=0.8, linestyle="--", label="floor")
            ax.set_xlabel("mode k", fontsize=8)
            ax.set_ylabel("weight", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.set_title("Preconditioner weights", fontsize=9)
        self._precond_canvas.draw_idle()

        # JSON
        diag_dict = {
            "run_id": result.get("run_id", ""),
            "status": result.get("status", ""),
            "metrics": {k: self._safe(v) for k, v in metrics.items()},
            "diagnostics": {k: self._safe(v) for k, v in diagnostics.items()},
        }
        self._last_diag_json = json.dumps(diag_dict, indent=2, default=str)
        self._json_view.setPlainText(self._last_diag_json)

    def _copy_json(self):
        QApplication.clipboard().setText(self._last_diag_json)

    def _export_latex(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save LaTeX table", "", "TeX (*.tex)")
        if not path:
            return
        result = self._last_result
        summary = result.get("summary", {})
        metrics = result.get("metrics", {})
        lines = [
            r"\begin{tabular}{lr}",
            r"\toprule",
            r"Metric & Value \\",
            r"\midrule",
        ]
        for label, key in [
            ("L2 error", "l2_error"), ("Max error", "max_error"),
            ("Coverage (90\\%)", "measurement_interval_coverage_90pct"),
            ("Samples", "n_posterior_samples"), ("Runtime (s)", "runtime_sec"),
        ]:
            val = summary.get(key) or metrics.get(key)
            val_str = f"{val:.4f}" if isinstance(val, float) else str(val or "—")
            lines.append(f"{label} & {val_str} \\\\")
        lines += [r"\bottomrule", r"\end{tabular}"]
        with open(path, "w") as f:
            f.write("\n".join(lines))

    def _export_csv(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save CSV", "", "CSV (*.csv)")
        if not path:
            return
        result = self._last_result
        summary = result.get("summary", {})
        metrics = result.get("metrics", {})
        import csv
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["metric", "value"])
            for key in ("l2_error", "max_error", "measurement_interval_coverage_90pct",
                        "n_posterior_samples", "runtime_sec"):
                val = summary.get(key) or metrics.get(key)
                w.writerow([key, val])

    @staticmethod
    def _safe(v):
        if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
            return None
        if isinstance(v, np.ndarray):
            return v.tolist()
        return v
