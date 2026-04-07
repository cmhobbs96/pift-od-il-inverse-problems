"""Export panel: PDF/PNG export, LaTeX figure blocks, methods paragraph, reproducibility."""

from __future__ import annotations

import subprocess
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any

from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from frontend.plotting.common import export_figure
from frontend.style import COLORS


class ExportPanel(QWidget):
    """Module 4.6: publication export — figures, LaTeX, methods, reproducibility."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)

        # -- Figure export --
        layout.addWidget(self._section_label("Figure export"))
        fig_row = QHBoxLayout()
        fig_row.addWidget(QLabel("Plot:"))
        self._plot_selector = QComboBox()
        self._plot_selector.addItems(["Top-left", "Top-right", "Bottom-left", "Bottom-right"])
        fig_row.addWidget(self._plot_selector)
        fig_row.addWidget(QLabel("Format:"))
        self._fmt_selector = QComboBox()
        self._fmt_selector.addItems(["pdf", "png", "svg"])
        fig_row.addWidget(self._fmt_selector)
        fig_row.addWidget(QLabel("DPI:"))
        self._dpi_spin = QSpinBox()
        self._dpi_spin.setRange(72, 600)
        self._dpi_spin.setValue(300)
        fig_row.addWidget(self._dpi_spin)
        self._btn_export_fig = QPushButton("Export figure")
        self._btn_export_fig.setFixedHeight(28)
        self._btn_export_fig.clicked.connect(self._export_figure)
        fig_row.addWidget(self._btn_export_fig)
        fig_row.addStretch()
        layout.addLayout(fig_row)

        # -- LaTeX figure block --
        layout.addWidget(self._section_label("LaTeX figure block"))
        self._btn_gen_latex = QPushButton("Generate LaTeX \\figure block")
        self._btn_gen_latex.setFixedHeight(28)
        self._btn_gen_latex.clicked.connect(self._generate_latex_block)
        layout.addWidget(self._btn_gen_latex)

        # -- Methods paragraph --
        layout.addWidget(self._section_label("Methods paragraph"))
        self._btn_gen_methods = QPushButton("Generate methods text from current config")
        self._btn_gen_methods.setFixedHeight(28)
        self._btn_gen_methods.clicked.connect(self._generate_methods)
        layout.addWidget(self._btn_gen_methods)

        # -- Reproducibility bundle --
        layout.addWidget(self._section_label("Reproducibility bundle"))
        self._btn_bundle = QPushButton("Export reproducibility ZIP")
        self._btn_bundle.setFixedHeight(28)
        self._btn_bundle.clicked.connect(self._export_bundle)
        layout.addWidget(self._btn_bundle)

        # Output area
        self._output = QPlainTextEdit()
        self._output.setReadOnly(True)
        self._output.setMaximumHeight(250)
        layout.addWidget(self._output, 1)

        layout.addStretch()

        self._figures: list = []  # set by parent
        self._last_result: dict[str, Any] = {}
        self._last_config: dict[str, Any] = {}
        self._example_key: str = "phase_a_forward"

    def _section_label(self, text: str) -> QLabel:
        lbl = QLabel(text.upper())
        lbl.setStyleSheet(
            f"font-size: 10px; font-weight: 500; color: {COLORS['text3']}; "
            "letter-spacing: 0.5px; margin-top: 8px;"
        )
        return lbl

    def set_figures(self, figures: list):
        self._figures = figures

    def set_result(self, result: dict[str, Any], config: dict[str, Any], example_key: str):
        self._last_result = result
        self._last_config = config
        self._example_key = example_key

    def _export_figure(self):
        idx = self._plot_selector.currentIndex()
        if idx >= len(self._figures):
            self._output.setPlainText("No figure available at that index.")
            return
        fig = self._figures[idx]
        fmt = self._fmt_selector.currentText()
        dpi = self._dpi_spin.value()
        ext = fmt if fmt != "svg" else "svg"
        path, _ = QFileDialog.getSaveFileName(
            self, "Save figure", f"figure.{ext}", f"{fmt.upper()} (*.{ext})"
        )
        if path:
            export_figure(fig, path, fmt=fmt, dpi=dpi)
            self._output.setPlainText(f"Figure saved to {path}")

    def _generate_latex_block(self):
        idx = self._plot_selector.currentIndex()
        cell_names = ["tl", "tr", "bl", "br"]
        cell_name = cell_names[idx] if idx < 4 else "plot"
        label = f"fig:{self._example_key}_{cell_name}"
        caption = f"Result from {self._example_key.replace('_', ' ')}, panel {idx+1}."
        block = (
            f"\\begin{{figure}}[htbp]\n"
            f"  \\centering\n"
            f"  \\includegraphics[width=0.48\\textwidth]{{figures/{label}.pdf}}\n"
            f"  \\caption{{{caption}}}\n"
            f"  \\label{{{label}}}\n"
            f"\\end{{figure}}"
        )
        self._output.setPlainText(block)

    def _generate_methods(self):
        cfg = self._last_config
        ek = self._example_key

        if ek == "phase_a_forward":
            text = (
                f"We infer the posterior field distribution using preconditioned SGLD "
                f"with {cfg.get('n_steps', '?')} steps (burn-in: {cfg.get('burn_in', '?')}, "
                f"thinning: {cfg.get('thin', '?')}). The physics energy weight is "
                f"β = {cfg.get('beta', '?')}, with {cfg.get('n_modes', '?')} sine basis modes "
                f"and {cfg.get('n_quad', '?')} Monte Carlo quadrature points per step. "
                f"The initial step size is {cfg.get('step_size0', '?')} with polynomial decay "
                f"exponent {cfg.get('decay', '?')}. The diagonal preconditioner uses condition "
                f"number clipping at {cfg.get('max_condition_number', '?')}. "
                f"Observations: {cfg.get('n_obs', '?')} points with σ = {cfg.get('noise_std', '?')}."
            )
        elif ek in ("phase_c_params", "phase_c_source"):
            text = (
                f"We use nested SGLD (Algorithm 3) to jointly infer the PDE parameters. "
                f"The outer loop runs {cfg.get('outer_steps', '?')} iterations with step size "
                f"{cfg.get('outer_step_size0', '?')} and Robbins-Monro decay {cfg.get('outer_decay', '?')}. "
                f"Each outer step performs {cfg.get('inner_T_prior', '?')} prior and "
                f"{cfg.get('inner_T_posterior', '?')} posterior inner SGLD steps. "
                f"Warm-up: {cfg.get('warmup_steps', '?')} steps at fixed λ₀. "
                f"Physics weight β = {cfg.get('beta', '?')}, with {cfg.get('n_obs', '?')} observations "
                f"(σ = {cfg.get('noise_std', '?')})."
            )
        elif ek == "phase_d_allen_cahn":
            text = (
                f"We sample the posterior using NumPyro's NUTS sampler with "
                f"{cfg.get('num_warmup', '?')} warmup and {cfg.get('num_samples', '?')} "
                f"posterior samples. Target acceptance probability: {cfg.get('target_accept_prob', '?')}. "
                f"The 2D field is parameterized with a Fourier basis (max frequency = {cfg.get('max_freq', '?')}). "
                f"Observations on {len(cfg.get('observed_boundaries', []))} boundaries, "
                f"{cfg.get('n_obs_per_boundary', '?')} points each, σ = {cfg.get('noise_std', '?')}. "
                f"Posterior modes separated via GMM with {cfg.get('n_gmm_components', '?')} components."
            )
        else:
            text = f"Configuration: {cfg}"

        self._output.setPlainText(text)

    def _export_bundle(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save bundle", "reproducibility.zip", "ZIP (*.zip)")
        if not path:
            return

        import json as json_mod
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
            # Config
            zf.writestr("config.json", json_mod.dumps(self._last_config, indent=2, default=str))
            # Result summary
            summary = self._last_result.get("summary", {})
            zf.writestr("summary.json", json_mod.dumps(summary, indent=2, default=str))
            # pip freeze
            try:
                freeze = subprocess.check_output(["pip", "freeze"], text=True)
                zf.writestr("requirements.txt", freeze)
            except Exception:
                pass
            # git hash
            try:
                git_hash = subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], text=True
                ).strip()
                zf.writestr("git_hash.txt", git_hash)
            except Exception:
                pass
            # Artifacts
            artifacts = self._last_result.get("artifacts", {})
            for name, fpath in artifacts.items():
                p = Path(fpath)
                if p.exists():
                    zf.write(p, f"artifacts/{p.name}")

        self._output.setPlainText(f"Bundle saved to {path}")
