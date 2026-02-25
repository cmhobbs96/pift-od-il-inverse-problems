#!/usr/bin/env python3
"""PySide6 frontend shell for PIFT run management and comparison."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import sys
from datetime import datetime
import time
import zipfile

import numpy as np

from PySide6.QtCore import QObject, Qt, QThread, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QProgressBar,
    QRadioButton,
    QScrollArea,
    QSplitter,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "outputs" / ".cache" / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / "outputs" / ".cache"))

from core.parameterizations import SineBasisField
from pipelines.phase_a import DEFAULT_PHASE_A_CONFIG
from run_manager import RunManager
from utils.validators import validate_observations

UI_DEFAULT_CONFIG = {
    **DEFAULT_PHASE_A_CONFIG,
    # More conservative defaults for stable SGLD behavior in the GUI.
    "step_size0": 1e-5,
    "decay": 0.70,
    "beta": 0.5,
    "n_quad": 512,
    "noise_std": 0.15,
    "n_steps": 30000,
    "burn_in": 5000,
    "thin": 10,
    "mc_proposal_std": 2e-3,
}

STABLE_PRESETS: dict[str, dict[str, float | int]] = {
    "PIFT (Phase A Forward Poisson)": {
        "step_size0": 5e-6,
        "decay": 0.75,
        "beta": 0.5,
        "noise_std": 0.15,
        "n_quad": 512,
        "n_steps": 12000,
        "burn_in": 3000,
        "thin": 10,
        "runtime_check_interval": 5,
    },
    "Bayesian PINNs": {
        "step_size0": 2e-6,
        "decay": 0.75,
        "noise_std": 0.15,
        "n_quad": 512,
        "n_steps": 12000,
        "burn_in": 3000,
        "thin": 10,
        "bpinn_sigma_r": 0.2,
        "bpinn_prior_prec": 1e-2,
        "runtime_check_interval": 5,
    },
    "Classical Monte Carlo": {
        "noise_std": 0.15,
        "beta": 0.5,
        "n_quad": 512,
        "n_steps": 12000,
        "burn_in": 3000,
        "thin": 10,
        "mc_proposal_std": 8e-4,
    },
    "ODIL Baseline": {
        "noise_std": 0.15,
        "n_quad": 512,
        "odil_steps": 6000,
        "odil_lr": 1e-4,
        "odil_phys_weight": 4.0,
    },
    "__batch__": {
        "step_size0": 5e-6,
        "decay": 0.75,
        "beta": 0.5,
        "noise_std": 0.15,
        "n_quad": 512,
        "n_steps": 12000,
        "burn_in": 3000,
        "thin": 10,
        "runtime_check_interval": 5,
        "mc_proposal_std": 8e-4,
        "bpinn_sigma_r": 0.2,
        "bpinn_prior_prec": 1e-2,
        "odil_steps": 6000,
        "odil_lr": 1e-4,
        "odil_phys_weight": 4.0,
    },
}


class RunWorker(QObject):
    progress = Signal(float, str, str, float)
    completed = Signal(dict)
    failed = Signal(str)

    def __init__(
        self,
        run_manager: RunManager,
        methods: list[str],
        config: dict,
        device: str,
        csv_path: str | None,
        obs_data,
    ) -> None:
        super().__init__()
        self.run_manager = run_manager
        self.methods = methods
        self.config = config
        self.device = device
        self.csv_path = csv_path
        self.obs_data = obs_data

    def run(self) -> None:
        try:
            total = max(1, len(self.methods))
            batch_started = time.monotonic()
            results: list[dict] = []

            for idx, method in enumerate(self.methods):
                self.progress.emit(
                    float((idx / total) * 100.0),
                    "batch",
                    f"Starting {method} ({idx+1}/{total})",
                    float(time.monotonic() - batch_started),
                )

                def _progress(p: float, s: str, d: str, _elapsed: float, _idx=idx, _method=method) -> None:
                    overall = ((_idx + (p / 100.0)) / total) * 100.0
                    self.progress.emit(
                        float(overall),
                        s,
                        f"{_method}: {d}",
                        float(time.monotonic() - batch_started),
                    )

                result = self.run_manager.run_forward(
                    method=method,
                    config=self.config,
                    device_preference=self.device,
                    obs_csv_path=self.csv_path,
                    obs_data=self.obs_data,
                    progress_callback=_progress,
                )
                result["_batch_method"] = method
                results.append(result)
                self.progress.emit(
                    float(((idx + 1) / total) * 100.0),
                    "batch",
                    f"Finished {method} ({idx+1}/{total})",
                    float(time.monotonic() - batch_started),
                )

            if len(results) == 1:
                self.completed.emit(results[0])
                return

            self.completed.emit(
                {
                    "status": "completed",
                    "results": results,
                    "runtime_sec": float(time.monotonic() - batch_started),
                }
            )
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))


class ImagePanel(QGroupBox):
    def __init__(self, title: str) -> None:
        super().__init__(title)
        self._path: str | None = None
        self._pix: QPixmap | None = None
        self.label = QLabel("No image")
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setMinimumHeight(220)
        layout = QVBoxLayout(self)
        layout.addWidget(self.label)

    def set_image(self, path: str | None) -> None:
        self._path = path
        self._pix = None
        if not path or not Path(path).exists():
            self.label.setText("No image")
            self.label.setPixmap(QPixmap())
            return
        self._pix = QPixmap(self._path)
        self._render()

    def resizeEvent(self, event):  # noqa: N802
        super().resizeEvent(event)
        if self._path:
            self._render()

    def _render(self) -> None:
        if not self._path:
            return
        pix = self._pix if self._pix is not None else QPixmap(self._path)
        if pix.isNull():
            self.label.setText("Failed to load image")
            self.label.setPixmap(QPixmap())
            return
        scaled = pix.scaled(
            self.label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.label.setText("")
        self.label.setPixmap(scaled)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("PIFT Workbench (PySide6)")
        self.resize(1600, 980)

        self.run_manager = RunManager(output_root=ROOT / "outputs")
        self.worker_thread: QThread | None = None
        self.worker: RunWorker | None = None

        self.obs_csv_path: str | None = None
        self.obs_data = None

        self.method_checks: dict[str, QCheckBox] = {}
        self.config_inputs: dict[str, QLineEdit] = {}
        self._compare_plot_path: Path | None = None
        self._pinned_overlay_path: Path | None = None
        self._latest_compare_csv_path: Path | None = None
        self.run_rows: list[dict] = []
        self.latest_compare_stats: dict[str, float | str] | None = None
        self.baseline_run_id: str | None = None
        self._left_panel_visible = True
        self.section_boxes: list[QGroupBox] = []

        self.available_methods = {
            "PIFT (Phase A Forward Poisson)": True,
            "PINNs": False,
            "Bayesian PINNs": True,
            "Classical Monte Carlo": True,
            "MCMC (HMC/SGHMC)": False,
            "Gaussian Process Regression": False,
            "ODIL Baseline": True,
            "Finite-Difference Deterministic Solver": False,
        }

        self._build_ui()
        self.on_method_selection_changed()
        self.refresh_runs()

    def _build_ui(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)

        controls_row = QHBoxLayout()
        self.btn_toggle_controls = QPushButton("Hide Controls")
        self.btn_toggle_controls.clicked.connect(self.toggle_left_controls)
        self.btn_widen_controls = QPushButton("Widen Controls")
        self.btn_widen_controls.clicked.connect(self.widen_left_controls)
        self.btn_expand_all = QPushButton("Expand All")
        self.btn_expand_all.clicked.connect(self.expand_all_sections)
        self.btn_collapse_all = QPushButton("Collapse All")
        self.btn_collapse_all.clicked.connect(self.collapse_all_sections)
        controls_row.addWidget(self.btn_toggle_controls)
        controls_row.addWidget(self.btn_widen_controls)
        controls_row.addWidget(self.btn_expand_all)
        controls_row.addWidget(self.btn_collapse_all)
        controls_row.addStretch(1)
        layout.addLayout(controls_row)

        self.main_splitter = QSplitter(Qt.Horizontal)
        self.main_splitter.setChildrenCollapsible(False)
        self.main_splitter.setHandleWidth(10)
        layout.addWidget(self.main_splitter)

        self.left_scroll = QScrollArea()
        self.left_scroll.setWidgetResizable(True)
        self.left_scroll.setMinimumWidth(360)
        self.left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        left = QWidget()
        left.setMinimumWidth(340)
        left_layout = QVBoxLayout(left)
        self.left_scroll.setWidget(left)
        self.main_splitter.addWidget(self.left_scroll)

        right_tabs = QTabWidget()
        self.main_splitter.addWidget(right_tabs)
        self.main_splitter.setStretchFactor(0, 0)
        self.main_splitter.setStretchFactor(1, 1)
        self.main_splitter.setSizes([460, 1140])

        methods_box, methods_content, methods_layout = self._build_collapsible_section("Method Selection")
        methods_layout.addWidget(QLabel("Disabled methods are roadmap-only."))
        for name, built in self.available_methods.items():
            cb = QCheckBox(name + ("" if built else " (not built)"))
            cb.setChecked(built and "PIFT" in name)
            cb.setEnabled(built)
            cb.toggled.connect(self.on_method_selection_changed)
            methods_layout.addWidget(cb)
            self.method_checks[name] = cb
        left_layout.addWidget(methods_box)

        input_box, input_content, _input_layout = self._build_collapsible_section("Input Variables")
        form = QFormLayout(input_content)
        form.setLabelAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        for key, val in UI_DEFAULT_CONFIG.items():
            inp = QLineEdit(str(val))
            inp.setMinimumWidth(170)
            self.config_inputs[key] = inp
            form.addRow(key, inp)
        left_layout.addWidget(input_box)

        data_box, data_content, data_layout = self._build_collapsible_section("Data Source")
        self.data_label = QLabel("Source: synthetic")
        self.data_validation_label = QLabel("Validation: pending")
        data_layout.addWidget(self.data_label)
        btn_row = QHBoxLayout()
        self.btn_load_csv = QPushButton("Load CSV")
        self.btn_seed = QPushButton("Use Seed Data")
        self.btn_synth = QPushButton("Use Synthetic")
        self.btn_load_csv.clicked.connect(self.load_csv)
        self.btn_seed.clicked.connect(self.use_seed_data)
        self.btn_synth.clicked.connect(self.use_synthetic)
        btn_row.addWidget(self.btn_load_csv)
        btn_row.addWidget(self.btn_seed)
        btn_row.addWidget(self.btn_synth)
        data_layout.addLayout(btn_row)
        data_layout.addWidget(self.data_validation_label)
        left_layout.addWidget(data_box)

        device_box, _device_content, device_layout = self._build_collapsible_section("Compute Device")
        device_row = QHBoxLayout()
        self.cpu_radio = QRadioButton("CPU")
        self.gpu_radio = QRadioButton("GPU")
        self.cpu_radio.setChecked(True)
        device_row.addWidget(self.cpu_radio)
        device_row.addWidget(self.gpu_radio)
        device_layout.addLayout(device_row)
        left_layout.addWidget(device_box)

        exec_box, _exec_content, exec_layout = self._build_collapsible_section("Execution Controls")
        exec_row = QHBoxLayout()
        self.btn_run = QPushButton("Run")
        self.btn_stop = QPushButton("Stop")
        self.btn_dup = QPushButton("Duplicate Config")
        self.btn_reset = QPushButton("Reset Defaults")
        self.btn_run.clicked.connect(self.start_run)
        self.btn_stop.clicked.connect(self.stop_run)
        self.btn_dup.clicked.connect(self.duplicate_config)
        self.btn_reset.clicked.connect(self.reset_defaults)
        exec_row.addWidget(self.btn_run)
        exec_row.addWidget(self.btn_stop)
        exec_row.addWidget(self.btn_dup)
        exec_row.addWidget(self.btn_reset)
        exec_layout.addLayout(exec_row)
        left_layout.addWidget(exec_box)

        prog_box, _prog_content, prog_layout = self._build_collapsible_section("Run Progress")
        self.progress_status = QLabel("Idle")
        self.progress_meta = QLabel("0.0% | Elapsed 00:00:00")
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 1000)
        self.progress_log = QPlainTextEdit()
        self.progress_log.setReadOnly(True)
        self.progress_log.setMaximumBlockCount(500)
        prog_layout.addWidget(self.progress_status)
        prog_layout.addWidget(self.progress_meta)
        prog_layout.addWidget(self.progress_bar)
        prog_layout.addWidget(self.progress_log)
        left_layout.addWidget(prog_box)

        left_layout.addStretch(1)

        plots_tab = QWidget()
        plots_layout = QVBoxLayout(plots_tab)
        self.model_panel = ImagePanel("Model Overlay")
        plots_layout.addWidget(self.model_panel)
        right_tabs.addTab(plots_tab, "Plots")

        diag_tab = QWidget()
        diag_layout = QVBoxLayout(diag_tab)
        self.diag_text = QPlainTextEdit()
        self.diag_text.setReadOnly(True)
        diag_layout.addWidget(self.diag_text)
        right_tabs.addTab(diag_tab, "Diagnostics")

        runs_tab = QWidget()
        runs_layout = QVBoxLayout(runs_tab)

        runs_filter_row = QHBoxLayout()
        runs_filter_row.addWidget(QLabel("Status filter:"))
        self.run_status_filter = QComboBox()
        self.run_status_filter.addItems(["all", "queued", "running", "failed", "completed", "stopped"])
        self.run_status_filter.currentTextChanged.connect(lambda _: self.refresh_runs())
        runs_filter_row.addWidget(self.run_status_filter)
        runs_filter_row.addWidget(QLabel("Method filter:"))
        self.run_method_filter = QComboBox()
        self.run_method_filter.addItems(["all"] + list(self.available_methods.keys()))
        self.run_method_filter.currentTextChanged.connect(lambda _: self.refresh_runs())
        runs_filter_row.addWidget(self.run_method_filter)
        runs_filter_row.addWidget(QLabel("Quality:"))
        self.run_quality_filter = QComboBox()
        self.run_quality_filter.addItems(["all", "Stable", "Needs Review", "Failed"])
        self.run_quality_filter.currentTextChanged.connect(lambda _: self.refresh_runs())
        runs_filter_row.addWidget(self.run_quality_filter)
        runs_filter_row.addWidget(QLabel("Run ID contains:"))
        self.run_search = QLineEdit()
        self.run_search.setPlaceholderText("substring")
        self.run_search.textChanged.connect(lambda _: self.refresh_runs())
        runs_filter_row.addWidget(self.run_search)
        btn_refresh = QPushButton("Refresh")
        btn_refresh.clicked.connect(self.refresh_runs)
        runs_filter_row.addWidget(btn_refresh)
        btn_export_runs = QPushButton("Export Runs CSV")
        btn_export_runs.clicked.connect(self.export_runs_csv)
        runs_filter_row.addWidget(btn_export_runs)
        btn_export_bundle = QPushButton("Export Bundle")
        btn_export_bundle.clicked.connect(self.export_run_bundle)
        runs_filter_row.addWidget(btn_export_bundle)
        runs_filter_row.addStretch(1)
        runs_layout.addLayout(runs_filter_row)

        self.best_run_label = QLabel("Best completed run: n/a")
        runs_layout.addWidget(self.best_run_label)

        self.runs_table = QTableWidget(0, 7)
        self.runs_table.setHorizontalHeaderLabels(
            ["run_id", "status", "quality", "method", "device", "created_at", "runtime_sec"]
        )
        self.runs_table.itemSelectionChanged.connect(self.on_run_selected)
        self.runs_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.runs_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.runs_table.setSortingEnabled(True)
        runs_layout.addWidget(self.runs_table)

        compare_box = QGroupBox("Compare Runs")
        compare_layout = QHBoxLayout(compare_box)
        self.compare_a = QComboBox()
        self.compare_b = QComboBox()
        self.btn_compare = QPushButton("Compare")
        self.btn_compare.clicked.connect(self.compare_runs)
        self.btn_pin_baseline = QPushButton("Pin Baseline (Selected)")
        self.btn_pin_baseline.clicked.connect(self.pin_baseline_from_selection)
        self.btn_compare_vs_baseline = QPushButton("Compare Selected vs Baseline")
        self.btn_compare_vs_baseline.clicked.connect(self.compare_selected_vs_baseline)
        self.btn_export_compare = QPushButton("Export Compare CSV")
        self.btn_export_compare.clicked.connect(self.export_compare_csv)
        compare_layout.addWidget(self.compare_a)
        compare_layout.addWidget(self.compare_b)
        compare_layout.addWidget(self.btn_compare)
        compare_layout.addWidget(self.btn_pin_baseline)
        compare_layout.addWidget(self.btn_compare_vs_baseline)
        compare_layout.addWidget(self.btn_export_compare)
        runs_layout.addWidget(compare_box)

        self.baseline_label = QLabel("Baseline: n/a")
        runs_layout.addWidget(self.baseline_label)

        self.compare_text = QPlainTextEdit()
        self.compare_text.setReadOnly(True)
        self.compare_text.setMaximumHeight(150)
        runs_layout.addWidget(self.compare_text)
        right_tabs.addTab(runs_tab, "Runs")

        summary_tab = QWidget()
        summary_layout = QVBoxLayout(summary_tab)
        self.raw_toggle = QCheckBox("Show raw JSON")
        self.raw_toggle.toggled.connect(self.toggle_raw_summary)
        summary_layout.addWidget(self.raw_toggle)
        self.summary_interpreted = QPlainTextEdit()
        self.summary_interpreted.setReadOnly(True)
        self.summary_raw = QPlainTextEdit()
        self.summary_raw.setReadOnly(True)
        self.summary_raw.setVisible(False)
        summary_layout.addWidget(self.summary_interpreted)
        summary_layout.addWidget(self.summary_raw)
        right_tabs.addTab(summary_tab, "Summary")

    def _build_collapsible_section(self, title: str) -> tuple[QGroupBox, QWidget, QVBoxLayout]:
        box = QGroupBox(title)
        box.setCheckable(True)
        box.setChecked(True)
        outer = QVBoxLayout(box)
        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(content)
        box.toggled.connect(content.setVisible)
        self.section_boxes.append(box)
        return box, content, content_layout

    def toggle_left_controls(self) -> None:
        if self._left_panel_visible:
            self.left_scroll.hide()
            self._left_panel_visible = False
            self.btn_toggle_controls.setText("Show Controls")
            return
        self.left_scroll.show()
        self._left_panel_visible = True
        self.btn_toggle_controls.setText("Hide Controls")
        self.main_splitter.setSizes([460, 1140])

    def widen_left_controls(self) -> None:
        if not self._left_panel_visible:
            self.left_scroll.show()
            self._left_panel_visible = True
            self.btn_toggle_controls.setText("Hide Controls")
        self.main_splitter.setSizes([560, 1040])

    def expand_all_sections(self) -> None:
        for box in self.section_boxes:
            box.setChecked(True)

    def collapse_all_sections(self) -> None:
        for box in self.section_boxes:
            box.setChecked(False)

    def selected_methods(self) -> list[str]:
        return [name for name, cb in self.method_checks.items() if cb.isChecked() and cb.isEnabled()]

    def selected_method(self) -> str | None:
        chosen = self.selected_methods()
        return chosen[0] if chosen else None

    def parse_config(self) -> dict:
        int_keys = {"seed", "n_modes", "n_obs", "n_steps", "burn_in", "thin", "n_quad", "n_grid"}
        cfg = {}
        for k, inp in self.config_inputs.items():
            txt = inp.text().strip()
            cfg[k] = int(txt) if k in int_keys else float(txt)
        return cfg

    def load_csv(self) -> None:
        from PySide6.QtWidgets import QFileDialog

        path, _ = QFileDialog.getOpenFileName(self, "Select observation CSV", str(ROOT), "CSV Files (*.csv)")
        if not path:
            return
        self.obs_csv_path = path
        self.obs_data = None
        self.data_label.setText(f"Source: CSV ({path})")
        self.data_validation_label.setText("Validation: CSV selected (validated on run)")

    def use_seed_data(self) -> None:
        try:
            n_obs = int(self.config_inputs["n_obs"].text().strip())
            noise_std = float(self.config_inputs["noise_std"].text().strip())
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Input error", f"Invalid n_obs/noise_std: {exc}")
            return
        n_obs = max(4, n_obs)
        x = np.linspace(0.05, 0.95, n_obs)
        y_true = np.sin(np.pi * x) + 0.35 * np.sin(2.0 * np.pi * x)
        rng = np.random.default_rng(2024)
        y = y_true + noise_std * rng.normal(size=n_obs)
        self.obs_data = (x, y)
        self.obs_csv_path = None
        self.data_label.setText(f"Source: seed data ({n_obs} points)")
        errs = validate_observations(x, y)
        self.data_validation_label.setText("Validation: " + ("OK" if not errs else "; ".join(errs)))

    def use_synthetic(self) -> None:
        self.obs_data = None
        self.obs_csv_path = None
        self.data_label.setText("Source: synthetic")
        self.data_validation_label.setText("Validation: synthetic source")

    def reset_defaults(self) -> None:
        for k, v in UI_DEFAULT_CONFIG.items():
            self.config_inputs[k].setText(str(v))
        self.expand_all_sections()
        self.use_synthetic()

    def on_method_selection_changed(self) -> None:
        methods = self.selected_methods()
        if not methods:
            return
        if len(methods) == 1:
            self.apply_stable_preset(methods[0])
        else:
            self.apply_stable_preset("__batch__")

    def apply_stable_preset(self, method_key: str) -> None:
        preset = STABLE_PRESETS.get(method_key)
        if not preset:
            return
        for k, v in preset.items():
            if k in self.config_inputs:
                self.config_inputs[k].setText(str(v))
        self.expand_all_sections()

    def duplicate_config(self) -> None:
        row = self.runs_table.currentRow()
        if row < 0:
            QMessageBox.information(self, "Duplicate", "Select a run row first.")
            return
        run_id = self.runs_table.item(row, 0).text()
        rec = self.run_manager.storage.get_run(run_id)
        if rec is None:
            QMessageBox.critical(self, "Duplicate", "Could not load run record.")
            return
        cfg = rec.get("config", {})
        for k, inp in self.config_inputs.items():
            if k in cfg:
                inp.setText(str(cfg[k]))
        QMessageBox.information(self, "Duplicate", f"Copied config from run {run_id}")

    def set_progress(self, percent: float, stage: str, detail: str, elapsed: float) -> None:
        self.progress_status.setText("Running")
        self.progress_meta.setText(f"{percent:5.1f}% | Elapsed {self.to_hms(elapsed)}")
        self.progress_bar.setValue(int(max(0.0, min(100.0, percent)) * 10))
        self.progress_log.appendPlainText(f"{percent:5.1f}% | {self.to_hms(elapsed)} | {stage} | {detail}")

    def start_run(self) -> None:
        methods = self.selected_methods()
        if not methods:
            QMessageBox.critical(self, "Method", "Select an implemented method.")
            return

        try:
            cfg = self.parse_config()
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Input", f"Invalid config values: {exc}")
            return

        self.progress_log.clear()
        self.progress_status.setText("Queued")
        self.progress_meta.setText("0.0% | Elapsed 00:00:00")
        self.progress_bar.setValue(0)
        self.progress_log.appendPlainText(f"Queued run(s): {', '.join(methods)}")
        self.btn_run.setEnabled(False)

        device = "gpu" if self.gpu_radio.isChecked() else "cpu"

        self.worker_thread = QThread(self)
        self.worker = RunWorker(
            run_manager=self.run_manager,
            methods=methods,
            config=cfg,
            device=device,
            csv_path=self.obs_csv_path,
            obs_data=self.obs_data,
        )
        self.worker.moveToThread(self.worker_thread)
        self.worker_thread.started.connect(self.worker.run)
        self.worker.progress.connect(self.set_progress)
        self.worker.completed.connect(self.on_run_completed)
        self.worker.failed.connect(self.on_run_failed)
        self.worker.completed.connect(self.worker_thread.quit)
        self.worker.failed.connect(self.worker_thread.quit)
        self.worker_thread.finished.connect(lambda: self.btn_run.setEnabled(True))
        self.worker_thread.start()

    def stop_run(self) -> None:
        active = self.run_manager.get_active_run_id()
        if not active:
            QMessageBox.information(self, "Stop", "No active run.")
            return
        ok = self.run_manager.request_stop(active)
        if ok:
            self.progress_log.appendPlainText("Stop requested")
        else:
            QMessageBox.warning(self, "Stop", "Failed to request stop")

    def on_run_completed(self, result: dict) -> None:
        if isinstance(result.get("results"), list):
            batch_results = result["results"]
            batch_runtime = float(result.get("runtime_sec", 0.0))
            self.progress_status.setText("Completed")
            self.progress_meta.setText(f"100.0% | Elapsed {self.to_hms(batch_runtime)}")
            self.progress_bar.setValue(1000)
            self.progress_log.appendPlainText("Batch finished")
            for r in batch_results:
                self.progress_log.appendPlainText(
                    f"- {r.get('_batch_method')}: status={r.get('status')} run_id={r.get('run_id')}"
                )

            run_ids = [str(r.get("run_id")) for r in batch_results if r.get("run_id")]
            run_recs = [self.run_manager.storage.get_run(rid) for rid in run_ids]
            run_recs = [r for r in run_recs if r is not None]
            if run_recs:
                try:
                    out = self.build_multi_overlay(run_recs)
                    self._pinned_overlay_path = out
                    self.model_panel.set_image(str(out))
                except Exception as exc:  # noqa: BLE001
                    self.progress_log.appendPlainText(f"Overlay generation failed: {exc}")
            self.render_summary(result)
            self.render_diagnostics(batch_results[-1])
            self.refresh_runs()

            if len(run_ids) >= 2:
                self.compare_a.setCurrentText(run_ids[0])
                self.compare_b.setCurrentText(run_ids[1])
                self.compare_runs()
                self.progress_log.appendPlainText(
                    f"Auto-compare loaded: {run_ids[0][:8]}... vs {run_ids[1][:8]}..."
                )
            return

        status = str(result.get("status", "completed"))
        self.progress_status.setText(status.capitalize())
        metrics = result.get("metrics", {})
        runtime = float(metrics.get("runtime_sec", 0.0))
        self.progress_meta.setText(f"100.0% | Elapsed {self.to_hms(runtime)}")
        self.progress_bar.setValue(1000)
        self.progress_log.appendPlainText(f"Run finished with status: {status}")

        artifacts = result.get("artifacts", {})
        self._pinned_overlay_path = None
        self.model_panel.set_image(artifacts.get("field_plot"))

        self.render_summary(result)
        self.render_diagnostics(result)
        self.refresh_runs()

    def on_run_failed(self, error: str) -> None:
        self.progress_status.setText("Failed")
        self.progress_log.appendPlainText(f"ERROR: {error}")
        QMessageBox.critical(self, "Run failed", error)
        self.refresh_runs()

    def render_summary(self, result: dict) -> None:
        if isinstance(result.get("results"), list):
            lines = [
                "Batch Run Summary",
                f"Total methods: {len(result['results'])}",
                f"Runtime (sec): {result.get('runtime_sec')}",
            ]
            for r in result["results"]:
                m = r.get("_batch_method", "method")
                s = r.get("status")
                rid = r.get("run_id")
                rm = r.get("metrics", {})
                lines.append(f"{m}: status={s}, run_id={rid}, runtime_sec={rm.get('runtime_sec')}")
            self.summary_interpreted.setPlainText("\n".join(lines))
            self.summary_raw.setPlainText(json.dumps(result, indent=2, default=str))
            return

        summary = result.get("summary", {})
        metrics = result.get("metrics", {})
        lines = [
            "Run Summary",
            f"Run ID: {result.get('run_id')}",
            f"Status: {result.get('status')}",
            f"Device: {summary.get('device_used')} (requested {summary.get('device_requested')})",
            f"Runtime (sec): {metrics.get('runtime_sec')}",
            f"Posterior samples: {summary.get('n_posterior_samples')}",
            f"L2 error: {summary.get('l2_error')}",
            f"Max error: {summary.get('max_error')}",
            f"Coverage (90%): {summary.get('measurement_interval_coverage_90pct')}",
        ]
        self.summary_interpreted.setPlainText("\n".join(lines))
        self.summary_raw.setPlainText(json.dumps(result, indent=2, default=str))

    def render_diagnostics(self, result: dict) -> None:
        self.diag_text.setPlainText(json.dumps(result.get("diagnostics", {}), indent=2, default=str))

    def toggle_raw_summary(self, checked: bool) -> None:
        self.summary_raw.setVisible(checked)

    def refresh_runs(self) -> None:
        filt = {}
        if self.run_status_filter.currentText() != "all":
            filt["status"] = self.run_status_filter.currentText()
        if self.run_method_filter.currentText() != "all":
            filt["method"] = self.run_method_filter.currentText()

        rows = self.run_manager.storage.list_runs_with_metrics_summary(filters=filt)
        enriched_rows: list[dict] = []
        search_txt = self.run_search.text().strip().lower()
        quality_filter = self.run_quality_filter.currentText()

        for r in rows:
            quality = self.compute_quality_label(r)
            merged = {**r, "quality": quality}
            if search_txt and search_txt not in str(r.get("run_id", "")).lower():
                continue
            if quality_filter != "all" and quality != quality_filter:
                continue
            enriched_rows.append(merged)

        self.run_rows = enriched_rows
        self.runs_table.setSortingEnabled(False)
        self.runs_table.setRowCount(len(enriched_rows))
        for i, r in enumerate(enriched_rows):
            vals = [
                r["run_id"],
                r["status"],
                r["quality"],
                r["method"],
                r.get("device_used") or r.get("device_requested") or "",
                r["created_at"],
                str(r.get("runtime_sec") or ""),
            ]
            for j, v in enumerate(vals):
                self.runs_table.setItem(i, j, QTableWidgetItem(str(v)))
        self.runs_table.setSortingEnabled(True)

        run_ids = [r["run_id"] for r in enriched_rows]
        self.compare_a.clear()
        self.compare_b.clear()
        self.compare_a.addItems(run_ids)
        self.compare_b.addItems(run_ids)
        self._refresh_best_run_badge(enriched_rows)

    def _refresh_best_run_badge(self, rows: list[dict]) -> None:
        best: tuple[str, float] | None = None
        for r in rows:
            if r.get("status") != "completed":
                continue
            try:
                ref_l2 = float(r.get("reference_l2_error"))
            except Exception:  # noqa: BLE001
                continue
            if np.isnan(ref_l2):
                continue
            if best is None or ref_l2 < best[1]:
                best = (r["run_id"], ref_l2)

        if best is None:
            self.best_run_label.setText("Best completed run: n/a")
        else:
            self.best_run_label.setText(f"Best completed run: {best[0][:8]}... | reference_l2_error={best[1]:.6f}")

    def compute_quality_label(self, rec: dict) -> str:
        status = str(rec.get("status", ""))
        if status in {"failed", "stopped"}:
            return "Failed"
        if status != "completed":
            return "Needs Review"

        try:
            ref_l2 = float(rec.get("reference_l2_error"))
        except Exception:  # noqa: BLE001
            ref_l2 = float("nan")
        try:
            coverage = float(rec.get("coverage_90"))
        except Exception:  # noqa: BLE001
            coverage = float("nan")

        unstable = any(
            bool(rec.get(k)) for k in ["nan_detected", "inf_detected", "divergence_detected", "hamiltonian_overflow"]
        )
        if unstable:
            return "Needs Review"
        if np.isnan(ref_l2) or np.isnan(coverage):
            return "Needs Review"
        if ref_l2 < 0.20 and 0.60 <= coverage <= 1.0:
            return "Stable"
        return "Needs Review"

    def on_run_selected(self) -> None:
        row = self.runs_table.currentRow()
        if row < 0:
            return
        run_id = self.runs_table.item(row, 0).text()
        rec = self.run_manager.storage.get_run(run_id)
        if rec is None:
            return

        # Keep the current overlay pinned so tab switching/row selection does not
        # replace the chart unexpectedly while reviewing or taking screenshots.
        if self._pinned_overlay_path is None:
            artifacts = rec.get("artifacts", {})
            self.model_panel.set_image(artifacts.get("field_plot"))

        self.summary_interpreted.setPlainText(
            "\n".join(
                [
                    "Run Summary",
                    f"Run ID: {rec.get('run_id')}",
                    f"Status: {rec.get('status')}",
                    f"Method: {rec.get('method')}",
                    f"Device: {rec.get('device_used')} (requested {rec.get('device_requested')})",
                    f"Runtime (sec): {rec.get('runtime_sec')}",
                ]
            )
        )
        self.summary_raw.setPlainText(json.dumps(rec, indent=2, default=str))

        metrics = rec.get("metrics", {})
        diag = metrics.get("diagnostics", {}) if isinstance(metrics, dict) else {}
        self.diag_text.setPlainText(json.dumps(diag or metrics, indent=2, default=str))

    def pin_baseline_from_selection(self) -> None:
        row = self.runs_table.currentRow()
        if row < 0:
            QMessageBox.information(self, "Baseline", "Select a run first.")
            return
        run_id = self.runs_table.item(row, 0).text()
        self.baseline_run_id = run_id
        self.baseline_label.setText(f"Baseline: {run_id}")

    def compare_selected_vs_baseline(self) -> None:
        if not self.baseline_run_id:
            QMessageBox.information(self, "Baseline", "Pin a baseline first.")
            return
        row = self.runs_table.currentRow()
        if row < 0:
            QMessageBox.information(self, "Compare", "Select a run row to compare.")
            return
        selected_run = self.runs_table.item(row, 0).text()
        if selected_run == self.baseline_run_id:
            QMessageBox.information(self, "Compare", "Selected run is the baseline.")
            return
        self.compare_a.setCurrentText(self.baseline_run_id)
        self.compare_b.setCurrentText(selected_run)
        self.compare_runs()

    def compare_runs(self) -> None:
        run_a = self.compare_a.currentText().strip()
        run_b = self.compare_b.currentText().strip()
        if not run_a or not run_b:
            QMessageBox.critical(self, "Compare", "Select two runs")
            return
        if run_a == run_b:
            QMessageBox.critical(self, "Compare", "Select two distinct runs")
            return

        rec_a = self.run_manager.storage.get_run(run_a)
        rec_b = self.run_manager.storage.get_run(run_b)
        if rec_a is None or rec_b is None:
            QMessageBox.critical(self, "Compare", "Failed to load selected runs")
            return

        m_a = rec_a.get("metrics", {})
        m_b = rec_b.get("metrics", {})

        def _num(v):
            try:
                return float(v)
            except Exception:  # noqa: BLE001
                return float("nan")

        l2_a = _num(m_a.get("l2_error") if isinstance(m_a, dict) else None)
        l2_b = _num(m_b.get("l2_error") if isinstance(m_b, dict) else None)
        ref_l2_a = _num(m_a.get("reference_l2_error") if isinstance(m_a, dict) else None)
        ref_l2_b = _num(m_b.get("reference_l2_error") if isinstance(m_b, dict) else None)
        cov_a = _num(m_a.get("measurement_interval_coverage_90pct") if isinstance(m_a, dict) else None)
        cov_b = _num(m_b.get("measurement_interval_coverage_90pct") if isinstance(m_b, dict) else None)
        rt_a = _num(rec_a.get("runtime_sec"))
        rt_b = _num(rec_b.get("runtime_sec"))

        stats = {
            "run_a": run_a,
            "run_b": run_b,
            "l2_delta_b_minus_a": l2_b - l2_a,
            "reference_l2_delta_b_minus_a": ref_l2_b - ref_l2_a,
            "coverage_delta_b_minus_a": cov_b - cov_a,
            "runtime_delta_sec_b_minus_a": rt_b - rt_a,
        }
        self.latest_compare_stats = stats

        self.compare_text.setPlainText(
            "\n".join(
                [
                    f"Compare {run_a} vs {run_b}",
                    f"L2 delta (B-A): {stats['l2_delta_b_minus_a']}",
                    f"Reference L2 delta (B-A): {stats['reference_l2_delta_b_minus_a']}",
                    f"Coverage delta (B-A): {stats['coverage_delta_b_minus_a']}",
                    f"Runtime delta sec (B-A): {stats['runtime_delta_sec_b_minus_a']}",
                ]
            )
        )

        try:
            out = self.build_multi_overlay([rec_a, rec_b])
            self._pinned_overlay_path = out
            self.model_panel.set_image(str(out))
        except Exception as exc:  # noqa: BLE001
            self.compare_text.appendPlainText(f"\nOverlay generation failed: {exc}")

    def build_multi_overlay(self, records: list[dict]) -> Path:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        x = np.linspace(0.0, 1.0, 300)
        truth = np.sin(np.pi * x) + 0.35 * np.sin(2.0 * np.pi * x)

        out_dir = ROOT / "outputs" / "compare"
        out_dir.mkdir(parents=True, exist_ok=True)
        run_sig = "_".join([str(r.get("run_id", ""))[:8] for r in records[:4]])
        out_path = out_dir / f"overlay_{run_sig}.png"

        plt.figure(figsize=(10, 4.2))
        cmap = plt.get_cmap("tab10")
        first_obs_plotted = False
        plt.plot(x, truth, color="black", linewidth=2.2, linestyle="--", label="ground truth")

        for idx, rec in enumerate(records):
            chain_path = rec.get("artifacts", {}).get("chain")
            if not chain_path:
                continue
            npz = np.load(chain_path)
            cfg = rec.get("config", {})
            n_modes = int(cfg.get("n_modes", 12))
            burn = int(cfg.get("burn_in", 0))
            thin = int(cfg.get("thin", 1))
            chain = npz["chain"][burn::max(1, thin)]
            basis = np.asarray(SineBasisField(n_modes).design_matrix(x))
            color = cmap(idx % 10)
            if chain.size:
                phi_samples = chain @ basis.T
                mean = np.mean(phi_samples, axis=0)
                low = np.quantile(phi_samples, 0.05, axis=0)
                high = np.quantile(phi_samples, 0.95, axis=0)
                plt.fill_between(
                    x,
                    low,
                    high,
                    color=color,
                    alpha=0.14,
                    linewidth=0.0,
                    label=f"{str(rec.get('method', 'method'))} 5-95%",
                )
            else:
                mean = np.full(x.shape, np.nan)
            method = str(rec.get("method", "method"))
            run_id = str(rec.get("run_id", ""))[:8]
            plt.plot(x, mean, color=color, linewidth=2.0, label=f"{method} mean ({run_id})")

            if not first_obs_plotted and "x_obs" in npz and "y_obs" in npz:
                x_obs = np.asarray(npz["x_obs"], dtype=float)
                y_obs = np.asarray(npz["y_obs"], dtype=float)
                plt.scatter(
                    x_obs,
                    y_obs,
                    s=26,
                    marker="x",
                    color="tab:red",
                    alpha=0.9,
                    label="measurements",
                    zorder=3,
                )
                first_obs_plotted = True

        plt.title("Model Posterior Overlay (Means + 5-95% Bands)")
        plt.xlabel("x")
        plt.ylabel("phi(x)")
        plt.legend(loc="best", fontsize=9)
        plt.tight_layout()
        plt.savefig(out_path, dpi=160)
        plt.close()

        self._compare_plot_path = out_path
        return out_path

    def export_runs_csv(self) -> None:
        export_dir = ROOT / "outputs" / "exports"
        export_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = export_dir / f"runs_{self.run_status_filter.currentText()}_{ts}.csv"

        cols = [
            "run_id",
            "status",
            "quality",
            "method",
            "device",
            "created_at",
            "runtime_sec",
            "l2_error",
            "reference_l2_error",
            "reference_max_error",
            "measurement_interval_coverage_90pct",
        ]
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=cols)
            writer.writeheader()
            for r in self.run_rows:
                writer.writerow(
                    {
                        "run_id": r.get("run_id"),
                        "status": r.get("status"),
                        "quality": r.get("quality"),
                        "method": r.get("method"),
                        "device": r.get("device_used") or r.get("device_requested") or "",
                        "created_at": r.get("created_at"),
                        "runtime_sec": r.get("runtime_sec"),
                        "l2_error": r.get("l2_error"),
                        "reference_l2_error": r.get("reference_l2_error"),
                        "reference_max_error": "",
                        "measurement_interval_coverage_90pct": r.get("coverage_90"),
                    }
                )

        QMessageBox.information(self, "Export", f"Runs CSV exported:\n{out_path}")

    def export_compare_csv(self) -> None:
        if not self.latest_compare_stats:
            QMessageBox.information(self, "Export", "Run a comparison first.")
            return
        export_dir = ROOT / "outputs" / "exports"
        export_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = export_dir / (
            f"compare_{self.latest_compare_stats['run_a'][:8]}_{self.latest_compare_stats['run_b'][:8]}_{ts}.csv"
        )
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "run_a",
                    "run_b",
                    "l2_delta_b_minus_a",
                    "reference_l2_delta_b_minus_a",
                    "coverage_delta_b_minus_a",
                    "runtime_delta_sec_b_minus_a",
                ],
            )
            writer.writeheader()
            writer.writerow(self.latest_compare_stats)
        self._latest_compare_csv_path = out_path
        QMessageBox.information(self, "Export", f"Compare CSV exported:\n{out_path}")

    def export_run_bundle(self) -> None:
        row = self.runs_table.currentRow()
        if row < 0:
            QMessageBox.information(self, "Export bundle", "Select a run first.")
            return
        run_id = self.runs_table.item(row, 0).text()
        rec = self.run_manager.storage.get_run(run_id)
        if rec is None:
            QMessageBox.critical(self, "Export bundle", "Failed to load selected run.")
            return

        export_dir = ROOT / "outputs" / "exports"
        export_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        bundle_path = export_dir / f"run_bundle_{run_id[:8]}_{ts}.zip"

        summary_json = json.dumps(rec, indent=2, default=str)
        compare_csv_path = self._latest_compare_csv_path

        with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("run_record.json", summary_json)
            artifacts = rec.get("artifacts", {})
            for art_name, art_path in artifacts.items():
                p = Path(str(art_path))
                if p.exists() and p.is_file():
                    zf.write(p, arcname=f"artifacts/{art_name}_{p.name}")
            if compare_csv_path and Path(compare_csv_path).exists():
                zf.write(compare_csv_path, arcname=f"exports/{Path(compare_csv_path).name}")

        QMessageBox.information(self, "Export bundle", f"Bundle exported:\n{bundle_path}")

    def to_hms(self, elapsed: float) -> str:
        dt = int(max(0.0, elapsed))
        hh = dt // 3600
        mm = (dt % 3600) // 60
        ss = dt % 60
        return f"{hh:02d}:{mm:02d}:{ss:02d}"


def main() -> None:
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
