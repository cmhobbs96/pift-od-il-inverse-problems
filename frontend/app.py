#!/usr/bin/env python3
"""PIFT Research Workbench — PySide6 GUI for all Phase 1 examples."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

# Set matplotlib backend BEFORE any matplotlib imports
import matplotlib
matplotlib.use("QtAgg")

import numpy as np
from PySide6.QtCore import QObject, Qt, QThread, Signal
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

# Ensure both the project root (for `frontend.*`) and src/ (for `run_manager` etc.) are on the path
_root = str(Path(__file__).resolve().parent.parent)
_src = str(Path(__file__).resolve().parent.parent / "src")
for _p in (_root, _src):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from frontend.registry import ExampleSpec, get_example
from frontend.style import GLOBAL_QSS
from frontend.widgets.sidebar import SidebarWidget
from frontend.widgets.plot_grid import PlotGridWidget
from frontend.widgets.bottom_strip import BottomStripWidget
from frontend.widgets.diagnostics_panel import DiagnosticsPanel
from frontend.widgets.compare_panel import ComparePanel
from frontend.widgets.export_panel import ExportPanel
from frontend.widgets.runs_panel import RunsPanel

from run_manager import RunManager


# ---------------------------------------------------------------------------
# RunWorker — executes pipeline in a background QThread
# ---------------------------------------------------------------------------

class RunWorker(QObject):
    """Runs one or more pipelines in a background thread."""

    progress = Signal(float, str, str, float)  # percent, stage, detail, elapsed
    completed = Signal(dict)  # full result dict (or batch result)
    failed = Signal(str)  # error message

    def __init__(
        self,
        run_manager: RunManager,
        example_key: str,
        methods: list[str | None],
        config: dict[str, Any],
        device: str,
        obs_info: dict[str, Any],
    ):
        super().__init__()
        self.run_manager = run_manager
        self.example_key = example_key
        self.methods = methods
        self.config = config
        self.device = device
        self.obs_info = obs_info

    def run(self):
        try:
            obs_csv = self.obs_info.get("csv_path")
            obs_data = self.obs_info.get("data")

            if len(self.methods) == 1:
                result = self.run_manager.run_example(
                    example_key=self.example_key,
                    method_key=self.methods[0],
                    config=self.config,
                    device_preference=self.device,
                    obs_csv_path=obs_csv,
                    obs_data=obs_data,
                    progress_callback=self._on_progress,
                )
                result["config"] = self.config
                self.completed.emit(result)
            else:
                all_results = []
                for i, method_key in enumerate(self.methods):
                    def _progress(pct, stage, detail, elapsed, m=method_key, idx=i):
                        total_pct = (idx * 100 + pct) / len(self.methods)
                        self.progress.emit(total_pct, stage, f"[{m}] {detail}", elapsed)

                    result = self.run_manager.run_example(
                        example_key=self.example_key,
                        method_key=method_key,
                        config=self.config,
                        device_preference=self.device,
                        obs_csv_path=obs_csv,
                        obs_data=obs_data,
                        progress_callback=_progress,
                    )
                    result["config"] = self.config
                    all_results.append(result)

                batch = {
                    "status": "completed",
                    "results": all_results,
                    "config": self.config,
                }
                self.completed.emit(batch)

        except Exception as exc:
            self.failed.emit(str(exc))

    def _on_progress(self, percent, stage, detail, elapsed):
        self.progress.emit(percent, stage, detail, elapsed)


# ---------------------------------------------------------------------------
# MainWindow
# ---------------------------------------------------------------------------

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PIFT Research Workbench")
        self.resize(1400, 900)
        self.setMinimumSize(900, 600)

        self.run_manager = RunManager(output_root="outputs")
        self._worker: RunWorker | None = None
        self._thread: QThread | None = None
        self._last_result: dict[str, Any] = {}
        self._baseline_run_id: str | None = None

        self._build_ui()
        self._connect_signals()
        self._load_initial_state()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # -- Top header bar --
        header = QWidget()
        header.setObjectName("topHeader")
        header.setFixedHeight(36)
        header_l = QHBoxLayout(header)
        header_l.setContentsMargins(12, 4, 12, 4)
        title = QLabel("PIFT Research Workbench")
        title.setStyleSheet("font-weight: 600; font-size: 12px;")
        header_l.addWidget(title)
        header_l.addStretch()
        self._btn_restart = QPushButton("\u21bb Restart GUI")
        self._btn_restart.setToolTip(
            "Quit and relaunch the workbench from scratch (in-progress runs will be aborted)."
        )
        self._btn_restart.setFixedHeight(26)
        self._btn_restart.setStyleSheet("font-size: 11px; padding: 2px 10px;")
        self._btn_restart.clicked.connect(self._restart_gui)
        header_l.addWidget(self._btn_restart)
        main_layout.addWidget(header)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left sidebar (fixed 220px)
        self.sidebar = SidebarWidget()
        splitter.addWidget(self.sidebar)

        # Right: tabs + bottom strip
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)

        self.tabs = QTabWidget()

        self.plot_grid = PlotGridWidget()
        self.tabs.addTab(self.plot_grid, "Plots")

        self.diagnostics_panel = DiagnosticsPanel()
        self.tabs.addTab(self.diagnostics_panel, "Diagnostics")

        self.runs_panel = RunsPanel()
        self.tabs.addTab(self.runs_panel, "Runs")

        self.compare_panel = ComparePanel()
        self.tabs.addTab(self.compare_panel, "Compare")

        self.export_panel = ExportPanel()
        self.tabs.addTab(self.export_panel, "Export")

        right_layout.addWidget(self.tabs, 1)

        self.bottom_strip = BottomStripWidget()
        right_layout.addWidget(self.bottom_strip)

        splitter.addWidget(right)
        splitter.setSizes([260, 1140])
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        main_layout.addWidget(splitter)
        self.setStyleSheet(GLOBAL_QSS)

    def _connect_signals(self):
        self.sidebar.example_changed.connect(self._on_example_changed)
        self.sidebar.run_requested.connect(self._start_run)
        self.sidebar.stop_requested.connect(self._stop_run)
        self.sidebar.reset_requested.connect(self._on_reset)

        # Bottom strip buttons
        self.bottom_strip.compare_clicked.connect(
            lambda: self.tabs.setCurrentWidget(self.compare_panel)
        )
        self.bottom_strip.pin_baseline.connect(self._on_pin_baseline)
        self.bottom_strip._btn_pin.clicked.connect(self._pin_from_history)

    def _load_initial_state(self):
        """Populate run history from SQLite on startup."""
        spec = self.sidebar.current_example
        if spec:
            self.plot_grid.set_example(spec.key)

        self.runs_panel.set_storage(self.run_manager.storage)
        self.runs_panel.refresh()

        try:
            runs = self.run_manager.storage.list_runs_with_metrics_summary()
            self.bottom_strip.update_history(runs[:6])
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Example change
    # ------------------------------------------------------------------

    def _on_example_changed(self, spec: ExampleSpec):
        self.plot_grid.set_example(spec.key)

    # ------------------------------------------------------------------
    # Run lifecycle
    # ------------------------------------------------------------------

    def _start_run(self, example_key: str, methods: list, config: dict,
                   device: str, obs_info: object):
        if self._thread and self._thread.isRunning():
            return

        self.sidebar.set_running()
        self.sidebar.append_log(f"Starting {example_key} with methods={methods}")

        worker = RunWorker(
            self.run_manager, example_key, methods, config, device, obs_info
        )
        thread = QThread()
        worker.moveToThread(thread)

        thread.started.connect(worker.run)
        worker.progress.connect(self._on_progress)
        worker.completed.connect(self._on_completed)
        worker.failed.connect(self._on_failed)
        worker.completed.connect(thread.quit)
        worker.failed.connect(thread.quit)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._on_thread_finished)

        self._worker = worker
        self._thread = thread
        thread.start()

    def _stop_run(self):
        run_id = self.run_manager.get_active_run_id()
        if run_id:
            self.run_manager.request_stop(run_id)
            self.sidebar.append_log("Stop requested.")
        else:
            self.sidebar.append_log("No active run to stop.")

    def _on_reset(self):
        """Clear the workbench view back to a clean slate (no run loaded)."""
        if self._thread and self._thread.isRunning():
            return
        self._last_result = {}
        self._baseline_run_id = None
        spec = self.sidebar.current_example
        if spec:
            # set_example() resets each cell back to its placeholder state.
            self.plot_grid.set_example(spec.key)
        else:
            for cell in self.plot_grid.cells:
                cell.clear()

    def _restart_gui(self):
        """Quit the current process and relaunch with the same arguments."""
        # Best-effort: ask any active run to stop before tearing down.
        try:
            run_id = self.run_manager.get_active_run_id()
            if run_id:
                self.run_manager.request_stop(run_id)
        except Exception:
            pass
        if self._thread and self._thread.isRunning():
            self._thread.quit()
            self._thread.wait(2000)
        QApplication.quit()
        os.execv(sys.executable, [sys.executable, *sys.argv])

    def _on_thread_finished(self):
        """Drop references to the finished worker/thread so we don't touch deleted C++ objects."""
        self._thread = None
        self._worker = None

    def _on_progress(self, percent: float, stage: str, detail: str, elapsed: float):
        self.sidebar.set_progress(percent, stage, detail, elapsed)

    def _on_completed(self, result: dict[str, Any]):
        self._last_result = result
        self.sidebar.set_idle("Completed")
        self.sidebar.append_log("Run completed.")

        spec = self.sidebar.current_example
        example_key = spec.key if spec else "phase_a_forward"

        # Update 2x2 plot grid
        self.plot_grid.update_plots(example_key, result)

        # Update bottom strip metrics
        first = result["results"][0] if "results" in result else result
        self.bottom_strip.update_metrics(first, example_key)

        # Update diagnostics tab
        self.diagnostics_panel.update_diagnostics(first)

        # Refresh run history in bottom strip and runs tab
        self.runs_panel.refresh()
        try:
            runs = self.run_manager.storage.list_runs_with_metrics_summary()
            self.bottom_strip.update_history(runs[:6])
        except Exception:
            pass

        # Update export panel with current figures
        config = result.get("config", {})
        self.export_panel.set_figures(self.plot_grid.get_figures())
        self.export_panel.set_result(result, config, example_key)

        # Update compare panel
        if "results" in result:
            self.compare_panel.update_overlay(result["results"])

    def _on_failed(self, error_msg: str):
        self.sidebar.set_idle("Failed")
        self.sidebar.append_log(f"ERROR: {error_msg}")
        self.runs_panel.refresh()
        try:
            runs = self.run_manager.storage.list_runs_with_metrics_summary()
            self.bottom_strip.update_history(runs[:6])
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Baseline pinning / comparison
    # ------------------------------------------------------------------

    def _pin_from_history(self):
        """Pin the most recent completed run as baseline."""
        try:
            runs = self.run_manager.storage.list_runs_with_metrics_summary(
                {"status": "completed"}
            )
            if runs:
                self._on_pin_baseline(runs[0]["run_id"])
        except Exception:
            pass

    def _on_pin_baseline(self, run_id: str):
        self._baseline_run_id = run_id
        self.sidebar.append_log(f"Baseline pinned: {run_id[:8]}")

        # Feed completed runs to compare panel
        try:
            runs = self.run_manager.storage.list_runs_with_metrics_summary(
                {"status": "completed"}
            )
            enriched = []
            for r in runs:
                full = self.run_manager.storage.get_run(r["run_id"])
                if full:
                    enriched.append(full)
            self.compare_panel.set_runs(enriched)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    root_path = str(Path(__file__).resolve().parent.parent)
    src_path = str(Path(__file__).resolve().parent.parent / "src")
    for p in (root_path, src_path):
        if p not in sys.path:
            sys.path.insert(0, p)

    app = QApplication(sys.argv)
    app.setApplicationName("PIFT Research Workbench")

    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
