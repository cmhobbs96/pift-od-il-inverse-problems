"""Left sidebar widget: example selector, methods, config, data source, progress."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSizePolicy,
    QToolTip,
    QVBoxLayout,
    QWidget,
)

from frontend.registry import (
    EXAMPLE_DISPLAY_NAMES,
    EXAMPLE_KEYS,
    ExampleSpec,
    MethodSpec,
    get_example,
    get_example_by_index,
)
from frontend.style import COLORS, badge_qss


class _ValidatedLineEdit(QLineEdit):
    """QLineEdit that validates numeric input on focus-out and shows warnings."""

    def __init__(self, key: str, text: str, default_val: Any, parent=None):
        super().__init__(text, parent)
        self._key = key
        self._default_val = default_val
        self._warning_label: QLabel | None = None
        self.setStyleSheet(
            "font-family: 'Courier New', monospace; font-size: 11px;"
        )
        self.setFixedHeight(24)

    def set_warning_label(self, lbl: QLabel):
        self._warning_label = lbl

    def focusOutEvent(self, event):
        super().focusOutEvent(event)
        self._validate()

    def _validate(self):
        raw = self.text().strip()
        warn = ""
        if isinstance(self._default_val, (int, float)) and raw:
            try:
                val = float(raw)
                if self._key == "step_size0" and val > 0.01:
                    warn = "step_size0 > 0.01 may cause instability"
                elif self._key == "n_steps" and val < 1000:
                    warn = "n_steps < 1000 may be too few samples"
            except ValueError:
                warn = "non-numeric value"

        if self._warning_label:
            self._warning_label.setText(warn)
            self._warning_label.setVisible(bool(warn))

        if warn:
            self.setStyleSheet(
                "font-family: 'Courier New', monospace; font-size: 11px; "
                f"border: 1px solid {COLORS['amber_text']};"
            )
        else:
            self.setStyleSheet(
                "font-family: 'Courier New', monospace; font-size: 11px;"
            )


class SidebarWidget(QWidget):
    """Left panel with example selector, methods, config, data, device, and progress."""

    # Signals
    example_changed = Signal(object)  # emits ExampleSpec
    run_requested = Signal(str, list, dict, str, object)  # example_key, methods, config, device, obs_info
    stop_requested = Signal()
    reset_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("sidebar")
        self.setFixedWidth(260)

        self._current_spec: ExampleSpec | None = None
        self._method_checks: list[tuple[MethodSpec, QCheckBox]] = []
        self._config_inputs: dict[str, _ValidatedLineEdit] = {}
        self._obs_csv_path: str | None = None
        self._obs_data: tuple[np.ndarray, np.ndarray] | None = None

        self._build_ui()
        # Initialize with first example
        self._on_example_changed(0)

    def _build_ui(self):
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        container = QWidget()
        self._layout = QVBoxLayout(container)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(6)

        # -- Example selector --
        sec = self._make_section("Example")
        self._example_combo = QComboBox()
        self._example_combo.addItems(EXAMPLE_DISPLAY_NAMES)
        self._example_combo.currentIndexChanged.connect(self._on_example_changed)
        sec.layout().addWidget(self._example_combo)
        self._layout.addWidget(sec)

        # -- Methods --
        self._methods_section = self._make_section("Methods")
        self._methods_container = QVBoxLayout()
        self._methods_section.layout().addLayout(self._methods_container)
        self._layout.addWidget(self._methods_section)

        # -- Config --
        self._config_section = self._make_section("Sampler config")
        self._config_form = QFormLayout()
        self._config_form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        self._config_section.layout().addLayout(self._config_form)
        self._config_warn = QLabel("")
        self._config_warn.setStyleSheet(
            f"font-size: 10px; color: {COLORS['amber_text']};"
        )
        self._config_warn.setVisible(False)
        self._config_section.layout().addWidget(self._config_warn)
        self._layout.addWidget(self._config_section)

        # -- Data source --
        self._data_section = self._make_section("Data source")
        data_btns = QHBoxLayout()
        self._btn_synthetic = QPushButton("Synthetic")
        self._btn_synthetic.setObjectName("btn-primary")
        self._btn_synthetic.clicked.connect(self._use_synthetic)
        self._btn_csv = QPushButton("CSV")
        self._btn_csv.clicked.connect(self._load_csv)
        self._btn_seed = QPushButton("Seed")
        self._btn_seed.clicked.connect(self._use_seed)
        for b in (self._btn_synthetic, self._btn_csv, self._btn_seed):
            b.setFixedHeight(26)
            b.setStyleSheet("font-size: 11px; padding: 4px 8px;")
            data_btns.addWidget(b)
        self._data_section.layout().addLayout(data_btns)
        self._data_label = QLabel("Synthetic observations")
        self._data_label.setStyleSheet("font-size: 11px; color: #888780;")
        self._data_section.layout().addWidget(self._data_label)
        self._layout.addWidget(self._data_section)

        # -- Device --
        sec = self._make_section("Compute device")
        dev_row = QHBoxLayout()
        self._auto_radio = QRadioButton("Auto")
        self._cpu_radio = QRadioButton("CPU")
        self._gpu_radio = QRadioButton("GPU")
        self._cpu_radio.setChecked(True)
        self._gpu_radio.toggled.connect(self._on_gpu_toggled)
        dev_row.addWidget(self._auto_radio)
        dev_row.addWidget(self._cpu_radio)
        dev_row.addWidget(self._gpu_radio)
        dev_row.addStretch()
        sec.layout().addLayout(dev_row)

        backend_row = QHBoxLayout()
        backend_row.addWidget(QLabel("Backend:"))
        self._backend_combo = QComboBox()
        self._backend_combo.addItems(["local", "modal"])
        backend_row.addWidget(self._backend_combo)
        backend_row.addStretch()
        sec.layout().addLayout(backend_row)
        self._layout.addWidget(sec)

        # -- Run/Stop/Reset --
        btn_row = QHBoxLayout()
        btn_row.setContentsMargins(12, 8, 12, 8)
        self._btn_run = QPushButton("Run")
        self._btn_run.setObjectName("btn-primary")
        self._btn_run.clicked.connect(self._emit_run)
        self._btn_stop = QPushButton("Stop")
        self._btn_stop.clicked.connect(self.stop_requested.emit)
        self._btn_reset = QPushButton("Reset")
        self._btn_reset.clicked.connect(self._reset)
        self._btn_run.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        for b in (self._btn_run, self._btn_stop, self._btn_reset):
            b.setFixedHeight(30)
            btn_row.addWidget(b)
        self._btn_run.setToolTip("Start a run with the current example, methods, and config.")
        self._btn_stop.setToolTip("Request the active run to stop after the current step.")
        self._btn_reset.setToolTip("Reset config to defaults and clear plots, logs, and progress.")
        self._btn_stop.setEnabled(False)
        self._layout.addLayout(btn_row)

        # -- Progress --
        prog_w = QWidget()
        prog_l = QVBoxLayout(prog_w)
        prog_l.setContentsMargins(12, 4, 12, 8)
        self._progress_label = QLabel("Idle")
        self._progress_label.setStyleSheet("font-size: 11px; color: #888780;")
        self._progress_meta = QLabel("")
        self._progress_meta.setStyleSheet("font-size: 11px; color: #888780;")
        top_row = QHBoxLayout()
        top_row.addWidget(self._progress_label)
        top_row.addStretch()
        top_row.addWidget(self._progress_meta)
        prog_l.addLayout(top_row)
        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 1000)
        self._progress_bar.setValue(0)
        self._progress_bar.setTextVisible(False)
        self._progress_bar.setFixedHeight(3)
        prog_l.addWidget(self._progress_bar)
        self._progress_detail = QLabel("")
        self._progress_detail.setStyleSheet("font-size: 10px; color: #888780; margin-top: 2px;")
        prog_l.addWidget(self._progress_detail)
        self._layout.addWidget(prog_w)

        # -- Log --
        self._log = QPlainTextEdit()
        self._log.setReadOnly(True)
        self._log.setMaximumBlockCount(500)
        self._log.setFixedHeight(120)
        self._log.setStyleSheet("font-size: 10px;")
        self._layout.addWidget(self._log)

        self._layout.addStretch()
        scroll.setWidget(container)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 12, 0, 12)
        outer.addWidget(scroll)

    def _make_section(self, title: str) -> QGroupBox:
        box = QGroupBox(title)
        box.setFlat(True)
        layout = QVBoxLayout()
        layout.setContentsMargins(12, 6, 12, 6)
        layout.setSpacing(4)
        box.setLayout(layout)
        return box

    # ------------------------------------------------------------------
    # Example change handler
    # ------------------------------------------------------------------

    def _on_example_changed(self, index: int):
        spec = get_example_by_index(index)
        self._current_spec = spec
        self._rebuild_methods(spec)
        self._rebuild_config(spec)
        self._data_section.setVisible(spec.has_observations)
        self.example_changed.emit(spec)

    def _rebuild_methods(self, spec: ExampleSpec):
        while self._methods_container.count():
            item = self._methods_container.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._method_checks.clear()

        for ms in spec.methods:
            row = QWidget()
            rl = QHBoxLayout(row)
            rl.setContentsMargins(0, 2, 0, 2)
            rl.setSpacing(7)

            cb = QCheckBox(ms.display_name)
            cb.setChecked(ms.checked)
            cb.setEnabled(ms.enabled)
            rl.addWidget(cb)

            badge = QLabel(ms.badge)
            badge.setStyleSheet(badge_qss(ms.badge))
            badge.setFixedHeight(16)
            rl.addStretch()
            rl.addWidget(badge)

            self._methods_container.addWidget(row)
            self._method_checks.append((ms, cb))

    def _rebuild_config(self, spec: ExampleSpec):
        while self._config_form.count():
            item = self._config_form.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._config_inputs.clear()
        self._config_warn.setVisible(False)

        for key in spec.config_display_keys:
            default_val = spec.default_config.get(key, "")
            inp = _ValidatedLineEdit(key, str(default_val), default_val)
            inp.set_warning_label(self._config_warn)
            self._config_form.addRow(key, inp)
            self._config_inputs[key] = inp

    # ------------------------------------------------------------------
    # GPU validation
    # ------------------------------------------------------------------

    def _on_gpu_toggled(self, checked: bool):
        if not checked:
            return
        try:
            import jax
            gpus = jax.devices("gpu")
            if not gpus:
                raise RuntimeError("no GPU devices found")
        except Exception:
            self._gpu_radio.setToolTip(
                "no GPU detected \u2014 install CUDA or use Colab"
            )
            QToolTip.showText(
                self._gpu_radio.mapToGlobal(self._gpu_radio.rect().center()),
                "no GPU detected \u2014 install CUDA or use Colab",
                self._gpu_radio,
            )
            self._cpu_radio.setChecked(True)

    # ------------------------------------------------------------------
    # Data source handlers
    # ------------------------------------------------------------------

    def _use_synthetic(self):
        self._obs_csv_path = None
        self._obs_data = None
        self._data_label.setText("Synthetic observations")

    def _load_csv(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load observation CSV", "", "CSV (*.csv)")
        if path:
            self._obs_csv_path = path
            self._obs_data = None
            self._data_label.setText(f"CSV: {Path(path).name}")

    def _use_seed(self):
        x = np.linspace(0.05, 0.95, 28)
        y_true = np.sin(np.pi * x) + 0.35 * np.sin(2 * np.pi * x)
        y = y_true + np.random.default_rng(7).normal(0, 0.08, size=x.shape)
        self._obs_data = (x, y)
        self._obs_csv_path = None
        self._data_label.setText(f"Seed data: {len(x)} obs, \u03c3=0.08, seed 7")

    def _reset(self):
        # Reset is only allowed when no run is active.
        if not self._btn_run.isEnabled():
            return
        if self._current_spec:
            self._rebuild_config(self._current_spec)
        self._use_synthetic()
        self._progress_bar.setValue(0)
        self._progress_label.setText("Idle")
        self._progress_meta.setText("")
        self._progress_detail.setText("")
        self._log.clear()
        self.reset_requested.emit()

    # ------------------------------------------------------------------
    # Run request
    # ------------------------------------------------------------------

    def _emit_run(self):
        if self._current_spec is None:
            return

        methods = []
        for ms, cb in self._method_checks:
            if cb.isChecked() and ms.runner_key is not None:
                methods.append(ms.runner_key)
        if not methods:
            methods = [None]

        config = dict(self._current_spec.default_config)
        for key, inp in self._config_inputs.items():
            raw = inp.text().strip()
            if not raw:
                continue
            default_val = config.get(key)
            try:
                if isinstance(default_val, list):
                    import ast
                    config[key] = ast.literal_eval(raw)
                elif isinstance(default_val, int):
                    config[key] = int(float(raw))
                elif isinstance(default_val, float):
                    config[key] = float(raw)
                else:
                    config[key] = raw
            except (ValueError, SyntaxError):
                config[key] = raw

        if self._auto_radio.isChecked():
            device = "auto"
        elif self._gpu_radio.isChecked():
            device = "gpu"
        else:
            device = "cpu"
        config["compute"] = {
            "backend": self._backend_combo.currentText(),
            "device": device,
        }
        obs_info = {
            "csv_path": self._obs_csv_path,
            "data": self._obs_data,
        }

        self.run_requested.emit(
            self._current_spec.key, methods, config, device, obs_info
        )

    # ------------------------------------------------------------------
    # Progress updates (called from main thread via signal)
    # ------------------------------------------------------------------

    def set_progress(self, percent: float, stage: str, detail: str, elapsed: float):
        self._progress_bar.setValue(int(percent * 10))
        self._progress_label.setText(stage.capitalize())
        mins, secs = divmod(int(elapsed), 60)
        self._progress_meta.setText(f"{percent:.0f}% \u00b7 {mins:02d}:{secs:02d}")
        self._progress_detail.setText(detail)

    def append_log(self, text: str):
        self._log.appendPlainText(text)

    def set_running(self):
        self._btn_run.setEnabled(False)
        self._btn_stop.setEnabled(True)
        self._btn_reset.setEnabled(False)
        self._progress_label.setText("Running")

    def set_idle(self, status: str = "Completed"):
        self._btn_run.setEnabled(True)
        self._btn_stop.setEnabled(False)
        self._btn_reset.setEnabled(True)
        self._progress_label.setText(status)

    @property
    def current_example(self) -> ExampleSpec | None:
        return self._current_spec
