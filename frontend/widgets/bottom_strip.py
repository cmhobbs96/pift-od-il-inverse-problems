"""Bottom strip: 3-column layout with metrics, energy diagnostics, and run history."""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from frontend.style import COLORS, badge_qss, metric_color


class MetricCard(QFrame):
    """Small card showing a label and a large value."""

    def __init__(self, label: str, parent=None):
        super().__init__(parent)
        self.setStyleSheet(
            f"background-color: {COLORS['bg3']}; border-radius: 8px; padding: 7px 10px;"
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 7, 10, 7)
        layout.setSpacing(2)

        self._label = QLabel(label.upper())
        self._label.setStyleSheet(
            f"font-size: 10px; color: {COLORS['text3']}; "
            "text-transform: uppercase; letter-spacing: 0.4px;"
        )
        self._value = QLabel("\u2014")
        self._value.setStyleSheet("font-size: 17px; font-weight: 500;")
        layout.addWidget(self._label)
        layout.addWidget(self._value)

    def set_value(self, text: str, color: str | None = None):
        self._value.setText(text)
        if color:
            self._value.setStyleSheet(f"font-size: 17px; font-weight: 500; color: {color};")
        else:
            self._value.setStyleSheet("font-size: 17px; font-weight: 500;")


class DiagRow(QFrame):
    """Key-value row with optional pill badge."""

    def __init__(self, label: str, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setStyleSheet(
            f"DiagRow {{ border-bottom: 1px solid {COLORS['border']}; }}"
        )
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 3, 0, 3)
        self._label = QLabel(label)
        self._label.setStyleSheet("font-size: 11px; border: none;")
        self._value = QLabel("\u2014")
        self._value.setStyleSheet(
            f"font-family: 'Courier New', monospace; font-size: 10px; "
            f"color: {COLORS['text2']}; border: none;"
        )
        self._value.setAlignment(Qt.AlignmentFlag.AlignRight)
        layout.addWidget(self._label)
        layout.addStretch()
        layout.addWidget(self._value)

    def set_value(self, text: str, color: str | None = None):
        style = (
            f"font-family: 'Courier New', monospace; font-size: 10px; "
            f"color: {color or COLORS['text2']}; border: none;"
        )
        self._value.setText(text)
        self._value.setStyleSheet(style)

    def set_pill(self, text: str, pill_type: str = "ok"):
        self._value.setText(text)
        self._value.setStyleSheet(badge_qss(pill_type))


class RunHistoryRow(QWidget):
    """Compact run entry: [badge] method run_id time."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 3, 0, 3)
        layout.setSpacing(8)
        self._badge = QLabel("\u2014")
        self._badge.setFixedWidth(52)
        self._badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._method = QLabel("")
        self._method.setStyleSheet(f"color: {COLORS['text2']}; font-size: 11px;")
        self._run_id = QLabel("")
        self._run_id.setStyleSheet(
            f"font-family: 'Courier New', monospace; font-size: 10px; color: {COLORS['text3']};"
        )
        self._run_id.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._time = QLabel("")
        self._time.setStyleSheet(f"font-size: 10px; color: {COLORS['text3']};")
        layout.addWidget(self._badge)
        layout.addWidget(self._method)
        layout.addWidget(self._run_id, 1)
        layout.addWidget(self._time)
        self.setStyleSheet(f"border-bottom: 1px solid {COLORS['border']};")

    def set_run(self, status: str, method: str, run_id: str, runtime_sec: float | None):
        badge_type = {"completed": "done", "failed": "failed", "running": "running",
                      "stopped": "stopped", "queued": "ready"}.get(status, "disabled")
        self._badge.setText(status[:6])
        self._badge.setStyleSheet(badge_qss(badge_type))
        self._method.setText(method[:12])
        self._run_id.setText(run_id[:8])
        if runtime_sec is not None:
            m, s = divmod(int(runtime_sec), 60)
            self._time.setText(f"{m}:{s:02d}")
        else:
            self._time.setText("\u2014")


class BottomStripWidget(QWidget):
    """3-column footer: metrics | energy diagnostics | run history."""

    pin_baseline = Signal(str)  # run_id
    compare_clicked = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("bottom-strip")
        self.setFixedHeight(280)

        outer = QHBoxLayout(self)
        outer.setContentsMargins(12, 8, 12, 8)
        outer.setSpacing(10)

        # -- Column 1: Posterior Metrics --
        col1 = QFrame()
        col1.setStyleSheet(f"background-color: {COLORS['bg2']};")
        c1l = QVBoxLayout(col1)
        c1l.setContentsMargins(14, 12, 14, 10)

        c1_title = QLabel("POSTERIOR METRICS")
        c1_title.setStyleSheet(
            f"font-size: 10px; font-weight: 500; color: {COLORS['text3']}; "
            "letter-spacing: 0.5px;"
        )
        c1l.addWidget(c1_title)

        cards = QGridLayout()
        cards.setSpacing(8)
        self._card_l2 = MetricCard("L2 error")
        self._card_max = MetricCard("Max error")
        self._card_cov = MetricCard("Coverage")
        self._card_samples = MetricCard("Samples")
        cards.addWidget(self._card_l2, 0, 0)
        cards.addWidget(self._card_max, 0, 1)
        cards.addWidget(self._card_cov, 0, 2)
        cards.addWidget(self._card_samples, 0, 3)
        c1l.addLayout(cards)

        self._flag_nan = DiagRow("NaN detected")
        self._flag_div = DiagRow("Divergence")
        self._flag_maxH = DiagRow("max |H|")
        c1l.addWidget(self._flag_nan)
        c1l.addWidget(self._flag_div)
        c1l.addWidget(self._flag_maxH)
        c1l.addStretch()
        outer.addWidget(col1, 1)

        # -- Column 2: Energy Diagnostics --
        col2 = QFrame()
        col2.setStyleSheet(f"background-color: {COLORS['bg2']};")
        c2l = QVBoxLayout(col2)
        c2l.setContentsMargins(14, 12, 14, 10)

        c2_title = QLabel("ENERGY DIAGNOSTICS")
        c2_title.setStyleSheet(
            f"font-size: 10px; font-weight: 500; color: {COLORS['text3']}; "
            "letter-spacing: 0.5px;"
        )
        c2l.addWidget(c2_title)

        self._energy_rows: dict[str, DiagRow] = {}
        for label in ("\u03b2", "initial U", "initial \u03a9", "\u03b2\u00b7U / \u03a9",
                       "Condition no. (raw)", "Condition no. (clipped)"):
            row = DiagRow(label)
            c2l.addWidget(row)
            self._energy_rows[label] = row
        c2l.addStretch()
        outer.addWidget(col2, 1)

        # -- Column 3: Run History --
        col3 = QFrame()
        col3.setStyleSheet(f"background-color: {COLORS['bg2']};")
        c3l = QVBoxLayout(col3)
        c3l.setContentsMargins(14, 12, 14, 10)

        c3_title = QLabel("RUN HISTORY")
        c3_title.setStyleSheet(
            f"font-size: 10px; font-weight: 500; color: {COLORS['text3']}; "
            "letter-spacing: 0.5px;"
        )
        c3l.addWidget(c3_title)

        self._history_rows: list[RunHistoryRow] = []
        for _ in range(6):
            row = RunHistoryRow()
            row.setVisible(False)
            c3l.addWidget(row)
            self._history_rows.append(row)

        c3l.addStretch()
        btn_row = QHBoxLayout()
        self._btn_pin = QPushButton("Pin baseline")
        self._btn_compare = QPushButton("Compare \u2197")
        for b in (self._btn_pin, self._btn_compare):
            b.setStyleSheet("font-size: 11px; padding: 4px 0;")
            btn_row.addWidget(b)
        self._btn_compare.clicked.connect(self.compare_clicked.emit)
        c3l.addLayout(btn_row)
        outer.addWidget(col3, 1)

    # ------------------------------------------------------------------
    # Update methods
    # ------------------------------------------------------------------

    def update_metrics(self, result: dict[str, Any], example_key: str = "phase_a_forward"):
        """Extract and display metrics from a pipeline result."""
        summary = result.get("summary", {})
        metrics = result.get("metrics", {})
        diagnostics = result.get("diagnostics", {})

        l2 = summary.get("l2_error") or metrics.get("l2_error")
        max_err = summary.get("max_error") or metrics.get("max_error")
        cov = summary.get("measurement_interval_coverage_90pct")
        n_samples = summary.get("n_posterior_samples") or metrics.get("n_posterior_samples")

        # L2 error — green < 0.1, amber < 0.2, red >= 0.2
        if l2 is not None:
            self._card_l2.set_value(f"{l2:.3f}", metric_color(l2, (0.1, 0.2)))
        else:
            self._card_l2.set_value("\u2014")

        # Max error
        if max_err is not None:
            self._card_max.set_value(f"{max_err:.3f}", metric_color(max_err, (0.1, 0.2)))
        else:
            self._card_max.set_value("\u2014")

        # Coverage — green > 0.9, amber > 0.7, red <= 0.7
        if cov is not None:
            pct = cov * 100 if cov <= 1 else cov
            if pct >= 90:
                color = COLORS["green_text"]
            elif pct >= 70:
                color = COLORS["amber_text"]
            else:
                color = COLORS["red_text"]
            self._card_cov.set_value(f"{pct:.0f}%", color)
        else:
            self._card_cov.set_value("\u2014")

        # Samples
        if n_samples is not None:
            self._card_samples.set_value(f"{int(n_samples):,}")
        else:
            self._card_samples.set_value("\u2014")

        # Stability flags
        flags = diagnostics.get("stability_flags", {})
        nan_det = flags.get("nan_detected", False)
        div_det = flags.get("divergence_detected", False)
        self._flag_nan.set_pill("true" if nan_det else "false", "failed" if nan_det else "ok")
        self._flag_div.set_pill("true" if div_det else "false", "failed" if div_det else "ok")

        # max |H|
        max_h = diagnostics.get("max_abs_hamiltonian")
        if max_h is not None:
            self._flag_maxH.set_value(self._fmt_sci(max_h))
        else:
            self._flag_maxH.set_value("\u2014")

        # Energy diagnostics
        beta_val = (
            summary.get("beta")
            or diagnostics.get("beta")
            or result.get("config", {}).get("beta", "\u2014")
        )
        init_U = diagnostics.get("initial_physics_energy")
        init_Omega = diagnostics.get("initial_likelihood_energy")
        ratio = diagnostics.get("ratio_beta_U_over_Omega")
        cond_raw = diagnostics.get("condition_number_raw")
        cond_clip = diagnostics.get("condition_number")

        rows = list(self._energy_rows.values())
        rows[0].set_value(str(beta_val))
        rows[1].set_value(self._fmt_sci(init_U))
        rows[2].set_value(self._fmt_sci(init_Omega))

        # Color ratio: red if > 10 (physics dominating) or < 0.01 (likelihood dominating)
        ratio_str = self._fmt_sci(ratio)
        ratio_color = None
        if ratio is not None:
            try:
                rv = float(ratio)
                if rv > 10 or (rv > 0 and rv < 0.01):
                    ratio_color = COLORS["red_text"]
                elif rv > 1 or rv < 0.1:
                    ratio_color = COLORS["amber_text"]
                else:
                    ratio_color = COLORS["green_text"]
            except (ValueError, TypeError):
                pass
        rows[3].set_value(ratio_str, ratio_color)

        rows[4].set_value(str(cond_raw) if cond_raw is not None else "\u2014")
        rows[5].set_value(
            str(cond_clip) if cond_clip is not None else "\u2014",
            COLORS["green_text"] if cond_clip is not None else None,
        )

    def update_history(self, run_records: list[dict[str, Any]]):
        """Show most recent runs in compact format."""
        for i, row_widget in enumerate(self._history_rows):
            if i < len(run_records):
                rec = run_records[i]
                row_widget.set_run(
                    rec.get("status", "?"),
                    rec.get("method", "?"),
                    rec.get("run_id", "?"),
                    rec.get("runtime_sec"),
                )
                row_widget.setVisible(True)
            else:
                row_widget.setVisible(False)

    @staticmethod
    def _fmt_sci(val) -> str:
        if val is None:
            return "\u2014"
        try:
            v = float(val)
            if abs(v) < 0.01 or abs(v) > 1e4:
                return f"{v:.2e}"
            return f"{v:.4f}"
        except (ValueError, TypeError):
            return str(val)
