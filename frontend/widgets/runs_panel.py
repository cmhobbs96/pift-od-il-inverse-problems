"""Runs panel: filterable table of run history with management actions."""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from frontend.style import COLORS, badge_qss


def _quality_label(rec: dict[str, Any]) -> str:
    status = rec.get("status", "")
    if status in ("failed", "stopped"):
        return "Failed"
    nan_det = rec.get("nan_detected")
    inf_det = rec.get("inf_detected")
    if nan_det or inf_det:
        return "Needs Review"
    l2 = rec.get("reference_l2_error") or rec.get("l2_error")
    cov = rec.get("coverage_90")
    if l2 is not None:
        try:
            if float(l2) >= 0.20:
                return "Needs Review"
        except (ValueError, TypeError):
            pass
    if cov is not None:
        try:
            c = float(cov)
            if c < 0.60 or c > 1.0:
                return "Needs Review"
        except (ValueError, TypeError):
            pass
    if status == "completed":
        return "Stable"
    return "—"


class RunsPanel(QWidget):
    """Run history table with filters, delete, and selection signals."""

    run_selected = Signal(str)  # run_id

    def __init__(self, storage=None, parent=None):
        super().__init__(parent)
        self._storage = storage
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)

        # Filters row
        filters = QHBoxLayout()
        filters.addWidget(QLabel("Status:"))
        self._status_filter = QComboBox()
        self._status_filter.addItems(["all", "completed", "running", "failed", "stopped", "queued"])
        self._status_filter.currentIndexChanged.connect(self._refresh)
        filters.addWidget(self._status_filter)

        filters.addWidget(QLabel("Example:"))
        self._example_filter = QComboBox()
        self._example_filter.addItems(["all"])
        from frontend.registry import EXAMPLE_KEYS
        self._example_filter.addItems(EXAMPLE_KEYS)
        self._example_filter.currentIndexChanged.connect(self._refresh)
        filters.addWidget(self._example_filter)

        filters.addWidget(QLabel("Search:"))
        self._search = QLineEdit()
        self._search.setPlaceholderText("run ID substring")
        self._search.textChanged.connect(self._refresh)
        self._search.setFixedWidth(120)
        filters.addWidget(self._search)

        btn_refresh = QPushButton("Refresh")
        btn_refresh.setFixedHeight(26)
        btn_refresh.clicked.connect(self._refresh)
        filters.addWidget(btn_refresh)

        btn_delete = QPushButton("Delete selected")
        btn_delete.setFixedHeight(26)
        btn_delete.clicked.connect(self._delete_selected)
        filters.addWidget(btn_delete)

        filters.addStretch()
        layout.addLayout(filters)

        # Table
        self._table = QTableWidget()
        self._table.setColumnCount(8)
        self._table.setHorizontalHeaderLabels([
            "Run ID", "Status", "Quality", "Example", "Method",
            "Device", "Runtime", "Created",
        ])
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self._table.setAlternatingRowColors(True)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QTableWidget.SelectionMode.ExtendedSelection)
        self._table.itemSelectionChanged.connect(self._on_selection_changed)
        layout.addWidget(self._table)

        # Best run label
        self._best_label = QLabel("")
        self._best_label.setStyleSheet(f"font-size: 11px; color: {COLORS['text3']};")
        layout.addWidget(self._best_label)

        self._runs: list[dict[str, Any]] = []

    def set_storage(self, storage):
        self._storage = storage

    def refresh(self):
        self._refresh()

    def _refresh(self):
        if self._storage is None:
            return

        filters: dict[str, Any] = {}
        status = self._status_filter.currentText()
        if status != "all":
            filters["status"] = status

        try:
            rows = self._storage.list_runs_with_metrics_summary(filters)
        except Exception:
            rows = self._storage.list_runs(filters)

        # Apply example filter
        example = self._example_filter.currentText()
        if example != "all":
            rows = [r for r in rows if r.get("example_key", "phase_a_forward") == example]

        # Apply search filter
        search = self._search.text().strip().lower()
        if search:
            rows = [r for r in rows if search in r.get("run_id", "").lower()]

        self._runs = rows
        self._table.setRowCount(len(rows))
        best_l2 = float("inf")
        best_id = ""

        for i, r in enumerate(rows):
            quality = _quality_label(r)
            items = [
                r.get("run_id", "")[:8],
                r.get("status", ""),
                quality,
                r.get("example_key", "phase_a_forward"),
                r.get("method", ""),
                r.get("device_used", ""),
                f"{r.get('runtime_sec', 0) or 0:.1f}s",
                (r.get("created_at", ""))[:19],
            ]
            for j, val in enumerate(items):
                item = QTableWidgetItem(str(val))
                self._table.setItem(i, j, item)

            # Track best run
            l2 = r.get("reference_l2_error") or r.get("l2_error")
            if l2 is not None and r.get("status") == "completed":
                try:
                    l2f = float(l2)
                    if l2f < best_l2:
                        best_l2 = l2f
                        best_id = r.get("run_id", "")[:8]
                except (ValueError, TypeError):
                    pass

        if best_id:
            self._best_label.setText(f"Best: {best_id} (L2 = {best_l2:.4f})")
        else:
            self._best_label.setText("")

    def _on_selection_changed(self):
        rows = self._table.selectionModel().selectedRows()
        if rows and rows[0].row() < len(self._runs):
            run_id = self._runs[rows[0].row()].get("run_id", "")
            self.run_selected.emit(run_id)

    def _delete_selected(self):
        rows = self._table.selectionModel().selectedRows()
        if not rows:
            return
        ids = [self._runs[r.row()].get("run_id") for r in rows if r.row() < len(self._runs)]
        if not ids:
            return
        reply = QMessageBox.question(
            self, "Delete runs",
            f"Delete {len(ids)} run(s)? This cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            for rid in ids:
                if rid and self._storage:
                    self._storage.delete_run(rid)
            self._refresh()

    def get_selected_run_ids(self) -> list[str]:
        rows = self._table.selectionModel().selectedRows()
        return [
            self._runs[r.row()].get("run_id", "")
            for r in rows if r.row() < len(self._runs)
        ]

    def get_runs(self) -> list[dict[str, Any]]:
        return self._runs
