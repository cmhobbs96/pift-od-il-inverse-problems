"""QSS stylesheet and color constants matching the GUI mock design."""

from __future__ import annotations

# Color palette — adapted from GUI_MOCK.html CSS variables
COLORS = {
    "blue": "#185fa5",
    "blue_bg": "#e6f1fb",
    "blue_text": "#0c447c",
    "green_bg": "#eaf3de",
    "green_text": "#3b6d11",
    "amber_bg": "#faeeda",
    "amber_text": "#854f0b",
    "red_bg": "#fcebeb",
    "red_text": "#a32d2d",
    "gray_bg": "#f1efe8",
    "gray_text": "#5f5e5a",
    "bg": "#ffffff",
    "bg2": "#f8f7f4",
    "bg3": "#f1efe8",
    "border": "rgba(0,0,0,30)",
    "text": "#1a1a18",
    "text2": "#5f5e5a",
    "text3": "#888780",
}

BADGE_STYLES = {
    "ok": f"background-color: {COLORS['green_bg']}; color: {COLORS['green_text']};",
    "ready": f"background-color: {COLORS['blue_bg']}; color: {COLORS['blue_text']};",
    "unstable": f"background-color: {COLORS['amber_bg']}; color: {COLORS['amber_text']};",
    "disabled": f"background-color: {COLORS['gray_bg']}; color: {COLORS['gray_text']};",
    "running": f"background-color: {COLORS['blue_bg']}; color: {COLORS['blue_text']};",
    "failed": f"background-color: {COLORS['red_bg']}; color: {COLORS['red_text']};",
    "done": f"background-color: {COLORS['green_bg']}; color: {COLORS['green_text']};",
    "stopped": f"background-color: {COLORS['amber_bg']}; color: {COLORS['amber_text']};",
}


def badge_qss(badge_type: str) -> str:
    base = BADGE_STYLES.get(badge_type, BADGE_STYLES["disabled"])
    return (
        f"{base} font-size: 10px; padding: 2px 8px; "
        "border: none; border-radius: 10px; font-weight: 500;"
    )


def metric_color(value: float, thresholds: tuple[float, float] = (0.05, 0.2)) -> str:
    """Return color string: green if below low, amber if below high, else red."""
    if value < thresholds[0]:
        return COLORS["green_text"]
    if value < thresholds[1]:
        return COLORS["amber_text"]
    return COLORS["red_text"]


GLOBAL_QSS = f"""
QMainWindow {{
    background-color: {COLORS['bg']};
}}

/* Section titles */
.section-title {{
    font-size: 10px;
    font-weight: 500;
    color: {COLORS['text3']};
    text-transform: uppercase;
    letter-spacing: 0.5px;
}}

/* Sidebar */
#sidebar {{
    background-color: {COLORS['bg2']};
    border-right: 1px solid {COLORS['border']};
}}

/* Bottom strip */
#bottom-strip {{
    background-color: {COLORS['bg2']};
    border-top: 1px solid {COLORS['border']};
}}

/* Metric cards */
.metric-card {{
    background-color: {COLORS['bg3']};
    border-radius: 8px;
    padding: 7px 10px;
}}

/* Tab bar */
QTabWidget::pane {{
    border: none;
    background-color: {COLORS['bg']};
    margin-top: 4px;
}}

QTabBar {{
    margin-left: 12px;
    margin-top: 8px;
}}

QTabBar::tab {{
    min-width: 80px;
    padding: 8px 16px;
    font-size: 12px;
    color: {COLORS['text3']};
    border: none;
    border-bottom: 2px solid transparent;
    background: {COLORS['bg2']};
}}

QTabBar::tab:selected {{
    color: {COLORS['text']};
    font-weight: 500;
    border-bottom-color: {COLORS['blue']};
}}

QTabBar::tab:hover:!selected {{
    color: {COLORS['text2']};
}}

/* Buttons */
QPushButton {{
    font-size: 12px;
    padding: 5px 12px;
    border: 1px solid {COLORS['border']};
    border-radius: 6px;
    background: transparent;
    color: {COLORS['text']};
}}

QPushButton:hover {{
    background-color: {COLORS['bg3']};
}}

QPushButton#btn-primary {{
    background-color: {COLORS['blue']};
    color: white;
    border-color: {COLORS['blue']};
}}

QPushButton#btn-primary:hover {{
    opacity: 0.9;
}}

/* Inputs */
QLineEdit, QComboBox {{
    font-size: 12px;
    padding: 4px 8px;
    border: 1px solid {COLORS['border']};
    border-radius: 6px;
    background-color: {COLORS['bg']};
    color: {COLORS['text']};
}}

QLineEdit:focus, QComboBox:focus {{
    border-color: {COLORS['blue']};
}}

/* Progress bar */
QProgressBar {{
    height: 3px;
    background-color: {COLORS['bg3']};
    border: none;
    border-radius: 2px;
}}

QProgressBar::chunk {{
    background-color: {COLORS['blue']};
    border-radius: 2px;
}}

/* Scroll area */
QScrollArea {{
    border: none;
}}

/* Group boxes */
QGroupBox {{
    font-size: 10px;
    font-weight: 500;
    color: {COLORS['text3']};
    border: none;
    margin-top: 8px;
    padding-top: 14px;
}}

QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 0 4px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}}

/* Table */
QTableWidget {{
    gridline-color: {COLORS['border']};
    font-size: 11px;
    background-color: {COLORS['bg']};
    alternate-background-color: {COLORS['bg2']};
}}

QHeaderView::section {{
    background-color: {COLORS['bg2']};
    border: 1px solid {COLORS['border']};
    padding: 4px 8px;
    font-size: 10px;
    font-weight: 500;
    color: {COLORS['text3']};
    text-transform: uppercase;
}}

/* Log area */
QPlainTextEdit {{
    font-family: 'Courier New', monospace;
    font-size: 11px;
    background-color: {COLORS['bg']};
    border: 1px solid {COLORS['border']};
    border-radius: 6px;
}}
"""
