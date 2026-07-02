
"""
modules/multiple_regression.py
SPS Studio — Multiple Regression Builder

PySide6 module integrated with SPS Studio.

Purpose:
- Multiple Linear Regression for one numeric response Y and multiple numeric predictors X.
- Optional predictor selection by checkboxes.
- Generates:
  1) Numerical Results: coefficients, ANOVA, model summary, VIF, correlation table.
  2) Model Graph: actual vs fitted, residual summary, model summary table.
  3) Residual Analysis: residuals vs fitted, histogram, Q-Q plot, residuals vs order, assumption conclusions.
  4) Summary: full text report.
- Supports copy/export graph and Analysis Journal integration.
- Integrates with core.plot_interaction_tools.enable_interaction.

Visible UI text is English.
"""

from __future__ import annotations

import base64
import io
import math
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import probplot, shapiro, levene

from PySide6.QtCore import Qt, QEvent
from PySide6.QtGui import QImage
from PySide6.QtWidgets import (
    QApplication,
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

try:
    from core.plot_interaction_tools import enable_interaction
except Exception:
    enable_interaction = None


class ReportScrollArea(QScrollArea):
    """
    Scrollable and zoomable Matplotlib report viewer.

    Mouse controls:
    - Wheel: vertical scroll
    - Shift + Wheel: horizontal scroll
    - Ctrl + Wheel: zoom in/out
    """
    def __init__(self, canvas: FigureCanvas, fig: Figure, initial_zoom: float = 0.92, parent=None):
        super().__init__(parent)
        self.canvas = canvas
        self.fig = fig
        self.base_dpi = float(fig.dpi)
        self.base_width_in = float(fig.get_figwidth())
        self.base_height_in = float(fig.get_figheight())
        self.zoom = float(initial_zoom)

        self.setWidgetResizable(False)
        self.setAlignment(Qt.AlignCenter)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setFrameShape(QScrollArea.NoFrame)
        self.setFocusPolicy(Qt.StrongFocus)

        self.container = QWidget()
        self.container_layout = QVBoxLayout(self.container)
        self.container_layout.setContentsMargins(8, 8, 8, 8)
        self.container_layout.addWidget(self.canvas, alignment=Qt.AlignCenter)
        self.container_layout.addStretch()
        self.setWidget(self.container)

        self.viewport().installEventFilter(self)
        self.container.installEventFilter(self)
        self.canvas.installEventFilter(self)
        self.canvas.setFocusPolicy(Qt.StrongFocus)

        self.apply_zoom(self.zoom, center=False)

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Wheel:
            self._handle_wheel_event(event)
            return True
        return super().eventFilter(obj, event)

    def wheelEvent(self, event):
        self._handle_wheel_event(event)

    def _wheel_delta(self, event):
        delta = event.pixelDelta()
        if not delta.isNull():
            return delta.x(), delta.y()
        delta = event.angleDelta()
        return delta.x(), delta.y()

    def _handle_wheel_event(self, event):
        dx, dy = self._wheel_delta(event)
        if dx == 0 and dy == 0:
            event.accept()
            return

        mods = event.modifiers()
        if mods & Qt.ControlModifier:
            factor = 1.08 if dy > 0 else 1 / 1.08
            self.apply_zoom(self.zoom * factor, center=True)
        elif mods & Qt.ShiftModifier:
            bar = self.horizontalScrollBar()
            bar.setValue(bar.value() - dy - dx)
        else:
            bar = self.verticalScrollBar()
            bar.setValue(bar.value() - dy)
        event.accept()

    def apply_zoom(self, zoom: float, center: bool = True):
        old_h = self.horizontalScrollBar().value()
        old_v = self.verticalScrollBar().value()
        old_hw = max(1, self.horizontalScrollBar().maximum())
        old_vh = max(1, self.verticalScrollBar().maximum())
        h_ratio = old_h / old_hw
        v_ratio = old_v / old_vh

        self.zoom = max(0.62, min(1.65, float(zoom)))
        self.fig.set_size_inches(self.base_width_in, self.base_height_in, forward=True)
        self.fig.set_dpi(self.base_dpi * self.zoom)

        w = int(self.base_width_in * self.base_dpi * self.zoom)
        h = int(self.base_height_in * self.base_dpi * self.zoom)
        self.canvas.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.canvas.setFixedSize(w, h)
        self.canvas.draw_idle()

        self.container.adjustSize()

        if center:
            self.horizontalScrollBar().setValue(int(self.horizontalScrollBar().maximum() * h_ratio))
            self.verticalScrollBar().setValue(int(self.verticalScrollBar().maximum() * v_ratio))

    def zoom_in(self):
        self.apply_zoom(self.zoom * 1.12)

    def zoom_out(self):
        self.apply_zoom(self.zoom / 1.12)

    def reset_zoom(self):
        self.apply_zoom(1.0)

    def fit_to_view(self):
        viewport_w = max(1, self.viewport().width() - 24)
        viewport_h = max(1, self.viewport().height() - 24)
        base_w = self.base_width_in * self.base_dpi
        base_h = self.base_height_in * self.base_dpi
        z = min(viewport_w / base_w, viewport_h / base_h, 1.0)
        self.apply_zoom(max(0.62, z), center=False)


# ─────────────────────────────────────────────
# Utility functions
# ─────────────────────────────────────────────

def now_string() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def fmt(value: Any, decimals: int = 4) -> str:
    if value is None:
        return "N/A"
    try:
        v = float(value)
        if np.isnan(v) or np.isinf(v):
            return "N/A"
        return f"{v:.{decimals}f}"
    except Exception:
        return str(value)


def figure_to_png_bytes(fig: Figure) -> bytes:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
    buffer.seek(0)
    return buffer.getvalue()


def figure_to_png_base64(fig: Figure) -> str:
    return base64.b64encode(figure_to_png_bytes(fig)).decode("ascii")


def serializable(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    if isinstance(obj, dict):
        return {str(k): serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [serializable(v) for v in obj]
    return obj


def p_decision(p_value: Any, alpha: float) -> str:
    try:
        return "Reject H0" if float(p_value) <= alpha else "Accept H0"
    except Exception:
        return "N/A"


def significance_label(p_value: float, alpha: float) -> str:
    if pd.isna(p_value):
        return "N/A"
    if p_value <= alpha:
        return "Significant"
    if p_value <= alpha * 2:
        return "Borderline"
    return "Not significant"


def r2_label(r2: float) -> str:
    if pd.isna(r2):
        return "N/A"
    if r2 < 0.10:
        return "Poor"
    if r2 < 0.30:
        return "Fair"
    if r2 < 0.60:
        return "Good"
    return "Excellent"


def safe_shapiro(values: np.ndarray) -> Tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 3:
        return np.nan, np.nan
    if len(values) > 5000:
        values = values[:5000]
    try:
        stat, p = shapiro(values)
        return float(stat), float(p)
    except Exception:
        return np.nan, np.nan


def durbin_watson_stat(residuals: np.ndarray) -> float:
    residuals = np.asarray(residuals, dtype=float)
    if residuals.size < 2:
        return np.nan
    denominator = float(np.sum(residuals ** 2))
    if denominator == 0:
        return np.nan
    return float(np.sum(np.diff(residuals) ** 2) / denominator)


def standardized_residuals(residuals: np.ndarray, mse: float, h: np.ndarray) -> np.ndarray:
    if pd.isna(mse) or mse <= 0:
        return np.full_like(residuals, np.nan, dtype=float)
    denom = np.sqrt(mse * np.maximum(1e-12, 1.0 - h))
    return residuals / denom


def studentized_deleted_residuals(residuals: np.ndarray, mse: float, h: np.ndarray, df_error: int) -> np.ndarray:
    # Externally studentized residual approximation.
    std_res = standardized_residuals(residuals, mse, h)
    out = np.full_like(std_res, np.nan, dtype=float)
    for i, r in enumerate(std_res):
        if pd.isna(r):
            continue
        denom = max(1e-12, df_error - r**2)
        out[i] = r * math.sqrt(max(0.0, (df_error - 1) / denom))
    return out


def variance_stability_test(fitted: np.ndarray, residuals: np.ndarray) -> Tuple[float, float]:
    fitted = np.asarray(fitted, dtype=float)
    residuals = np.asarray(residuals, dtype=float)
    mask = ~(np.isnan(fitted) | np.isnan(residuals))
    fitted = fitted[mask]
    residuals = residuals[mask]
    if fitted.size < 8:
        return np.nan, np.nan

    q1, q2 = np.quantile(fitted, [1/3, 2/3])
    low = residuals[fitted <= q1]
    mid = residuals[(fitted > q1) & (fitted <= q2)]
    high = residuals[fitted > q2]
    groups = [g for g in [low, mid, high] if len(g) >= 3]
    if len(groups) < 2:
        return np.nan, np.nan

    try:
        stat, p = levene(*groups, center="median")
        return float(stat), float(p)
    except Exception:
        return np.nan, np.nan


def correlation_strength(r: float) -> str:
    if pd.isna(r):
        return "N/A"
    a = abs(float(r))
    if a < 0.20:
        return "Very weak"
    if a < 0.40:
        return "Weak"
    if a < 0.60:
        return "Moderate"
    if a < 0.80:
        return "Strong"
    return "Very strong"


def compute_vif(X_predictors: np.ndarray, predictor_names: List[str]) -> List[Dict[str, Any]]:
    rows = []
    if X_predictors.shape[1] == 1:
        return [{"Predictor": predictor_names[0], "R-sq": 0.0, "VIF": 1.0, "Conclusion": "No multicollinearity with one predictor."}]

    for idx, name in enumerate(predictor_names):
        y = X_predictors[:, idx]
        others = np.delete(X_predictors, idx, axis=1)
        X_aux = np.column_stack([np.ones(len(y)), others])
        try:
            beta, *_ = np.linalg.lstsq(X_aux, y, rcond=None)
            fitted = X_aux @ beta
            ss_res = float(np.sum((y - fitted) ** 2))
            ss_tot = float(np.sum((y - np.mean(y)) ** 2))
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
            vif = 1.0 / (1.0 - r2) if pd.notna(r2) and r2 < 1 else np.inf
        except Exception:
            r2, vif = np.nan, np.nan

        if pd.isna(vif):
            conclusion = "N/A"
        elif vif < 5:
            conclusion = "Acceptable"
        elif vif < 10:
            conclusion = "Review"
        else:
            conclusion = "High multicollinearity"

        rows.append({"Predictor": name, "R-sq": r2, "VIF": vif, "Conclusion": conclusion})
    return rows




def residual_conclusion(fit: Dict[str, Any], alpha: float, decimals: int) -> str:
    res = fit.get("residual_analysis", {})
    lines = []
    recs = []

    if res.get("normality_ok"):
        lines.append(f"• Normality: PASS. Shapiro-Wilk p={fmt(res.get('shapiro_p'), decimals)} > α={fmt(alpha, 3)}.")
    else:
        lines.append(f"• Normality: REVIEW. Shapiro-Wilk p={fmt(res.get('shapiro_p'), decimals)} ≤ α={fmt(alpha, 3)}.")
        recs.append("Review Q-Q plot, outliers, transformations, or non-normal response behavior.")

    if res.get("constant_variance_ok"):
        lines.append(f"• Constant variance: PASS. Levene p={fmt(res.get('levene_p'), decimals)} > α={fmt(alpha, 3)}.")
    else:
        lines.append(f"• Constant variance: REVIEW. Levene p={fmt(res.get('levene_p'), decimals)} ≤ α={fmt(alpha, 3)}.")
        recs.append("Consider transformation, weighted regression, or missing predictor structure.")

    if res.get("independence_ok"):
        lines.append(f"• Independence: PASS. Durbin-Watson={fmt(res.get('durbin_watson'), decimals)}.")
    else:
        lines.append(f"• Independence: REVIEW. Durbin-Watson={fmt(res.get('durbin_watson'), decimals)}.")
        recs.append("Check run order, time drift, batching, autocorrelation, or repeated measurements.")

    if res.get("outliers_ok"):
        lines.append("• Severe outliers: PASS. No standardized residuals beyond ±3 were detected.")
    else:
        lines.append(f"• Severe outliers: REVIEW. {res.get('outliers_n', 0)} standardized residual(s) beyond ±3 detected.")
        recs.append("Investigate special causes, data entry issues, or influential observations.")

    if res.get("high_leverage_ok"):
        lines.append(f"• High leverage: PASS. No points above leverage threshold {fmt(res.get('high_leverage_threshold'), decimals)}.")
    else:
        lines.append(f"• High leverage: REVIEW. {res.get('high_leverage_n', 0)} point(s) above leverage threshold {fmt(res.get('high_leverage_threshold'), decimals)}.")
        recs.append("Review leverage/Cook-like influence before final model acceptance.")

    vif_review = [r for r in fit.get("vif_rows", []) if pd.notna(r.get("VIF")) and r.get("VIF") >= 5]
    if not vif_review:
        lines.append("• Multicollinearity: PASS. VIF values are below 5.")
    else:
        lines.append(f"• Multicollinearity: REVIEW. {len(vif_review)} predictor(s) have VIF ≥ 5.")
        recs.append("Review predictor redundancy; consider removing or combining correlated predictors.")

    if not recs:
        recs.append("Model assumptions are acceptable. Continue validating with new data.")

    return (
        "Residual and Model Assumption Conclusions\n"
        "────────────────────────────────────────\n"
        + "\n".join(lines)
        + "\n\nRecommendations\n"
        "───────────────\n"
        + "\n".join(f"• {r}" for r in recs)
    )


@dataclass
class MultipleRegressionOptions:
    alpha: float = 0.05
    decimals: int = 4
    show_grid: bool = True
    auto_journal_save: bool = True


# ─────────────────────────────────────────────
# Figure builders
# ─────────────────────────────────────────────



def build_residual_figure(analysis: Dict[str, Any], decimals: int = 4) -> Figure:
    fit = analysis["fit"]
    fitted = np.asarray(fit["fitted"], dtype=float)
    residuals = np.asarray(fit["residuals"], dtype=float)
    std_resid = np.asarray(fit["std_residuals"], dtype=float)
    leverage = np.asarray(fit["leverage"], dtype=float)
    order = np.arange(1, len(residuals) + 1)
    show_grid = bool(analysis["options"].get("show_grid", True))

    fig = Figure(figsize=(11.2, 6.6), dpi=120)
    fig.patch.set_facecolor("white")

    title_ax = fig.add_axes([0.04, 0.875, 0.92, 0.065])
    title_ax.axis("off")
    title_ax.set_title("Multiple Regression Residual Analysis", fontsize=16, fontweight="bold", pad=2)

    gs = fig.add_gridspec(
        2, 3,
        left=0.06, right=0.97, top=0.81, bottom=0.08,
        wspace=0.32, hspace=0.40,
    )

    ax_rf = fig.add_subplot(gs[0, 0])
    ax_hist = fig.add_subplot(gs[0, 1])
    ax_qq = fig.add_subplot(gs[0, 2])
    ax_order = fig.add_subplot(gs[1, 0])
    ax_scale = fig.add_subplot(gs[1, 1])
    ax_summary = fig.add_subplot(gs[1, 2])

    ax_rf.scatter(fitted, residuals, s=42, alpha=0.80, edgecolors="white", linewidths=0.6)
    ax_rf.axhline(0, color="#C62828", linestyle="--", linewidth=1.4)
    ax_rf.set_title("Residuals vs Fitted", fontsize=10, fontweight="bold")
    ax_rf.set_xlabel("Fitted")
    ax_rf.set_ylabel("Residual")
    ax_rf.grid(show_grid, alpha=0.18)

    ax_hist.hist(residuals, bins=min(16, max(6, len(residuals) // 3)), alpha=0.75, edgecolor="white")
    ax_hist.axvline(np.mean(residuals), color="#C62828", linestyle="--", linewidth=1.2)
    ax_hist.set_title("Histogram of Residuals", fontsize=10, fontweight="bold")
    ax_hist.set_xlabel("Residual")
    ax_hist.set_ylabel("Frequency")
    ax_hist.grid(show_grid, alpha=0.18)

    probplot(residuals, dist="norm", plot=ax_qq)
    ax_qq.set_title("Normal Q-Q Plot", fontsize=10, fontweight="bold")
    ax_qq.grid(show_grid, alpha=0.18)

    ax_order.plot(order, residuals, marker="o", linewidth=0.9, alpha=0.80)
    ax_order.axhline(0, color="#C62828", linestyle="--", linewidth=1.2)
    ax_order.set_title("Residuals vs Observation Order", fontsize=10, fontweight="bold")
    ax_order.set_xlabel("Observation order")
    ax_order.set_ylabel("Residual")
    ax_order.grid(show_grid, alpha=0.18)

    ax_scale.scatter(fitted, np.sqrt(np.abs(std_resid)), s=42, alpha=0.80, edgecolors="white", linewidths=0.6)
    ax_scale.set_title("Scale-Location", fontsize=10, fontweight="bold")
    ax_scale.set_xlabel("Fitted")
    ax_scale.set_ylabel("√|Standardized Residual|")
    ax_scale.grid(show_grid, alpha=0.18)

    res = fit["residual_analysis"]
    ax_summary.axis("off")
    rows = [
        ["Normality", "PASS" if res.get("normality_ok") else "REVIEW", f"p={fmt(res.get('shapiro_p'), decimals)}"],
        ["Constant variance", "PASS" if res.get("constant_variance_ok") else "REVIEW", f"p={fmt(res.get('levene_p'), decimals)}"],
        ["Independence", "PASS" if res.get("independence_ok") else "REVIEW", f"DW={fmt(res.get('durbin_watson'), decimals)}"],
        ["Outliers", "PASS" if res.get("outliers_ok") else "REVIEW", f"Count={res.get('outliers_n', 0)}"],
        ["High leverage", "PASS" if res.get("high_leverage_ok") else "REVIEW", f"Count={res.get('high_leverage_n', 0)}"],
    ]
    tbl = ax_summary.table(
        cellText=rows,
        colLabels=["Assumption", "Result", "Evidence"],
        loc="upper center",
        cellLoc="center",
        colWidths=[0.35, 0.22, 0.43],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.2)
    tbl.scale(1.0, 1.50)
    for c in range(3):
        tbl[0, c].set_facecolor("#17324D")
        tbl[0, c].set_text_props(color="white", fontweight="bold")
    ax_summary.set_title("Assumption Summary", fontsize=10, fontweight="bold", pad=8)

    ax_summary.text(0.02, 0.20, analysis.get("residual_conclusion", ""), fontsize=7.6, va="top", wrap=True)

    return fig


# ─────────────────────────────────────────────
# Main widget
# ─────────────────────────────────────────────

class MultipleRegressionBuilder(QWidget):
    def __init__(self, app_window):
        super().__init__()
        self.app_window = app_window
        self.current_df = pd.DataFrame()
        self.current_analysis: Optional[Dict[str, Any]] = None

        self.model_figure: Optional[Figure] = None
        self.model_canvas: Optional[FigureCanvas] = None
        self.residual_figure: Optional[Figure] = None
        self.residual_canvas: Optional[FigureCanvas] = None

        self.current_journal_entry_id: Optional[str] = None
        self.interaction_managers: List[Any] = []

        self.model_scroll_area = None
        self.residual_scroll_area = None
        self.last_canvas_width = 0
        self.last_canvas_height = 0

        self._build_ui()
        self.refresh_columns()

    # UI

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)

        header = QHBoxLayout()
        title = QLabel("Multiple Regression Builder")
        title.setStyleSheet("font-size:20px; font-weight:bold; color:#17324D;")
        header.addWidget(title)
        header.addStretch()
        self.lbl_ws = QLabel("Worksheet: -")
        header.addWidget(self.lbl_ws)
        root.addLayout(header)

        self.tabs = QTabWidget()
        root.addWidget(self.tabs)

        self._build_setup_tab()
        self._build_numeric_tab()
        self._build_model_graph_tab()
        self._build_residual_tab()
        self._build_summary_tab()
        self._build_debug_tab()

    def _build_setup_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(8, 8, 8, 8)

        top = QHBoxLayout()

        var_box = QGroupBox("1. Variables")
        var_layout = QGridLayout(var_box)

        self.cmb_y = QComboBox()
        self.lst_x = QListWidget()
        self.lst_x.setSelectionMode(QAbstractItemView.MultiSelection)
        self.lst_x.setMinimumHeight(220)

        var_layout.addWidget(QLabel("Response Y"), 0, 0)
        var_layout.addWidget(self.cmb_y, 0, 1)
        var_layout.addWidget(QLabel("Predictors X"), 1, 0)
        var_layout.addWidget(self.lst_x, 1, 1)

        opt_box = QGroupBox("2. Options")
        opt = QGridLayout(opt_box)

        self.spn_alpha = QDoubleSpinBox()
        self.spn_alpha.setRange(0.001, 0.999)
        self.spn_alpha.setDecimals(3)
        self.spn_alpha.setSingleStep(0.005)
        self.spn_alpha.setValue(0.05)

        self.spn_decimals = QSpinBox()
        self.spn_decimals.setRange(0, 8)
        self.spn_decimals.setValue(4)

        self.chk_grid = QCheckBox("Show grid")
        self.chk_grid.setChecked(True)

        self.chk_journal = QCheckBox("Auto journal save")
        self.chk_journal.setChecked(True)

        opt.addWidget(QLabel("Alpha"), 0, 0)
        opt.addWidget(self.spn_alpha, 0, 1)
        opt.addWidget(QLabel("Decimals"), 1, 0)
        opt.addWidget(self.spn_decimals, 1, 1)
        opt.addWidget(self.chk_grid, 2, 1)
        opt.addWidget(self.chk_journal, 3, 1)

        top.addWidget(var_box, 2)
        top.addWidget(opt_box, 1)
        layout.addLayout(top)

        run_box = QGroupBox("3. Run")
        run_layout = QVBoxLayout(run_box)
        buttons = QHBoxLayout()

        self.btn_refresh = QPushButton("Refresh Columns")
        self.btn_run = QPushButton("Run Multiple Regression")
        self.btn_run.setStyleSheet("background:#2F5D87; color:white; font-weight:bold; padding:8px; border-radius:6px;")

        self.btn_refresh.clicked.connect(self.refresh_columns)
        self.btn_run.clicked.connect(self.run_analysis)

        buttons.addWidget(self.btn_refresh)
        buttons.addWidget(self.btn_run)
        run_layout.addLayout(buttons)

        note = QLabel(
            "Select one numeric response Y and two or more numeric predictors X. "
            "The module reports model fit, coefficients, VIF, residual diagnostics and predictor impact."
        )
        note.setWordWrap(True)
        note.setStyleSheet("color:#4F6680;")
        run_layout.addWidget(note)

        layout.addWidget(run_box)
        layout.addStretch()
        self.tabs.addTab(tab, "Setup")

    def _build_numeric_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        self.tbl_model_summary = QTableWidget()
        self.tbl_model_summary.setColumnCount(7)
        self.tbl_model_summary.setHorizontalHeaderLabels(["S", "R-sq", "R-sq(adj)", "R-sq(pred)", "PRESS", "Model p", "Conclusion"])

        self.tbl_anova = QTableWidget()
        self.tbl_anova.setColumnCount(6)
        self.tbl_anova.setHorizontalHeaderLabels(["Source", "DF", "Adj SS", "Adj MS", "F-Value", "P-Value"])

        self.tbl_coefficients = QTableWidget()
        self.tbl_coefficients.setColumnCount(9)
        self.tbl_coefficients.setHorizontalHeaderLabels(["Term", "Coef", "SE Coef", "T-Value", "P-Value", "Lower CI", "Upper CI", "Std Coef", "Conclusion"])

        self.tbl_vif = QTableWidget()
        self.tbl_vif.setColumnCount(4)
        self.tbl_vif.setHorizontalHeaderLabels(["Predictor", "R-sq", "VIF", "Conclusion"])

        self.tbl_corr = QTableWidget()
        self.tbl_corr.setColumnCount(4)
        self.tbl_corr.setHorizontalHeaderLabels(["Predictor", "Pearson r", "P-Value", "Strength"])

        layout.addWidget(QLabel("Model Summary"))
        layout.addWidget(self.tbl_model_summary)
        layout.addWidget(QLabel("Analysis of Variance"))
        layout.addWidget(self.tbl_anova)
        layout.addWidget(QLabel("Coefficients"))
        layout.addWidget(self.tbl_coefficients)
        layout.addWidget(QLabel("Variance Inflation Factors"))
        layout.addWidget(self.tbl_vif)
        layout.addWidget(QLabel("Correlation with Response"))
        layout.addWidget(self.tbl_corr)

        self.tabs.addTab(tab, "Numerical Results")

    def _build_model_graph_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        toolbar = QHBoxLayout()
        self.btn_copy_model = QPushButton("Copy Graph")
        self.btn_export_model = QPushButton("Export PNG")
        self.btn_update_journal_1 = QPushButton("Update Journal Entry")

        self.btn_copy_model.clicked.connect(lambda: self.copy_graph("model"))
        self.btn_export_model.clicked.connect(lambda: self.export_png("model"))
        self.btn_update_journal_1.clicked.connect(self.update_journal_entry)

        toolbar.addStretch()
        toolbar.addWidget(self.btn_copy_model)
        toolbar.addWidget(self.btn_export_model)
        toolbar.addWidget(self.btn_update_journal_1)
        layout.addLayout(toolbar)

        self.model_container = QWidget()
        self.model_layout = QVBoxLayout(self.model_container)
        layout.addWidget(self.model_container)

        self.tabs.addTab(tab, "Model Graph")

    def _build_residual_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        toolbar = QHBoxLayout()
        self.btn_copy_residual = QPushButton("Copy Graph")
        self.btn_export_residual = QPushButton("Export PNG")
        self.btn_update_journal_2 = QPushButton("Update Journal Entry")
        self.btn_zoom_out_residual = QPushButton("Zoom -")
        self.btn_zoom_fit_residual = QPushButton("Fit")
        self.btn_zoom_reset_residual = QPushButton("100%")
        self.btn_zoom_in_residual = QPushButton("Zoom +")

        self.btn_copy_residual.clicked.connect(lambda: self.copy_graph("residual"))
        self.btn_export_residual.clicked.connect(lambda: self.export_png("residual"))
        self.btn_update_journal_2.clicked.connect(self.update_journal_entry)
        self.btn_zoom_out_residual.clicked.connect(lambda: self.zoom_graph("residual", "out"))
        self.btn_zoom_fit_residual.clicked.connect(lambda: self.zoom_graph("residual", "fit"))
        self.btn_zoom_reset_residual.clicked.connect(lambda: self.zoom_graph("residual", "reset"))
        self.btn_zoom_in_residual.clicked.connect(lambda: self.zoom_graph("residual", "in"))

        toolbar.addStretch()
        toolbar.addWidget(self.btn_zoom_out_residual)
        toolbar.addWidget(self.btn_zoom_fit_residual)
        toolbar.addWidget(self.btn_zoom_reset_residual)
        toolbar.addWidget(self.btn_zoom_in_residual)
        toolbar.addWidget(self.btn_copy_residual)
        toolbar.addWidget(self.btn_export_residual)
        toolbar.addWidget(self.btn_update_journal_2)
        layout.addLayout(toolbar)

        self.residual_container = QWidget()
        self.residual_layout = QVBoxLayout(self.residual_container)
        layout.addWidget(self.residual_container)

        self.txt_residual = QTextEdit()
        self.txt_residual.setReadOnly(True)
        self.txt_residual.setMaximumHeight(180)
        layout.addWidget(self.txt_residual)

        self.tabs.addTab(tab, "Residual Analysis")

    def _build_summary_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        top = QHBoxLayout()
        self.btn_copy_summary = QPushButton("Copy Summary")
        self.btn_copy_summary.clicked.connect(lambda: QApplication.clipboard().setText(self.txt_summary.toPlainText()))
        top.addStretch()
        top.addWidget(self.btn_copy_summary)
        layout.addLayout(top)

        self.txt_summary = QTextEdit()
        self.txt_summary.setReadOnly(True)
        layout.addWidget(self.txt_summary)

        self.tabs.addTab(tab, "Summary")

    def _build_debug_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        self.txt_debug = QTextEdit()
        self.txt_debug.setReadOnly(True)
        layout.addWidget(self.txt_debug)

        self.tabs.addTab(tab, "Debug")

    # Data and analysis

    def log(self, message: str):
        self.txt_debug.append(f"[{now_string()}] {message}")

    def refresh_columns(self):
        self.current_df = self.app_window.get_active_dataframe()
        self.lbl_ws.setText(f"Worksheet: {self.app_window.active_worksheet_name()}")

        cols = []
        if not self.current_df.empty:
            for col in self.current_df.columns:
                s = pd.to_numeric(self.current_df[col], errors="coerce")
                if s.notna().sum() >= 4:
                    cols.append(str(col))

        self.cmb_y.clear()
        self.lst_x.clear()

        self.cmb_y.addItems(cols)
        for col in cols:
            self.lst_x.addItem(col)

        if len(cols) >= 3:
            self.cmb_y.setCurrentIndex(0)
            for i in range(1, self.lst_x.count()):
                self.lst_x.item(i).setSelected(True)

        self.log(f"Numeric columns refreshed: {cols}")

    def _options(self) -> MultipleRegressionOptions:
        return MultipleRegressionOptions(
            alpha=float(self.spn_alpha.value()),
            decimals=int(self.spn_decimals.value()),
            show_grid=self.chk_grid.isChecked(),
            auto_journal_save=self.chk_journal.isChecked(),
        )

    def _selected_predictors(self) -> List[str]:
        return [item.text() for item in self.lst_x.selectedItems()]

    def run_analysis(self):
        try:
            opts = self._options()

            if self.current_df.empty:
                raise ValueError("No active worksheet data found.")

            y_col = self.cmb_y.currentText()
            x_cols = self._selected_predictors()

            if not y_col:
                raise ValueError("Select a response Y column.")
            if len(x_cols) < 2:
                raise ValueError("Select at least two predictor X columns.")
            if y_col in x_cols:
                raise ValueError("Response Y cannot also be selected as a predictor X.")

            fit = multiple_regression_fit(self.current_df, y_col, x_cols, opts.alpha)
            res_txt = residual_conclusion(fit, opts.alpha, opts.decimals)

            analysis = {
                "id": str(uuid.uuid4()),
                "module": "Multiple Regression",
                "type": "Multiple Regression",
                "worksheet": self.app_window.active_worksheet_name(),
                "options": asdict(opts),
                "fit": fit,
                "residual_conclusion": res_txt,
                "created_at": now_string(),
            }
            analysis["summary"] = self._build_summary_text(analysis)

            self.current_analysis = analysis
            self._populate_numeric_results(analysis)
            self._render_figures(analysis)
            self.txt_summary.setPlainText(analysis["summary"])
            self.txt_residual.setPlainText(res_txt)

            if opts.auto_journal_save:
                self.save_to_journal()

            self.tabs.setCurrentIndex(1)
            self.log("Multiple Regression completed successfully.")

        except Exception as exc:
            self.log(f"ERROR: {exc}")
            QMessageBox.critical(self, "Multiple Regression Error", str(exc))

    def _populate_numeric_results(self, analysis: Dict[str, Any]):
        d = int(analysis["options"]["decimals"])
        fit = analysis["fit"]
        alpha = float(analysis["options"].get("alpha", 0.05))

        # Model Summary
        ms_values = [[
            fmt(fit.get("s"), d),
            f"{fmt(100 * fit.get('r2'), 2)}%",
            f"{fmt(100 * fit.get('adj_r2'), 2)}%",
            f"{fmt(100 * fit.get('pred_r2'), 2)}%",
            fmt(fit.get("press"), d),
            fmt(fit.get("p_model"), d),
            significance_label(fit.get("p_model"), alpha),
        ]]
        self.tbl_model_summary.setRowCount(1)
        for c, value in enumerate(ms_values[0]):
            self.tbl_model_summary.setItem(0, c, QTableWidgetItem(str(value)))
        self.tbl_model_summary.resizeColumnsToContents()

        # ANOVA
        anova_rows = [
            ["Regression", fit["df_reg"], fmt(fit["ss_reg"], d), fmt(fit["ms_reg"], d), fmt(fit["f_value"], d), fmt(fit["p_model"], d)],
            ["Error", fit["df_err"], fmt(fit["ss_error"], d), fmt(fit["ms_error"], d), "", ""],
            ["Total", fit["df_total"], fmt(fit["ss_total"], d), "", "", ""],
        ]
        self.tbl_anova.setRowCount(len(anova_rows))
        for r, row in enumerate(anova_rows):
            for c, value in enumerate(row):
                self.tbl_anova.setItem(r, c, QTableWidgetItem(str(value)))
        self.tbl_anova.resizeColumnsToContents()

        # Coefficients
        coef_rows = fit.get("coefficient_rows", [])
        self.tbl_coefficients.setRowCount(len(coef_rows))
        for r, row in enumerate(coef_rows):
            values = [
                row.get("Term", ""),
                fmt(row.get("Coef"), d),
                fmt(row.get("SE Coef"), d),
                fmt(row.get("T-Value"), d),
                fmt(row.get("P-Value"), d),
                fmt(row.get("Lower CI"), d),
                fmt(row.get("Upper CI"), d),
                fmt(row.get("Std Coef"), d) if row.get("Term") != "Constant" else "",
                row.get("Conclusion", ""),
            ]
            for c, value in enumerate(values):
                self.tbl_coefficients.setItem(r, c, QTableWidgetItem(str(value)))
        self.tbl_coefficients.resizeColumnsToContents()

        # VIF
        vif_rows = fit.get("vif_rows", [])
        self.tbl_vif.setRowCount(len(vif_rows))
        for r, row in enumerate(vif_rows):
            values = [row.get("Predictor", ""), fmt(row.get("R-sq"), d), fmt(row.get("VIF"), d), row.get("Conclusion", "")]
            for c, value in enumerate(values):
                self.tbl_vif.setItem(r, c, QTableWidgetItem(str(value)))
        self.tbl_vif.resizeColumnsToContents()

        # Correlations
        corr_rows = fit.get("correlation_rows", [])
        self.tbl_corr.setRowCount(len(corr_rows))
        for r, row in enumerate(corr_rows):
            values = [row.get("Predictor", ""), fmt(row.get("Pearson r"), d), fmt(row.get("P-Value"), d), row.get("Strength", "")]
            for c, value in enumerate(values):
                self.tbl_corr.setItem(r, c, QTableWidgetItem(str(value)))
        self.tbl_corr.resizeColumnsToContents()

    def _render_figures(self, analysis: Dict[str, Any]):
        self._clear_layout(self.model_layout)
        self._clear_layout(self.residual_layout)

        d = int(analysis["options"]["decimals"])

        self.model_figure = build_model_figure(analysis, d)
        self.model_canvas = FigureCanvas(self.model_figure)
        self.model_layout.addWidget(self.model_canvas)
        self.model_canvas.draw()

        self.residual_figure = build_residual_figure(analysis, d)
        self.residual_canvas = FigureCanvas(self.residual_figure)
        self.residual_layout.addWidget(self.residual_canvas)
        self.residual_canvas.draw()

        self._enable_interactions()

    def _enable_interactions(self):
        self.interaction_managers.clear()
        if enable_interaction is None:
            return

        for fig, canvas in [(self.model_figure, self.model_canvas), (self.residual_figure, self.residual_canvas)]:
            if fig is None or canvas is None:
                continue
            try:
                manager = enable_interaction(
                    host=self,
                    figure_getter=lambda f=fig: f,
                    canvas_getter=lambda c=canvas: c,
                    decimals_getter=lambda: int(self.spn_decimals.value()),
                    on_status=lambda msg: self.app_window.statusBar().showMessage(msg),
                )
                manager.register_axes()
                self.interaction_managers.append(manager)
            except Exception as exc:
                self.log(f"Plot interaction was not enabled: {exc}")

    def _clear_layout(self, layout):
        while layout.count():
            child = layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

    def _build_summary_text(self, analysis: Dict[str, Any]) -> str:
        d = int(analysis["options"]["decimals"])
        fit = analysis["fit"]
        alpha = float(analysis["options"].get("alpha", 0.05))

        lines = [
            "=" * 92,
            "MULTIPLE REGRESSION REPORT — SPS STUDIO",
            "=" * 92,
            f"Worksheet: {analysis.get('worksheet', '')}",
            f"Response Y: {fit.get('y_name', '')}",
            f"Predictors X: {', '.join(fit.get('x_names', []))}",
            f"N: {fit.get('n')}",
            f"Alpha: {fmt(alpha, 3)}",
            "",
            "MODEL SUMMARY",
            "-" * 92,
            f"S={fmt(fit.get('s'), d)} | R-sq={fmt(100 * fit.get('r2'), 2)}% | "
            f"R-sq(adj)={fmt(100 * fit.get('adj_r2'), 2)}% | R-sq(pred)={fmt(100 * fit.get('pred_r2'), 2)}%",
            f"Model p-value={fmt(fit.get('p_model'), d)} | Decision={p_decision(fit.get('p_model'), alpha)} | {significance_label(fit.get('p_model'), alpha)}",
            "",
            "REGRESSION EQUATION",
            "-" * 92,
            fit.get("equation", ""),
            "",
            "ANALYSIS OF VARIANCE",
            "-" * 92,
            f"{'Source':<14}{'DF':<8}{'Adj SS':<14}{'Adj MS':<14}{'F-Value':<14}{'P-Value':<14}",
            f"{'Regression':<14}{fit['df_reg']:<8}{fmt(fit['ss_reg'], d):<14}{fmt(fit['ms_reg'], d):<14}{fmt(fit['f_value'], d):<14}{fmt(fit['p_model'], d):<14}",
            f"{'Error':<14}{fit['df_err']:<8}{fmt(fit['ss_error'], d):<14}{fmt(fit['ms_error'], d):<14}{'':<14}{'':<14}",
            f"{'Total':<14}{fit['df_total']:<8}{fmt(fit['ss_total'], d):<14}{'':<14}{'':<14}{'':<14}",
            "",
            "COEFFICIENTS",
            "-" * 92,
            f"{'Term':<24}{'Coef':<14}{'SE Coef':<14}{'T':<12}{'P':<12}{'Conclusion':<24}",
        ]

        for row in fit.get("coefficient_rows", []):
            lines.append(
                f"{row.get('Term',''):<24}{fmt(row.get('Coef'), d):<14}{fmt(row.get('SE Coef'), d):<14}"
                f"{fmt(row.get('T-Value'), d):<12}{fmt(row.get('P-Value'), d):<12}{row.get('Conclusion',''):<24}"
            )

        lines += [
            "",
            "MULTICOLLINEARITY / VIF",
            "-" * 92,
            f"{'Predictor':<24}{'VIF':<14}{'Conclusion':<24}",
        ]
        for row in fit.get("vif_rows", []):
            lines.append(f"{row.get('Predictor',''):<24}{fmt(row.get('VIF'), d):<14}{row.get('Conclusion',''):<24}")

        lines += [
            "",
            "RESIDUAL ANALYSIS",
            "-" * 92,
            analysis.get("residual_conclusion", ""),
        ]

        return "\n".join(lines)

    # Clipboard / Export / Journal

    def _figure_by_name(self, graph_type: str) -> Optional[Figure]:
        if graph_type == "model":
            return self.model_figure
        if graph_type == "residual":
            return self.residual_figure
        return None

    def copy_graph(self, graph_type: str):
        fig = self._figure_by_name(graph_type)
        if fig is None:
            QMessageBox.information(self, "No graph", "Run an analysis first.")
            return
        try:
            image = QImage()
            if not image.loadFromData(figure_to_png_bytes(fig), "PNG"):
                raise ValueError("Could not convert graph to clipboard image.")
            QApplication.clipboard().setImage(image)
            self.app_window.statusBar().showMessage("Graph copied to clipboard as image.")
        except Exception as exc:
            QMessageBox.critical(self, "Copy failed", str(exc))

    def export_png(self, graph_type: str):
        fig = self._figure_by_name(graph_type)
        if fig is None:
            QMessageBox.information(self, "No graph", "Run an analysis first.")
            return

        path, _ = QFileDialog.getSaveFileName(self, "Export PNG", "", "PNG Image (*.png)")
        if not path:
            return
        if not path.lower().endswith(".png"):
            path += ".png"
        fig.savefig(path, dpi=160, bbox_inches="tight")
        self.app_window.statusBar().showMessage(f"Graph exported: {path}")

    def _journal_entry_payload(self) -> Dict[str, Any]:
        if self.current_analysis is None:
            raise ValueError("No analysis has been run.")

        model_b64 = figure_to_png_base64(self.model_figure) if self.model_figure is not None else ""
        residual_b64 = figure_to_png_base64(self.residual_figure) if self.residual_figure is not None else ""

        fit = self.current_analysis.get("fit", {})
        name = f"Multiple Regression - {fit.get('y_name', '')}"

        return {
            "id": self.current_journal_entry_id or str(uuid.uuid4()),
            "name": name,
            "type": "Multiple Regression",
            "module": "Multiple Regression",
            "worksheet": self.current_analysis.get("worksheet", ""),
            "status": p_decision(fit.get("p_model"), float(self.current_analysis.get("options", {}).get("alpha", 0.05))),
            "created_at": self.current_analysis.get("created_at", now_string()),
            "modified_at": now_string(),
            "summary": self.txt_summary.toPlainText(),
            "figure_png_base64": model_b64,
            "residual_png_base64": residual_b64,
            "payload": serializable(self.current_analysis),
        }

    def save_to_journal(self):
        try:
            entry = self._journal_entry_payload()
            if self.current_journal_entry_id:
                self.app_window.update_journal_entry(self.current_journal_entry_id, entry)
            else:
                self.current_journal_entry_id = entry["id"]
                self.app_window.add_journal_entry(entry)
            self.log("Journal entry saved.")
        except Exception as exc:
            self.log(f"Journal save failed: {exc}")

    def update_journal_entry(self):
        if not self.current_analysis:
            QMessageBox.information(self, "No analysis", "Run an analysis before updating the journal.")
            return
        self.save_to_journal()
        self.app_window.statusBar().showMessage("Multiple Regression journal entry updated.")

    def load_from_journal_entry(self, entry: Dict[str, Any]):
        self.current_journal_entry_id = entry.get("id")
        payload = entry.get("payload", {})
        self.current_analysis = payload
        self.txt_summary.setPlainText(entry.get("summary", payload.get("summary", "")))
        self._populate_numeric_results(payload)
        self._render_figures(payload)
        self.txt_residual.setPlainText(payload.get("residual_conclusion", ""))


# =============================================================================
# SPS Studio patch — Multiple Regression Minitab-style output + Prediction tab
# =============================================================================

def r2_explanation_label(r2: float) -> str:
    if pd.isna(r2):
        return "N/A"
    pct = 100.0 * float(r2)
    if pct < 30.0:
        return "Low"
    if pct < 70.0:
        return "Medium"
    return "High"


def fit_linear_model_arrays(y: np.ndarray, X_raw: np.ndarray) -> Dict[str, Any]:
    n = len(y)
    p = X_raw.shape[1]
    X = np.column_stack([np.ones(n), X_raw])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    fitted = X @ beta
    residuals = y - fitted
    ss_error = float(np.sum(residuals ** 2))
    ss_total = float(np.sum((y - np.mean(y)) ** 2))
    df_error = n - p - 1
    ms_error = ss_error / df_error if df_error > 0 else np.nan
    xtx_inv = np.linalg.pinv(X.T @ X)
    h = np.sum((X @ xtx_inv) * X, axis=1)
    return {
        "beta": beta,
        "fitted": fitted,
        "residuals": residuals,
        "ss_error": ss_error,
        "ss_total": ss_total,
        "df_error": df_error,
        "ms_error": ms_error,
        "x_design": X,
        "leverage": h,
        "xtx_inv": xtx_inv,
    }


def lack_of_fit_table(data: pd.DataFrame, y_col: str, x_cols: List[str], ss_error: float, df_error: int) -> Dict[str, Any]:
    try:
        grouped = data.groupby(x_cols, dropna=False)[y_col]
        pure_error_ss = 0.0
        pure_error_df = 0
        for _, g in grouped:
            vals = g.to_numpy(dtype=float)
            if len(vals) > 1:
                pure_error_ss += float(np.sum((vals - np.mean(vals)) ** 2))
                pure_error_df += len(vals) - 1

        lof_ss = float(ss_error - pure_error_ss)
        lof_df = int(df_error - pure_error_df)

        if pure_error_df > 0 and lof_df > 0:
            lof_ms = lof_ss / lof_df
            pe_ms = pure_error_ss / pure_error_df
            lof_f = lof_ms / pe_ms if pe_ms > 0 else np.nan
            lof_p = float(stats.f.sf(lof_f, lof_df, pure_error_df)) if pd.notna(lof_f) else np.nan
        else:
            lof_ms = np.nan
            pe_ms = pure_error_ss / pure_error_df if pure_error_df > 0 else np.nan
            lof_f = np.nan
            lof_p = np.nan

        return {
            "lack_of_fit_ss": lof_ss if lof_df > 0 else np.nan,
            "lack_of_fit_df": lof_df if lof_df > 0 else 0,
            "lack_of_fit_ms": lof_ms,
            "lack_of_fit_f": lof_f,
            "lack_of_fit_p": lof_p,
            "pure_error_ss": pure_error_ss if pure_error_df > 0 else np.nan,
            "pure_error_df": pure_error_df,
            "pure_error_ms": pe_ms,
        }
    except Exception:
        return {
            "lack_of_fit_ss": np.nan,
            "lack_of_fit_df": 0,
            "lack_of_fit_ms": np.nan,
            "lack_of_fit_f": np.nan,
            "lack_of_fit_p": np.nan,
            "pure_error_ss": np.nan,
            "pure_error_df": 0,
            "pure_error_ms": np.nan,
        }


def multiple_regression_fit(df: pd.DataFrame, y_col: str, x_cols: List[str], alpha: float) -> Dict[str, Any]:
    data = df[[y_col] + x_cols].copy()
    for col in [y_col] + x_cols:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data = data.dropna(subset=[y_col] + x_cols)

    if len(data) < len(x_cols) + 3:
        raise ValueError("Not enough valid rows. Multiple regression requires n > predictors + 2.")

    y = data[y_col].to_numpy(dtype=float)
    X_raw = data[x_cols].to_numpy(dtype=float)

    if np.any(np.nanstd(X_raw, axis=0) == 0):
        bad = [x_cols[i] for i, s in enumerate(np.nanstd(X_raw, axis=0)) if s == 0]
        raise ValueError(f"Predictor(s) with zero variation are not valid: {', '.join(bad)}")

    n = len(y)
    p = len(x_cols)
    full = fit_linear_model_arrays(y, X_raw)
    beta = full["beta"]
    fitted = full["fitted"]
    residuals = full["residuals"]
    ss_error = full["ss_error"]
    ss_total = full["ss_total"]
    ms_error = full["ms_error"]
    h = full["leverage"]
    xtx_inv = full["xtx_inv"]
    rank = int(np.linalg.matrix_rank(full["x_design"]))

    df_reg = p
    df_err = n - p - 1
    df_total = n - 1
    ss_reg = float(ss_total - ss_error)
    ms_reg = ss_reg / df_reg if df_reg > 0 else np.nan
    f_value = ms_reg / ms_error if pd.notna(ms_error) and ms_error > 0 else np.nan
    p_model = float(stats.f.sf(f_value, df_reg, df_err)) if pd.notna(f_value) else np.nan

    r2 = 1.0 - ss_error / ss_total if ss_total > 0 else np.nan
    adj_r2 = 1.0 - (1.0 - r2) * (n - 1) / df_err if df_err > 0 and pd.notna(r2) else np.nan
    s = math.sqrt(ms_error) if pd.notna(ms_error) and ms_error >= 0 else np.nan

    press = 0.0
    for i in range(n):
        denom = 1.0 - h[i]
        if abs(denom) < 1e-12:
            press = np.nan
            break
        press += float((residuals[i] / denom) ** 2)
    pred_r2 = 1.0 - press / ss_total if pd.notna(press) and ss_total > 0 else np.nan

    se_beta = np.sqrt(np.diag(xtx_inv) * ms_error) if pd.notna(ms_error) and ms_error >= 0 else np.full(p + 1, np.nan)
    t_values = beta / se_beta
    p_values = 2 * stats.t.sf(np.abs(t_values), df_err)
    tcrit = stats.t.ppf(1 - alpha / 2, df_err) if df_err > 0 else np.nan

    coefficient_rows = []
    term_names = ["Constant"] + x_cols
    for i, term in enumerate(term_names):
        lo = beta[i] - tcrit * se_beta[i] if pd.notna(tcrit) else np.nan
        hi = beta[i] + tcrit * se_beta[i] if pd.notna(tcrit) else np.nan
        coefficient_rows.append({
            "Term": term,
            "Coef": float(beta[i]),
            "SE Coef": float(se_beta[i]) if pd.notna(se_beta[i]) else np.nan,
            "T-Value": float(t_values[i]) if pd.notna(t_values[i]) else np.nan,
            "P-Value": float(p_values[i]) if pd.notna(p_values[i]) else np.nan,
            "Lower CI": float(lo) if pd.notna(lo) else np.nan,
            "Upper CI": float(hi) if pd.notna(hi) else np.nan,
            "Decision": p_decision(p_values[i], alpha),
            "Conclusion": "Significant" if i > 0 and pd.notna(p_values[i]) and p_values[i] <= alpha else ("Intercept" if i == 0 else "Not significant"),
        })

    y_sd = np.std(y, ddof=1)
    x_sds = np.std(X_raw, axis=0, ddof=1)
    for i, row in enumerate(coefficient_rows[1:]):
        row["Std Coef"] = float(beta[i + 1] * x_sds[i] / y_sd) if y_sd > 0 and x_sds[i] > 0 else np.nan

    # Adjusted SS per term using reduced model comparison.
    term_anova_rows = []
    for j, x_name in enumerate(x_cols):
        reduced_cols = [i for i in range(p) if i != j]
        if reduced_cols:
            reduced = fit_linear_model_arrays(y, X_raw[:, reduced_cols])
            ss_reduced_error = reduced["ss_error"]
        else:
            ss_reduced_error = float(np.sum((y - np.mean(y)) ** 2))
        ss_term = float(ss_reduced_error - ss_error)
        df_term = 1
        ms_term = ss_term
        f_term = ms_term / ms_error if pd.notna(ms_error) and ms_error > 0 else np.nan
        p_term = float(stats.f.sf(f_term, df_term, df_err)) if pd.notna(f_term) else np.nan
        term_anova_rows.append({
            "source": x_name,
            "df": df_term,
            "adj_ss": ss_term,
            "adj_ms": ms_term,
            "f_value": f_term,
            "p_value": p_term,
        })

    lof = lack_of_fit_table(data, y_col, x_cols, ss_error, df_err)

    anova_rows = [
        {"source": "Regression", "df": df_reg, "adj_ss": ss_reg, "adj_ms": ms_reg, "f_value": f_value, "p_value": p_model},
        *term_anova_rows,
        {"source": "Error", "df": df_err, "adj_ss": ss_error, "adj_ms": ms_error, "f_value": np.nan, "p_value": np.nan},
    ]
    if lof.get("lack_of_fit_df", 0) > 0:
        anova_rows.append({
            "source": "Lack-of-Fit",
            "df": lof["lack_of_fit_df"],
            "adj_ss": lof["lack_of_fit_ss"],
            "adj_ms": lof["lack_of_fit_ms"],
            "f_value": lof["lack_of_fit_f"],
            "p_value": lof["lack_of_fit_p"],
        })
    if lof.get("pure_error_df", 0) > 0:
        anova_rows.append({
            "source": "Pure Error",
            "df": lof["pure_error_df"],
            "adj_ss": lof["pure_error_ss"],
            "adj_ms": lof["pure_error_ms"],
            "f_value": np.nan,
            "p_value": np.nan,
        })
    anova_rows.append({"source": "Total", "df": df_total, "adj_ss": ss_total, "adj_ms": np.nan, "f_value": np.nan, "p_value": np.nan})

    std_resid = standardized_residuals(residuals, ms_error, h)
    stud_deleted = studentized_deleted_residuals(residuals, ms_error, h, df_err)
    sh_stat, sh_p = safe_shapiro(residuals)
    lev_stat, lev_p = variance_stability_test(fitted, residuals)
    dw = durbin_watson_stat(residuals)
    outliers_n = int(np.sum(np.abs(std_resid) > 3)) if len(std_resid) else 0
    high_leverage_threshold = 2.0 * (p + 1) / n
    high_leverage_n = int(np.sum(h > high_leverage_threshold)) if len(h) else 0

    vif_rows = compute_vif(X_raw, x_cols)
    vif_by_name = {r["Predictor"]: r["VIF"] for r in vif_rows}

    for row in coefficient_rows:
        if row["Term"] == "Constant":
            row["VIF"] = ""
        else:
            row["VIF"] = vif_by_name.get(row["Term"], np.nan)

    corr_rows = []
    for x_col in x_cols:
        r, p_corr = stats.pearsonr(data[x_col].to_numpy(dtype=float), y)
        corr_rows.append({
            "Predictor": x_col,
            "Pearson r": float(r),
            "P-Value": float(p_corr),
            "Strength": correlation_strength(r),
        })

    abs_std = np.array([abs(r.get("Std Coef", np.nan)) for r in coefficient_rows[1:]], dtype=float)
    if np.nansum(abs_std) > 0:
        for idx, row in enumerate(coefficient_rows[1:]):
            row["Relative Impact %"] = float(100.0 * abs_std[idx] / np.nansum(abs_std))
    else:
        for row in coefficient_rows[1:]:
            row["Relative Impact %"] = np.nan

    unusual_rows = []
    for i in range(n):
        is_unusual = bool(abs(std_resid[i]) > 2.0 or h[i] > high_leverage_threshold)
        if is_unusual:
            flags = []
            if abs(std_resid[i]) > 2.0:
                flags.append("R")
            if h[i] > high_leverage_threshold:
                flags.append("X")
            unusual_rows.append({
                "Obs": int(i + 1),
                y_col: float(y[i]),
                "Fit": float(fitted[i]),
                "Resid": float(residuals[i]),
                "Std Resid": float(std_resid[i]),
                "Flag": " ".join(flags),
            })

    equation = f"{y_col} = " + fmt(beta[0], 4)
    for coef, name in zip(beta[1:], x_cols):
        sign = "+" if coef >= 0 else "-"
        equation += f" {sign} {fmt(abs(coef), 4)} {name}"

    return {
        "n": int(n),
        "p": int(p),
        "rank": rank,
        "y_name": y_col,
        "x_names": x_cols,
        "data": data,
        "x_ranges": {col: {"min": float(data[col].min()), "max": float(data[col].max()), "mean": float(data[col].mean())} for col in x_cols},
        "y_range": {"min": float(np.min(y)), "max": float(np.max(y)), "mean": float(np.mean(y))},
        "y_values": y,
        "x_matrix": X_raw,
        "x_design": full["x_design"],
        "beta": beta,
        "term_names": term_names,
        "equation": equation,
        "fitted": fitted,
        "residuals": residuals,
        "std_residuals": std_resid,
        "studentized_deleted_residuals": stud_deleted,
        "leverage": h,
        "df_reg": int(df_reg),
        "df_err": int(df_err),
        "df_total": int(df_total),
        "ss_reg": ss_reg,
        "ss_error": ss_error,
        "ss_total": ss_total,
        "ms_reg": ms_reg,
        "ms_error": ms_error,
        "f_value": float(f_value) if pd.notna(f_value) else np.nan,
        "p_model": p_model,
        "s": s,
        "r2": float(r2) if pd.notna(r2) else np.nan,
        "adj_r2": float(adj_r2) if pd.notna(adj_r2) else np.nan,
        "pred_r2": float(pred_r2) if pd.notna(pred_r2) else np.nan,
        "press": float(press) if pd.notna(press) else np.nan,
        "anova_rows": anova_rows,
        "coefficient_rows": coefficient_rows,
        "vif_rows": vif_rows,
        "correlation_rows": corr_rows,
        "unusual_rows": unusual_rows,
        "residual_analysis": {
            "shapiro_stat": sh_stat,
            "shapiro_p": sh_p,
            "normality_ok": pd.notna(sh_p) and sh_p > alpha,
            "levene_stat": lev_stat,
            "levene_p": lev_p,
            "constant_variance_ok": pd.notna(lev_p) and lev_p > alpha,
            "durbin_watson": dw,
            "independence_ok": pd.notna(dw) and 1.5 <= dw <= 2.5,
            "outliers_n": outliers_n,
            "outliers_ok": outliers_n == 0,
            "high_leverage_n": high_leverage_n,
            "high_leverage_threshold": high_leverage_threshold,
            "high_leverage_ok": high_leverage_n == 0,
        }
    }


def predict_from_fit(fit: Dict[str, Any], x_values: Dict[str, float], enabled: Optional[Dict[str, bool]] = None) -> float:
    beta = np.asarray(fit["beta"], dtype=float)
    x_names = fit["x_names"]
    y_hat = float(beta[0])
    for i, name in enumerate(x_names):
        use_term = True if enabled is None else bool(enabled.get(name, True))
        if use_term:
            x_val = float(x_values.get(name, fit["x_ranges"][name]["mean"]))
        else:
            # Disabled predictors remain fixed at their historical mean.
            x_val = float(fit["x_ranges"][name]["mean"])
        y_hat += float(beta[i + 1]) * x_val
    return y_hat


def optimize_prediction(fit: Dict[str, Any], objective: str, target: Optional[float] = None, enabled: Optional[Dict[str, bool]] = None) -> Dict[str, Any]:
    beta = np.asarray(fit["beta"], dtype=float)
    x_names = fit["x_names"]
    enabled = enabled or {name: True for name in x_names}

    x_result = {}
    for i, name in enumerate(x_names):
        rng = fit["x_ranges"][name]
        lo, hi, mean = float(rng["min"]), float(rng["max"]), float(rng["mean"])
        if not enabled.get(name, True):
            x_result[name] = mean
            continue

        coef = float(beta[i + 1])
        if objective == "Maximize Y":
            x_result[name] = hi if coef >= 0 else lo
        elif objective == "Minimize Y":
            x_result[name] = lo if coef >= 0 else hi
        else:
            x_result[name] = mean

    if objective == "Target Y":
        if target is None:
            target = float(fit["y_range"]["mean"])

        adjustable = [name for name in x_names if enabled.get(name, True)]
        if not adjustable:
            y_pred = predict_from_fit(fit, x_result, enabled)
            return {"x_values": x_result, "predicted_y": y_pred, "objective_error": abs(y_pred - target)}

        # Coordinate descent inside observed ranges. This is bounded interpolation, not extrapolation.
        for _ in range(80):
            improved = False
            current_error = abs(predict_from_fit(fit, x_result, enabled) - target)
            for name in adjustable:
                rng = fit["x_ranges"][name]
                lo, hi = float(rng["min"]), float(rng["max"])
                candidates = np.linspace(lo, hi, 51)
                best_val = x_result[name]
                best_err = current_error
                for val in candidates:
                    trial = dict(x_result)
                    trial[name] = float(val)
                    err = abs(predict_from_fit(fit, trial, enabled) - target)
                    if err < best_err:
                        best_err = err
                        best_val = float(val)
                if best_err + 1e-12 < current_error:
                    x_result[name] = best_val
                    current_error = best_err
                    improved = True
            if not improved:
                break

    y_pred = predict_from_fit(fit, x_result, enabled)
    err = abs(y_pred - target) if objective == "Target Y" and target is not None else np.nan
    return {"x_values": x_result, "predicted_y": y_pred, "objective_error": err}




def _mr_build_ui(self):
    root = QVBoxLayout(self)
    root.setContentsMargins(0, 0, 0, 0)

    header = QHBoxLayout()
    title = QLabel("Multiple Regression Builder")
    title.setStyleSheet("font-size:20px; font-weight:bold; color:#17324D;")
    header.addWidget(title)
    header.addStretch()
    self.lbl_ws = QLabel("Worksheet: -")
    header.addWidget(self.lbl_ws)
    root.addLayout(header)

    self.tabs = QTabWidget()
    root.addWidget(self.tabs)

    self._build_setup_tab()
    self._build_numeric_tab()
    self._build_model_graph_tab()
    self._build_prediction_tab()
    self._build_residual_tab()
    self._build_summary_tab()
    self._build_debug_tab()


def _mr_build_numeric_tab(self):
    tab = QWidget()
    layout = QVBoxLayout(tab)

    self.txt_equation = QTextEdit()
    self.txt_equation.setReadOnly(True)
    self.txt_equation.setMaximumHeight(75)

    self.tbl_coefficients = QTableWidget()
    self.tbl_coefficients.setColumnCount(6)
    self.tbl_coefficients.setHorizontalHeaderLabels(["Term", "Coef", "SE Coef", "T-Value", "P-Value", "VIF"])

    self.tbl_model_summary = QTableWidget()
    self.tbl_model_summary.setColumnCount(4)
    self.tbl_model_summary.setHorizontalHeaderLabels(["S", "R-sq", "R-sq(adj)", "R-sq(pred)"])

    self.tbl_anova = QTableWidget()
    self.tbl_anova.setColumnCount(6)
    self.tbl_anova.setHorizontalHeaderLabels(["Source", "DF", "Adj SS", "Adj MS", "F-Value", "P-Value"])

    self.tbl_unusual = QTableWidget()
    self.tbl_unusual.setColumnCount(6)
    self.tbl_unusual.setHorizontalHeaderLabels(["Obs", "Response", "Fit", "Resid", "Std Resid", "Flag"])

    layout.addWidget(QLabel("Regression Equation"))
    layout.addWidget(self.txt_equation)
    layout.addWidget(QLabel("Coefficients"))
    layout.addWidget(self.tbl_coefficients)
    layout.addWidget(QLabel("Model Summary"))
    layout.addWidget(self.tbl_model_summary)
    layout.addWidget(QLabel("Analysis of Variance"))
    layout.addWidget(self.tbl_anova)
    layout.addWidget(QLabel("Fits and Diagnostics for Unusual Observations"))
    layout.addWidget(self.tbl_unusual)

    self.tabs.addTab(tab, "Numerical Results")


def _mr_build_model_graph_tab(self):
    tab = QWidget()
    layout = QVBoxLayout(tab)

    toolbar = QHBoxLayout()
    self.btn_copy_model = QPushButton("Copy Graph")
    self.btn_export_model = QPushButton("Export PNG")
    self.btn_update_journal_1 = QPushButton("Update Journal Entry")
    self.btn_zoom_out_model = QPushButton("Zoom -")
    self.btn_zoom_fit_model = QPushButton("Fit")
    self.btn_zoom_reset_model = QPushButton("100%")
    self.btn_zoom_in_model = QPushButton("Zoom +")

    self.btn_copy_model.clicked.connect(lambda: self.copy_graph("model"))
    self.btn_export_model.clicked.connect(lambda: self.export_png("model"))
    self.btn_update_journal_1.clicked.connect(self.update_journal_entry)
    self.btn_zoom_out_model.clicked.connect(lambda: self.zoom_graph("model", "out"))
    self.btn_zoom_fit_model.clicked.connect(lambda: self.zoom_graph("model", "fit"))
    self.btn_zoom_reset_model.clicked.connect(lambda: self.zoom_graph("model", "reset"))
    self.btn_zoom_in_model.clicked.connect(lambda: self.zoom_graph("model", "in"))

    toolbar.addStretch()
    toolbar.addWidget(self.btn_zoom_out_model)
    toolbar.addWidget(self.btn_zoom_fit_model)
    toolbar.addWidget(self.btn_zoom_reset_model)
    toolbar.addWidget(self.btn_zoom_in_model)
    toolbar.addWidget(self.btn_copy_model)
    toolbar.addWidget(self.btn_export_model)
    toolbar.addWidget(self.btn_update_journal_1)
    layout.addLayout(toolbar)

    self.model_container = QWidget()
    self.model_layout = QVBoxLayout(self.model_container)
    layout.addWidget(self.model_container)

    self.tabs.addTab(tab, "Graphs")


def _mr_build_prediction_tab(self):
    tab = QWidget()
    layout = QVBoxLayout(tab)

    controls_box = QGroupBox("Predictor Controls")
    self.pred_grid = QGridLayout(controls_box)
    self.pred_grid.addWidget(QLabel("Use"), 0, 0)
    self.pred_grid.addWidget(QLabel("Predictor"), 0, 1)
    self.pred_grid.addWidget(QLabel("Min"), 0, 2)
    self.pred_grid.addWidget(QLabel("Input X"), 0, 3)
    self.pred_grid.addWidget(QLabel("Max"), 0, 4)

    self.prediction_controls = {}

    action_box = QGroupBox("Manual Prediction and Optimization")
    action_layout = QGridLayout(action_box)

    self.btn_predict_y = QPushButton("Predict Y")
    self.cmb_objective = QComboBox()
    self.cmb_objective.addItems(["Maximize Y", "Minimize Y", "Target Y"])
    self.spn_target_y = QDoubleSpinBox()
    self.spn_target_y.setRange(-1_000_000_000, 1_000_000_000)
    self.spn_target_y.setDecimals(6)
    self.btn_optimize = QPushButton("Optimize")
    self.txt_prediction = QTextEdit()
    self.txt_prediction.setReadOnly(True)

    self.btn_predict_y.clicked.connect(self.predict_y_from_inputs)
    self.btn_optimize.clicked.connect(self.optimize_y_from_inputs)

    action_layout.addWidget(QLabel("Objective"), 0, 0)
    action_layout.addWidget(self.cmb_objective, 0, 1)
    action_layout.addWidget(QLabel("Target Y"), 1, 0)
    action_layout.addWidget(self.spn_target_y, 1, 1)
    action_layout.addWidget(self.btn_predict_y, 2, 0)
    action_layout.addWidget(self.btn_optimize, 2, 1)
    action_layout.addWidget(self.txt_prediction, 3, 0, 1, 2)

    layout.addWidget(controls_box)
    layout.addWidget(action_box)
    self.tabs.addTab(tab, "Prediction & Optimization")


def _mr_run_analysis(self):
    try:
        opts = self._options()
        if self.current_df.empty:
            raise ValueError("No active worksheet data found.")

        y_col = self.cmb_y.currentText()
        x_cols = self._selected_predictors()

        if not y_col:
            raise ValueError("Select a response Y column.")
        if len(x_cols) < 2:
            raise ValueError("Select at least two predictor X columns.")
        if y_col in x_cols:
            raise ValueError("Response Y cannot also be selected as a predictor X.")

        fit = multiple_regression_fit(self.current_df, y_col, x_cols, opts.alpha)
        res_txt = residual_conclusion(fit, opts.alpha, opts.decimals)

        analysis = {
            "id": str(uuid.uuid4()),
            "module": "Multiple Regression",
            "type": "Multiple Regression",
            "worksheet": self.app_window.active_worksheet_name(),
            "options": asdict(opts),
            "fit": fit,
            "residual_conclusion": res_txt,
            "created_at": now_string(),
        }
        analysis["summary"] = self._build_summary_text(analysis)

        self.current_analysis = analysis
        self._populate_numeric_results(analysis)
        self._render_figures(analysis)
        self._build_prediction_controls_from_fit(fit)
        self.txt_summary.setPlainText(analysis["summary"])
        self.txt_residual.setPlainText(res_txt)

        if opts.auto_journal_save:
            self.save_to_journal()

        self.tabs.setCurrentIndex(1)
        self.log("Multiple Regression completed successfully.")

    except Exception as exc:
        self.log(f"ERROR: {exc}")
        QMessageBox.critical(self, "Multiple Regression Error", str(exc))


def _mr_populate_numeric_results(self, analysis: Dict[str, Any]):
    d = int(analysis["options"]["decimals"])
    fit = analysis["fit"]

    self.txt_equation.setPlainText(fit.get("equation", ""))

    coef_rows = fit.get("coefficient_rows", [])
    self.tbl_coefficients.setRowCount(len(coef_rows))
    for r, row in enumerate(coef_rows):
        values = [
            row.get("Term", ""),
            fmt(row.get("Coef"), d),
            fmt(row.get("SE Coef"), d),
            fmt(row.get("T-Value"), d),
            fmt(row.get("P-Value"), d),
            fmt(row.get("VIF"), d) if row.get("Term") != "Constant" else "",
        ]
        for c, value in enumerate(values):
            self.tbl_coefficients.setItem(r, c, QTableWidgetItem(str(value)))
    self.tbl_coefficients.resizeColumnsToContents()

    ms_values = [[
        fmt(fit.get("s"), d),
        f"{fmt(100 * fit.get('r2'), 2)}%",
        f"{fmt(100 * fit.get('adj_r2'), 2)}%",
        f"{fmt(100 * fit.get('pred_r2'), 2)}%",
    ]]
    self.tbl_model_summary.setRowCount(1)
    for c, value in enumerate(ms_values[0]):
        self.tbl_model_summary.setItem(0, c, QTableWidgetItem(str(value)))
    self.tbl_model_summary.resizeColumnsToContents()

    anova_rows = fit.get("anova_rows", [])
    self.tbl_anova.setRowCount(len(anova_rows))
    for r, row in enumerate(anova_rows):
        values = [
            row.get("source", ""),
            row.get("df", ""),
            fmt(row.get("adj_ss"), d),
            fmt(row.get("adj_ms"), d),
            fmt(row.get("f_value"), d),
            fmt(row.get("p_value"), d),
        ]
        for c, value in enumerate(values):
            self.tbl_anova.setItem(r, c, QTableWidgetItem(str(value)))
    self.tbl_anova.resizeColumnsToContents()

    unusual_rows = fit.get("unusual_rows", [])
    self.tbl_unusual.setRowCount(len(unusual_rows))
    for r, row in enumerate(unusual_rows):
        values = [
            row.get("Obs", ""),
            fmt(row.get(fit["y_name"]), d),
            fmt(row.get("Fit"), d),
            fmt(row.get("Resid"), d),
            fmt(row.get("Std Resid"), d),
            row.get("Flag", ""),
        ]
        for c, value in enumerate(values):
            self.tbl_unusual.setItem(r, c, QTableWidgetItem(str(value)))
    self.tbl_unusual.resizeColumnsToContents()


def _mr_canvas_pixel_size(fig: Figure) -> Tuple[int, int]:
    return int(fig.get_figwidth() * fig.dpi), int(fig.get_figheight() * fig.dpi)


def _mr_wrap_canvas_in_scroll_area(canvas: FigureCanvas, fig: Figure, initial_zoom: float = 0.92) -> ReportScrollArea:
    return ReportScrollArea(canvas=canvas, fig=fig, initial_zoom=initial_zoom)


def _mr_render_figures(self, analysis: Dict[str, Any]):
    self._clear_layout(self.model_layout)
    self._clear_layout(self.residual_layout)

    d = int(analysis["options"]["decimals"])

    self.model_figure = build_model_figure(analysis, d)
    self.model_canvas = FigureCanvas(self.model_figure)
    self.model_scroll_area = _mr_wrap_canvas_in_scroll_area(self.model_canvas, self.model_figure, initial_zoom=0.92)
    self.model_layout.addWidget(self.model_scroll_area)
    self.model_canvas.draw_idle()

    self.residual_figure = build_residual_figure(analysis, d)
    self.residual_canvas = FigureCanvas(self.residual_figure)
    self.residual_scroll_area = _mr_wrap_canvas_in_scroll_area(self.residual_canvas, self.residual_figure, initial_zoom=0.92)
    self.residual_layout.addWidget(self.residual_scroll_area)
    self.residual_canvas.draw_idle()

    self._enable_interactions()


def _mr_zoom_graph(self, graph_type: str, action: str):
    scroll = self.model_scroll_area if graph_type == "model" else self.residual_scroll_area
    if scroll is None or not hasattr(scroll, "zoom_in"):
        return
    if action == "in":
        scroll.zoom_in()
    elif action == "out":
        scroll.zoom_out()
    elif action == "reset":
        scroll.reset_zoom()
    elif action == "fit":
        scroll.fit_to_view()


def _mr_build_prediction_controls_from_fit(self, fit: Dict[str, Any]):
    # Clear old rows except header.
    while self.pred_grid.count() > 5:
        child = self.pred_grid.takeAt(5)
        if child.widget():
            child.widget().deleteLater()

    self.prediction_controls = {}
    for r, name in enumerate(fit["x_names"], start=1):
        rng = fit["x_ranges"][name]
        chk = QCheckBox()
        chk.setChecked(True)

        spn = QDoubleSpinBox()
        spn.setRange(float(rng["min"]), float(rng["max"]))
        spn.setDecimals(6)
        spn.setValue(float(rng["mean"]))

        self.pred_grid.addWidget(chk, r, 0)
        self.pred_grid.addWidget(QLabel(name), r, 1)
        self.pred_grid.addWidget(QLabel(fmt(rng["min"], 4)), r, 2)
        self.pred_grid.addWidget(spn, r, 3)
        self.pred_grid.addWidget(QLabel(fmt(rng["max"], 4)), r, 4)

        self.prediction_controls[name] = {"enabled": chk, "input": spn}

    self.spn_target_y.setValue(float(fit.get("y_range", {}).get("mean", 0.0)))


def _mr_get_prediction_inputs(self):
    fit = self.current_analysis["fit"]
    x_values = {}
    enabled = {}
    for name in fit["x_names"]:
        ctrl = self.prediction_controls.get(name)
        if ctrl:
            x_values[name] = float(ctrl["input"].value())
            enabled[name] = bool(ctrl["enabled"].isChecked())
        else:
            x_values[name] = float(fit["x_ranges"][name]["mean"])
            enabled[name] = True
    return x_values, enabled


def _mr_predict_y_from_inputs(self):
    if not self.current_analysis:
        QMessageBox.information(self, "No model", "Run a regression model first.")
        return
    fit = self.current_analysis["fit"]
    x_values, enabled = self._get_prediction_inputs()
    y_pred = predict_from_fit(fit, x_values, enabled)
    lines = [
        "Manual Prediction",
        "─────────────────",
        f"Predicted {fit['y_name']}: {fmt(y_pred, int(self.spn_decimals.value()))}",
        "",
        "Predictor settings:",
    ]
    for name in fit["x_names"]:
        status = "Enabled" if enabled.get(name, True) else "Disabled / held at historical mean"
        lines.append(f"• {name}: {fmt(x_values[name], int(self.spn_decimals.value()))} ({status})")
    lines.append("")
    lines.append("Note: input ranges are locked to the observed data ranges to avoid extrapolation.")
    self.txt_prediction.setPlainText("\n".join(lines))


def _mr_optimize_y_from_inputs(self):
    if not self.current_analysis:
        QMessageBox.information(self, "No model", "Run a regression model first.")
        return
    fit = self.current_analysis["fit"]
    _, enabled = self._get_prediction_inputs()
    objective = self.cmb_objective.currentText()
    target = float(self.spn_target_y.value()) if objective == "Target Y" else None
    result = optimize_prediction(fit, objective, target=target, enabled=enabled)

    d = int(self.spn_decimals.value())
    lines = [
        "Optimization Result",
        "──────────────────",
        f"Objective: {objective}",
        f"Predicted {fit['y_name']}: {fmt(result['predicted_y'], d)}",
    ]
    if objective == "Target Y":
        lines.append(f"Target {fit['y_name']}: {fmt(target, d)}")
        lines.append(f"Absolute error: {fmt(result['objective_error'], d)}")
    lines.append("")
    lines.append("Recommended predictor settings:")
    for name, value in result["x_values"].items():
        status = "Enabled" if enabled.get(name, True) else "Disabled / held at historical mean"
        lines.append(f"• {name}: {fmt(value, d)} ({status})")
    lines.append("")
    lines.append("Optimization is bounded within observed min/max predictor ranges.")
    self.txt_prediction.setPlainText("\n".join(lines))

    for name, value in result["x_values"].items():
        ctrl = self.prediction_controls.get(name)
        if ctrl:
            ctrl["input"].setValue(float(value))


def _mr_build_summary_text(self, analysis: Dict[str, Any]) -> str:
    d = int(analysis["options"]["decimals"])
    fit = analysis["fit"]
    alpha = float(analysis["options"].get("alpha", 0.05))
    sig_predictors = [r["Term"] for r in fit.get("coefficient_rows", [])[1:] if pd.notna(r.get("P-Value")) and r.get("P-Value") <= alpha]

    lines = [
        "=" * 92,
        "MULTIPLE REGRESSION REPORT — SPS STUDIO",
        "=" * 92,
        f"Worksheet: {analysis.get('worksheet', '')}",
        f"Response Y: {fit.get('y_name', '')}",
        f"Predictors X: {', '.join(fit.get('x_names', []))}",
        f"N: {fit.get('n')}",
        f"Alpha: {fmt(alpha, 3)}",
        "",
        "REGRESSION EQUATION",
        "-" * 92,
        fit.get("equation", ""),
        "",
        "MODEL SUMMARY",
        "-" * 92,
        f"S={fmt(fit.get('s'), d)} | R-sq={fmt(100 * fit.get('r2'), 2)}% | "
        f"R-sq(adj)={fmt(100 * fit.get('adj_r2'), 2)}% | R-sq(pred)={fmt(100 * fit.get('pred_r2'), 2)}%",
        f"The model explains {fmt(100 * fit.get('r2'), 2)}% of the observed variation in {fit.get('y_name', 'Y')}.",
        f"Variation explained classification: {r2_explanation_label(fit.get('r2'))}",
        f"Model p-value={fmt(fit.get('p_model'), d)} | {significance_label(fit.get('p_model'), alpha)}",
        "",
        "ANALYSIS OF VARIANCE",
        "-" * 92,
        f"{'Source':<16}{'DF':<8}{'Adj SS':<14}{'Adj MS':<14}{'F-Value':<14}{'P-Value':<14}",
    ]

    for row in fit.get("anova_rows", []):
        lines.append(
            f"{row.get('source',''):<16}{str(row.get('df','')):<8}{fmt(row.get('adj_ss'), d):<14}"
            f"{fmt(row.get('adj_ms'), d):<14}{fmt(row.get('f_value'), d):<14}{fmt(row.get('p_value'), d):<14}"
        )

    lines += [
        "",
        "COEFFICIENTS",
        "-" * 92,
        f"{'Term':<24}{'Coef':<14}{'SE Coef':<14}{'T':<12}{'P':<12}{'VIF':<10}",
    ]
    for row in fit.get("coefficient_rows", []):
        lines.append(
            f"{row.get('Term',''):<24}{fmt(row.get('Coef'), d):<14}{fmt(row.get('SE Coef'), d):<14}"
            f"{fmt(row.get('T-Value'), d):<12}{fmt(row.get('P-Value'), d):<12}"
            f"{fmt(row.get('VIF'), d) if row.get('Term') != 'Constant' else '':<10}"
        )

    lines += [
        "",
        "CONCLUSIONS",
        "-" * 92,
        f"Model is statistically significant? {'Yes' if fit.get('p_model', np.nan) <= alpha else 'No'}",
        f"Significant factors: {', '.join(sig_predictors) if sig_predictors else 'None'}",
        f"R² interpretation: {r2_explanation_label(fit.get('r2'))} explanatory power.",
        "",
        "RESIDUAL ANALYSIS",
        "-" * 92,
        analysis.get("residual_conclusion", ""),
    ]
    return "\n".join(lines)


def _mr_load_from_journal_entry(self, entry: Dict[str, Any]):
    self.current_journal_entry_id = entry.get("id")
    payload = entry.get("payload", {})
    self.current_analysis = payload
    self.txt_summary.setPlainText(entry.get("summary", payload.get("summary", "")))
    self._populate_numeric_results(payload)
    self._render_figures(payload)
    self._build_prediction_controls_from_fit(payload["fit"])
    self.txt_residual.setPlainText(payload.get("residual_conclusion", ""))


# Apply overrides to keep compatibility with the original class name/import.
MultipleRegressionBuilder._build_ui = _mr_build_ui
MultipleRegressionBuilder._build_numeric_tab = _mr_build_numeric_tab
MultipleRegressionBuilder._build_model_graph_tab = _mr_build_model_graph_tab
MultipleRegressionBuilder._build_prediction_tab = _mr_build_prediction_tab
MultipleRegressionBuilder.run_analysis = _mr_run_analysis
MultipleRegressionBuilder._populate_numeric_results = _mr_populate_numeric_results
MultipleRegressionBuilder._render_figures = _mr_render_figures
MultipleRegressionBuilder.zoom_graph = _mr_zoom_graph
MultipleRegressionBuilder._build_prediction_controls_from_fit = _mr_build_prediction_controls_from_fit
MultipleRegressionBuilder._get_prediction_inputs = _mr_get_prediction_inputs
MultipleRegressionBuilder.predict_y_from_inputs = _mr_predict_y_from_inputs
MultipleRegressionBuilder.optimize_y_from_inputs = _mr_optimize_y_from_inputs
MultipleRegressionBuilder._build_summary_text = _mr_build_summary_text
MultipleRegressionBuilder.load_from_journal_entry = _mr_load_from_journal_entry


# =============================================================================
# SPS Studio patch v2.1 — Multiple Regression table copy + graph spacing fixes
# =============================================================================

def build_model_figure(analysis: Dict[str, Any], decimals: int = 4) -> Figure:
    fit = analysis["fit"]
    alpha = float(analysis["options"].get("alpha", 0.05))
    show_grid = bool(analysis["options"].get("show_grid", True))
    y_name = fit["y_name"]

    fig = Figure(figsize=(11.2, 6.6), dpi=120)
    fig.patch.set_facecolor("white")

    title_ax = fig.add_axes([0.04, 0.885, 0.92, 0.06])
    title_ax.axis("off")
    title_ax.set_title("Multiple Regression Report", fontsize=16, fontweight="bold", pad=2)

    gs = fig.add_gridspec(
        2, 2,
        left=0.07, right=0.965, top=0.82, bottom=0.08,
        wspace=0.33, hspace=0.44,
        width_ratios=[1.17, 1.10],
        height_ratios=[0.95, 1.05],
    )

    ax_pareto = fig.add_subplot(gs[:, 0])
    ax_main = fig.add_subplot(gs[0, 1])
    ax_conc = fig.add_subplot(gs[1, 1])

    coef_rows = fit.get("coefficient_rows", [])[1:]
    effects = np.array([abs(r.get("T-Value", np.nan)) for r in coef_rows], dtype=float)
    names = [r["Term"] for r in coef_rows]
    order = np.argsort(effects)
    effects_sorted = effects[order]
    names_sorted = [names[i] for i in order]
    pvals_sorted = [coef_rows[i].get("P-Value", np.nan) for i in order]

    colors = ["#C62828" if pd.notna(pv) and pv <= alpha else "#7EA6D8" for pv in pvals_sorted]
    y_pos = np.arange(len(names_sorted))
    ax_pareto.barh(y_pos, effects_sorted, color=colors, edgecolor="#333333", linewidth=0.4)
    ax_pareto.set_yticks(y_pos)
    ax_pareto.set_yticklabels(names_sorted)
    ax_pareto.set_xlabel("Standardized Effect")
    # Main title
    ax_pareto.set_title(
        "Pareto Chart of the Standardized Effects",
        fontsize=10,
        fontweight="bold",
        pad=18,
    )

    # Subtitle
    ax_pareto.text(
        0.5,
        1.015,
        f"(response is {y_name}, α = {fmt(alpha, 3)})",
        transform=ax_pareto.transAxes,
        ha="center",
        va="bottom",
        fontsize=8,
        fontweight="normal",
        color="#444444",
    )

    tcrit = stats.t.ppf(1 - alpha / 2, fit["df_err"]) if fit.get("df_err", 0) > 0 else np.nan
    max_effect = float(np.nanmax(effects_sorted)) if len(effects_sorted) else 1.0
    # Keep the threshold close to the right border, Minitab-style.
    if pd.notna(tcrit):
        x_upper = max(max_effect * 1.12, tcrit * 1.06)
    else:
        x_upper = max_effect * 1.12

    ax_pareto.set_xlim(0, x_upper)

    if pd.notna(tcrit):
        ax_pareto.axvline(
            tcrit,
            color="#C62828",
            linestyle="--",
            linewidth=1.0,
            alpha=0.95,
            zorder=3,
        )

        # Threshold label near the top border, aligned with the vertical line.
        ax_pareto.text(
            tcrit,
            1.0,
            fmt(tcrit, 3),
            transform=ax_pareto.get_xaxis_transform(),
            color="#C62828",
            ha="center",
            va="bottom",
            fontsize=8,
            clip_on=False,
            zorder=4,
        )

    ax_pareto.grid(show_grid, axis="x", alpha=0.18)

    # Main Effects Plot: bounded line from observed min to max for each predictor, holding others at mean.
    x_names = fit["x_names"]
    for i, nm in enumerate(x_names):
        if i >= 6:
            break
        x_rng = fit["x_ranges"][nm]
        x_vals = np.array([x_rng["min"], x_rng["max"]], dtype=float)
        y_vals = []
        for xv in x_vals:
            x_point = {n: fit["x_ranges"][n]["mean"] for n in x_names}
            x_point[nm] = float(xv)
            y_vals.append(predict_from_fit(fit, x_point))
        ax_main.plot([0, 1], y_vals, marker="o", linewidth=1.8, label=nm)

    ax_main.set_xticks([0, 1])
    ax_main.set_xticklabels(["Low", "High"])
    ax_main.set_title(f"Main Effects Plot for {y_name}", fontsize=10, fontweight="bold")
    ax_main.set_ylabel(f"Mean of {y_name}")
    ax_main.legend(fontsize=7, frameon=False)
    ax_main.grid(show_grid, alpha=0.18)

    ax_conc.axis("off")
    sig_predictors = [r["Term"] for r in coef_rows if pd.notna(r.get("P-Value")) and r.get("P-Value") <= alpha]
    model_sig = fit.get("p_model", np.nan) <= alpha if pd.notna(fit.get("p_model", np.nan)) else False
    r2_pct = 100 * fit.get("r2", np.nan)
    r2_class = r2_explanation_label(fit.get("r2", np.nan))

    rows = [
        ["Model p-value", fmt(fit.get("p_model"), decimals), "Significant" if model_sig else "Not significant"],
        ["R²", f"{fmt(r2_pct, 2)}%", f"{r2_class} variation explained"],
        ["R²(adj)", f"{fmt(100 * fit.get('adj_r2'), 2)}%", "Complexity-adjusted"],
        ["R²(pred)", f"{fmt(100 * fit.get('pred_r2'), 2)}%", "Prediction capability"],
        ["Significant factors", ", ".join(sig_predictors) if sig_predictors else "None", "Based on p-value"],
    ]

    tbl = ax_conc.table(
        cellText=rows,
        colLabels=["Factor", "Result", "Conclusion"],
        loc="upper center",
        cellLoc="center",
        colWidths=[0.30, 0.25, 0.45],
        bbox=[0.00, 0.48, 1.00, 0.46],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7.6)
    tbl.scale(1.0, 1.35)

    for c in range(3):
        tbl[0, c].set_facecolor("#17324D")
        tbl[0, c].set_text_props(color="white", fontweight="bold")

    ax_conc.set_title("Conclusion Summary", fontsize=10, fontweight="bold", pad=8)

    # Factor significance list below the table.
    ax_conc.text(
        0.01,
        0.40,
        "Factor significance",
        fontsize=9,
        fontweight="bold",
        transform=ax_conc.transAxes,
        va="top",
    )
    y = 0.32
    for row in coef_rows:
        term = row.get("Term", "")
        p_value = row.get("P-Value", np.nan)
        is_sig = pd.notna(p_value) and p_value <= alpha
        status = "Significant" if is_sig else "Not significant"
        color = "#C62828" if is_sig else "#1B1B1B"
        line = f"{term}: p={fmt(p_value, decimals)} | α={fmt(alpha, 3)} | {status}"
        ax_conc.text(
            0.02,
            y,
            line,
            fontsize=8,
            color=color,
            transform=ax_conc.transAxes,
            va="top",
        )
        y -= 0.085

    return fig


def _mr_build_numeric_tab_v21(self):
    tab = QWidget()
    layout = QVBoxLayout(tab)

    toolbar = QHBoxLayout()
    self.cmb_table_copy = QComboBox()
    self.cmb_table_copy.addItems([
        "Regression Equation",
        "Coefficients",
        "Model Summary",
        "Analysis of Variance",
        "Unusual Observations",
    ])
    self.btn_copy_table_text = QPushButton("Copy Table as Text")
    self.btn_copy_table_image = QPushButton("Copy Table as Image")
    self.btn_copy_table_text.clicked.connect(self.copy_selected_table_as_text)
    self.btn_copy_table_image.clicked.connect(self.copy_selected_table_as_image)

    toolbar.addWidget(QLabel("Table"))
    toolbar.addWidget(self.cmb_table_copy)
    toolbar.addStretch()
    toolbar.addWidget(self.btn_copy_table_text)
    toolbar.addWidget(self.btn_copy_table_image)
    layout.addLayout(toolbar)

    self.txt_equation = QTextEdit()
    self.txt_equation.setReadOnly(True)
    self.txt_equation.setMaximumHeight(75)

    self.tbl_coefficients = QTableWidget()
    self.tbl_coefficients.setColumnCount(6)
    self.tbl_coefficients.setHorizontalHeaderLabels(["Term", "Coef", "SE Coef", "T-Value", "P-Value", "VIF"])

    self.tbl_model_summary = QTableWidget()
    self.tbl_model_summary.setColumnCount(4)
    self.tbl_model_summary.setHorizontalHeaderLabels(["S", "R-sq", "R-sq(adj)", "R-sq(pred)"])

    self.tbl_anova = QTableWidget()
    self.tbl_anova.setColumnCount(6)
    self.tbl_anova.setHorizontalHeaderLabels(["Source", "DF", "Adj SS", "Adj MS", "F-Value", "P-Value"])

    self.tbl_unusual = QTableWidget()
    self.tbl_unusual.setColumnCount(6)
    self.tbl_unusual.setHorizontalHeaderLabels(["Obs", "Response", "Fit", "Resid", "Std Resid", "Flag"])

    layout.addWidget(QLabel("Regression Equation"))
    layout.addWidget(self.txt_equation)
    layout.addWidget(QLabel("Coefficients"))
    layout.addWidget(self.tbl_coefficients)
    layout.addWidget(QLabel("Model Summary"))
    layout.addWidget(self.tbl_model_summary)
    layout.addWidget(QLabel("Analysis of Variance"))
    layout.addWidget(self.tbl_anova)
    layout.addWidget(QLabel("Fits and Diagnostics for Unusual Observations"))
    layout.addWidget(self.tbl_unusual)

    self.tabs.addTab(tab, "Numerical Results")


def _mr_selected_table_widget(self):
    name = self.cmb_table_copy.currentText()
    if name == "Regression Equation":
        return self.txt_equation
    if name == "Coefficients":
        return self.tbl_coefficients
    if name == "Model Summary":
        return self.tbl_model_summary
    if name == "Analysis of Variance":
        return self.tbl_anova
    if name == "Unusual Observations":
        return self.tbl_unusual
    return None


def _table_to_text(table: QTableWidget) -> str:
    headers = []
    for c in range(table.columnCount()):
        item = table.horizontalHeaderItem(c)
        headers.append(item.text() if item else f"C{c + 1}")

    lines = ["\t".join(headers)]
    for r in range(table.rowCount()):
        row = []
        for c in range(table.columnCount()):
            item = table.item(r, c)
            row.append(item.text() if item else "")
        lines.append("\t".join(row))
    return "\n".join(lines)


def _mr_copy_selected_table_as_text(self):
    widget = self._selected_table_widget()
    if widget is None:
        QMessageBox.information(self, "No table", "No table is available to copy.")
        return

    if isinstance(widget, QTextEdit):
        QApplication.clipboard().setText(widget.toPlainText())
    elif isinstance(widget, QTableWidget):
        QApplication.clipboard().setText(_table_to_text(widget))
    else:
        QApplication.clipboard().setText("")

    self.app_window.statusBar().showMessage("Table copied to clipboard as text.")


def _mr_copy_selected_table_as_image(self):
    widget = self._selected_table_widget()
    if widget is None:
        QMessageBox.information(self, "No table", "No table is available to copy.")
        return

    try:
        pixmap = widget.grab()
        QApplication.clipboard().setPixmap(pixmap)
        self.app_window.statusBar().showMessage("Table copied to clipboard as image.")
    except Exception as exc:
        QMessageBox.critical(self, "Copy failed", str(exc))


MultipleRegressionBuilder._build_numeric_tab = _mr_build_numeric_tab_v21
MultipleRegressionBuilder._selected_table_widget = _mr_selected_table_widget
MultipleRegressionBuilder.copy_selected_table_as_text = _mr_copy_selected_table_as_text
MultipleRegressionBuilder.copy_selected_table_as_image = _mr_copy_selected_table_as_image
