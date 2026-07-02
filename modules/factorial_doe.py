
"""
modules/factorial_doe.py
SPS Studio — Factorial DOE Builder

PySide6 module integrated with SPS Studio.

Scope:
- 2-Level Full Factorial DOE
- 2-Level Fractional Factorial DOE using generator definitions
- Center points support
- Factor coding to -1 / +1 / 0
- Main effects and interaction effects
- Regression/ANOVA backend
- Pareto chart of standardized effects
- Main Effects Plot
- Interaction Plot
- Normal Plot of Effects
- Residual Analysis
- Prediction and Optimization
- Alias / generator / confounding summary for fractional designs
- Analysis Journal integration
- Scrollable and zoomable report viewer

Visible UI text is English.
"""

from __future__ import annotations

import base64
import io
import itertools
import math
import re
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


# ─────────────────────────────────────────────
# Scrollable / zoomable report viewer
# ─────────────────────────────────────────────

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

        self.zoom = max(0.62, min(1.75, float(zoom)))
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
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    if isinstance(obj, pd.Series):
        return obj.tolist()
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


def significance_label(p_value: Any, alpha: float) -> str:
    try:
        p = float(p_value)
        if np.isnan(p):
            return "N/A"
        if p <= alpha:
            return "Significant"
        if p <= alpha * 2:
            return "Borderline"
        return "Not significant"
    except Exception:
        return "N/A"


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


def term_label(combo: Tuple[str, ...]) -> str:
    return "*".join(combo)


def factor_code_map(factors: List[str]) -> Dict[str, str]:
    """Map selected factor names to DOE letters A, B, C... for compact reporting."""
    return {str(name): chr(ord("A") + i) for i, name in enumerate(factors)}


def coded_term_label(term: str, code_map: Dict[str, str]) -> str:
    """Convert a model term from long factor names to coded DOE notation."""
    if not term or term == "Constant":
        return term
    parts = str(term).split("*")
    return "*".join(code_map.get(part, part) for part in parts)


def is_main_effect(term: str) -> bool:
    return bool(term) and term != "Constant" and "*" not in str(term)


def parse_generator_string(text: str) -> List[Tuple[str, str]]:
    """
    Parse generator definitions such as:
    D=ABC; E=BCD
    """
    out = []
    if not text.strip():
        return out

    parts = re.split(r"[;,]\s*", text.strip())
    for part in parts:
        if not part.strip():
            continue
        if "=" not in part:
            raise ValueError(f"Invalid generator '{part}'. Expected format like D=ABC.")
        lhs, rhs = part.split("=", 1)
        lhs = lhs.strip().upper()
        rhs = rhs.strip().upper().replace("*", "").replace(" ", "")
        if not re.fullmatch(r"[A-Z]", lhs):
            raise ValueError(f"Invalid generated factor '{lhs}'. Use one letter, e.g. D.")
        if not re.fullmatch(r"[A-Z]+", rhs):
            raise ValueError(f"Invalid generator expression '{rhs}'. Use letters only, e.g. ABC.")
        if lhs in rhs:
            raise ValueError(f"Invalid generator {lhs}={rhs}. Generated factor cannot appear on the right side.")
        out.append((lhs, rhs))
    return out


def generator_word(lhs: str, rhs: str) -> str:
    letters = "".join(sorted(lhs + rhs))
    return letters


def multiply_words(a: str, b: str) -> str:
    """
    Multiply alias words in two-level DOE algebra.
    Repeated letters cancel because X^2 = I.
    """
    counts: Dict[str, int] = {}
    for ch in a + b:
        counts[ch] = counts.get(ch, 0) + 1
    return "".join(sorted([ch for ch, count in counts.items() if count % 2 == 1])) or "I"


def generate_defining_relation(generator_words: List[str]) -> List[str]:
    words = {"I"}
    for r in range(1, len(generator_words) + 1):
        for combo in itertools.combinations(generator_words, r):
            w = "I"
            for item in combo:
                w = multiply_words(w if w != "I" else "", item)
            words.add(w or "I")
    return sorted(words, key=lambda x: (len(x), x))


def alias_for_term(term: str, defining_relation: List[str]) -> List[str]:
    aliases = set()
    base = term.replace("*", "").replace(" ", "").upper()
    for word in defining_relation:
        if word == "I":
            aliases.add(base or "I")
        else:
            aliases.add(multiply_words(base, word))
    return sorted(aliases, key=lambda x: (len(x), x))


def code_two_level_factor(series: pd.Series, factor_name: str) -> Tuple[pd.Series, Dict[str, Any]]:
    """
    Convert a factor to coded -1 / +1 / 0.

    Numeric factor:
    - min -> -1
    - max -> +1
    - values between min and max are linearly coded
    - center point is near 0

    Categorical factor with two levels:
    - first sorted level -> -1
    - second sorted level -> +1
    """
    raw = series.copy()
    numeric = pd.to_numeric(raw, errors="coerce")

    if numeric.notna().sum() == raw.notna().sum():
        vals = numeric.dropna()
        if vals.empty:
            raise ValueError(f"Factor '{factor_name}' has no valid values.")
        lo = float(vals.min())
        hi = float(vals.max())
        if math.isclose(lo, hi):
            raise ValueError(f"Factor '{factor_name}' has zero variation.")
        coded = 2.0 * (numeric - lo) / (hi - lo) - 1.0
        meta = {
            "type": "numeric",
            "low": lo,
            "high": hi,
            "center": (lo + hi) / 2.0,
            "scale": (hi - lo) / 2.0,
            "low_label": fmt(lo, 4),
            "high_label": fmt(hi, 4),
        }
        return coded.astype(float), meta

    levels = sorted([str(v) for v in raw.dropna().unique()])
    if len(levels) < 2:
        raise ValueError(f"Factor '{factor_name}' needs at least two levels.")
    if len(levels) > 2:
        raise ValueError(
            f"Factor '{factor_name}' has more than two levels. "
            "This module currently supports 2-level factors plus numeric center points."
        )

    mapping = {levels[0]: -1.0, levels[1]: 1.0}
    coded = raw.astype(str).map(mapping)
    meta = {
        "type": "categorical",
        "low": levels[0],
        "high": levels[1],
        "center": None,
        "scale": None,
        "low_label": levels[0],
        "high_label": levels[1],
    }
    return coded.astype(float), meta


def build_interaction_matrix(coded_df: pd.DataFrame, factors: List[str], max_order: int) -> pd.DataFrame:
    X = pd.DataFrame(index=coded_df.index)
    max_order = max(1, min(int(max_order), len(factors)))
    for order in range(1, max_order + 1):
        for combo in itertools.combinations(factors, order):
            name = term_label(combo)
            values = np.ones(len(coded_df), dtype=float)
            for f in combo:
                values *= coded_df[f].to_numpy(dtype=float)
            X[name] = values
    return X


def fit_ols(y: np.ndarray, X_raw: np.ndarray, term_names_no_intercept: List[str], alpha: float) -> Dict[str, Any]:
    y = np.asarray(y, dtype=float)
    X_raw = np.asarray(X_raw, dtype=float)
    n = len(y)
    p = X_raw.shape[1]

    if n <= p + 1:
        raise ValueError(
            f"Not enough degrees of freedom. Valid rows={n}, model terms={p}. "
            "Reduce interaction order, use fewer factors, add replicates, or add center points."
        )

    X = np.column_stack([np.ones(n), X_raw])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    fitted = X @ beta
    residuals = y - fitted
    xtx_inv = np.linalg.pinv(X.T @ X)
    h = np.sum((X @ xtx_inv) * X, axis=1)
    rank = int(np.linalg.matrix_rank(X))

    df_reg = p
    df_err = n - p - 1
    df_total = n - 1
    ss_total = float(np.sum((y - np.mean(y)) ** 2))
    ss_error = float(np.sum(residuals ** 2))
    ss_reg = float(ss_total - ss_error)
    ms_reg = ss_reg / df_reg if df_reg > 0 else np.nan
    ms_error = ss_error / df_err if df_err > 0 else np.nan
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
    all_terms = ["Constant"] + term_names_no_intercept
    for i, term in enumerate(all_terms):
        coef = float(beta[i])
        effect = np.nan if i == 0 else float(2.0 * beta[i])
        lo = beta[i] - tcrit * se_beta[i] if pd.notna(tcrit) else np.nan
        hi = beta[i] + tcrit * se_beta[i] if pd.notna(tcrit) else np.nan
        p_val = float(p_values[i]) if pd.notna(p_values[i]) else np.nan
        coefficient_rows.append({
            "Term": term,
            "Effect": effect,
            "Coef": coef,
            "SE Coef": float(se_beta[i]) if pd.notna(se_beta[i]) else np.nan,
            "T-Value": float(t_values[i]) if pd.notna(t_values[i]) else np.nan,
            "P-Value": p_val,
            "Lower CI": float(lo) if pd.notna(lo) else np.nan,
            "Upper CI": float(hi) if pd.notna(hi) else np.nan,
            "Decision": p_decision(p_val, alpha),
            "Conclusion": "Intercept" if i == 0 else significance_label(p_val, alpha),
        })

    term_anova_rows = []
    for j, term in enumerate(term_names_no_intercept):
        cols = [i for i in range(p) if i != j]
        if cols:
            X_reduced = X_raw[:, cols]
            Xr = np.column_stack([np.ones(n), X_reduced])
            br, *_ = np.linalg.lstsq(Xr, y, rcond=None)
            resid_r = y - Xr @ br
            ss_reduced_error = float(np.sum(resid_r ** 2))
        else:
            ss_reduced_error = float(np.sum((y - np.mean(y)) ** 2))

        ss_term = float(ss_reduced_error - ss_error)
        df_term = 1
        ms_term = ss_term
        f_term = ms_term / ms_error if pd.notna(ms_error) and ms_error > 0 else np.nan
        p_term = float(stats.f.sf(f_term, df_term, df_err)) if pd.notna(f_term) else np.nan
        term_anova_rows.append({
            "source": term,
            "df": df_term,
            "adj_ss": ss_term,
            "adj_ms": ms_term,
            "f_value": f_term,
            "p_value": p_term,
        })

    anova_rows = [
        {"source": "Model", "df": df_reg, "adj_ss": ss_reg, "adj_ms": ms_reg, "f_value": f_value, "p_value": p_model},
        *term_anova_rows,
        {"source": "Error", "df": df_err, "adj_ss": ss_error, "adj_ms": ms_error, "f_value": np.nan, "p_value": np.nan},
        {"source": "Total", "df": df_total, "adj_ss": ss_total, "adj_ms": np.nan, "f_value": np.nan, "p_value": np.nan},
    ]

    std_res = residuals / math.sqrt(ms_error) if pd.notna(ms_error) and ms_error > 0 else np.full_like(residuals, np.nan)
    sh_stat, sh_p = safe_shapiro(residuals)
    lev_stat, lev_p = variance_stability_test(fitted, residuals)
    dw = durbin_watson_stat(residuals)
    high_lev_threshold = 2.0 * (p + 1) / n if n else np.nan
    high_lev_n = int(np.sum(h > high_lev_threshold)) if pd.notna(high_lev_threshold) else 0
    outliers_n = int(np.sum(np.abs(std_res) > 3)) if len(std_res) else 0

    return {
        "n": int(n),
        "p": int(p),
        "rank": rank,
        "x_design": X,
        "x_matrix": X_raw,
        "beta": beta,
        "term_names": all_terms,
        "model_terms": term_names_no_intercept,
        "fitted": fitted,
        "residuals": residuals,
        "std_residuals": std_res,
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
            "high_leverage_n": high_lev_n,
            "high_leverage_threshold": high_lev_threshold,
            "high_leverage_ok": high_lev_n == 0,
        },
    }


@dataclass
class FactorialDOEOptions:
    doe_type: str = "2-Level Full Factorial"
    alpha: float = 0.05
    decimals: int = 4
    max_interaction_order: int = 2
    include_interactions: bool = True
    show_grid: bool = True
    coded_units: bool = True
    include_center_points: bool = True
    auto_journal_save: bool = True
    generators: str = ""


def factorial_doe_fit(
    df: pd.DataFrame,
    response_col: str,
    factor_cols: List[str],
    options: FactorialDOEOptions,
) -> Dict[str, Any]:
    if not response_col:
        raise ValueError("Select a response column.")
    if len(factor_cols) < 2:
        raise ValueError("Select at least two factors.")

    data = df[[response_col] + factor_cols].copy()
    data[response_col] = pd.to_numeric(data[response_col], errors="coerce")

    coded = pd.DataFrame(index=data.index)
    factor_meta: Dict[str, Any] = {}
    for col in factor_cols:
        coded[col], factor_meta[col] = code_two_level_factor(data[col], col)

    working = pd.concat([data[[response_col]], coded], axis=1)
    working = working.dropna(subset=[response_col] + factor_cols)

    if working.empty:
        raise ValueError("No valid DOE rows after coding response and factors.")

    center_mask = pd.Series(False, index=working.index)
    if options.include_center_points:
        abs_sum = working[factor_cols].abs().sum(axis=1)
        center_mask = abs_sum <= 0.20

    factorial_rows = working.loc[~center_mask].copy()
    center_rows = working.loc[center_mask].copy()

    if len(factorial_rows) < len(factor_cols) + 2:
        raise ValueError("Not enough factorial rows. Check factor levels and response data.")

    effective_factors = list(factor_cols)
    generators = parse_generator_string(options.generators) if "Fractional" in options.doe_type else []

    # Validate fractional generator names against selected factors.
    if generators:
        selected_letters = [chr(ord("A") + i) for i in range(len(factor_cols))]
        factor_letter_map = {selected_letters[i]: factor_cols[i] for i in range(len(factor_cols))}
        for lhs, rhs in generators:
            if lhs not in factor_letter_map:
                raise ValueError(f"Generator factor {lhs} is not available. Selected factors map to {', '.join(selected_letters)}.")
            for ch in rhs:
                if ch not in factor_letter_map:
                    raise ValueError(f"Generator term {rhs} uses factor {ch}, which is not available.")

    if not getattr(options, "include_interactions", True):
        max_order = 1
    else:
        max_order = max(1, min(options.max_interaction_order, len(factor_cols)))
    X_df = build_interaction_matrix(factorial_rows[factor_cols], factor_cols, max_order=max_order)

    y = factorial_rows[response_col].to_numpy(dtype=float)
    ols = fit_ols(y, X_df.to_numpy(dtype=float), list(X_df.columns), alpha=options.alpha)

    design_matrix = factorial_rows[[response_col] + factor_cols].copy()
    design_matrix.insert(0, "Run", np.arange(1, len(design_matrix) + 1))
    design_matrix_coded = factorial_rows[factor_cols].copy()
    design_matrix_coded.insert(0, "Run", np.arange(1, len(design_matrix_coded) + 1))
    design_matrix_coded[response_col] = factorial_rows[response_col].to_numpy(dtype=float)

    generator_words = [generator_word(lhs, rhs) for lhs, rhs in generators]
    defining_relation = generate_defining_relation(generator_words) if generator_words else ["I"]

    alias_rows = []
    if generator_words:
        for term in X_df.columns:
            compact = term.replace("*", "")
            aliases = alias_for_term(compact, defining_relation)
            alias_rows.append({
                "Term": term,
                "Aliases": " = ".join(aliases),
            })

    center_summary = {}
    if len(center_rows) > 0:
        center_summary = {
            "center_points_n": int(len(center_rows)),
            "center_mean": float(center_rows[response_col].mean()),
            "factorial_mean": float(factorial_rows[response_col].mean()),
            "center_delta": float(center_rows[response_col].mean() - factorial_rows[response_col].mean()),
        }
    else:
        center_summary = {
            "center_points_n": 0,
            "center_mean": np.nan,
            "factorial_mean": float(factorial_rows[response_col].mean()),
            "center_delta": np.nan,
        }

    fit = {
        **ols,
        "response_name": response_col,
        "factor_names": factor_cols,
        "factor_code_map": factor_code_map(factor_cols),
        "code_factor_map": {v: k for k, v in factor_code_map(factor_cols).items()},
        "factor_meta": factor_meta,
        "coded_data": factorial_rows,
        "center_data": center_rows,
        "design_matrix": design_matrix,
        "design_matrix_coded": design_matrix_coded,
        "x_terms": X_df,
        "doe_type": options.doe_type,
        "generators": generators,
        "generator_words": generator_words,
        "defining_relation": defining_relation,
        "alias_rows": alias_rows,
        "center_summary": center_summary,
    }
    return fit


def residual_conclusion(fit: Dict[str, Any], alpha: float, decimals: int) -> str:
    res = fit.get("residual_analysis", {})
    lines = []
    recs = []

    if res.get("normality_ok"):
        lines.append(f"• Normality: PASS. Shapiro-Wilk p={fmt(res.get('shapiro_p'), decimals)} > α={fmt(alpha, 3)}.")
    else:
        lines.append(f"• Normality: REVIEW. Shapiro-Wilk p={fmt(res.get('shapiro_p'), decimals)} ≤ α={fmt(alpha, 3)}.")
        recs.append("Review normal plot of residuals, unusual observations, or transformation options.")

    if res.get("constant_variance_ok"):
        lines.append(f"• Constant variance: PASS. Levene p={fmt(res.get('levene_p'), decimals)} > α={fmt(alpha, 3)}.")
    else:
        lines.append(f"• Constant variance: REVIEW. Levene p={fmt(res.get('levene_p'), decimals)} ≤ α={fmt(alpha, 3)}.")
        recs.append("Check factor settings, blocking, missing factors, or transformation needs.")

    if res.get("independence_ok"):
        lines.append(f"• Independence: PASS. Durbin-Watson={fmt(res.get('durbin_watson'), decimals)}.")
    else:
        lines.append(f"• Independence: REVIEW. Durbin-Watson={fmt(res.get('durbin_watson'), decimals)}.")
        recs.append("Review run order, randomization, time drift, batch effects, or measurement sequence.")

    if res.get("outliers_ok"):
        lines.append("• Severe outliers: PASS. No standardized residuals beyond ±3 were detected.")
    else:
        lines.append(f"• Severe outliers: REVIEW. {res.get('outliers_n', 0)} standardized residual(s) beyond ±3 detected.")
        recs.append("Investigate special causes or data entry issues before final DOE conclusions.")

    if res.get("high_leverage_ok"):
        lines.append(f"• High leverage: PASS. No points above leverage threshold {fmt(res.get('high_leverage_threshold'), decimals)}.")
    else:
        lines.append(f"• High leverage: REVIEW. {res.get('high_leverage_n', 0)} point(s) above leverage threshold.")
        recs.append("Check influential design points and replicate structure.")

    if not recs:
        recs.append("Model diagnostics are acceptable. Use confirmation runs before process release.")

    return (
        "DOE Residual and Model Assumption Conclusions\n"
        "──────────────────────────────────────────────\n"
        + "\n".join(lines)
        + "\n\nRecommendations\n"
        "───────────────\n"
        + "\n".join(f"• {r}" for r in recs)
    )


def predict_doe_response(fit: Dict[str, Any], factor_settings: Dict[str, float]) -> float:
    beta = np.asarray(fit["beta"], dtype=float)
    factors = fit["factor_names"]
    max_order = max(len(t.split("*")) for t in fit["model_terms"]) if fit.get("model_terms") else 1

    coded = {}
    for f in factors:
        meta = fit["factor_meta"][f]
        val = factor_settings.get(f, 1.0)
        if meta["type"] == "numeric":
            low = float(meta["low"])
            high = float(meta["high"])
            coded[f] = 2.0 * (float(val) - low) / (high - low) - 1.0
        else:
            if str(val) == str(meta["low"]):
                coded[f] = -1.0
            elif str(val) == str(meta["high"]):
                coded[f] = 1.0
            else:
                coded[f] = float(val)

    x_values = []
    for term in fit["model_terms"]:
        parts = term.split("*")
        prod = 1.0
        for p in parts:
            prod *= coded[p]
        x_values.append(prod)

    return float(beta[0] + np.dot(beta[1:], np.asarray(x_values, dtype=float)))


def optimize_doe_response(
    fit: Dict[str, Any],
    objective: str,
    target: Optional[float],
) -> Dict[str, Any]:
    factors = fit["factor_names"]
    candidate_levels = []
    for f in factors:
        meta = fit["factor_meta"][f]
        if meta["type"] == "numeric":
            vals = [float(meta["low"]), float(meta["high"])]
            if fit.get("center_summary", {}).get("center_points_n", 0) > 0:
                vals.append(float(meta["center"]))
        else:
            vals = [meta["low"], meta["high"]]
        candidate_levels.append(vals)

    best = None
    rows = []
    for combo in itertools.product(*candidate_levels):
        settings = dict(zip(factors, combo))
        y_hat = predict_doe_response(fit, settings)

        if objective == "Maximize Y":
            score = y_hat
            desirability = 1.0
        elif objective == "Minimize Y":
            score = -y_hat
            desirability = 1.0
        else:
            if target is None:
                target = float(fit.get("center_summary", {}).get("factorial_mean", 0.0))
            err = abs(y_hat - target)
            spread = max(1e-9, float(np.nanstd(fit["coded_data"][fit["response_name"]].to_numpy(dtype=float))))
            desirability = max(0.0, 1.0 - err / (3 * spread))
            score = desirability

        row = {"Predicted Y": y_hat, "Desirability": desirability, **settings}
        rows.append(row)
        if best is None or score > best["score"]:
            best = {"score": score, "settings": settings, "predicted_y": y_hat, "desirability": desirability}

    return {
        "best_settings": best["settings"] if best else {},
        "predicted_y": best["predicted_y"] if best else np.nan,
        "desirability": best["desirability"] if best else np.nan,
        "candidate_rows": rows,
    }


# ─────────────────────────────────────────────
# Figure builders
# ─────────────────────────────────────────────

def _as_dataframe(value: Any) -> pd.DataFrame:
    """Rebuild DataFrames after Journal JSON serialization."""
    if isinstance(value, pd.DataFrame):
        return value
    if isinstance(value, list):
        return pd.DataFrame(value)
    if isinstance(value, dict):
        return pd.DataFrame(value)
    return pd.DataFrame()


def build_doe_graphs_figure(analysis: Dict[str, Any], decimals: int = 4) -> Figure:
    """
    Main DOE report figure.

    Includes:
    - Pareto chart of standardized effects
    - Main effects plot
    - DOE summary table and factor significance list

    Interaction plots are intentionally handled in a separate tab because they
    consume space quickly as the number of factors increases.
    """
    fit = analysis["fit"]
    alpha = float(analysis["options"].get("alpha", 0.05))
    show_grid = bool(analysis["options"].get("show_grid", True))

    factors_for_codes = list(fit.get("factor_names", []))
    code_map_for_terms = fit.get("factor_code_map") or factor_code_map(factors_for_codes)
    coef_rows = [r for r in fit.get("coefficient_rows", []) if r.get("Term") != "Constant"]
    terms = [coded_term_label(r["Term"], code_map_for_terms) for r in coef_rows]
    tvals = np.array([abs(r.get("T-Value", np.nan)) for r in coef_rows], dtype=float)
    pvals = np.array([r.get("P-Value", np.nan) for r in coef_rows], dtype=float)
    df_err = int(fit.get("df_err", 1))
    tcrit = stats.t.ppf(1 - alpha / 2, df_err) if df_err > 0 else np.nan

    order = np.argsort(np.nan_to_num(tvals, nan=-np.inf))
    terms_sorted = [terms[i] for i in order]
    tvals_sorted = tvals[order]
    pvals_sorted = pvals[order]

    fig = Figure(figsize=(11.2, 6.6), dpi=120)
    fig.patch.set_facecolor("white")

    title_ax = fig.add_axes([0.04, 0.905, 0.92, 0.055])
    title_ax.axis("off")
    title_ax.set_title(f"Factorial DOE Report — {fit['response_name']}", fontsize=15, fontweight="bold", pad=1)

    gs = fig.add_gridspec(
        2, 2,
        left=0.07, right=0.97, top=0.82, bottom=0.08,
        wspace=0.38, hspace=0.46,
        width_ratios=[1.22, 1.0],
        height_ratios=[1.0, 1.0],
    )

    ax_pareto = fig.add_subplot(gs[:, 0])
    ax_main = fig.add_subplot(gs[0, 1])
    ax_summary = fig.add_subplot(gs[1, 1])

    # Pareto chart
    y_pos = np.arange(len(terms_sorted))
    bar_colors = ["#C62828" if pd.notna(pv) and pv <= alpha else "#2F5D87" for pv in pvals_sorted]
    ax_pareto.barh(y_pos, tvals_sorted, color=bar_colors, alpha=0.88)
    ax_pareto.set_yticks(y_pos)
    ax_pareto.set_yticklabels(terms_sorted, fontsize=8)
    ax_pareto.set_xlabel("Absolute standardized effect")
    ax_pareto.set_title(
        "Pareto Chart of Standardized Effects",
        fontsize=10,
        fontweight="bold",
        pad=24,
    )

    ax_pareto.text(
        0.5,
        1.035,
        f"(response is {fit['response_name']}, α = {fmt(alpha, 3)})",
        transform=ax_pareto.transAxes,
        ha="center",
        va="bottom",
        fontsize=8,
        fontweight=400,
        color="#444444",
    )
    if pd.notna(tcrit):
        xmax = max(np.nanmax(tvals_sorted) if len(tvals_sorted) else 0, tcrit) * 1.10 + 0.05
        ax_pareto.set_xlim(0, xmax)
        ax_pareto.axvline(tcrit, color="#C62828", linestyle="--", linewidth=1.0)
        ax_pareto.text(
            tcrit, 1.0, fmt(tcrit, 3),
            transform=ax_pareto.get_xaxis_transform(),
            color="#C62828", ha="center", va="bottom",
            fontsize=8, clip_on=False,
        )
    ax_pareto.grid(show_grid, axis="x", alpha=0.18)

    data = _as_dataframe(fit.get("coded_data"))
    factors = list(fit.get("factor_names", []))
    code_map = fit.get("factor_code_map") or factor_code_map(factors)
    y_name = fit.get("response_name", "Y")

    # Main effects plot
    if not data.empty and factors:
        for f in factors:
            if f not in data.columns or y_name not in data.columns:
                continue
            lo = data.loc[data[f] <= 0, y_name].mean()
            hi = data.loc[data[f] > 0, y_name].mean()
            code = code_map.get(f, f)
            ax_main.plot([0, 1], [lo, hi], marker="o", linewidth=1.6, label=code)
            if pd.notna(hi):
                ax_main.text(1.02, hi, code, fontsize=7, va="center")
    ax_main.set_xticks([0, 1])
    ax_main.set_xticklabels(["Low", "High"])
    ax_main.set_title("Main Effects Plot", fontsize=10, fontweight="bold")
    ax_main.set_ylabel(f"Mean of {y_name}")
    ax_main.grid(show_grid, alpha=0.18)

    # DOE Summary
    ax_summary.axis("off")
    main_effect_rows = [r for r in coef_rows if is_main_effect(r.get("Term", ""))]
    sig_main = [r for r in main_effect_rows if pd.notna(r.get("P-Value")) and r.get("P-Value") <= alpha]
    center_n = fit.get("center_summary", {}).get("center_points_n", 0)
    include_interactions = bool(analysis.get("options", {}).get("include_interactions", True))


    summary_rows = [
        ["DOE Type", fit.get("doe_type", "")],
        ["Factors", str(len(factors))],
        ["Terms", str(len(fit.get("model_terms", [])))],
        ["S", fmt(fit.get("s"), decimals)],
        ["R²", f"{fmt(100 * fit.get('r2'), 2)}%"],
        ["R²(adj)", f"{fmt(100 * fit.get('adj_r2'), 2)}%"],
        ["Model p", fmt(fit.get("p_model"), decimals)],
        ["Center pts", str(center_n)],
    ]

    sig_rows = []
    for factor in factors:
        code = code_map.get(factor, factor)
        row = next((r for r in main_effect_rows if r.get("Term") == factor), None)
        if row is None or pd.isna(row.get("P-Value")):
            sig_rows.append([code, factor, "N/A", fmt(alpha, 3), "N/A"])
        else:
            pval = row.get("P-Value")
            flag = "Yes" if pval <= alpha else "No"
            sig_rows.append([code, factor, fmt(pval, decimals), fmt(alpha, 3), flag])

    # DOE Summary
    ax_summary.text(
        0.5, 0.985, "DOE Summary",
        transform=ax_summary.transAxes,
        ha="center",
        va="top",
        fontsize=10,
        fontweight="bold",
        color="#111111",
    )

    summary_tbl = ax_summary.table(
        cellText=summary_rows,
        colLabels=["Metric", "Result"],
        bbox=[0.05, 0.53, 0.90, 0.37],
        cellLoc="center",
        colWidths=[0.42, 0.58],
    )

    summary_tbl.auto_set_font_size(False)
    summary_tbl.set_fontsize(7.2)
    summary_tbl.scale(1.0, 1.25)

    for (r, c), cell in summary_tbl.get_celld().items():
        cell.set_edgecolor("#B8C7D6")
        cell.set_linewidth(0.35)
        cell.PAD = 0.035

        if r == 0:
            cell.set_facecolor("#17324D")
            cell.set_text_props(color="white", fontweight="bold", fontsize=7.4)
        else:
            cell.set_facecolor("#FFFFFF" if r % 2 else "#F6F9FC")
            cell.set_text_props(color="#1E2A36", fontsize=7.2)

            if c == 0:
                cell.set_text_props(color="#17324D", fontweight="bold", fontsize=7.2)

    ax_summary.text(
        0.5, 0.455, "Main Factor Significance",
        transform=ax_summary.transAxes,
        ha="center",
        va="top",
        fontsize=9.2,
        fontweight="bold",
        color="#111111",
    )

# HASA AQUI

    sig_tbl = ax_summary.table(
        cellText=sig_rows,
        colLabels=["Code", "Variable", "P-Value", "Alpha", "Significant"],
        bbox=[0.02, 0.04, 0.96, 0.34],
        cellLoc="center",
        colWidths=[0.12, 0.38, 0.18, 0.14, 0.18],
    )
    sig_tbl.auto_set_font_size(False)
    sig_tbl.set_fontsize(6.6)
    for c in range(5):
        sig_tbl[0, c].set_facecolor("#17324D")
        sig_tbl[0, c].set_text_props(color="white", fontweight="bold")
    for (r, c), cell in sig_tbl.get_celld().items():
        cell.set_edgecolor("#222222")
        cell.set_linewidth(0.45)
        if r > 0 and r % 2 == 0:
            cell.set_facecolor("#F4F7FA")
        if r > 0 and c == 0:
            cell.set_text_props(fontweight="bold", color="#17324D")
        if r > 0 and c == 4:
            value = cell.get_text().get_text().strip().lower()
            if value == "yes":
                cell.set_facecolor("#DFF3E5")
                cell.set_text_props(color="#1A7F37", fontweight="bold")
            elif value == "no":
                cell.set_facecolor("#F6E1E1")
                cell.set_text_props(color="#A61B1B", fontweight="bold")

    # Only main factors are listed here. Interaction significance remains visible in the Pareto chart
    # and in the numerical Effect Estimates table.

    return fig


def build_interaction_plots_figure(analysis: Dict[str, Any], decimals: int = 4) -> Figure:
    """Create a separate figure with all pairwise interaction plots."""
    fit = analysis["fit"]
    show_grid = bool(analysis["options"].get("show_grid", True))
    data = _as_dataframe(fit.get("coded_data"))
    factors = list(fit.get("factor_names", []))
    code_map = fit.get("factor_code_map") or factor_code_map(factors)
    y_name = fit.get("response_name", "Y")

    pairs = list(itertools.combinations(factors, 2))
    n_pairs = len(pairs)
    if n_pairs == 0:
        ncols, nrows = 1, 1
    else:
        ncols = min(3, max(1, n_pairs))
        nrows = int(math.ceil(n_pairs / ncols))

    fig_h = max(6.6, 2.75 * nrows + 1.1)
    fig = Figure(figsize=(11.2, fig_h), dpi=120)
    fig.patch.set_facecolor("white")

    # Lower title to avoid clipping in the Interaction Plots tab.
    title_ax = fig.add_axes([0.04, 0.860, 0.92, 0.055])
    title_ax.axis("off")
    title_ax.set_title(f"Factorial DOE Interaction Plots — {y_name}", fontsize=14, fontweight="bold", pad=1)

    if data.empty or n_pairs == 0:
        ax = fig.add_axes([0.08, 0.12, 0.84, 0.76])
        ax.axis("off")
        ax.text(0.5, 0.5, "Interaction plots require at least two valid factors.", ha="center", va="center", fontsize=12)
        return fig

    top = 0.76
    bottom = 0.08
    gs = fig.add_gridspec(
        nrows, ncols,
        left=0.07, right=0.97, top=top, bottom=bottom,
        wspace=0.34, hspace=0.48,
    )

    axes = []
    for idx in range(nrows * ncols):
        ax = fig.add_subplot(gs[idx // ncols, idx % ncols])
        axes.append(ax)

    for ax, (f1, f2) in zip(axes, pairs):
        if f1 not in data.columns or f2 not in data.columns or y_name not in data.columns:
            ax.axis("off")
            continue

        c1 = code_map.get(f1, f1)
        c2 = code_map.get(f2, f2)
        for level, label in [(-1, f"{c2} Low"), (1, f"{c2} High")]:
            subset = data.loc[np.sign(data[f2]) == level]
            vals = []
            for xlevel in [-1, 1]:
                vals.append(subset.loc[np.sign(subset[f1]) == xlevel, y_name].mean())
            ax.plot([0, 1], vals, marker="o", linewidth=1.5, label=label)

        ax.set_xticks([0, 1])
        ax.set_xticklabels([f"{c1} Low", f"{c1} High"], fontsize=8)
        ax.set_title(f"{c1} × {c2}", fontsize=10, fontweight="bold")
        ax.set_ylabel(f"Mean of {y_name}")
        ax.legend(fontsize=7, frameon=False)
        ax.grid(show_grid, alpha=0.18)

    for ax in axes[n_pairs:]:
        ax.axis("off")

    return fig


def build_cube_plot_figure(analysis: Dict[str, Any], decimals: int = 4) -> Figure:
    """Create a separate Cube Plot tab figure."""
    fit = analysis["fit"]
    show_grid = bool(analysis["options"].get("show_grid", True))
    data = _as_dataframe(fit.get("coded_data"))
    factors = list(fit.get("factor_names", []))
    y_name = fit.get("response_name", "Y")

    fig = Figure(figsize=(11.2, 6.6), dpi=120)
    fig.patch.set_facecolor("white")
    title_ax = fig.add_axes([0.04, 0.905, 0.92, 0.055])
    title_ax.axis("off")
    title_ax.set_title(f"Factorial DOE Cube Plot — {y_name}", fontsize=15, fontweight="bold", pad=1)

    ax = fig.add_axes([0.08, 0.10, 0.84, 0.74])
    ax.set_title("Cube Plot of Mean Response", fontsize=11, fontweight="bold")
    ax.axis("off")

    if data.empty or len(factors) < 2:
        ax.text(0.5, 0.5, "Cube plot requires at least two factors.", ha="center", va="center", fontsize=12)
        return fig

    f1, f2 = factors[0], factors[1]
    f3 = factors[2] if len(factors) >= 3 else None

    # Coordinates for 2D square or 3D-style cube projection.
    if f3:
        coords = {
            (-1, -1, -1): (0.20, 0.20), (1, -1, -1): (0.58, 0.20),
            (-1, 1, -1): (0.20, 0.56), (1, 1, -1): (0.58, 0.56),
            (-1, -1, 1): (0.36, 0.36), (1, -1, 1): (0.74, 0.36),
            (-1, 1, 1): (0.36, 0.72), (1, 1, 1): (0.74, 0.72),
        }
        edges = [
            ((-1,-1,-1),(1,-1,-1)), ((-1,1,-1),(1,1,-1)), ((-1,-1,-1),(-1,1,-1)), ((1,-1,-1),(1,1,-1)),
            ((-1,-1,1),(1,-1,1)), ((-1,1,1),(1,1,1)), ((-1,-1,1),(-1,1,1)), ((1,-1,1),(1,1,1)),
            ((-1,-1,-1),(-1,-1,1)), ((1,-1,-1),(1,-1,1)), ((-1,1,-1),(-1,1,1)), ((1,1,-1),(1,1,1)),
        ]
        for a, b in edges:
            ax.plot([coords[a][0], coords[b][0]], [coords[a][1], coords[b][1]], color="#7A8793", linewidth=1.0)
        for levels, (x, y) in coords.items():
            mask = (np.sign(data[f1]) == levels[0]) & (np.sign(data[f2]) == levels[1]) & (np.sign(data[f3]) == levels[2])
            val = data.loc[mask, y_name].mean()
            label = fmt(val, decimals) if pd.notna(val) else "N/A"
            ax.scatter([x], [y], s=90, color="#2F5D87", alpha=0.90, edgecolors="white", linewidths=0.8)
            ax.text(x, y + 0.035, label, ha="center", va="bottom", fontsize=8, fontweight="bold")
        ax.text(0.39, 0.09, f"{f1}: Low → High", ha="center", fontsize=9)
        ax.text(0.04, 0.40, f"{f2}: Low → High", rotation=90, va="center", fontsize=9)
        ax.text(0.83, 0.52, f"{f3}: Low → High", rotation=35, va="center", fontsize=9)
    else:
        coords = {(-1, -1): (0.25, 0.25), (1, -1): (0.75, 0.25), (-1, 1): (0.25, 0.75), (1, 1): (0.75, 0.75)}
        edges = [((-1,-1),(1,-1)),((-1,1),(1,1)),((-1,-1),(-1,1)),((1,-1),(1,1))]
        for a, b in edges:
            ax.plot([coords[a][0], coords[b][0]], [coords[a][1], coords[b][1]], color="#7A8793", linewidth=1.0)
        for levels, (x, y) in coords.items():
            mask = (np.sign(data[f1]) == levels[0]) & (np.sign(data[f2]) == levels[1])
            val = data.loc[mask, y_name].mean()
            label = fmt(val, decimals) if pd.notna(val) else "N/A"
            ax.scatter([x], [y], s=90, color="#2F5D87", alpha=0.90, edgecolors="white", linewidths=0.8)
            ax.text(x, y + 0.04, label, ha="center", va="bottom", fontsize=8, fontweight="bold")
        ax.text(0.50, 0.12, f"{f1}: Low → High", ha="center", fontsize=9)
        ax.text(0.10, 0.50, f"{f2}: Low → High", rotation=90, va="center", fontsize=9)

    note = "Cube plot uses the first 2 or 3 selected factors. Values are mean response at each observed factor setting."
    ax.text(0.5, 0.02, note, ha="center", va="bottom", fontsize=8, color="#4F6680")
    if show_grid:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    return fig


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

    title_ax = fig.add_axes([0.04, 0.905, 0.92, 0.055])
    title_ax.axis("off")
    title_ax.set_title("Factorial DOE Residual Analysis", fontsize=15, fontweight="bold", pad=1)

    gs = fig.add_gridspec(
        2, 3,
        left=0.065, right=0.97, top=0.82, bottom=0.08,
        wspace=0.38, hspace=0.46,
    )

    ax_rf = fig.add_subplot(gs[0, 0])
    ax_hist = fig.add_subplot(gs[0, 1])
    ax_qq = fig.add_subplot(gs[0, 2])
    ax_order = fig.add_subplot(gs[1, 0])
    ax_lev = fig.add_subplot(gs[1, 1])
    ax_summary = fig.add_subplot(gs[1, 2])

    ax_rf.scatter(fitted, residuals, s=42, alpha=0.80, edgecolors="white", linewidths=0.6)
    ax_rf.axhline(0, color="#C62828", linestyle="--", linewidth=1.2)
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
    ax_order.set_title("Residuals vs Run Order", fontsize=10, fontweight="bold")
    ax_order.set_xlabel("Run order")
    ax_order.set_ylabel("Residual")
    ax_order.grid(show_grid, alpha=0.18)

    ax_lev.scatter(leverage, np.abs(std_resid), s=42, alpha=0.80, edgecolors="white", linewidths=0.6)
    threshold = fit.get("residual_analysis", {}).get("high_leverage_threshold", np.nan)
    if pd.notna(threshold):
        ax_lev.axvline(threshold, color="#C62828", linestyle="--", linewidth=1.0)
    ax_lev.axhline(2, color="#C62828", linestyle=":", linewidth=1.0)
    ax_lev.set_title("Leverage vs |Std Residual|", fontsize=10, fontweight="bold")
    ax_lev.set_xlabel("Leverage")
    ax_lev.set_ylabel("|Std Residual|")
    ax_lev.grid(show_grid, alpha=0.18)

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
    tbl.set_fontsize(7.6)
    tbl.scale(1.0, 1.45)
    for c in range(3):
        tbl[0, c].set_facecolor("#17324D")
        tbl[0, c].set_text_props(color="white", fontweight="bold")
    ax_summary.set_title("Assumption Summary", fontsize=10, fontweight="bold", pad=8)
    ax_summary.text(0.02, 0.22, analysis.get("residual_conclusion", ""), fontsize=7.0, va="top", wrap=True)

    return fig


# ─────────────────────────────────────────────
# Main widget
# ─────────────────────────────────────────────

class FactorialDOEBuilder(QWidget):
    def __init__(self, app_window):
        super().__init__()
        self.app_window = app_window
        self.current_df = pd.DataFrame()
        self.current_analysis: Optional[Dict[str, Any]] = None

        self.graph_figure: Optional[Figure] = None
        self.graph_canvas: Optional[FigureCanvas] = None
        self.graph_scroll_area: Optional[ReportScrollArea] = None

        self.cube_figure: Optional[Figure] = None
        self.cube_canvas: Optional[FigureCanvas] = None
        self.cube_scroll_area: Optional[ReportScrollArea] = None

        self.interaction_figure: Optional[Figure] = None
        self.interaction_canvas: Optional[FigureCanvas] = None
        self.interaction_scroll_area: Optional[ReportScrollArea] = None

        self.residual_figure: Optional[Figure] = None
        self.residual_canvas: Optional[FigureCanvas] = None
        self.residual_scroll_area: Optional[ReportScrollArea] = None

        self.current_journal_entry_id: Optional[str] = None
        self.interaction_managers: List[Any] = []

        self._build_ui()
        self.refresh_columns()

    # UI

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)

        header = QHBoxLayout()
        title = QLabel("Factorial DOE Builder")
        title.setStyleSheet("font-size:20px; font-weight:bold; color:#17324D;")
        header.addWidget(title)
        header.addStretch()
        self.lbl_ws = QLabel("Worksheet: -")
        header.addWidget(self.lbl_ws)
        root.addLayout(header)

        self.tabs = QTabWidget()
        root.addWidget(self.tabs)

        self._build_setup_tab()
        self._build_design_tab()
        self._build_numeric_tab()
        self._build_graph_tab()
        self._build_interactions_tab()
        self._build_cube_tab()
        self._build_residual_tab()
        self._build_optimization_tab()
        self._build_advanced_tab()
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
        self.lst_factors = QListWidget()
        self.lst_factors.setSelectionMode(QAbstractItemView.MultiSelection)
        self.lst_factors.setMinimumHeight(240)

        var_layout.addWidget(QLabel("Response Y"), 0, 0)
        var_layout.addWidget(self.cmb_y, 0, 1)
        var_layout.addWidget(QLabel("Factors"), 1, 0)
        var_layout.addWidget(self.lst_factors, 1, 1)

        opt_box = QGroupBox("2. DOE Options")
        opt = QGridLayout(opt_box)

        self.cmb_doe_type = QComboBox()
        self.cmb_doe_type.addItems(["2-Level Full Factorial", "2-Level Fractional Factorial"])

        self.spn_alpha = QDoubleSpinBox()
        self.spn_alpha.setRange(0.001, 0.999)
        self.spn_alpha.setDecimals(3)
        self.spn_alpha.setSingleStep(0.005)
        self.spn_alpha.setValue(0.05)

        self.spn_decimals = QSpinBox()
        self.spn_decimals.setRange(0, 8)
        self.spn_decimals.setValue(4)

        self.spn_interaction_order = QSpinBox()
        self.spn_interaction_order.setRange(1, 6)
        self.spn_interaction_order.setValue(2)

        self.chk_include_interactions = QCheckBox("Include all interactions")
        self.chk_include_interactions.setChecked(True)
        self.chk_include_interactions.setToolTip("If unchecked, the DOE model uses main effects only.")
        self.chk_include_interactions.toggled.connect(self.spn_interaction_order.setEnabled)

        self.chk_grid = QCheckBox("Show grid")
        self.chk_grid.setChecked(True)

        self.chk_coded_units = QCheckBox("Use coded units (-1, +1)")
        self.chk_coded_units.setChecked(True)

        self.chk_center_points = QCheckBox("Detect center points")
        self.chk_center_points.setChecked(True)

        self.chk_journal = QCheckBox("Auto journal save")
        self.chk_journal.setChecked(True)

        self.txt_generators = QTextEdit()
        self.txt_generators.setMaximumHeight(55)
        self.txt_generators.setPlaceholderText("Optional fractional generators, e.g. D=ABC; E=BCD")

        opt.addWidget(QLabel("DOE Type"), 0, 0)
        opt.addWidget(self.cmb_doe_type, 0, 1)
        opt.addWidget(QLabel("Alpha"), 1, 0)
        opt.addWidget(self.spn_alpha, 1, 1)
        opt.addWidget(QLabel("Decimals"), 2, 0)
        opt.addWidget(self.spn_decimals, 2, 1)
        opt.addWidget(QLabel("Max interaction order"), 3, 0)
        opt.addWidget(self.spn_interaction_order, 3, 1)
        opt.addWidget(self.chk_include_interactions, 4, 1)
        opt.addWidget(self.chk_grid, 5, 1)
        opt.addWidget(self.chk_coded_units, 6, 1)
        opt.addWidget(self.chk_center_points, 7, 1)
        opt.addWidget(self.chk_journal, 8, 1)
        opt.addWidget(QLabel("Generators"), 9, 0)
        opt.addWidget(self.txt_generators, 9, 1)

        top.addWidget(var_box, 2)
        top.addWidget(opt_box, 1)
        layout.addLayout(top)

        run_box = QGroupBox("3. Run")
        run_layout = QVBoxLayout(run_box)
        buttons = QHBoxLayout()

        self.btn_refresh = QPushButton("Refresh Columns")
        self.btn_run = QPushButton("Run Factorial DOE")
        self.btn_run.setStyleSheet("background:#2F5D87; color:white; font-weight:bold; padding:8px; border-radius:6px;")

        self.btn_refresh.clicked.connect(self.refresh_columns)
        self.btn_run.clicked.connect(self.run_analysis)

        buttons.addWidget(self.btn_refresh)
        buttons.addWidget(self.btn_run)
        run_layout.addLayout(buttons)

        note = QLabel(
            "Select one numeric response and two or more 2-level factors. "
            "Numeric factors are coded from observed low/high values. "
            "Categorical factors must have exactly two levels."
        )
        note.setWordWrap(True)
        note.setStyleSheet("color:#4F6680;")
        run_layout.addWidget(note)

        layout.addWidget(run_box)
        layout.addStretch()

        self.tabs.addTab(tab, "Setup")

    def _build_design_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        self.tbl_design = QTableWidget()
        self.tbl_design_coded = QTableWidget()

        layout.addWidget(QLabel("Design Matrix - Original Units"))
        layout.addWidget(self.tbl_design)
        layout.addWidget(QLabel("Design Matrix - Coded Units"))
        layout.addWidget(self.tbl_design_coded)

        self.tabs.addTab(tab, "Design Matrix")

    def _build_numeric_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        self.tbl_effects = QTableWidget()
        self.tbl_effects.setColumnCount(8)
        self.tbl_effects.setHorizontalHeaderLabels(["Term", "Effect", "Coef", "SE Coef", "T-Value", "P-Value", "Decision", "Conclusion"])

        self.tbl_anova = QTableWidget()
        self.tbl_anova.setColumnCount(6)
        self.tbl_anova.setHorizontalHeaderLabels(["Source", "DF", "Adj SS", "Adj MS", "F-Value", "P-Value"])

        self.tbl_model_summary = QTableWidget()
        self.tbl_model_summary.setColumnCount(7)
        self.tbl_model_summary.setHorizontalHeaderLabels(["S", "R-sq", "R-sq(adj)", "R-sq(pred)", "PRESS", "Model p", "Conclusion"])

        layout.addWidget(QLabel("Effect Estimates"))
        layout.addWidget(self.tbl_effects)
        layout.addWidget(QLabel("Analysis of Variance"))
        layout.addWidget(self.tbl_anova)
        layout.addWidget(QLabel("Model Summary"))
        layout.addWidget(self.tbl_model_summary)

        self.tabs.addTab(tab, "Numerical Results")

    def _build_graph_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        toolbar = QHBoxLayout()
        self.btn_zoom_out_graph = QPushButton("Zoom -")
        self.btn_zoom_fit_graph = QPushButton("Fit")
        self.btn_zoom_reset_graph = QPushButton("100%")
        self.btn_zoom_in_graph = QPushButton("Zoom +")
        self.btn_copy_graph = QPushButton("Copy Graph")
        self.btn_export_graph = QPushButton("Export PNG")
        self.btn_update_journal_1 = QPushButton("Update Journal Entry")

        self.btn_zoom_out_graph.clicked.connect(lambda: self.zoom_graph("graph", "out"))
        self.btn_zoom_fit_graph.clicked.connect(lambda: self.zoom_graph("graph", "fit"))
        self.btn_zoom_reset_graph.clicked.connect(lambda: self.zoom_graph("graph", "reset"))
        self.btn_zoom_in_graph.clicked.connect(lambda: self.zoom_graph("graph", "in"))
        self.btn_copy_graph.clicked.connect(lambda: self.copy_graph("graph"))
        self.btn_export_graph.clicked.connect(lambda: self.export_png("graph"))
        self.btn_update_journal_1.clicked.connect(self.update_journal_entry)

        toolbar.addStretch()
        toolbar.addWidget(self.btn_zoom_out_graph)
        toolbar.addWidget(self.btn_zoom_fit_graph)
        toolbar.addWidget(self.btn_zoom_reset_graph)
        toolbar.addWidget(self.btn_zoom_in_graph)
        toolbar.addWidget(self.btn_copy_graph)
        toolbar.addWidget(self.btn_export_graph)
        toolbar.addWidget(self.btn_update_journal_1)
        layout.addLayout(toolbar)

        self.graph_container = QWidget()
        self.graph_layout = QVBoxLayout(self.graph_container)
        layout.addWidget(self.graph_container)

        self.tabs.addTab(tab, "Graphs")

    def _build_interactions_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        toolbar = QHBoxLayout()
        self.btn_zoom_out_interaction = QPushButton("Zoom -")
        self.btn_zoom_fit_interaction = QPushButton("Fit")
        self.btn_zoom_reset_interaction = QPushButton("100%")
        self.btn_zoom_in_interaction = QPushButton("Zoom +")
        self.btn_copy_interaction = QPushButton("Copy Interaction Plots")
        self.btn_export_interaction = QPushButton("Export PNG")
        self.btn_update_journal_interaction = QPushButton("Update Journal Entry")

        self.btn_zoom_out_interaction.clicked.connect(lambda: self.zoom_graph("interaction", "out"))
        self.btn_zoom_fit_interaction.clicked.connect(lambda: self.zoom_graph("interaction", "fit"))
        self.btn_zoom_reset_interaction.clicked.connect(lambda: self.zoom_graph("interaction", "reset"))
        self.btn_zoom_in_interaction.clicked.connect(lambda: self.zoom_graph("interaction", "in"))
        self.btn_copy_interaction.clicked.connect(lambda: self.copy_graph("interaction"))
        self.btn_export_interaction.clicked.connect(lambda: self.export_png("interaction"))
        self.btn_update_journal_interaction.clicked.connect(self.update_journal_entry)

        toolbar.addStretch()
        toolbar.addWidget(self.btn_zoom_out_interaction)
        toolbar.addWidget(self.btn_zoom_fit_interaction)
        toolbar.addWidget(self.btn_zoom_reset_interaction)
        toolbar.addWidget(self.btn_zoom_in_interaction)
        toolbar.addWidget(self.btn_copy_interaction)
        toolbar.addWidget(self.btn_export_interaction)
        toolbar.addWidget(self.btn_update_journal_interaction)
        layout.addLayout(toolbar)

        self.interaction_container = QWidget()
        self.interaction_layout = QVBoxLayout(self.interaction_container)
        layout.addWidget(self.interaction_container)

        self.tabs.addTab(tab, "Interaction Plots")

    def _build_cube_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        toolbar = QHBoxLayout()
        self.btn_zoom_out_cube = QPushButton("Zoom -")
        self.btn_zoom_fit_cube = QPushButton("Fit")
        self.btn_zoom_reset_cube = QPushButton("100%")
        self.btn_zoom_in_cube = QPushButton("Zoom +")
        self.btn_copy_cube = QPushButton("Copy Cube Plot")
        self.btn_export_cube = QPushButton("Export PNG")
        self.btn_update_journal_cube = QPushButton("Update Journal Entry")

        self.btn_zoom_out_cube.clicked.connect(lambda: self.zoom_graph("cube", "out"))
        self.btn_zoom_fit_cube.clicked.connect(lambda: self.zoom_graph("cube", "fit"))
        self.btn_zoom_reset_cube.clicked.connect(lambda: self.zoom_graph("cube", "reset"))
        self.btn_zoom_in_cube.clicked.connect(lambda: self.zoom_graph("cube", "in"))
        self.btn_copy_cube.clicked.connect(lambda: self.copy_graph("cube"))
        self.btn_export_cube.clicked.connect(lambda: self.export_png("cube"))
        self.btn_update_journal_cube.clicked.connect(self.update_journal_entry)

        toolbar.addStretch()
        toolbar.addWidget(self.btn_zoom_out_cube)
        toolbar.addWidget(self.btn_zoom_fit_cube)
        toolbar.addWidget(self.btn_zoom_reset_cube)
        toolbar.addWidget(self.btn_zoom_in_cube)
        toolbar.addWidget(self.btn_copy_cube)
        toolbar.addWidget(self.btn_export_cube)
        toolbar.addWidget(self.btn_update_journal_cube)
        layout.addLayout(toolbar)

        self.cube_container = QWidget()
        self.cube_layout = QVBoxLayout(self.cube_container)
        layout.addWidget(self.cube_container)

        self.tabs.addTab(tab, "Cube Plot")

    def _build_residual_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        toolbar = QHBoxLayout()
        self.btn_zoom_out_residual = QPushButton("Zoom -")
        self.btn_zoom_fit_residual = QPushButton("Fit")
        self.btn_zoom_reset_residual = QPushButton("100%")
        self.btn_zoom_in_residual = QPushButton("Zoom +")
        self.btn_copy_residual = QPushButton("Copy Graph")
        self.btn_export_residual = QPushButton("Export PNG")
        self.btn_update_journal_2 = QPushButton("Update Journal Entry")

        self.btn_zoom_out_residual.clicked.connect(lambda: self.zoom_graph("residual", "out"))
        self.btn_zoom_fit_residual.clicked.connect(lambda: self.zoom_graph("residual", "fit"))
        self.btn_zoom_reset_residual.clicked.connect(lambda: self.zoom_graph("residual", "reset"))
        self.btn_zoom_in_residual.clicked.connect(lambda: self.zoom_graph("residual", "in"))
        self.btn_copy_residual.clicked.connect(lambda: self.copy_graph("residual"))
        self.btn_export_residual.clicked.connect(lambda: self.export_png("residual"))
        self.btn_update_journal_2.clicked.connect(self.update_journal_entry)

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
        self.txt_residual.setMaximumHeight(165)
        layout.addWidget(self.txt_residual)

        self.tabs.addTab(tab, "Residual Analysis")

    def _build_optimization_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        top = QHBoxLayout()
        box = QGroupBox("Optimization Settings")
        grid = QGridLayout(box)

        self.cmb_objective = QComboBox()
        self.cmb_objective.addItems(["Maximize Y", "Minimize Y", "Target Y"])

        self.spn_target_y = QDoubleSpinBox()
        self.spn_target_y.setRange(-1_000_000_000, 1_000_000_000)
        self.spn_target_y.setDecimals(6)

        self.btn_optimize = QPushButton("Optimize")
        self.btn_optimize.setStyleSheet("background:#2F5D87; color:white; font-weight:bold; padding:6px; border-radius:5px;")
        self.btn_optimize.clicked.connect(self.optimize_response)

        grid.addWidget(QLabel("Objective"), 0, 0)
        grid.addWidget(self.cmb_objective, 0, 1)
        grid.addWidget(QLabel("Target Y"), 1, 0)
        grid.addWidget(self.spn_target_y, 1, 1)
        grid.addWidget(self.btn_optimize, 2, 1)

        self.txt_optimization = QTextEdit()
        self.txt_optimization.setReadOnly(True)

        self.tbl_optimization = QTableWidget()

        top.addWidget(box, 1)
        top.addWidget(self.txt_optimization, 2)
        layout.addLayout(top)
        layout.addWidget(QLabel("Candidate Settings"))
        layout.addWidget(self.tbl_optimization)

        self.tabs.addTab(tab, "Optimization")

    def _build_advanced_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        self.txt_advanced = QTextEdit()
        self.txt_advanced.setReadOnly(True)
        layout.addWidget(self.txt_advanced)

        self.tbl_alias = QTableWidget()
        self.tbl_alias.setColumnCount(2)
        self.tbl_alias.setHorizontalHeaderLabels(["Term", "Aliases"])
        layout.addWidget(QLabel("Alias Structure / Confounding"))
        layout.addWidget(self.tbl_alias)

        self.tabs.addTab(tab, "Advanced DOE")

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

        cols_all = []
        cols_numeric = []
        if not self.current_df.empty:
            for col in self.current_df.columns:
                col_name = str(col).strip()
                if not col_name:
                    continue

                series = self.current_df[col]
                nonempty = series.dropna()
                nonempty = nonempty[nonempty.astype(str).str.strip() != ""]
                valid_count = int(len(nonempty))

                # Ignore truly empty worksheet columns.
                if valid_count < 2:
                    continue

                cols_all.append(str(col))

                numeric = pd.to_numeric(series, errors="coerce")
                if int(numeric.notna().sum()) >= 4:
                    cols_numeric.append(str(col))

        self.cmb_y.clear()
        self.lst_factors.clear()

        self.cmb_y.addItems(cols_numeric)
        for col in cols_all:
            self.lst_factors.addItem(col)

        if cols_numeric:
            self.cmb_y.setCurrentIndex(0)

        response = self.cmb_y.currentText()
        selected_count = 0
        for i in range(self.lst_factors.count()):
            item = self.lst_factors.item(i)
            if item.text() != response and selected_count < 4:
                item.setSelected(True)
                selected_count += 1

        self.log(f"Columns refreshed. Numeric={cols_numeric}; Factor candidates={cols_all}")

    def _options(self) -> FactorialDOEOptions:
        return FactorialDOEOptions(
            doe_type=self.cmb_doe_type.currentText(),
            alpha=float(self.spn_alpha.value()),
            decimals=int(self.spn_decimals.value()),
            max_interaction_order=int(self.spn_interaction_order.value()),
            include_interactions=self.chk_include_interactions.isChecked(),
            show_grid=self.chk_grid.isChecked(),
            coded_units=self.chk_coded_units.isChecked(),
            include_center_points=self.chk_center_points.isChecked(),
            auto_journal_save=self.chk_journal.isChecked(),
            generators=self.txt_generators.toPlainText().strip(),
        )

    def _selected_factors(self) -> List[str]:
        return [item.text() for item in self.lst_factors.selectedItems()]

    def run_analysis(self):
        try:
            opts = self._options()

            if self.current_df.empty:
                raise ValueError("No active worksheet data found.")

            y_col = self.cmb_y.currentText()
            factors = self._selected_factors()

            if y_col in factors:
                raise ValueError("Response Y cannot also be selected as a factor.")

            fit = factorial_doe_fit(self.current_df, y_col, factors, opts)
            res_txt = residual_conclusion(fit, opts.alpha, opts.decimals)

            analysis = {
                "id": str(uuid.uuid4()),
                "module": "Factorial DOE",
                "type": "Factorial DOE",
                "worksheet": self.app_window.active_worksheet_name(),
                "options": asdict(opts),
                "fit": fit,
                "residual_conclusion": res_txt,
                "created_at": now_string(),
            }
            analysis["summary"] = self._build_summary_text(analysis)

            self.current_analysis = analysis
            self._populate_design_matrix(analysis)
            self._populate_numeric_results(analysis)
            self._populate_advanced(analysis)
            self._render_figures(analysis)
            self.txt_summary.setPlainText(analysis["summary"])
            self.txt_residual.setPlainText(res_txt)

            if opts.auto_journal_save:
                self.save_to_journal()

            self.tabs.setCurrentIndex(2)
            self.log("Factorial DOE completed successfully.")

        except Exception as exc:
            self.log(f"ERROR: {exc}")
            QMessageBox.critical(self, "Factorial DOE Error", str(exc))

    def _populate_table_from_dataframe(self, table: QTableWidget, df: Any, decimals: int = 4):
        df = _as_dataframe(df)
        table.clear()
        table.setRowCount(len(df))
        table.setColumnCount(len(df.columns))
        table.setHorizontalHeaderLabels([str(c) for c in df.columns])
        for r in range(len(df)):
            for c, col in enumerate(df.columns):
                val = df.iloc[r, c]
                if isinstance(val, (int, float, np.integer, np.floating)):
                    text = fmt(val, decimals)
                else:
                    text = str(val)
                table.setItem(r, c, QTableWidgetItem(text))
        table.resizeColumnsToContents()

    def _populate_design_matrix(self, analysis: Dict[str, Any]):
        d = int(analysis["options"]["decimals"])
        fit = analysis["fit"]
        self._populate_table_from_dataframe(self.tbl_design, fit["design_matrix"], d)
        self._populate_table_from_dataframe(self.tbl_design_coded, fit["design_matrix_coded"], d)

    def _populate_numeric_results(self, analysis: Dict[str, Any]):
        d = int(analysis["options"]["decimals"])
        fit = analysis["fit"]
        alpha = float(analysis["options"].get("alpha", 0.05))

        coef_rows = fit.get("coefficient_rows", [])
        self.tbl_effects.setRowCount(len(coef_rows))
        for r, row in enumerate(coef_rows):
            values = [
                row.get("Term", ""),
                fmt(row.get("Effect"), d) if row.get("Term") != "Constant" else "",
                fmt(row.get("Coef"), d),
                fmt(row.get("SE Coef"), d),
                fmt(row.get("T-Value"), d),
                fmt(row.get("P-Value"), d),
                row.get("Decision", ""),
                row.get("Conclusion", ""),
            ]
            for c, value in enumerate(values):
                self.tbl_effects.setItem(r, c, QTableWidgetItem(str(value)))
        self.tbl_effects.resizeColumnsToContents()

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

    def _populate_advanced(self, analysis: Dict[str, Any]):
        fit = analysis["fit"]
        lines = [
            "ADVANCED DOE SUMMARY",
            "────────────────────",
            f"DOE Type: {fit.get('doe_type', '')}",
            f"Factors: {', '.join(fit.get('factor_names', []))}",
            "Factor codes: " + ", ".join([f"{code}={name}" for name, code in (fit.get('factor_code_map') or factor_code_map(fit.get('factor_names', []))).items()]),
            f"Model terms: {', '.join([coded_term_label(t, fit.get('factor_code_map') or factor_code_map(fit.get('factor_names', []))) for t in fit.get('model_terms', [])])}",
            "",
            "Generators",
            "──────────",
        ]

        if fit.get("generators"):
            for lhs, rhs in fit.get("generators", []):
                lines.append(f"{lhs} = {rhs}")
        else:
            lines.append("No fractional generators defined. Full factorial or no alias structure.")

        lines += [
            "",
            "Defining Relation",
            "─────────────────",
            " = ".join(fit.get("defining_relation", ["I"])),
            "",
            "Center Points",
            "─────────────",
        ]
        cs = fit.get("center_summary", {})
        lines.append(f"Center points detected: {cs.get('center_points_n', 0)}")
        lines.append(f"Center mean: {fmt(cs.get('center_mean'), int(analysis['options']['decimals']))}")
        lines.append(f"Factorial mean: {fmt(cs.get('factorial_mean'), int(analysis['options']['decimals']))}")
        lines.append(f"Center delta: {fmt(cs.get('center_delta'), int(analysis['options']['decimals']))}")

        self.txt_advanced.setPlainText("\n".join(lines))

        alias_rows = fit.get("alias_rows", [])
        self.tbl_alias.setRowCount(len(alias_rows))
        for r, row in enumerate(alias_rows):
            self.tbl_alias.setItem(r, 0, QTableWidgetItem(str(row.get("Term", ""))))
            self.tbl_alias.setItem(r, 1, QTableWidgetItem(str(row.get("Aliases", ""))))
        self.tbl_alias.resizeColumnsToContents()

    def _render_figures(self, analysis: Dict[str, Any]):
        self._clear_layout(self.graph_layout)
        self._clear_layout(self.interaction_layout)
        self._clear_layout(self.cube_layout)
        self._clear_layout(self.residual_layout)

        d = int(analysis["options"]["decimals"])

        self.graph_figure = build_doe_graphs_figure(analysis, d)
        self.graph_canvas = FigureCanvas(self.graph_figure)
        self.graph_scroll_area = ReportScrollArea(self.graph_canvas, self.graph_figure, initial_zoom=0.92)
        self.graph_layout.addWidget(self.graph_scroll_area)
        self.graph_scroll_area.fit_to_view()

        self.interaction_figure = build_interaction_plots_figure(analysis, d)
        self.interaction_canvas = FigureCanvas(self.interaction_figure)
        self.interaction_scroll_area = ReportScrollArea(self.interaction_canvas, self.interaction_figure, initial_zoom=0.92)
        self.interaction_layout.addWidget(self.interaction_scroll_area)
        self.interaction_scroll_area.fit_to_view()

        self.cube_figure = build_cube_plot_figure(analysis, d)
        self.cube_canvas = FigureCanvas(self.cube_figure)
        self.cube_scroll_area = ReportScrollArea(self.cube_canvas, self.cube_figure, initial_zoom=0.92)
        self.cube_layout.addWidget(self.cube_scroll_area)
        self.cube_scroll_area.fit_to_view()

        self.residual_figure = build_residual_figure(analysis, d)
        self.residual_canvas = FigureCanvas(self.residual_figure)
        self.residual_scroll_area = ReportScrollArea(self.residual_canvas, self.residual_figure, initial_zoom=0.92)
        self.residual_layout.addWidget(self.residual_scroll_area)
        self.residual_scroll_area.fit_to_view()

        self._enable_interactions()

    def _enable_interactions(self):
        self.interaction_managers.clear()
        if enable_interaction is None:
            return

        for fig, canvas in [(self.graph_figure, self.graph_canvas), (self.interaction_figure, self.interaction_canvas), (self.cube_figure, self.cube_canvas), (self.residual_figure, self.residual_canvas)]:
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

    def zoom_graph(self, graph_type: str, action: str):
        if graph_type == "graph":
            scroll = self.graph_scroll_area
        elif graph_type == "interaction":
            scroll = self.interaction_scroll_area
        elif graph_type == "cube":
            scroll = self.cube_scroll_area
        else:
            scroll = self.residual_scroll_area
        if scroll is None:
            return
        if action == "in":
            scroll.zoom_in()
        elif action == "out":
            scroll.zoom_out()
        elif action == "reset":
            scroll.reset_zoom()
        elif action == "fit":
            scroll.fit_to_view()

    def optimize_response(self):
        if self.current_analysis is None:
            QMessageBox.information(self, "No DOE", "Run a Factorial DOE analysis first.")
            return

        fit = self.current_analysis["fit"]
        d = int(self.current_analysis["options"]["decimals"])
        objective = self.cmb_objective.currentText()
        target = float(self.spn_target_y.value()) if objective == "Target Y" else None

        result = optimize_doe_response(fit, objective, target)
        best = result.get("best_settings", {})

        lines = [
            "DOE Optimization Result",
            "───────────────────────",
            f"Objective: {objective}",
            f"Predicted {fit['response_name']}: {fmt(result.get('predicted_y'), d)}",
            f"Desirability: {fmt(result.get('desirability'), d)}",
            "",
            "Optimal Factor Settings",
            "───────────────────────",
        ]
        for k, v in best.items():
            lines.append(f"• {k}: {v}")
        self.txt_optimization.setPlainText("\n".join(lines))

        rows = result.get("candidate_rows", [])
        if rows:
            df = pd.DataFrame(rows)
            self._populate_table_from_dataframe(self.tbl_optimization, df, d)

    def _build_summary_text(self, analysis: Dict[str, Any]) -> str:
        d = int(analysis["options"]["decimals"])
        fit = analysis["fit"]
        alpha = float(analysis["options"].get("alpha", 0.05))

        sig_terms = [
            r for r in fit.get("coefficient_rows", [])
            if r.get("Term") != "Constant" and pd.notna(r.get("P-Value")) and r.get("P-Value") <= alpha
        ]

        lines = [
            "=" * 92,
            "FACTORIAL DOE REPORT — SPS STUDIO",
            "=" * 92,
            f"Worksheet: {analysis.get('worksheet', '')}",
            f"DOE Type: {fit.get('doe_type', '')}",
            f"Response Y: {fit.get('response_name', '')}",
            f"Factors: {', '.join(fit.get('factor_names', []))}",
            f"N: {fit.get('n')}",
            f"Alpha: {fmt(alpha, 3)}",
            "",
            "MODEL SUMMARY",
            "-" * 92,
            f"S={fmt(fit.get('s'), d)} | R-sq={fmt(100 * fit.get('r2'), 2)}% | "
            f"R-sq(adj)={fmt(100 * fit.get('adj_r2'), 2)}% | R-sq(pred)={fmt(100 * fit.get('pred_r2'), 2)}%",
            f"Model p-value={fmt(fit.get('p_model'), d)} | Decision={p_decision(fit.get('p_model'), alpha)} | {significance_label(fit.get('p_model'), alpha)}",
            "",
            "SIGNIFICANT TERMS",
            "-" * 92,
        ]

        if sig_terms:
            for row in sig_terms:
                lines.append(f"• {row['Term']}: Effect={fmt(row.get('Effect'), d)}, P={fmt(row.get('P-Value'), d)}")
        else:
            lines.append("No statistically significant terms were detected at the selected alpha.")

        lines += [
            "",
            "EFFECT ESTIMATES",
            "-" * 92,
            f"{'Term':<24}{'Effect':<14}{'Coef':<14}{'T':<12}{'P':<12}{'Conclusion':<20}",
        ]
        for row in fit.get("coefficient_rows", []):
            lines.append(
                f"{row.get('Term',''):<24}{fmt(row.get('Effect'), d):<14}{fmt(row.get('Coef'), d):<14}"
                f"{fmt(row.get('T-Value'), d):<12}{fmt(row.get('P-Value'), d):<12}{row.get('Conclusion',''):<20}"
            )

        lines += [
            "",
            "ANALYSIS OF VARIANCE",
            "-" * 92,
            f"{'Source':<24}{'DF':<8}{'Adj SS':<14}{'Adj MS':<14}{'F-Value':<14}{'P-Value':<14}",
        ]
        for row in fit.get("anova_rows", []):
            lines.append(
                f"{row.get('source',''):<24}{str(row.get('df','')):<8}{fmt(row.get('adj_ss'), d):<14}"
                f"{fmt(row.get('adj_ms'), d):<14}{fmt(row.get('f_value'), d):<14}{fmt(row.get('p_value'), d):<14}"
            )

        lines += [
            "",
            "RESIDUAL ANALYSIS",
            "-" * 92,
            analysis.get("residual_conclusion", ""),
            "",
            "ADVANCED DOE",
            "-" * 92,
            f"Defining relation: {' = '.join(fit.get('defining_relation', ['I']))}",
            f"Center points: {fit.get('center_summary', {}).get('center_points_n', 0)}",
        ]

        return "\n".join(lines)

    # Clipboard / Export / Journal

    def _figure_by_name(self, graph_type: str) -> Optional[Figure]:
        if graph_type == "graph":
            return self.graph_figure
        if graph_type == "interaction":
            return self.interaction_figure
        if graph_type == "cube":
            return self.cube_figure
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

        graph_b64 = figure_to_png_base64(self.graph_figure) if self.graph_figure is not None else ""
        interaction_b64 = figure_to_png_base64(self.interaction_figure) if self.interaction_figure is not None else ""
        residual_b64 = figure_to_png_base64(self.residual_figure) if self.residual_figure is not None else ""
        cube_b64 = figure_to_png_base64(self.cube_figure) if self.cube_figure is not None else ""

        fit = self.current_analysis.get("fit", {})
        name = f"Factorial DOE - {fit.get('response_name', '')}"

        return {
            "id": self.current_journal_entry_id or str(uuid.uuid4()),
            "name": name,
            "type": "Factorial DOE",
            "module": "Factorial DOE",
            "worksheet": self.current_analysis.get("worksheet", ""),
            "status": p_decision(fit.get("p_model"), float(self.current_analysis.get("options", {}).get("alpha", 0.05))),
            "created_at": self.current_analysis.get("created_at", now_string()),
            "modified_at": now_string(),
            "summary": self.txt_summary.toPlainText(),
            "figure_png_base64": graph_b64,
            "interaction_png_base64": interaction_b64,
            "residual_png_base64": residual_b64,
            "cube_png_base64": cube_b64,
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
        self.app_window.statusBar().showMessage("Factorial DOE journal entry updated.")

    def _restore_payload_after_json(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Restore DataFrames serialized as list[dict] inside Analysis Journal payload."""
        if not payload:
            return payload
        fit = payload.get("fit", {})
        for key in ["coded_data", "center_data", "design_matrix", "design_matrix_coded", "x_terms"]:
            if key in fit and not isinstance(fit[key], pd.DataFrame):
                fit[key] = _as_dataframe(fit[key])
        return payload

    def load_from_journal_entry(self, entry: Dict[str, Any]):
        self.current_journal_entry_id = entry.get("id")
        payload = entry.get("payload", {})
        payload = self._restore_payload_after_json(payload)
        self.current_analysis = payload

        if self.current_analysis:
            self._populate_design_matrix(self.current_analysis)
            self._populate_numeric_results(self.current_analysis)
            self._populate_advanced(self.current_analysis)
            self._render_figures(self.current_analysis)

        self.txt_summary.setPlainText(entry.get("summary", payload.get("summary", "")))
        self.txt_residual.setPlainText(payload.get("residual_conclusion", ""))
