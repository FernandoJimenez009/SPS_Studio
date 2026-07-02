
"""
modules/simple_regression.py
SPS Studio — Simple Regression Builder

PySide6 module integrated with SPS Studio.

Purpose:
- Simple regression analysis for one Y variable vs one X variable.
- Compares Linear, Quadratic and Cubic models.
- Uses checkboxes for model selection.
- Auto mode runs Linear + Quadratic + Cubic and recommends the best model.
- Shows numerical results, relationship graph and residual diagnostics.
- Integrates with Analysis Journal and core.plot_interaction_tools.

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

from PySide6.QtGui import QImage
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
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


MODEL_ORDER = ["Linear", "Quadratic", "Cubic"]
MODEL_DEGREE = {"Linear": 1, "Quadratic": 2, "Cubic": 3}


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
    if isinstance(obj, np.poly1d):
        return str(obj)
    if isinstance(obj, (np.floating, float)):
        return None if np.isnan(obj) or np.isinf(obj) else float(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, dict):
        return {str(k): serializable(v) for k, v in obj.items() if k != "poly"}
    if isinstance(obj, (list, tuple)):
        return [serializable(v) for v in obj]
    return obj


def pearson_r_p(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    if len(x) < 3:
        return np.nan, np.nan
    if np.nanstd(x) == 0 or np.nanstd(y) == 0:
        return np.nan, np.nan
    try:
        r, p = stats.pearsonr(x, y)
        return float(r), float(p)
    except Exception:
        return np.nan, np.nan


def correlation_label(r: float) -> str:
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


def significance_label(p_value: float, alpha: float) -> str:
    if pd.isna(p_value):
        return "N/A"
    if p_value <= alpha:
        return "Significant"
    if p_value <= alpha * 2:
        return "Borderline"
    return "Not significant"


def equation_text(coeffs: np.ndarray, decimals: int = 4) -> str:
    deg = len(coeffs) - 1
    parts = []
    for i, c in enumerate(coeffs):
        power = deg - i
        value = f"{float(c):.{decimals}f}"
        if power == 0:
            parts.append(value)
        elif power == 1:
            parts.append(f"{value}·x")
        else:
            parts.append(f"{value}·x^{power}")
    return "y = " + " + ".join(parts).replace("+ -", "− ")


def adjusted_r2(r2: float, n: int, predictors: int) -> float:
    if n <= predictors + 1 or pd.isna(r2):
        return np.nan
    return 1.0 - (1.0 - r2) * (n - 1) / (n - predictors - 1)


def durbin_watson_stat(residuals: np.ndarray) -> float:
    residuals = np.asarray(residuals, dtype=float)
    if residuals.size < 2:
        return np.nan
    numerator = np.sum(np.diff(residuals) ** 2)
    denominator = np.sum(residuals ** 2)
    return float(numerator / denominator) if denominator != 0 else np.nan


def safe_shapiro(residuals: np.ndarray) -> Tuple[float, float]:
    residuals = np.asarray(residuals, dtype=float)
    residuals = residuals[~np.isnan(residuals)]
    if residuals.size < 3:
        return np.nan, np.nan
    if residuals.size > 5000:
        residuals = residuals[:5000]
    try:
        stat, p = shapiro(residuals)
        return float(stat), float(p)
    except Exception:
        return np.nan, np.nan


def variance_stability_test(fitted: np.ndarray, residuals: np.ndarray) -> Tuple[float, float]:
    fitted = np.asarray(fitted, dtype=float)
    residuals = np.asarray(residuals, dtype=float)

    mask = ~(np.isnan(fitted) | np.isnan(residuals))
    fitted = fitted[mask]
    residuals = residuals[mask]

    if fitted.size < 6:
        return np.nan, np.nan

    med = np.median(fitted)
    low = residuals[fitted <= med]
    high = residuals[fitted > med]

    if low.size < 3 or high.size < 3:
        return np.nan, np.nan

    try:
        stat, p = levene(low, high, center="median")
        return float(stat), float(p)
    except Exception:
        return np.nan, np.nan


def standardized_residuals(residuals: np.ndarray) -> np.ndarray:
    residuals = np.asarray(residuals, dtype=float)
    s = np.std(residuals, ddof=1) if residuals.size > 1 else np.nan
    if pd.isna(s) or s == 0:
        return np.full(residuals.shape, np.nan)
    return residuals / s


def fit_polynomial_model(x: np.ndarray, y: np.ndarray, degree: int, alpha: float = 0.05) -> Optional[Dict[str, Any]]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if len(x) < degree + 2:
        return None
    if np.unique(x).size < degree + 1:
        return None

    try:
        coeffs = np.polyfit(x, y, degree)
    except Exception:
        return None

    poly = np.poly1d(coeffs)
    fitted = poly(x)
    residuals = y - fitted

    n = len(y)
    p = degree
    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    ss_reg = ss_tot - ss_res
    r2 = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan
    adj = adjusted_r2(r2, n, p)
    mse = ss_res / (n - p - 1) if n > p + 1 else np.nan
    rmse = math.sqrt(mse) if mse >= 0 else np.nan

    df_reg = degree
    df_err = n - degree - 1
    ms_reg = ss_reg / df_reg if df_reg > 0 else np.nan
    ms_err = ss_res / df_err if df_err > 0 else np.nan
    f_stat = ms_reg / ms_err if ms_err and ms_err > 0 else np.nan
    p_model = stats.f.sf(f_stat, df_reg, df_err) if not pd.isna(f_stat) else np.nan

    sh_stat, sh_p = safe_shapiro(residuals)
    lev_stat, lev_p = variance_stability_test(fitted, residuals)
    dw = durbin_watson_stat(residuals)
    std_resid = standardized_residuals(residuals)
    outliers_n = int(np.sum(np.abs(std_resid) > 3)) if std_resid.size else 0

    if len(x) >= 5 and np.nanstd(residuals) > 0:
        try:
            curvature_r, curvature_p = pearson_r_p((x - np.mean(x)) ** 2, residuals)
        except Exception:
            curvature_r, curvature_p = np.nan, np.nan
    else:
        curvature_r, curvature_p = np.nan, np.nan

    return {
        "model": {1: "Linear", 2: "Quadratic", 3: "Cubic"}[degree],
        "degree": degree,
        "n": int(n),
        "coefficients": coeffs,
        "poly": poly,
        "equation": equation_text(coeffs, 4),
        "fitted": fitted,
        "residuals": residuals,
        "std_residuals": std_resid,
        "ss_reg": ss_reg,
        "ss_res": ss_res,
        "ss_tot": ss_tot,
        "df_reg": int(df_reg),
        "df_err": int(df_err),
        "ms_reg": ms_reg,
        "ms_err": ms_err,
        "f_stat": float(f_stat) if not pd.isna(f_stat) else np.nan,
        "p_model": float(p_model) if not pd.isna(p_model) else np.nan,
        "r2": float(r2),
        "adj_r2": float(adj),
        "mse": float(mse) if not pd.isna(mse) else np.nan,
        "rmse": float(rmse) if not pd.isna(rmse) else np.nan,
        "shapiro_stat": sh_stat,
        "shapiro_p": sh_p,
        "levene_stat": lev_stat,
        "levene_p": lev_p,
        "durbin_watson": dw,
        "outliers_n": outliers_n,
        "curvature_r": curvature_r,
        "curvature_p": curvature_p,
        "normality_ok": pd.notna(sh_p) and sh_p > alpha,
        "homoscedasticity_ok": pd.notna(lev_p) and lev_p > alpha,
        "independence_ok": pd.notna(dw) and 1.5 <= dw <= 2.5,
        "outliers_ok": outliers_n == 0,
        "linearity_ok": degree > 1 or (pd.isna(curvature_p) or curvature_p > alpha),
    }


def model_score(model: Dict[str, Any]) -> Tuple[int, float, int]:
    checks = [
        bool(model.get("normality_ok")),
        bool(model.get("homoscedasticity_ok")),
        bool(model.get("independence_ok")),
        bool(model.get("outliers_ok")),
        bool(model.get("linearity_ok")),
    ]
    # Higher diagnostics and adj R² are better; lower degree is preferred as tie-breaker.
    return (sum(checks), float(model.get("adj_r2", -np.inf)), -int(model.get("degree", 99)))


def best_available_model(models: Dict[str, Optional[Dict[str, Any]]], visible_models: List[str]) -> str:
    candidates = [m for name, m in models.items() if name in visible_models and m is not None]
    if not candidates:
        return "N/A"
    return max(candidates, key=model_score)["model"]


def choose_best_model(models: Dict[str, Optional[Dict[str, Any]]], alpha: float, visible_models: Optional[List[str]] = None) -> Tuple[str, str]:
    if visible_models is None:
        visible_models = MODEL_ORDER

    candidates = [m for name, m in models.items() if name in visible_models and m is not None]
    if not candidates:
        return "N/A", "No model could be fitted."

    # Base candidate by diagnostic score and adjusted R².
    best = max(candidates, key=model_score)

    # Complexity policy: only escalate when improvement is material.
    linear = models.get("Linear") if "Linear" in visible_models else None
    quadratic = models.get("Quadratic") if "Quadratic" in visible_models else None
    cubic = models.get("Cubic") if "Cubic" in visible_models else None

    if linear is not None:
        best = linear

    if quadratic is not None and best is not None:
        if (quadratic["adj_r2"] - best["adj_r2"] >= 0.03) or not best.get("linearity_ok", True):
            best = quadratic

    if cubic is not None and best is not None:
        if (cubic["adj_r2"] - best["adj_r2"] >= 0.04) and model_score(cubic)[0] >= model_score(best)[0]:
            best = cubic

    if best is None:
        best = max(candidates, key=model_score)

    if best["model"] == "Linear":
        reason = "Recommended model: Linear. It is the simplest selected model and residual diagnostics do not justify higher-order curvature."
    elif best["model"] == "Quadratic":
        reason = "Recommended model: Quadratic. Curvature or adjusted R² improvement suggests a second-order relationship."
    else:
        reason = "Recommended model: Cubic. The cubic fit provides meaningful improvement and acceptable residual behavior."

    return best["model"], reason


def residual_conclusion(model: Optional[Dict[str, Any]], alpha: float, decimals: int = 4) -> str:
    if not model:
        return "Residual analysis is not available."

    lines = []
    recs = []

    if model.get("linearity_ok"):
        lines.append("• Model form / linearity: PASS. Residuals do not show strong remaining curvature.")
    else:
        lines.append(f"• Model form / linearity: FAIL. Residual curvature is significant (p={fmt(model.get('curvature_p'), decimals)}).")
        recs.append("Evaluate Quadratic or Cubic model; inspect residuals vs fitted.")

    sh_p = model.get("shapiro_p")
    if pd.isna(sh_p):
        lines.append("• Normality: N/A. Not enough valid residuals to evaluate Shapiro-Wilk.")
    elif model.get("normality_ok"):
        lines.append(f"• Normality: PASS. Shapiro-Wilk p={fmt(sh_p, decimals)} > α.")
    else:
        lines.append(f"• Normality: FAIL. Shapiro-Wilk p={fmt(sh_p, decimals)} ≤ α.")
        recs.append("Investigate outliers, skewness, transformations, or a non-normal response behavior.")

    lev_p = model.get("levene_p")
    if pd.isna(lev_p):
        lines.append("• Constant variance: N/A. Residual variance stability could not be evaluated reliably.")
    elif model.get("homoscedasticity_ok"):
        lines.append(f"• Constant variance: PASS. Levene p={fmt(lev_p, decimals)} > α.")
    else:
        lines.append(f"• Constant variance: FAIL. Levene p={fmt(lev_p, decimals)} ≤ α.")
        recs.append("Consider transformation, weighted regression, or segmenting the process.")

    dw = model.get("durbin_watson")
    if pd.isna(dw):
        lines.append("• Independence: N/A. Durbin-Watson could not be evaluated.")
    elif model.get("independence_ok"):
        lines.append(f"• Independence: PASS. Durbin-Watson={fmt(dw, decimals)}.")
    else:
        lines.append(f"• Independence: FAIL. Durbin-Watson={fmt(dw, decimals)}.")
        recs.append("Check observation order, time effects, drift, batching, or autocorrelation.")

    outliers_n = int(model.get("outliers_n", 0))
    if outliers_n == 0:
        lines.append("• Severe outliers: PASS. No standardized residuals beyond ±3 were detected.")
    else:
        lines.append(f"• Severe outliers: FAIL. {outliers_n} standardized residual(s) beyond ±3 detected.")
        recs.append("Investigate special causes, data entry issues, or influential observations.")

    pass_count = sum([
        bool(model.get("linearity_ok")),
        bool(model.get("normality_ok")),
        bool(model.get("homoscedasticity_ok")),
        bool(model.get("independence_ok")),
        bool(model.get("outliers_ok")),
    ])

    if pass_count >= 4:
        lines.append("")
        lines.append("Overall residual conclusion: model assumptions are mostly acceptable.")
    elif pass_count == 3:
        lines.append("")
        lines.append("Overall residual conclusion: model is usable with caution; at least one assumption needs review.")
    else:
        lines.append("")
        lines.append("Overall residual conclusion: model adequacy is questionable; evaluate another model or transformation.")

    if not recs:
        recs.append("Keep the selected model and continue monitoring with new data.")

    return (
        "Residual Assumption Conclusions\n"
        "────────────────────────────────\n"
        + "\n".join(lines)
        + "\n\nRecommendations\n"
        "───────────────\n"
        + "\n".join(f"• {r}" for r in recs)
    )


@dataclass
class RegressionOptions:
    alpha: float = 0.05
    decimals: int = 4
    auto_model: bool = True
    selected_models: List[str] = None
    show_grid: bool = True
    auto_journal_save: bool = True


# ─────────────────────────────────────────────
# Figure builders
# ─────────────────────────────────────────────

def build_relationship_figure(analysis: Dict[str, Any], decimals: int = 4) -> Figure:
    x = np.asarray(analysis["x_values"], dtype=float)
    y = np.asarray(analysis["y_values"], dtype=float)
    x_name = analysis["x_name"]
    y_name = analysis["y_name"]
    alpha = float(analysis["options"]["alpha"])
    show_grid = bool(analysis["options"].get("show_grid", True))
    best_model_name = analysis["best_model"]
    selected_model_name = analysis["selected_model_effective"]
    visible_models = analysis.get("visible_models", MODEL_ORDER)
    model = analysis["models"].get(selected_model_name) or analysis["models"].get(best_model_name)

    fig = Figure(figsize=(13.8, 8.4), dpi=120)
    fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(
        2, 2,
        left=0.07, right=0.96, top=0.84, bottom=0.08,
        wspace=0.28, hspace=0.36,
        height_ratios=[3.0, 1.35],
    )

    title_ax = fig.add_axes([0.04, 0.875, 0.92, 0.065])
    title_ax.axis("off")
    title_ax.set_title(f"Simple Regression — {y_name} vs {x_name}", fontsize=16, fontweight="bold", pad=2)

    ax_scatter = fig.add_subplot(gs[0, 0])
    ax_summary = fig.add_subplot(gs[0, 1])
    ax_table = fig.add_subplot(gs[1, :])

    ax_scatter.scatter(x, y, s=45, alpha=0.80, edgecolors="white", linewidths=0.6, label="Data")

    xs = np.linspace(np.min(x), np.max(x), 300)
    red_styles = {
        "Linear": "-",
        "Quadratic": "--",
        "Cubic": ":",
    }
    plotted_model_names = []
    for model_name in visible_models:
        m = analysis["models"].get(model_name)
        if m is None:
            continue
        ys = np.poly1d(m["coefficients"])(xs)
        ax_scatter.plot(
            xs,
            ys,
            color="#C62828",
            linestyle=red_styles.get(model_name, "-"),
            linewidth=2.4,
            label=f"{model_name} fit",
        )
        plotted_model_names.append(model_name)

    subtitle = "Models: " + (", ".join(plotted_model_names) if plotted_model_names else "N/A")
    ax_scatter.set_title(f"Scatterplot with Fitted Model\n{subtitle}", fontsize=11, fontweight="bold")
    ax_scatter.set_xlabel(x_name)
    ax_scatter.set_ylabel(y_name)
    ax_scatter.grid(show_grid, alpha=0.18)
    ax_scatter.legend(fontsize=8, frameon=False)

    r, p = analysis["pearson_r"], analysis["pearson_p"]
    r2 = model["r2"] if model else np.nan
    adj = model["adj_r2"] if model else np.nan
    sig = significance_label(p, alpha)
    rel = correlation_label(r)
    fit_label = r2_label(r2)

    ax_summary.axis("off")
    ax_summary.set_title("Relationship Summary", fontsize=11, fontweight="bold", pad=8)

    summary_rows = [
        ["Pearson r", fmt(r, decimals), f"{rel} relationship"],
        ["p-value", fmt(p, decimals), sig],
        ["Selected model", selected_model_name, "Used for residual analysis"],
        ["R²", fmt(r2, decimals), f"Fit quality: {fit_label}"],
        ["Adjusted R²", fmt(adj, decimals), "Complexity-adjusted fit"],
    ]
    tbl_summary = ax_summary.table(
        cellText=summary_rows,
        colLabels=["Factor", "Result", "Conclusion"],
        loc="center",
        cellLoc="center",
        colWidths=[0.32, 0.24, 0.44],
    )
    tbl_summary.auto_set_font_size(False)
    tbl_summary.set_fontsize(8.4)
    tbl_summary.scale(1.0, 1.65)
    for c in range(3):
        tbl_summary[0, c].set_facecolor("#17324D")
        tbl_summary[0, c].set_text_props(color="white", fontweight="bold")
    for r_idx in range(1, len(summary_rows) + 1):
        for c in range(3):
            tbl_summary[r_idx, c].set_facecolor("#FFFFFF" if r_idx % 2 else "#F6F9FC")
            if c == 2:
                tbl_summary[r_idx, c].set_text_props(fontweight="bold")

    ax_table.axis("off")
    rows = []
    for model_name in visible_models:
        m = analysis["models"].get(model_name)
        if not m:
            rows.append([model_name, "N/A", "N/A", "N/A", "N/A", "N/A"])
        else:
            rows.append([
                model_name,
                fmt(m["r2"], decimals),
                fmt(m["adj_r2"], decimals),
                fmt(m["p_model"], decimals),
                fmt(m["rmse"], decimals),
                "PASS" if m["linearity_ok"] and m["outliers_ok"] else "Review",
            ])

    tbl = ax_table.table(
        cellText=rows,
        colLabels=["Model", "R²", "Adj R²", "Model p", "RMSE", "Residual Check"],
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1.0, 1.6)
    for c in range(6):
        tbl[0, c].set_facecolor("#17324D")
        tbl[0, c].set_text_props(color="white", fontweight="bold")

    return fig


def build_residual_figure(analysis: Dict[str, Any], decimals: int = 4) -> Figure:
    selected_model_name = analysis["selected_model_effective"]
    model = analysis["models"].get(selected_model_name)
    show_grid = bool(analysis["options"].get("show_grid", True))

    fig = Figure(figsize=(13.5, 8.5), dpi=120)
    fig.patch.set_facecolor("white")

    title_ax = fig.add_axes([0.04, 0.885, 0.92, 0.07])
    title_ax.axis("off")
    title_ax.set_title(f"Residual Analysis — {selected_model_name} Model", fontsize=16, fontweight="bold", pad=2)

    gs = fig.add_gridspec(2, 3, left=0.07, right=0.96, top=0.84, bottom=0.08, wspace=0.32, hspace=0.38)
    ax_rf = fig.add_subplot(gs[0, 0])
    ax_hist = fig.add_subplot(gs[0, 1])
    ax_qq = fig.add_subplot(gs[0, 2])
    ax_order = fig.add_subplot(gs[1, 0])
    ax_table = fig.add_subplot(gs[1, 1:])
    ax_table.axis("off")

    if not model:
        ax_table.text(0.5, 0.5, "Residual analysis is not available.", ha="center", va="center")
        return fig

    fitted = np.asarray(model["fitted"], dtype=float)
    resid = np.asarray(model["residuals"], dtype=float)
    order = np.arange(1, len(resid) + 1)

    ax_rf.scatter(fitted, resid, s=42, alpha=0.80, edgecolors="white", linewidths=0.6)
    ax_rf.axhline(0, linestyle="--", linewidth=1.2)
    ax_rf.set_title("Residuals vs Fitted", fontsize=10, fontweight="bold")
    ax_rf.set_xlabel("Fitted values")
    ax_rf.set_ylabel("Residuals")
    ax_rf.grid(show_grid, alpha=0.18)

    ax_hist.hist(resid, bins=min(14, max(6, len(resid) // 3)), alpha=0.75, edgecolor="white")
    ax_hist.axvline(np.mean(resid), linestyle="--", linewidth=1.2)
    ax_hist.set_title("Histogram of Residuals", fontsize=10, fontweight="bold")
    ax_hist.set_xlabel("Residual")
    ax_hist.set_ylabel("Frequency")
    ax_hist.grid(show_grid, alpha=0.18)

    probplot(resid, dist="norm", plot=ax_qq)
    ax_qq.set_title("Normal Q-Q Plot", fontsize=10, fontweight="bold")
    ax_qq.grid(show_grid, alpha=0.18)

    ax_order.plot(order, resid, marker="o", linewidth=0.9, alpha=0.80)
    ax_order.axhline(0, linestyle="--", linewidth=1.2)
    ax_order.set_title("Residuals vs Observation Order", fontsize=10, fontweight="bold")
    ax_order.set_xlabel("Observation order")
    ax_order.set_ylabel("Residual")
    ax_order.grid(show_grid, alpha=0.18)

    assumption_rows = [
        ["Model form", "PASS" if model["linearity_ok"] else "FAIL", f"Curvature p={fmt(model.get('curvature_p'), decimals)}"],
        ["Normality", "PASS" if model["normality_ok"] else "FAIL", f"Shapiro p={fmt(model.get('shapiro_p'), decimals)}"],
        ["Constant variance", "PASS" if model["homoscedasticity_ok"] else "FAIL", f"Levene p={fmt(model.get('levene_p'), decimals)}"],
        ["Independence", "PASS" if model["independence_ok"] else "FAIL", f"DW={fmt(model.get('durbin_watson'), decimals)}"],
        ["Severe outliers", "PASS" if model["outliers_ok"] else "FAIL", f"Count={model.get('outliers_n', 0)}"],
    ]

    tbl = ax_table.table(
        cellText=assumption_rows,
        colLabels=["Assumption", "Result", "Evidence"],
        loc="upper center",
        cellLoc="center",
        colWidths=[0.32, 0.18, 0.50],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1.0, 1.55)
    for c in range(3):
        tbl[0, c].set_facecolor("#17324D")
        tbl[0, c].set_text_props(color="white", fontweight="bold")

    ax_table.text(0.02, 0.18, analysis["residual_conclusion"], fontsize=8.5, va="top", wrap=True)

    return fig


# ─────────────────────────────────────────────
# PySide6 Builder
# ─────────────────────────────────────────────

class SimpleRegressionBuilder(QWidget):
    def __init__(self, app_window):
        super().__init__()
        self.app_window = app_window
        self.current_df = pd.DataFrame()
        self.current_analysis: Optional[Dict[str, Any]] = None

        self.relationship_figure: Optional[Figure] = None
        self.relationship_canvas: Optional[FigureCanvas] = None
        self.residual_figure: Optional[Figure] = None
        self.residual_canvas: Optional[FigureCanvas] = None

        self.current_journal_entry_id: Optional[str] = None
        self.interaction_managers: List[Any] = []

        self._build_ui()
        self.refresh_columns()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)

        header = QHBoxLayout()
        title = QLabel("Simple Regression Builder")
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
        self._build_relationship_tab()
        self._build_residual_tab()
        self._build_summary_tab()
        self._build_debug_tab()

    def _build_setup_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(8, 8, 8, 8)

        top = QHBoxLayout()

        variables = QGroupBox("1. Variables")
        grid = QGridLayout(variables)
        self.cmb_y = QComboBox()
        self.cmb_x = QComboBox()
        grid.addWidget(QLabel("Response Y"), 0, 0)
        grid.addWidget(self.cmb_y, 0, 1)
        grid.addWidget(QLabel("Predictor X"), 1, 0)
        grid.addWidget(self.cmb_x, 1, 1)

        options = QGroupBox("2. Options")
        opt = QGridLayout(options)

        self.chk_auto_model = QCheckBox("Auto model selection")
        self.chk_auto_model.setChecked(False)

        self.chk_linear = QCheckBox("Linear")
        self.chk_quadratic = QCheckBox("Quadratic")
        self.chk_cubic = QCheckBox("Cubic")
        self.chk_linear.setChecked(True)
        self.chk_quadratic.setChecked(False)
        self.chk_cubic.setChecked(False)

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

        self.chk_auto_model.stateChanged.connect(self._sync_model_checkboxes)

        opt.addWidget(QLabel("Model selection"), 0, 0)
        opt.addWidget(self.chk_auto_model, 0, 1)
        opt.addWidget(QLabel("Models"), 1, 0)
        models_row = QHBoxLayout()
        models_row.addWidget(self.chk_linear)
        models_row.addWidget(self.chk_quadratic)
        models_row.addWidget(self.chk_cubic)
        models_container = QWidget()
        models_container.setLayout(models_row)
        opt.addWidget(models_container, 1, 1)

        opt.addWidget(QLabel("Alpha"), 2, 0)
        opt.addWidget(self.spn_alpha, 2, 1)
        opt.addWidget(QLabel("Decimals"), 3, 0)
        opt.addWidget(self.spn_decimals, 3, 1)
        opt.addWidget(self.chk_grid, 4, 1)
        opt.addWidget(self.chk_journal, 5, 1)

        top.addWidget(variables, 2)
        top.addWidget(options, 1)
        layout.addLayout(top)

        run_group = QGroupBox("3. Run")
        run_layout = QVBoxLayout(run_group)
        buttons = QHBoxLayout()
        self.btn_refresh = QPushButton("Refresh Columns")
        self.btn_run = QPushButton("Run Simple Regression")
        self.btn_run.setStyleSheet("background:#2F5D87; color:white; font-weight:bold; padding:8px; border-radius:6px;")
        self.btn_refresh.clicked.connect(self.refresh_columns)
        self.btn_run.clicked.connect(self.run_analysis)
        buttons.addWidget(self.btn_refresh)
        buttons.addWidget(self.btn_run)
        run_layout.addLayout(buttons)

        note = QLabel("Auto runs Linear, Quadratic and Cubic models. Disable Auto to choose specific models.")
        note.setWordWrap(True)
        note.setStyleSheet("color:#4F6680;")
        run_layout.addWidget(note)

        layout.addWidget(run_group)
        layout.addStretch()
        self.tabs.addTab(tab, "Setup")

        self._sync_model_checkboxes()

    def _sync_model_checkboxes(self):
        auto = self.chk_auto_model.isChecked()
        for chk in [self.chk_linear, self.chk_quadratic, self.chk_cubic]:
            chk.setEnabled(not auto)
            if auto:
                chk.setChecked(True)

    def _selected_model_names(self) -> List[str]:
        if self.chk_auto_model.isChecked():
            return MODEL_ORDER.copy()

        selected = []
        if self.chk_linear.isChecked():
            selected.append("Linear")
        if self.chk_quadratic.isChecked():
            selected.append("Quadratic")
        if self.chk_cubic.isChecked():
            selected.append("Cubic")
        return selected

    def _build_numeric_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        self.tbl_models = QTableWidget()
        self.tbl_models.setColumnCount(8)
        self.tbl_models.setHorizontalHeaderLabels(["Model", "Equation", "R²", "Adj R²", "RMSE", "F", "p-value", "Recommendation"])

        self.tbl_anova = QTableWidget()
        self.tbl_anova.setColumnCount(5)
        self.tbl_anova.setHorizontalHeaderLabels(["Source", "DF", "SS", "MS", "F"])

        self.txt_numeric = QTextEdit()
        self.txt_numeric.setReadOnly(True)

        layout.addWidget(QLabel("Model Comparison"))
        layout.addWidget(self.tbl_models)
        layout.addWidget(QLabel("ANOVA / Numerical Summary"))
        layout.addWidget(self.tbl_anova)
        layout.addWidget(self.txt_numeric)

        self.tabs.addTab(tab, "Numerical Results")

    def _build_relationship_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        toolbar = QHBoxLayout()
        self.btn_copy_relationship = QPushButton("Copy Graph")
        self.btn_export_relationship = QPushButton("Export PNG")
        self.btn_update_journal_1 = QPushButton("Update Journal Entry")
        self.btn_copy_relationship.clicked.connect(lambda: self.copy_graph("relationship"))
        self.btn_export_relationship.clicked.connect(lambda: self.export_png("relationship"))
        self.btn_update_journal_1.clicked.connect(self.update_journal_entry)
        toolbar.addStretch()
        toolbar.addWidget(self.btn_copy_relationship)
        toolbar.addWidget(self.btn_export_relationship)
        toolbar.addWidget(self.btn_update_journal_1)
        layout.addLayout(toolbar)

        self.relationship_container = QWidget()
        self.relationship_layout = QVBoxLayout(self.relationship_container)
        layout.addWidget(self.relationship_container)

        self.tabs.addTab(tab, "Relationship Graph")

    def _build_residual_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        toolbar = QHBoxLayout()
        self.btn_copy_residual = QPushButton("Copy Graph")
        self.btn_export_residual = QPushButton("Export PNG")
        self.btn_update_journal_2 = QPushButton("Update Journal Entry")
        self.btn_copy_residual.clicked.connect(lambda: self.copy_graph("residual"))
        self.btn_export_residual.clicked.connect(lambda: self.export_png("residual"))
        self.btn_update_journal_2.clicked.connect(self.update_journal_entry)
        toolbar.addStretch()
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

    def log(self, message: str):
        self.txt_debug.append(f"[{now_string()}] {message}")

    def refresh_columns(self):
        self.current_df = self.app_window.get_active_dataframe()
        self.lbl_ws.setText(f"Worksheet: {self.app_window.active_worksheet_name()}")

        cols = []
        if not self.current_df.empty:
            for col in self.current_df.columns:
                series = pd.to_numeric(self.current_df[col], errors="coerce")
                if series.notna().sum() >= 3:
                    cols.append(str(col))

        self.cmb_y.clear()
        self.cmb_x.clear()
        self.cmb_y.addItems(cols)
        self.cmb_x.addItems(cols)

        if len(cols) >= 2:
            self.cmb_y.setCurrentIndex(0)
            self.cmb_x.setCurrentIndex(1)

        self.log(f"Numeric columns refreshed: {cols}")

    def _options(self) -> RegressionOptions:
        return RegressionOptions(
            alpha=float(self.spn_alpha.value()),
            decimals=int(self.spn_decimals.value()),
            auto_model=self.chk_auto_model.isChecked(),
            selected_models=self._selected_model_names(),
            show_grid=self.chk_grid.isChecked(),
            auto_journal_save=self.chk_journal.isChecked(),
        )

    def run_analysis(self):
        try:
            opts = self._options()

            if self.current_df.empty:
                raise ValueError("No active worksheet data found.")

            y_col = self.cmb_y.currentText()
            x_col = self.cmb_x.currentText()

            if not y_col or not x_col:
                raise ValueError("Select both Y and X columns.")
            if y_col == x_col:
                raise ValueError("Y and X must be different columns.")

            visible_models = opts.selected_models or []
            if not visible_models:
                raise ValueError("Select at least one regression model or enable Auto model selection.")

            data = self.current_df[[x_col, y_col]].copy()
            data[x_col] = pd.to_numeric(data[x_col], errors="coerce")
            data[y_col] = pd.to_numeric(data[y_col], errors="coerce")
            data = data.dropna(subset=[x_col, y_col])

            if len(data) < 4:
                raise ValueError("At least 4 valid numeric data points are required.")

            x = data[x_col].to_numpy(dtype=float)
            y = data[y_col].to_numpy(dtype=float)

            r, p = pearson_r_p(x, y)

            models = {
                name: fit_polynomial_model(x, y, MODEL_DEGREE[name], alpha=opts.alpha)
                for name in MODEL_ORDER
            }

            available_visible_models = [name for name in visible_models if models.get(name) is not None]
            if not available_visible_models:
                raise ValueError("The selected regression model(s) could not be fitted with the available data.")

            best_model, best_reason = choose_best_model(models, opts.alpha, visible_models=available_visible_models)
            selected_model_effective = best_model

            selected_model = models.get(selected_model_effective)
            residual_txt = residual_conclusion(selected_model, opts.alpha, opts.decimals) if selected_model else "Residual analysis is not available."

            relationship_conclusion = (
                f"Correlation strength is {correlation_label(r).lower()} "
                f"(r={fmt(r, opts.decimals)}). "
                f"The selected model has {r2_label(selected_model.get('r2') if selected_model else np.nan).lower()} explanatory power "
                f"(R²={fmt(selected_model.get('r2') if selected_model else np.nan, opts.decimals)})."
            )

            analysis = {
                "id": str(uuid.uuid4()),
                "module": "Simple Regression",
                "type": "Simple Regression",
                "worksheet": self.app_window.active_worksheet_name(),
                "x_name": x_col,
                "y_name": y_col,
                "x_values": x,
                "y_values": y,
                "n": int(len(data)),
                "options": asdict(opts),
                "pearson_r": r,
                "pearson_p": p,
                "correlation_strength": correlation_label(r),
                "models": models,
                "visible_models": visible_models,
                "available_visible_models": available_visible_models,
                "best_model": best_model,
                "best_reason": best_reason,
                "selected_model_effective": selected_model_effective,
                "relationship_conclusion": relationship_conclusion,
                "residual_conclusion": residual_txt,
                "created_at": now_string(),
            }

            analysis["summary"] = self._build_summary_text(analysis)

            self.current_analysis = analysis
            self._populate_numeric_results(analysis)
            self._render_figures(analysis)
            self.txt_summary.setPlainText(analysis["summary"])
            self.txt_residual.setPlainText(residual_txt)

            if opts.auto_journal_save:
                self.save_to_journal()

            self.tabs.setCurrentIndex(1)
            self.log("Simple Regression completed successfully.")

        except Exception as exc:
            self.log(f"ERROR: {exc}")
            QMessageBox.critical(self, "Regression Error", str(exc))

    def _populate_numeric_results(self, analysis: Dict[str, Any]):
        d = int(analysis["options"]["decimals"])
        models = analysis["models"]
        best_model = analysis["best_model"]

        rows = []
        for name in analysis.get("visible_models", MODEL_ORDER):
            model = models.get(name)
            if model is None:
                rows.append([name, "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "Not available"])
            else:
                rows.append([
                    name,
                    equation_text(model["coefficients"], d),
                    fmt(model["r2"], d),
                    fmt(model["adj_r2"], d),
                    fmt(model["rmse"], d),
                    fmt(model["f_stat"], d),
                    fmt(model["p_model"], d),
                    "Recommended" if name == best_model else "",
                ])

        self.tbl_models.setRowCount(len(rows))
        for r, row in enumerate(rows):
            for c, value in enumerate(row):
                self.tbl_models.setItem(r, c, QTableWidgetItem(str(value)))
        self.tbl_models.resizeColumnsToContents()

        model = analysis["models"].get(analysis["selected_model_effective"])
        anova_rows = []
        if model:
            anova_rows = [
                ["Regression", model["df_reg"], fmt(model["ss_reg"], d), fmt(model["ms_reg"], d), fmt(model["f_stat"], d)],
                ["Error", model["df_err"], fmt(model["ss_res"], d), fmt(model["ms_err"], d), ""],
                ["Total", model["df_reg"] + model["df_err"], fmt(model["ss_tot"], d), "", ""],
            ]

        self.tbl_anova.setRowCount(len(anova_rows))
        for r, row in enumerate(anova_rows):
            for c, value in enumerate(row):
                self.tbl_anova.setItem(r, c, QTableWidgetItem(str(value)))
        self.tbl_anova.resizeColumnsToContents()

        numeric_txt = [
            "Numerical Summary",
            "─────────────────",
            f"Worksheet: {analysis['worksheet']}",
            f"Response Y: {analysis['y_name']}",
            f"Predictor X: {analysis['x_name']}",
            f"N: {analysis['n']}",
            "",
            f"Pearson r: {fmt(analysis['pearson_r'], d)}",
            f"Pearson p-value: {fmt(analysis['pearson_p'], d)}",
            f"Correlation strength: {analysis['correlation_strength']}",
            "",
            f"Models analyzed: {', '.join(analysis.get('visible_models', MODEL_ORDER))}",
            f"Best model: {analysis['best_model']}",
            analysis["best_reason"],
        ]

        if model:
            numeric_txt += [
                "",
                f"Selected model for residual analysis: {analysis['selected_model_effective']}",
                f"Equation: {equation_text(model['coefficients'], d)}",
                f"R²: {fmt(model['r2'], d)}",
                f"Adjusted R²: {fmt(model['adj_r2'], d)}",
                f"RMSE: {fmt(model['rmse'], d)}",
                f"Model p-value: {fmt(model['p_model'], d)}",
            ]
        self.txt_numeric.setPlainText("\n".join(numeric_txt))

    def _render_figures(self, analysis: Dict[str, Any]):
        self._clear_layout(self.relationship_layout)
        self._clear_layout(self.residual_layout)

        d = int(analysis["options"]["decimals"])

        self.relationship_figure = build_relationship_figure(analysis, d)
        self.relationship_canvas = FigureCanvas(self.relationship_figure)
        self.relationship_layout.addWidget(self.relationship_canvas)
        self.relationship_canvas.draw()

        self.residual_figure = build_residual_figure(analysis, d)
        self.residual_canvas = FigureCanvas(self.residual_figure)
        self.residual_layout.addWidget(self.residual_canvas)
        self.residual_canvas.draw()

        self._enable_interactions()

    def _enable_interactions(self):
        self.interaction_managers.clear()
        if enable_interaction is None:
            return

        for fig, canvas in [(self.relationship_figure, self.relationship_canvas), (self.residual_figure, self.residual_canvas)]:
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
        model = analysis["models"].get(analysis["selected_model_effective"])

        lines = [
            "=" * 88,
            "SIMPLE REGRESSION REPORT — SPS STUDIO",
            "=" * 88,
            f"Worksheet: {analysis['worksheet']}",
            f"Response Y: {analysis['y_name']}",
            f"Predictor X: {analysis['x_name']}",
            f"N: {analysis['n']}",
            "",
            "RELATIONSHIP",
            "-" * 40,
            f"Pearson r: {fmt(analysis['pearson_r'], d)}",
            f"Correlation strength: {analysis['correlation_strength']}",
            f"Pearson p-value: {fmt(analysis['pearson_p'], d)}",
            f"Relationship conclusion: {analysis['relationship_conclusion']}",
            "",
            "MODEL COMPARISON",
            "-" * 40,
        ]

        for name in analysis.get("visible_models", MODEL_ORDER):
            m = analysis["models"].get(name)
            if m:
                lines.append(
                    f"{name:<10} R²={fmt(m['r2'], d)} | Adj R²={fmt(m['adj_r2'], d)} | "
                    f"RMSE={fmt(m['rmse'], d)} | p={fmt(m['p_model'], d)} | {equation_text(m['coefficients'], d)}"
                )
            else:
                lines.append(f"{name:<10} N/A")

        lines += [
            "",
            f"Recommended model: {analysis['best_model']}",
            analysis["best_reason"],
            "",
        ]

        if model:
            lines += [
                f"Selected model for residual analysis: {analysis['selected_model_effective']}",
                f"Selected equation: {equation_text(model['coefficients'], d)}",
                "",
            ]

        lines += [
            "RESIDUAL ANALYSIS",
            "-" * 40,
            analysis["residual_conclusion"],
        ]
        return "\n".join(lines)

    def _figure_by_name(self, graph_type: str) -> Optional[Figure]:
        if graph_type == "relationship":
            return self.relationship_figure
        if graph_type == "residual":
            return self.residual_figure
        return None

    def copy_graph(self, graph_type: str):
        fig = self._figure_by_name(graph_type)
        if fig is None:
            QMessageBox.information(self, "No graph", "Run an analysis first.")
            return
        try:
            raw = figure_to_png_bytes(fig)
            image = QImage()
            if not image.loadFromData(raw, "PNG"):
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

        relationship_b64 = figure_to_png_base64(self.relationship_figure) if self.relationship_figure is not None else ""
        residual_b64 = figure_to_png_base64(self.residual_figure) if self.residual_figure is not None else ""

        name = f"Simple Regression - {self.current_analysis.get('y_name', '')} vs {self.current_analysis.get('x_name', '')}"

        return {
            "id": self.current_journal_entry_id or str(uuid.uuid4()),
            "name": name,
            "type": "Simple Regression",
            "module": "Simple Regression",
            "worksheet": self.current_analysis.get("worksheet", ""),
            "status": self.current_analysis.get("best_model", "N/A"),
            "created_at": self.current_analysis.get("created_at", now_string()),
            "modified_at": now_string(),
            "summary": self.txt_summary.toPlainText(),
            "figure_png_base64": relationship_b64,
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
        self.app_window.statusBar().showMessage("Simple Regression journal entry updated.")

    def load_from_journal_entry(self, entry: Dict[str, Any]):
        self.current_journal_entry_id = entry.get("id")
        payload = entry.get("payload", {})
        self.current_analysis = payload

        self.txt_summary.setPlainText(entry.get("summary", payload.get("summary", "")))
        self._populate_numeric_results(payload)
        self._render_figures(payload)
        self.txt_residual.setPlainText(payload.get("residual_conclusion", ""))
