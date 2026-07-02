"""
modules/anova_analysis.py
SPS Studio — ANOVA Analysis Builder

One-way ANOVA workflow for SPS Studio.
Visible UI text is English.
"""

from __future__ import annotations

import base64
import io
import itertools
import math
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import probplot, shapiro, levene, bartlett, kruskal, f_oneway

from PySide6.QtGui import QImage
from PySide6.QtWidgets import (
    QApplication, QAbstractItemView, QCheckBox, QComboBox, QDoubleSpinBox,
    QFileDialog, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QListWidget,
    QMessageBox, QPushButton, QRadioButton, QSpinBox, QTableWidget,
    QTableWidgetItem, QTabWidget, QTextEdit, QVBoxLayout, QWidget,
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

try:
    from core.plot_interaction_tools import enable_interaction
except Exception:
    enable_interaction = None


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


def clean_numeric(series: pd.Series) -> np.ndarray:
    values = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
    return values[np.isfinite(values)]


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


def yes_no(decision: str) -> str:
    return "Yes" if decision == "Reject H0" else "No"


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


def group_descriptives(groups: Dict[str, np.ndarray]) -> Dict[str, Dict[str, Any]]:
    output = {}
    for name, values in groups.items():
        values = np.asarray(values, dtype=float)
        n = len(values)
        output[name] = {
            "N": int(n),
            "Mean": float(np.mean(values)) if n else np.nan,
            "Std Dev": float(np.std(values, ddof=1)) if n > 1 else np.nan,
            "Variance": float(np.var(values, ddof=1)) if n > 1 else np.nan,
            "Median": float(np.median(values)) if n else np.nan,
            "Min": float(np.min(values)) if n else np.nan,
            "Max": float(np.max(values)) if n else np.nan,
            "Q1": float(np.percentile(values, 25)) if n else np.nan,
            "Q3": float(np.percentile(values, 75)) if n else np.nan,
        }
    return output


def normality_by_group(groups: Dict[str, np.ndarray], alpha: float) -> Dict[str, Any]:
    rows = {}
    for name, values in groups.items():
        stat, p = safe_shapiro(values)
        decision = p_decision(p, alpha)
        normal = bool(pd.notna(p) and p > alpha)
        rows[name] = {
            "test": "Shapiro-Wilk", "statistic": stat, "p_value": p,
            "decision": decision, "normal": normal,
            "conclusion": "Statistically normal? Yes" if normal else "Statistically normal? No",
        }
    all_normal = bool(rows) and all(r.get("normal", False) for r in rows.values())
    return {"groups": rows, "all_normal": all_normal}


def f_test_two_std(g1: np.ndarray, g2: np.ndarray, alpha: float, alternative: str) -> Dict[str, Any]:
    v1 = float(np.var(g1, ddof=1))
    v2 = float(np.var(g2, ddof=1))
    if v1 <= 0 or v2 <= 0:
        raise ValueError("Standard deviation test requires positive variance in both groups.")
    f_stat = v1 / v2
    df1 = len(g1) - 1
    df2 = len(g2) - 1
    if alternative == "group1 > group2":
        p = stats.f.sf(f_stat, df1, df2)
        h0, h1 = "H0: σ1 ≤ σ2", "H1: σ1 > σ2"
        conclusion = f"Statistically greater? {yes_no(p_decision(p, alpha))}"
    elif alternative == "group1 < group2":
        p = stats.f.cdf(f_stat, df1, df2)
        h0, h1 = "H0: σ1 ≥ σ2", "H1: σ1 < σ2"
        conclusion = f"Statistically less? {yes_no(p_decision(p, alpha))}"
    else:
        cdf = stats.f.cdf(f_stat, df1, df2)
        p = min(1.0, 2.0 * min(cdf, 1.0 - cdf))
        h0, h1 = "H0: σ1 = σ2", "H1: σ1 ≠ σ2"
        conclusion = f"Statistically different? {yes_no(p_decision(p, alpha))}"
    return {
        "section": "Standard Deviation", "test": "F Test for Standard Deviations",
        "statistic_name": "F", "statistic": float(f_stat), "p_value": float(p),
        "decision": p_decision(p, alpha), "h0": h0, "h1": h1,
        "conclusion": conclusion, "equal_std": p > alpha,
    }


def std_homogeneity_test(groups: Dict[str, np.ndarray], all_normal: bool, alpha: float, alternative: str) -> Dict[str, Any]:
    names = list(groups.keys())
    values = [groups[n] for n in names]
    if len(names) < 2:
        raise ValueError("At least two groups are required.")
    if len(names) == 2 and alternative in {"group1 > group2", "group1 < group2"} and all_normal:
        out = f_test_two_std(values[0], values[1], alpha, alternative)
        out["group_1"], out["group_2"] = names[0], names[1]
        return out
    if all_normal:
        stat, p = bartlett(*values)
        method = "Bartlett"
    else:
        stat, p = levene(*values, center="median")
        method = "Brown-Forsythe / Levene"
    decision = p_decision(p, alpha)
    return {
        "section": "Standard Deviation", "test": method,
        "statistic_name": "χ²" if method == "Bartlett" else "W",
        "statistic": float(stat), "p_value": float(p), "decision": decision,
        "h0": "H0: All standard deviations are equal.",
        "h1": "H1: At least one standard deviation is different.",
        "conclusion": f"Standard deviations different? {yes_no(decision)}",
        "equal_std": p > alpha,
    }


def welch_anova(groups: Dict[str, np.ndarray]) -> Dict[str, Any]:
    values = [np.asarray(v, dtype=float) for v in groups.values()]
    k = len(values)
    n_i = np.array([len(v) for v in values], dtype=float)
    means = np.array([np.mean(v) for v in values], dtype=float)
    variances = np.array([np.var(v, ddof=1) for v in values], dtype=float)
    if np.any(n_i < 2) or np.any(variances <= 0):
        return {"statistic": np.nan, "p_value": np.nan, "df1": np.nan, "df2": np.nan}
    w_i = n_i / variances
    w_sum = np.sum(w_i)
    weighted_mean = np.sum(w_i * means) / w_sum
    numerator = np.sum(w_i * (means - weighted_mean) ** 2) / (k - 1)
    correction = 1 + (2 * (k - 2) / (k**2 - 1)) * np.sum((1 / (n_i - 1)) * (1 - w_i / w_sum) ** 2)
    f_stat = numerator / correction
    df1 = k - 1
    df2 = (k**2 - 1) / (3 * np.sum((1 / (n_i - 1)) * (1 - w_i / w_sum) ** 2))
    p = stats.f.sf(f_stat, df1, df2)
    return {"statistic": float(f_stat), "p_value": float(p), "df1": float(df1), "df2": float(df2)}


def means_test(groups: Dict[str, np.ndarray], all_normal: bool, equal_std: bool, alpha: float, alternative: str) -> Dict[str, Any]:
    names = list(groups.keys())
    values = [groups[n] for n in names]
    k = len(values)
    if k < 2:
        raise ValueError("At least two groups are required.")
    if k == 2 and alternative in {"group1 > group2", "group1 < group2"}:
        alt = "greater" if alternative == "group1 > group2" else "less"
        if all_normal:
            stat, p = stats.ttest_ind(values[0], values[1], equal_var=equal_std, alternative=alt)
            method = "2-Sample t (Pooled)" if equal_std else "Welch 2-Sample t"
            stat_name = "t"
        else:
            stat, p = stats.mannwhitneyu(values[0], values[1], alternative=alt, method="auto")
            method = "Mann-Whitney U"
            stat_name = "U"
        decision = p_decision(p, alpha)
        if alternative == "group1 > group2":
            conclusion, h0, h1 = f"Group 1 mean greater? {yes_no(decision)}", "H0: μ1 ≤ μ2", "H1: μ1 > μ2"
        else:
            conclusion, h0, h1 = f"Group 1 mean less? {yes_no(decision)}", "H0: μ1 ≥ μ2", "H1: μ1 < μ2"
        return {"section": "Means", "test": method, "statistic_name": stat_name, "statistic": float(stat), "p_value": float(p), "decision": decision, "h0": h0, "h1": h1, "conclusion": conclusion}
    if all_normal and equal_std:
        stat, p = f_oneway(*values)
        method, stat_name = "One-Way ANOVA", "F"
    elif all_normal and not equal_std:
        w = welch_anova(groups)
        stat, p = w["statistic"], w["p_value"]
        method, stat_name = "Welch ANOVA", "F"
    else:
        stat, p = kruskal(*values)
        method, stat_name = "Kruskal-Wallis", "H"
    decision = p_decision(p, alpha)
    return {
        "section": "Means", "test": method, "statistic_name": stat_name,
        "statistic": float(stat), "p_value": float(p), "decision": decision,
        "h0": "H0: All group means are equal.",
        "h1": "H1: At least one group mean is different.",
        "conclusion": f"At least one mean different? {yes_no(decision)}",
    }



def classic_anova_table(groups: Dict[str, np.ndarray]) -> Dict[str, Any]:
    names = list(groups.keys())
    values = [np.asarray(groups[n], dtype=float) for n in names]
    all_values = np.concatenate(values)
    grand_mean = float(np.mean(all_values))
    k = len(values)
    n_total = len(all_values)

    ss_factor = float(sum(len(v) * (float(np.mean(v)) - grand_mean) ** 2 for v in values))
    ss_error = float(sum(np.sum((v - float(np.mean(v))) ** 2) for v in values))
    ss_total = float(np.sum((all_values - grand_mean) ** 2))

    df_factor = k - 1
    df_error = n_total - k
    df_total = n_total - 1

    ms_factor = ss_factor / df_factor if df_factor > 0 else np.nan
    ms_error = ss_error / df_error if df_error > 0 else np.nan
    f_value = ms_factor / ms_error if ms_error and ms_error > 0 else np.nan
    p_value = float(stats.f.sf(f_value, df_factor, df_error)) if pd.notna(f_value) else np.nan

    s = math.sqrt(ms_error) if pd.notna(ms_error) and ms_error >= 0 else np.nan
    r_sq = 100.0 * ss_factor / ss_total if ss_total > 0 else np.nan
    r_sq_adj = 100.0 * (1.0 - (ms_error / (ss_total / df_total))) if ss_total > 0 and df_total > 0 and pd.notna(ms_error) else np.nan

    press = 0.0
    valid_press = True
    for v in values:
        if len(v) <= 1:
            valid_press = False
            break
        for i, obs in enumerate(v):
            loo = np.delete(v, i)
            pred = float(np.mean(loo))
            press += float((obs - pred) ** 2)

    r_sq_pred = 100.0 * (1.0 - press / ss_total) if valid_press and ss_total > 0 else np.nan

    return {
        "source_rows": [
            {"source": "Factor", "df": df_factor, "adj_ss": ss_factor, "adj_ms": ms_factor, "f_value": f_value, "p_value": p_value},
            {"source": "Error", "df": df_error, "adj_ss": ss_error, "adj_ms": ms_error, "f_value": np.nan, "p_value": np.nan},
            {"source": "Total", "df": df_total, "adj_ss": ss_total, "adj_ms": np.nan, "f_value": np.nan, "p_value": np.nan},
        ],
        "model_summary": {
            "S": s,
            "R-sq": r_sq,
            "R-sq(adj)": r_sq_adj,
            "R-sq(pred)": r_sq_pred,
        },
    }


def pairwise_std_tests(groups: Dict[str, np.ndarray], alpha: float, all_normal: bool) -> List[Dict[str, Any]]:
    rows = []
    names = list(groups.keys())

    for a, b in itertools.combinations(names, 2):
        x, y = np.asarray(groups[a], dtype=float), np.asarray(groups[b], dtype=float)
        try:
            if all_normal and len(x) > 1 and len(y) > 1 and np.var(x, ddof=1) > 0 and np.var(y, ddof=1) > 0:
                vx, vy = np.var(x, ddof=1), np.var(y, ddof=1)
                f_stat = vx / vy
                cdf = stats.f.cdf(f_stat, len(x) - 1, len(y) - 1)
                p = min(1.0, 2.0 * min(cdf, 1.0 - cdf))
                test = "F Test"
                stat_name = "F"
                stat = f_stat
            else:
                stat, p = levene(x, y, center="median")
                test = "Brown-Forsythe"
                stat_name = "W"
            rows.append({
                "group_a": a,
                "group_b": b,
                "test": test,
                "statistic_name": stat_name,
                "statistic": float(stat),
                "p_value": float(p),
                "adjusted_alpha": float(alpha),
                "decision": "Different" if p <= alpha else "Not different",
                "std_a": float(np.std(x, ddof=1)),
                "std_b": float(np.std(y, ddof=1)),
            })
        except Exception:
            rows.append({
                "group_a": a, "group_b": b, "test": "N/A", "statistic_name": "",
                "statistic": np.nan, "p_value": np.nan, "adjusted_alpha": float(alpha),
                "decision": "N/A", "std_a": np.nan, "std_b": np.nan,
            })
    return rows


def grouped_difference_summary(
    pairwise_rows: List[Dict[str, Any]],
    metric_label: str,
    group_order: Optional[List[str]] = None,
    group_codes: Optional[Dict[str, str]] = None,
) -> List[str]:
    """Return compact Minitab-style difference summary using short sample codes.

    Example:
    Which means differ?
    #   Sample      Differs from
    1   Variable_a  5
    2   Variable_e
    ...
    """
    if group_order is None:
        group_order = sorted(
            set([str(r.get("group_a", "")) for r in pairwise_rows]
                + [str(r.get("group_b", "")) for r in pairwise_rows])
        )
    if group_codes is None:
        group_codes = {g: str(i + 1) for i, g in enumerate(group_order)}

    title = "Which standard deviations differ?" if metric_label == "std dev" else "Which means differ?"
    lines = [title, "#   Sample                 Differs from"]

    for g in group_order:
        different_codes = []
        for r in pairwise_rows:
            if r.get("decision") != "Different":
                continue
            ga = str(r.get("group_a"))
            gb = str(r.get("group_b"))
            if ga == g:
                different_codes.append(group_codes.get(gb, gb))
            elif gb == g:
                different_codes.append(group_codes.get(ga, ga))

        sample_name = str(g)
        if len(sample_name) > 22:
            sample_name = sample_name[:19] + "..."
        differs = ", ".join(different_codes) if different_codes else ""
        lines.append(f"{group_codes.get(g, g):<3} {sample_name:<22} {differs}")

    return lines

def pairwise_tests(groups: Dict[str, np.ndarray], alpha: float, all_normal: bool, equal_std: bool) -> List[Dict[str, Any]]:
    rows = []
    names = list(groups.keys())
    for a, b in itertools.combinations(names, 2):
        x, y = groups[a], groups[b]
        try:
            if all_normal:
                stat, p = stats.ttest_ind(x, y, equal_var=equal_std)
                test = "t-test" if equal_std else "Welch t-test"
                stat_name = "t"
            else:
                stat, p = stats.mannwhitneyu(x, y, alternative="two-sided", method="auto")
                test, stat_name = "Mann-Whitney U", "U"
            rows.append({"group_a": a, "group_b": b, "test": test, "statistic_name": stat_name, "statistic": float(stat), "p_value": float(p), "adjusted_alpha": float(alpha), "decision": "Different" if p <= alpha else "Not different", "mean_diff": float(np.mean(x) - np.mean(y))})
        except Exception:
            rows.append({"group_a": a, "group_b": b, "test": "N/A", "statistic_name": "", "statistic": np.nan, "p_value": np.nan, "adjusted_alpha": float(alpha), "decision": "N/A", "mean_diff": np.nan})
    return rows


def residuals_from_group_means(groups: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    residuals, fitted, order, labels = [], [], [], []
    idx = 1
    for name, values in groups.items():
        mean = float(np.mean(values))
        for v in values:
            fitted.append(mean); residuals.append(float(v - mean)); order.append(idx); labels.append(name); idx += 1
    return np.array(residuals), np.array(fitted), np.array(order), labels


def durbin_watson(residuals: np.ndarray) -> float:
    if len(residuals) < 2:
        return np.nan
    denom = np.sum(residuals**2)
    return float(np.sum(np.diff(residuals)**2) / denom) if denom else np.nan


def residual_analysis(groups: Dict[str, np.ndarray], alpha: float) -> Dict[str, Any]:
    resid, fitted, order, labels = residuals_from_group_means(groups)
    sh_stat, sh_p = safe_shapiro(resid)
    try:
        lev_stat, lev_p = levene(*list(groups.values()), center="median") if len(groups) >= 2 else (np.nan, np.nan)
    except Exception:
        lev_stat, lev_p = np.nan, np.nan
    dw = durbin_watson(resid)
    std = np.std(resid, ddof=1) if len(resid) > 1 else np.nan
    std_resid = resid / std if std and std > 0 else np.full(resid.shape, np.nan)
    outliers = int(np.sum(np.abs(std_resid) > 3)) if len(std_resid) else 0
    return {"residuals": resid, "fitted": fitted, "order": order, "labels": labels, "shapiro_stat": sh_stat, "shapiro_p": sh_p, "normality_ok": pd.notna(sh_p) and sh_p > alpha, "levene_stat": float(lev_stat) if pd.notna(lev_stat) else np.nan, "levene_p": float(lev_p) if pd.notna(lev_p) else np.nan, "constant_variance_ok": pd.notna(lev_p) and lev_p > alpha, "durbin_watson": dw, "independence_ok": pd.notna(dw) and 1.5 <= dw <= 2.5, "outliers_n": outliers, "outliers_ok": outliers == 0}


def residual_conclusion(res: Dict[str, Any], decimals: int) -> str:
    lines, recs = [], []
    if res.get("normality_ok"):
        lines.append(f"• Residual normality: PASS. Shapiro-Wilk p={fmt(res.get('shapiro_p'), decimals)}.")
    else:
        lines.append(f"• Residual normality: REVIEW. Shapiro-Wilk p={fmt(res.get('shapiro_p'), decimals)}.")
        recs.append("Review Q-Q plot, outliers, skewed groups, or consider Kruskal-Wallis if normality is not acceptable.")
    if res.get("constant_variance_ok"):
        lines.append(f"• Constant variance: PASS. Levene p={fmt(res.get('levene_p'), decimals)}.")
    else:
        lines.append(f"• Constant variance: REVIEW. Levene p={fmt(res.get('levene_p'), decimals)}.")
        recs.append("Use Welch ANOVA when standard deviations are not equal.")
    if res.get("independence_ok"):
        lines.append(f"• Independence: PASS. Durbin-Watson={fmt(res.get('durbin_watson'), decimals)}.")
    else:
        lines.append(f"• Independence: REVIEW. Durbin-Watson={fmt(res.get('durbin_watson'), decimals)}.")
        recs.append("Check run order, time drift, batching, or repeated-measure structure.")
    if res.get("outliers_ok"):
        lines.append("• Severe outliers: PASS. No standardized residuals beyond ±3 were detected.")
    else:
        lines.append(f"• Severe outliers: REVIEW. {res.get('outliers_n', 0)} severe residual outlier(s) detected.")
        recs.append("Investigate special causes or data entry issues before final inference.")
    if not recs:
        recs.append("Assumptions are acceptable for the selected ANOVA workflow.")
    return "Residual Assumption Conclusions\n────────────────────────────────\n" + "\n".join(lines) + "\n\nRecommendations\n───────────────\n" + "\n".join(f"• {r}" for r in recs)


@dataclass
class AnovaOptions:
    alpha_normality: float = 0.05
    alpha_std: float = 0.05
    alpha_means: float = 0.05
    decimals: int = 4
    data_mode: str = "Long"
    run_normality: bool = True
    run_std: bool = True
    run_means: bool = True
    show_grid: bool = True
    auto_journal_save: bool = True


def draw_text_panel(ax, title: str, lines: List[str], header_color: str = "#17324D"):
    ax.axis("off")
    ax.set_title(title, fontsize=10.5, fontweight="bold", pad=8)
    y = 0.92
    line_h = 0.075 if len(lines) <= 9 else 0.06
    for i, line in enumerate(lines[:14]):
        is_header = str(line).endswith(":")
        ax.text(
            0.02,
            y,
            str(line),
            fontsize=8.0 if not is_header else 8.4,
            fontweight="bold" if is_header else "normal",
            color=header_color if is_header else "#111111",
            va="top",
            wrap=True,
        )
        y -= line_h
        if y < 0.03:
            break


def normality_summary_lines(analysis: Dict[str, Any], decimals: int) -> List[str]:
    rows = []
    normality = analysis.get("normality", {}).get("groups", {})
    if not normality:
        return ["Normality tests not selected."]

    group_order = list(analysis.get("groups", {}).keys())
    group_codes = analysis.get("group_codes") or {g: str(i + 1) for i, g in enumerate(group_order)}

    alpha_n = float(analysis.get("options", {}).get("alpha_normality", analysis.get("options", {}).get("alpha", 0.05)))
    all_normal = analysis.get("normality", {}).get("all_normal", False)
    rows.append(f"Overall: {'All groups normal' if all_normal else 'Review normality'}")
    rows.append(f"Alpha: {fmt(alpha_n, decimals)}")
    rows.append("")
    rows.append("#   Sample                 Result")
    for name in group_order:
        row = normality.get(name, {})
        p = row.get("p_value")
        conclusion = "Normal" if row.get("normal") else "Not normal"
        sample_name = str(name)
        if len(sample_name) > 22:
            sample_name = sample_name[:19] + "..."
        rows.append(f"{group_codes.get(name, name):<3} {sample_name:<22} p={fmt(p, decimals)} → {conclusion}")
    return rows

def std_summary_lines(analysis: Dict[str, Any], decimals: int) -> List[str]:
    std_result = analysis.get("std_result")
    pairwise_std = analysis.get("pairwise_std", [])
    group_order = list(analysis.get("groups", {}).keys())
    group_codes = analysis.get("group_codes") or {g: str(i + 1) for i, g in enumerate(group_order)}

    lines = []
    alpha_s = float(analysis.get("options", {}).get("alpha_std", analysis.get("options", {}).get("alpha", 0.05)))
    if std_result:
        lines.append(f"Omnibus: p={fmt(std_result.get('p_value'), decimals)} | α={fmt(alpha_s, decimals)}")
        lines.append(std_result.get("conclusion", ""))
        lines.append("")
    else:
        lines.append("Std Dev test not selected.")

    if pairwise_std:
        lines.extend(grouped_difference_summary(pairwise_std, "std dev", group_order, group_codes))
    return lines


def means_summary_lines(analysis: Dict[str, Any], decimals: int) -> List[str]:
    means_result = analysis.get("means_result")
    pairwise = analysis.get("pairwise", [])
    group_order = list(analysis.get("groups", {}).keys())
    group_codes = analysis.get("group_codes") or {g: str(i + 1) for i, g in enumerate(group_order)}

    lines = []
    alpha_m = float(analysis.get("options", {}).get("alpha_means", analysis.get("options", {}).get("alpha", 0.05)))
    if means_result:
        lines.append(f"Omnibus: p={fmt(means_result.get('p_value'), decimals)} | α={fmt(alpha_m, decimals)}")
        lines.append(means_result.get("conclusion", ""))
        lines.append("")
    else:
        lines.append("Means test not selected.")

    if pairwise:
        lines.extend(grouped_difference_summary(pairwise, "mean", group_order, group_codes))
    return lines


# ─────────────────────────────────────────────
# Figure builders
# ─────────────────────────────────────────────

def draw_decision_table(ax, rows: List[Dict[str, Any]]):
    ax.axis("off")
    ax.set_title("Decision Summary", fontsize=11, fontweight="bold", pad=8)
    col_labels = ["Step", "p-value", "H0", "Conclusion"]
    table_rows = [[r.get("step", ""), r.get("result", ""), r.get("h0", ""), r.get("conclusion", "")] for r in rows]
    if not table_rows:
        table_rows = [["No test selected", "", "", ""]]
    tbl = ax.table(cellText=table_rows, colLabels=col_labels, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7.7)
    tbl.scale(1.0, 2.0)
    widths = [0.28, 0.18, 0.18, 0.36]
    for c, w in enumerate(widths):
        for r in range(len(table_rows) + 1):
            tbl[r, c].set_width(w)
    for c in range(len(col_labels)):
        cell = tbl[0, c]
        cell.set_facecolor("#17324D")
        cell.set_text_props(color="white", fontweight="bold")
    for r_idx, row in enumerate(rows, start=1):
        reject = str(row.get("h0", "")).lower().startswith("reject")
        for c in range(len(col_labels)):
            cell = tbl[r_idx, c]
            if c == 2:
                cell.set_facecolor("#FDE2E1" if reject else "#DFF3E5")
                cell.set_text_props(color="#C62828" if reject else "#1A7F37", fontweight="bold")
            else:
                cell.set_facecolor("#FFFFFF" if r_idx % 2 else "#F6F9FC")



def groups_with_differences(pairwise_rows: List[Dict[str, Any]]) -> set:
    out = set()
    for r in pairwise_rows or []:
        if r.get("decision") == "Different":
            out.add(str(r.get("group_a")))
            out.add(str(r.get("group_b")))
    return out


def std_dev_ci(values: np.ndarray, alpha: float = 0.05) -> Tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    n = len(values)
    sd = float(np.std(values, ddof=1)) if n > 1 else np.nan
    if n <= 1 or pd.isna(sd):
        return sd, np.nan, np.nan
    df = n - 1
    var = sd ** 2
    lower_var = df * var / stats.chi2.ppf(1 - alpha / 2, df)
    upper_var = df * var / stats.chi2.ppf(alpha / 2, df)
    return sd, math.sqrt(max(0.0, lower_var)), math.sqrt(max(0.0, upper_var))

def build_anova_summary_figure(analysis: Dict[str, Any], decimals: int = 4) -> Figure:
    groups = {k: np.asarray(v, dtype=float) for k, v in analysis["groups"].items()}
    names = list(groups.keys())
    values = [groups[n] for n in names]
    show_grid = bool(analysis["options"].get("show_grid", True))
    alpha = float(analysis["options"].get("alpha_std", analysis["options"].get("alpha", 0.05)))

    mean_diff_groups = groups_with_differences(analysis.get("pairwise", []))
    std_diff_groups = groups_with_differences(analysis.get("pairwise_std", []))
    mean_colors = ["#C62828" if n in mean_diff_groups else "#17324D" for n in names]
    std_colors = ["#C62828" if n in std_diff_groups else "#17324D" for n in names]

    fig = Figure(figsize=(15.5, 8.8), dpi=120)
    fig.patch.set_facecolor("white")
    title_ax = fig.add_axes([0.04, 0.875, 0.92, 0.055])
    title_ax.axis("off")
    title_ax.set_title("ANOVA Analysis Report", fontsize=16, fontweight="bold", pad=2)

    gs = fig.add_gridspec(
        2, 3,
        left=0.055, right=0.985, top=0.81, bottom=0.08,
        wspace=0.30, hspace=0.42,
        width_ratios=[1.0, 1.0, 1.25],
    )

    ax_box = fig.add_subplot(gs[0, 0])
    ax_std_plot = fig.add_subplot(gs[0, 1])
    ax_means = fig.add_subplot(gs[0, 2])
    ax_norm = fig.add_subplot(gs[1, 0])
    ax_std = fig.add_subplot(gs[1, 1])
    ax_mean_summary = fig.add_subplot(gs[1, 2])

    bp = ax_box.boxplot(values, labels=names, patch_artist=True)
    for patch, name in zip(bp["boxes"], names):
        patch.set_facecolor("#F4B6B3" if name in mean_diff_groups else "#A7C7E7")
        patch.set_edgecolor("#C62828" if name in mean_diff_groups else "#17324D")
    means = [float(np.mean(v)) for v in values]
    ax_box.plot(range(1, len(names) + 1), means, marker="o", linewidth=1.8, color="#17324D", label="Mean")
    ax_box.scatter(range(1, len(names) + 1), means, s=65, c=mean_colors, zorder=5)
    ax_box.set_title("Boxplot by Group", fontsize=10, fontweight="bold")
    ax_box.set_ylabel("Response")
    ax_box.tick_params(axis="x", rotation=25)
    ax_box.legend(fontsize=8, frameon=False)
    ax_box.grid(show_grid, alpha=0.18)

    # Standard deviation comparison plot, replacing Violin Plot.
    # Sort only this chart from lowest standard deviation to highest standard deviation.
    std_records = []
    for name, v in zip(names, values):
        sd, lo, hi = std_dev_ci(v, alpha)
        std_records.append((name, v, sd, lo, hi))
    std_records = sorted(
        std_records,
        key=lambda item: float("inf") if pd.isna(item[2]) else item[2],
    )
    std_names = [r[0] for r in std_records]
    y_pos = np.arange(len(std_records))
    for i, (name, v, sd, lo, hi) in enumerate(std_records):
        if pd.isna(sd):
            continue
        left_err = max(0.0, sd - lo) if pd.notna(lo) else 0.0
        right_err = max(0.0, hi - sd) if pd.notna(hi) else 0.0
        color = "#C62828" if name in std_diff_groups else "#1565C0"
        ax_std_plot.errorbar(
            sd, i,
            xerr=[[left_err], [right_err]],
            fmt="o", color=color, ecolor=color,
            elinewidth=1.4, capsize=4, markersize=5,
        )
    ax_std_plot.set_yticks(y_pos)
    ax_std_plot.set_yticklabels(std_names)
    ax_std_plot.invert_yaxis()
    ax_std_plot.set_xlabel("Confidence Intervals for Std Devs")
    ax_std_plot.set_ylabel("Factor")
    ax_std_plot.set_title("Test for Equal Standard Deviations", fontsize=10, fontweight="bold")
    ax_std_plot.grid(show_grid, alpha=0.18)

    errors = [stats.sem(v) if len(v) > 1 else 0 for v in values]
    for i, (m, e, name) in enumerate(zip(means, errors, names), start=1):
        color = "#C62828" if name in mean_diff_groups else "#17324D"
        ax_means.errorbar(i, m, yerr=e, marker="o", linewidth=1.8, capsize=5, color=color)
    ax_means.plot(range(1, len(names) + 1), means, linewidth=1.2, color="#777777", alpha=0.65)
    ax_means.set_xticks(range(1, len(names) + 1))
    ax_means.set_xticklabels(names, rotation=25)
    ax_means.set_title("Means Plot ± SE", fontsize=10, fontweight="bold")
    ax_means.set_ylabel("Mean Response")
    ax_means.grid(show_grid, alpha=0.18)

    draw_text_panel(
        ax_norm,
        "Normality Summary",
        normality_summary_lines(analysis, decimals),
        header_color="#17324D",
    )

    draw_text_panel(
        ax_std,
        "Test for Equal Standard Deviations",
        std_summary_lines(analysis, decimals),
        header_color="#17324D",
    )

    draw_text_panel(
        ax_mean_summary,
        "One-way ANOVA",
        means_summary_lines(analysis, decimals),
        header_color="#17324D",
    )

    return fig


def build_residual_figure(analysis: Dict[str, Any], decimals: int = 4) -> Figure:
    res = analysis.get("residual_analysis", {})
    show_grid = bool(analysis["options"].get("show_grid", True))
    residuals = np.asarray(res.get("residuals", []), dtype=float)
    fitted = np.asarray(res.get("fitted", []), dtype=float)
    order = np.asarray(res.get("order", []), dtype=float)
    fig = Figure(figsize=(14.0, 8.5), dpi=120)
    fig.patch.set_facecolor("white")
    title_ax = fig.add_axes([0.04, 0.875, 0.92, 0.055])
    title_ax.axis("off")
    title_ax.set_title("ANOVA Residual Analysis", fontsize=16, fontweight="bold", pad=2)
    gs = fig.add_gridspec(2, 3, left=0.06, right=0.97, top=0.81, bottom=0.08, wspace=0.32, hspace=0.38)
    ax_rf = fig.add_subplot(gs[0, 0])
    ax_hist = fig.add_subplot(gs[0, 1])
    ax_qq = fig.add_subplot(gs[0, 2])
    ax_order = fig.add_subplot(gs[1, 0])
    ax_table = fig.add_subplot(gs[1, 1:])
    ax_table.axis("off")
    if residuals.size == 0:
        ax_table.text(0.5, 0.5, "Residual analysis is not available.", ha="center", va="center")
        return fig
    ax_rf.scatter(fitted, residuals, s=42, alpha=0.80, edgecolors="white", linewidths=0.6)
    ax_rf.axhline(0, linestyle="--", linewidth=1.2)
    ax_rf.set_title("Residuals vs Fitted", fontsize=10, fontweight="bold")
    ax_rf.set_xlabel("Fitted group mean")
    ax_rf.set_ylabel("Residual")
    ax_rf.grid(show_grid, alpha=0.18)
    ax_hist.hist(residuals, bins=min(14, max(6, len(residuals) // 3)), alpha=0.75, edgecolor="white")
    ax_hist.axvline(np.mean(residuals), linestyle="--", linewidth=1.2)
    ax_hist.set_title("Histogram of Residuals", fontsize=10, fontweight="bold")
    ax_hist.set_xlabel("Residual")
    ax_hist.set_ylabel("Frequency")
    ax_hist.grid(show_grid, alpha=0.18)
    probplot(residuals, dist="norm", plot=ax_qq)
    ax_qq.set_title("Normal Q-Q Plot", fontsize=10, fontweight="bold")
    ax_qq.grid(show_grid, alpha=0.18)
    ax_order.plot(order, residuals, marker="o", linewidth=0.9, alpha=0.80)
    ax_order.axhline(0, linestyle="--", linewidth=1.2)
    ax_order.set_title("Residuals vs Observation Order", fontsize=10, fontweight="bold")
    ax_order.set_xlabel("Observation order")
    ax_order.set_ylabel("Residual")
    ax_order.grid(show_grid, alpha=0.18)
    rows = [
        ["Residual normality", "PASS" if res.get("normality_ok") else "REVIEW", f"Shapiro p={fmt(res.get('shapiro_p'), decimals)}"],
        ["Constant variance", "PASS" if res.get("constant_variance_ok") else "REVIEW", f"Levene p={fmt(res.get('levene_p'), decimals)}"],
        ["Independence", "PASS" if res.get("independence_ok") else "REVIEW", f"DW={fmt(res.get('durbin_watson'), decimals)}"],
        ["Severe outliers", "PASS" if res.get("outliers_ok") else "REVIEW", f"Count={res.get('outliers_n', 0)}"],
    ]
    tbl = ax_table.table(cellText=rows, colLabels=["Assumption", "Result", "Evidence"], loc="upper center", cellLoc="center", colWidths=[0.32, 0.18, 0.50])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1.0, 1.55)
    for c in range(3):
        tbl[0, c].set_facecolor("#17324D")
        tbl[0, c].set_text_props(color="white", fontweight="bold")
    ax_table.text(0.02, 0.18, analysis.get("residual_conclusion", ""), fontsize=8.5, va="top", wrap=True)
    return fig

# ─────────────────────────────────────────────
# Main widget
# ─────────────────────────────────────────────

class AnovaAnalysisBuilder(QWidget):
    def __init__(self, app_window):
        super().__init__()
        self.app_window = app_window
        self.current_df = pd.DataFrame()
        self.current_analysis: Optional[Dict[str, Any]] = None
        self.summary_figure: Optional[Figure] = None
        self.summary_canvas: Optional[FigureCanvas] = None
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
        title = QLabel("ANOVA Analysis Builder")
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
        self._build_graph_tab()
        self._build_residual_tab()
        self._build_summary_tab()
        self._build_debug_tab()

    def _build_setup_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(8, 8, 8, 8)
        top = QHBoxLayout()
        data_box = QGroupBox("1. Data Structure")
        data_layout = QVBoxLayout(data_box)
        self.rad_long = QRadioButton("Long format: Response column + Factor column")
        self.rad_wide = QRadioButton("Wide format: Multiple response columns; each column is a group")
        self.rad_long.setChecked(True)
        self.rad_long.toggled.connect(self._sync_data_mode)
        self.rad_wide.toggled.connect(self._sync_data_mode)
        data_layout.addWidget(self.rad_long)
        data_layout.addWidget(self.rad_wide)
        long_grid = QGridLayout()
        self.cmb_response = QComboBox()
        self.cmb_factor = QComboBox()
        long_grid.addWidget(QLabel("Response Y"), 0, 0)
        long_grid.addWidget(self.cmb_response, 0, 1)
        long_grid.addWidget(QLabel("Factor"), 1, 0)
        long_grid.addWidget(self.cmb_factor, 1, 1)
        data_layout.addLayout(long_grid)
        data_layout.addWidget(QLabel("Wide format columns"))
        self.lst_group_columns = QListWidget()
        self.lst_group_columns.setSelectionMode(QAbstractItemView.MultiSelection)
        self.lst_group_columns.setMinimumHeight(150)
        data_layout.addWidget(self.lst_group_columns)
        opt_box = QGroupBox("2. Analysis Options")
        opt = QGridLayout(opt_box)
        self.spn_alpha_normality = QDoubleSpinBox(); self.spn_alpha_normality.setRange(0.001, 0.999); self.spn_alpha_normality.setDecimals(3); self.spn_alpha_normality.setSingleStep(0.005); self.spn_alpha_normality.setValue(0.05)
        self.spn_alpha_std = QDoubleSpinBox(); self.spn_alpha_std.setRange(0.001, 0.999); self.spn_alpha_std.setDecimals(3); self.spn_alpha_std.setSingleStep(0.005); self.spn_alpha_std.setValue(0.05)
        self.spn_alpha_means = QDoubleSpinBox(); self.spn_alpha_means.setRange(0.001, 0.999); self.spn_alpha_means.setDecimals(3); self.spn_alpha_means.setSingleStep(0.005); self.spn_alpha_means.setValue(0.05)
        self.spn_decimals = QSpinBox(); self.spn_decimals.setRange(0, 8); self.spn_decimals.setValue(4)
        self.chk_normality = QCheckBox("Run normality tests")
        self.chk_std = QCheckBox("Run standard deviation test")
        self.chk_means = QCheckBox("Run means test")
        self.chk_grid = QCheckBox("Show grid")
        self.chk_journal = QCheckBox("Auto journal save")
        for chk in [self.chk_normality, self.chk_std, self.chk_means, self.chk_grid, self.chk_journal]: chk.setChecked(True)
        opt.addWidget(QLabel("Normality alpha"), 0, 0); opt.addWidget(self.spn_alpha_normality, 0, 1)
        opt.addWidget(QLabel("Std Dev alpha"), 1, 0); opt.addWidget(self.spn_alpha_std, 1, 1)
        opt.addWidget(QLabel("Means alpha"), 2, 0); opt.addWidget(self.spn_alpha_means, 2, 1)
        opt.addWidget(QLabel("Decimals"), 3, 0); opt.addWidget(self.spn_decimals, 3, 1)
        opt.addWidget(self.chk_normality, 4, 1); opt.addWidget(self.chk_std, 5, 1); opt.addWidget(self.chk_means, 6, 1); opt.addWidget(self.chk_grid, 7, 1); opt.addWidget(self.chk_journal, 8, 1)
        top.addWidget(data_box, 2); top.addWidget(opt_box, 1); layout.addLayout(top)
        run_box = QGroupBox("3. Run")
        run_layout = QVBoxLayout(run_box)
        buttons = QHBoxLayout()
        self.btn_refresh = QPushButton("Refresh Columns")
        self.btn_run = QPushButton("Run ANOVA Analysis")
        self.btn_run.setStyleSheet("background:#2F5D87; color:white; font-weight:bold; padding:8px; border-radius:6px;")
        self.btn_refresh.clicked.connect(self.refresh_columns); self.btn_run.clicked.connect(self.run_analysis)
        buttons.addWidget(self.btn_refresh); buttons.addWidget(self.btn_run); run_layout.addLayout(buttons)
        note = QLabel("Default workflow runs normality, standard deviation and means tests. ANOVA evaluates whether groups are statistically different; directional greater/less alternatives are not used for this method.")
        note.setWordWrap(True); note.setStyleSheet("color:#4F6680;")
        run_layout.addWidget(note)
        layout.addWidget(run_box); layout.addStretch()
        self.tabs.addTab(tab, "Setup")
        self._sync_data_mode()

    def _build_numeric_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        self.tbl_desc = QTableWidget()
        self.tbl_desc.setColumnCount(9)
        self.tbl_desc.setHorizontalHeaderLabels(["Group", "N", "Mean", "Std Dev", "Variance", "Median", "Min", "Max", "IQR"])

        self.tbl_anova = QTableWidget()
        self.tbl_anova.setColumnCount(6)
        self.tbl_anova.setHorizontalHeaderLabels(["Source", "DF", "Adj SS", "Adj MS", "F-Value", "P-Value"])

        self.tbl_model_summary = QTableWidget()
        self.tbl_model_summary.setColumnCount(4)
        self.tbl_model_summary.setHorizontalHeaderLabels(["S", "R-sq", "R-sq(adj)", "R-sq(pred)"])

        self.tbl_tests = QTableWidget()
        self.tbl_tests.setColumnCount(6)
        self.tbl_tests.setHorizontalHeaderLabels(["Step", "Test", "Statistic", "p-value", "H0", "Conclusion"])

        self.tbl_pairwise_std = QTableWidget()
        self.tbl_pairwise_std.setColumnCount(7)
        self.tbl_pairwise_std.setHorizontalHeaderLabels(["Group A", "Group B", "Test", "Statistic", "p-value", "Alpha", "Decision"])

        self.tbl_pairwise = QTableWidget()
        self.tbl_pairwise.setColumnCount(7)
        self.tbl_pairwise.setHorizontalHeaderLabels(["Group A", "Group B", "Test", "Statistic", "p-value", "Alpha", "Decision"])

        layout.addWidget(QLabel("Descriptive Statistics"))
        layout.addWidget(self.tbl_desc)
        layout.addWidget(QLabel("Analysis of Variance"))
        layout.addWidget(self.tbl_anova)
        layout.addWidget(QLabel("Model Summary"))
        layout.addWidget(self.tbl_model_summary)
        layout.addWidget(QLabel("Hypothesis Tests"))
        layout.addWidget(self.tbl_tests)
        layout.addWidget(QLabel("Pairwise Standard Deviation Comparisons"))
        layout.addWidget(self.tbl_pairwise_std)
        layout.addWidget(QLabel("Pairwise Mean Comparisons"))
        layout.addWidget(self.tbl_pairwise)

        self.tabs.addTab(tab, "Numerical Results")

    def _build_graph_tab(self):
        tab = QWidget(); layout = QVBoxLayout(tab)
        toolbar = QHBoxLayout()
        self.btn_copy_graph = QPushButton("Copy Graph"); self.btn_export_graph = QPushButton("Export PNG"); self.btn_update_journal_1 = QPushButton("Update Journal Entry")
        self.btn_copy_graph.clicked.connect(lambda: self.copy_graph("summary")); self.btn_export_graph.clicked.connect(lambda: self.export_png("summary")); self.btn_update_journal_1.clicked.connect(self.update_journal_entry)
        toolbar.addStretch(); toolbar.addWidget(self.btn_copy_graph); toolbar.addWidget(self.btn_export_graph); toolbar.addWidget(self.btn_update_journal_1); layout.addLayout(toolbar)
        self.summary_container = QWidget(); self.summary_layout = QVBoxLayout(self.summary_container); layout.addWidget(self.summary_container)
        self.tabs.addTab(tab, "Graphs")

    def _build_residual_tab(self):
        tab = QWidget(); layout = QVBoxLayout(tab)
        toolbar = QHBoxLayout()
        self.btn_copy_residual = QPushButton("Copy Graph"); self.btn_export_residual = QPushButton("Export PNG"); self.btn_update_journal_2 = QPushButton("Update Journal Entry")
        self.btn_copy_residual.clicked.connect(lambda: self.copy_graph("residual")); self.btn_export_residual.clicked.connect(lambda: self.export_png("residual")); self.btn_update_journal_2.clicked.connect(self.update_journal_entry)
        toolbar.addStretch(); toolbar.addWidget(self.btn_copy_residual); toolbar.addWidget(self.btn_export_residual); toolbar.addWidget(self.btn_update_journal_2); layout.addLayout(toolbar)
        self.residual_container = QWidget(); self.residual_layout = QVBoxLayout(self.residual_container); layout.addWidget(self.residual_container)
        self.txt_residual = QTextEdit(); self.txt_residual.setReadOnly(True); self.txt_residual.setMaximumHeight(180); layout.addWidget(self.txt_residual)
        self.tabs.addTab(tab, "Residual Analysis")

    def _build_summary_tab(self):
        tab = QWidget(); layout = QVBoxLayout(tab); top = QHBoxLayout()
        self.btn_copy_summary = QPushButton("Copy Summary"); self.btn_copy_summary.clicked.connect(lambda: QApplication.clipboard().setText(self.txt_summary.toPlainText()))
        top.addStretch(); top.addWidget(self.btn_copy_summary); layout.addLayout(top)
        self.txt_summary = QTextEdit(); self.txt_summary.setReadOnly(True); layout.addWidget(self.txt_summary)
        self.tabs.addTab(tab, "Summary")

    def _build_debug_tab(self):
        tab = QWidget(); layout = QVBoxLayout(tab); self.txt_debug = QTextEdit(); self.txt_debug.setReadOnly(True); layout.addWidget(self.txt_debug); self.tabs.addTab(tab, "Debug")

    def _sync_data_mode(self):
        wide = self.rad_wide.isChecked(); self.cmb_response.setEnabled(not wide); self.cmb_factor.setEnabled(not wide); self.lst_group_columns.setEnabled(wide)

    def log(self, message: str): self.txt_debug.append(f"[{now_string()}] {message}")

    def refresh_columns(self):
        self.current_df = self.app_window.get_active_dataframe(); self.lbl_ws.setText(f"Worksheet: {self.app_window.active_worksheet_name()}")
        all_cols = list(self.current_df.columns) if not self.current_df.empty else []
        numeric_cols = []
        for col in all_cols:
            s = pd.to_numeric(self.current_df[col], errors="coerce")
            if s.notna().sum() >= 2: numeric_cols.append(str(col))
        self.cmb_response.clear(); self.cmb_factor.clear(); self.lst_group_columns.clear()
        self.cmb_response.addItems(numeric_cols); self.cmb_factor.addItems([str(c) for c in all_cols])
        for col in numeric_cols: self.lst_group_columns.addItem(col)
        for i in range(self.lst_group_columns.count()): self.lst_group_columns.item(i).setSelected(True)
        self.log(f"Columns refreshed. Numeric columns: {numeric_cols}. All columns: {all_cols}")

    def _options(self) -> AnovaOptions:
        return AnovaOptions(
            alpha_normality=float(self.spn_alpha_normality.value()),
            alpha_std=float(self.spn_alpha_std.value()),
            alpha_means=float(self.spn_alpha_means.value()),
            decimals=int(self.spn_decimals.value()),
            data_mode="Wide" if self.rad_wide.isChecked() else "Long",
            run_normality=self.chk_normality.isChecked(),
            run_std=self.chk_std.isChecked(),
            run_means=self.chk_means.isChecked(),
            show_grid=self.chk_grid.isChecked(),
            auto_journal_save=self.chk_journal.isChecked(),
        )

    def _selected_groups(self, opts: AnovaOptions) -> Dict[str, np.ndarray]:
        if self.current_df.empty: raise ValueError("No active worksheet data found.")
        if opts.data_mode == "Long":
            y_col, f_col = self.cmb_response.currentText(), self.cmb_factor.currentText()
            if not y_col or not f_col: raise ValueError("Select response and factor columns.")
            if y_col == f_col: raise ValueError("Response and factor columns must be different.")
            df = self.current_df[[y_col, f_col]].copy(); df[y_col] = pd.to_numeric(df[y_col], errors="coerce"); df[f_col] = df[f_col].astype(str); df = df.dropna(subset=[y_col, f_col])
            groups = {}
            for level, gdf in df.groupby(f_col, dropna=True):
                arr = gdf[y_col].dropna().to_numpy(dtype=float)
                if len(arr) >= 2: groups[str(level)] = arr
            if len(groups) < 2: raise ValueError("At least two factor levels with two or more observations are required.")
            return groups
        selected_cols = [item.text() for item in self.lst_group_columns.selectedItems()]
        if len(selected_cols) < 2: raise ValueError("Select at least two numeric columns for wide format ANOVA.")
        groups = {str(col): clean_numeric(self.current_df[col]) for col in selected_cols}
        groups = {k: v for k, v in groups.items() if len(v) >= 2}
        if len(groups) < 2: raise ValueError("At least two selected columns must have two or more numeric observations.")
        return groups

    def run_analysis(self):
        try:
            opts = self._options()
            if not opts.run_normality and not opts.run_std and not opts.run_means: raise ValueError("Select at least one analysis block: normality, standard deviation, or means.")
            groups = self._selected_groups(opts); d = int(opts.decimals)
            desc = group_descriptives(groups)
            normality = normality_by_group(groups, opts.alpha_normality) if opts.run_normality else {"groups": {}, "all_normal": True}
            all_normal = bool(normality.get("all_normal", True))
            tests, std_result, means_result, pairwise, pairwise_std = [], None, None, [], []
            if opts.run_std:
                std_result = std_homogeneity_test(groups, all_normal, opts.alpha_std, "two-sided")
                tests.append(std_result)
                pairwise_std = pairwise_std_tests(groups, opts.alpha_std, all_normal)
            equal_std = bool(std_result.get("equal_std", True)) if std_result else True
            if opts.run_means:
                means_result = means_test(groups, all_normal, equal_std, opts.alpha_means, "two-sided")
                tests.append(means_result)
                pairwise = pairwise_tests(groups, opts.alpha_means, all_normal, equal_std)

            anova_table = classic_anova_table(groups)
            res = residual_analysis(groups, opts.alpha_normality)
            res_txt = residual_conclusion(res, d)

            analysis = {
                "id": str(uuid.uuid4()),
                "module": "ANOVA Analysis",
                "type": "ANOVA Analysis",
                "worksheet": self.app_window.active_worksheet_name(),
                "options": asdict(opts),
                "groups": groups,
                "group_codes": {g: str(i + 1) for i, g in enumerate(groups.keys())},
                "descriptives": desc,
                "anova_table": anova_table,
                "model_summary": anova_table.get("model_summary", {}),
                "normality": normality,
                "tests": tests,
                "std_result": std_result,
                "means_result": means_result,
                "pairwise_std": pairwise_std,
                "pairwise": pairwise,
                "residual_analysis": res,
                "residual_conclusion": res_txt,
                "decision_rows": self._build_decision_rows(normality, tests, opts),
                "created_at": now_string(),
            }
            analysis["summary"] = self._build_summary_text(analysis)
            self.current_analysis = analysis; self._populate_numeric_results(analysis); self._render_figures(analysis); self.txt_summary.setPlainText(analysis["summary"]); self.txt_residual.setPlainText(res_txt)
            if opts.auto_journal_save: self.save_to_journal()
            self.tabs.setCurrentIndex(1); self.log("ANOVA Analysis completed successfully.")
        except Exception as exc:
            self.log(f"ERROR: {exc}"); QMessageBox.critical(self, "ANOVA Error", str(exc))

    def _build_decision_rows(self, normality, tests, opts):
        rows, d = [], int(opts.decimals)
        for name, row in normality.get("groups", {}).items():
            rows.append({
                "step": f"Normality - {name}",
                "result": f"p={fmt(row.get('p_value'), d)} | α={fmt(opts.alpha_normality, d)}",
                "h0": row.get("decision", ""),
                "conclusion": row.get("conclusion", ""),
            })
        for t in tests:
            section = t.get("section", t.get("test", ""))
            alpha_used = opts.alpha_std if section == "Standard Deviation" else opts.alpha_means if section == "Means" else opts.alpha_normality
            rows.append({
                "step": section,
                "result": f"p={fmt(t.get('p_value'), d)} | α={fmt(alpha_used, d)}",
                "h0": t.get("decision", ""),
                "conclusion": t.get("conclusion", ""),
            })
        return rows

    def _populate_numeric_results(self, analysis):
        d = int(analysis["options"]["decimals"])
        desc = analysis.get("descriptives", {})

        rows = []
        for group, m in desc.items():
            rows.append([
                group,
                m.get("N", ""),
                fmt(m.get("Mean"), d),
                fmt(m.get("Std Dev"), d),
                fmt(m.get("Variance"), d),
                fmt(m.get("Median"), d),
                fmt(m.get("Min"), d),
                fmt(m.get("Max"), d),
                fmt(m.get("Q3", np.nan) - m.get("Q1", np.nan), d),
            ])

        self.tbl_desc.setRowCount(len(rows))
        for r, row in enumerate(rows):
            for c, value in enumerate(row):
                self.tbl_desc.setItem(r, c, QTableWidgetItem(str(value)))
        self.tbl_desc.resizeColumnsToContents()

        anova_rows = analysis.get("anova_table", {}).get("source_rows", [])
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

        ms = analysis.get("model_summary", {})
        self.tbl_model_summary.setRowCount(1 if ms else 0)
        if ms:
            values = [
                fmt(ms.get("S"), d),
                f"{fmt(ms.get('R-sq'), 2)}%",
                f"{fmt(ms.get('R-sq(adj)'), 2)}%",
                f"{fmt(ms.get('R-sq(pred)'), 2)}%",
            ]
            for c, value in enumerate(values):
                self.tbl_model_summary.setItem(0, c, QTableWidgetItem(str(value)))
        self.tbl_model_summary.resizeColumnsToContents()

        test_rows = []
        for n, row in analysis.get("normality", {}).get("groups", {}).items():
            test_rows.append([f"Normality - {n}", row.get("test", ""), fmt(row.get("statistic"), d), fmt(row.get("p_value"), d), row.get("decision", ""), row.get("conclusion", "")])
        for t in analysis.get("tests", []):
            test_rows.append([t.get("section", ""), t.get("test", ""), fmt(t.get("statistic"), d), fmt(t.get("p_value"), d), t.get("decision", ""), t.get("conclusion", "")])
        self.tbl_tests.setRowCount(len(test_rows))
        for r, row in enumerate(test_rows):
            for c, value in enumerate(row):
                self.tbl_tests.setItem(r, c, QTableWidgetItem(str(value)))
        self.tbl_tests.resizeColumnsToContents()

        pair_std_rows = []
        for p in analysis.get("pairwise_std", []):
            pair_std_rows.append([
                p.get("group_a", ""),
                p.get("group_b", ""),
                p.get("test", ""),
                fmt(p.get("statistic"), d),
                fmt(p.get("p_value"), d),
                fmt(p.get("adjusted_alpha"), d),
                p.get("decision", ""),
            ])
        self.tbl_pairwise_std.setRowCount(len(pair_std_rows))
        for r, row in enumerate(pair_std_rows):
            for c, value in enumerate(row):
                self.tbl_pairwise_std.setItem(r, c, QTableWidgetItem(str(value)))
        self.tbl_pairwise_std.resizeColumnsToContents()

        pair_rows = []
        for p in analysis.get("pairwise", []):
            pair_rows.append([
                p.get("group_a", ""),
                p.get("group_b", ""),
                p.get("test", ""),
                fmt(p.get("statistic"), d),
                fmt(p.get("p_value"), d),
                fmt(p.get("adjusted_alpha"), d),
                p.get("decision", ""),
            ])
        self.tbl_pairwise.setRowCount(len(pair_rows))
        for r, row in enumerate(pair_rows):
            for c, value in enumerate(row):
                self.tbl_pairwise.setItem(r, c, QTableWidgetItem(str(value)))
        self.tbl_pairwise.resizeColumnsToContents()

    def _render_figures(self, analysis):
        self._clear_layout(self.summary_layout); self._clear_layout(self.residual_layout); d = int(analysis["options"]["decimals"])
        self.summary_figure = build_anova_summary_figure(analysis, d); self.summary_canvas = FigureCanvas(self.summary_figure); self.summary_layout.addWidget(self.summary_canvas); self.summary_canvas.draw()
        self.residual_figure = build_residual_figure(analysis, d); self.residual_canvas = FigureCanvas(self.residual_figure); self.residual_layout.addWidget(self.residual_canvas); self.residual_canvas.draw()
        self._enable_interactions()

    def _enable_interactions(self):
        self.interaction_managers.clear()
        if enable_interaction is None: return
        for fig, canvas in [(self.summary_figure, self.summary_canvas), (self.residual_figure, self.residual_canvas)]:
            if fig is None or canvas is None: continue
            try:
                manager = enable_interaction(host=self, figure_getter=lambda f=fig: f, canvas_getter=lambda c=canvas: c, decimals_getter=lambda: int(self.spn_decimals.value()), on_status=lambda msg: self.app_window.statusBar().showMessage(msg))
                manager.register_axes(); self.interaction_managers.append(manager)
            except Exception as exc: self.log(f"Plot interaction was not enabled: {exc}")

    def _clear_layout(self, layout):
        while layout.count():
            child = layout.takeAt(0)
            if child.widget(): child.widget().deleteLater()

    def _build_summary_text(self, analysis):
        d = int(analysis["options"]["decimals"])
        lines = [
            "=" * 92,
            "ANOVA ANALYSIS REPORT — SPS STUDIO",
            "=" * 92,
            f"Worksheet: {analysis.get('worksheet', '')}",
            f"Data mode: {analysis.get('options', {}).get('data_mode', '')}",
            f"Normality alpha: {fmt(analysis.get('options', {}).get('alpha_normality', analysis.get('options', {}).get('alpha', 0.05)), d)}",
            f"Std Dev alpha: {fmt(analysis.get('options', {}).get('alpha_std', analysis.get('options', {}).get('alpha', 0.05)), d)}",
            f"Means alpha: {fmt(analysis.get('options', {}).get('alpha_means', analysis.get('options', {}).get('alpha', 0.05)), d)}",
            f"Groups: {', '.join(analysis.get('groups', {}).keys())}",
            f"Sample codes: {', '.join([f'{v}={k}' for k, v in analysis.get('group_codes', {}).items()])}",
            "",
            "DESCRIPTIVE STATISTICS",
            "-" * 92,
        ]

        for group, m in analysis.get("descriptives", {}).items():
            lines.append(
                f"{group:<24} N={m.get('N')} | Mean={fmt(m.get('Mean'), d)} | "
                f"Std Dev={fmt(m.get('Std Dev'), d)} | Median={fmt(m.get('Median'), d)}"
            )

        lines += ["", "ANALYSIS OF VARIANCE", "-" * 92]
        lines.append(f"{'Source':<14}{'DF':<8}{'Adj SS':<14}{'Adj MS':<14}{'F-Value':<14}{'P-Value':<14}")
        for row in analysis.get("anova_table", {}).get("source_rows", []):
            lines.append(
                f"{row.get('source',''):<14}{row.get('df',''):<8}"
                f"{fmt(row.get('adj_ss'), d):<14}{fmt(row.get('adj_ms'), d):<14}"
                f"{fmt(row.get('f_value'), d):<14}{fmt(row.get('p_value'), d):<14}"
            )

        ms = analysis.get("model_summary", {})
        if ms:
            lines += ["", "MODEL SUMMARY", "-" * 92]
            lines.append(f"S={fmt(ms.get('S'), d)} | R-sq={fmt(ms.get('R-sq'), 2)}% | R-sq(adj)={fmt(ms.get('R-sq(adj)'), 2)}% | R-sq(pred)={fmt(ms.get('R-sq(pred)'), 2)}%")

        lines += ["", "STEP-BY-STEP WORKFLOW", "-" * 92]
        for row in analysis.get("decision_rows", []):
            lines.append(f"{row.get('step'):<35} {row.get('result'):<14} {row.get('h0'):<12} {row.get('conclusion')}")

        means = analysis.get("means_result")
        if means:
            lines += [
                "",
                "MEANS TEST CONCLUSION",
                "-" * 92,
                f"Test: {means.get('test')}",
                f"p-value: {fmt(means.get('p_value'), d)}",
                f"Alpha: {fmt(analysis.get('options', {}).get('alpha_means', analysis.get('options', {}).get('alpha', 0.05)), d)}",
                f"Decision: {means.get('decision')}",
                f"Conclusion: {means.get('conclusion')}",
            ]

        if analysis.get("pairwise_std"):
            lines += ["", "PAIRWISE STANDARD DEVIATION COMPARISONS", "-" * 92]
            for p in analysis["pairwise_std"]:
                lines.append(f"{p.get('group_a')} vs {p.get('group_b')}: p={fmt(p.get('p_value'), d)} | adj α={fmt(p.get('adjusted_alpha'), d)} | {p.get('decision')}")

        if analysis.get("pairwise"):
            lines += ["", "PAIRWISE MEAN COMPARISONS", "-" * 92]
            for p in analysis["pairwise"]:
                lines.append(f"{p.get('group_a')} vs {p.get('group_b')}: p={fmt(p.get('p_value'), d)} | adj α={fmt(p.get('adjusted_alpha'), d)} | {p.get('decision')}")

        lines += ["", "RESIDUAL ANALYSIS", "-" * 92, analysis.get("residual_conclusion", "")]
        return "\n".join(lines)

    def _figure_by_name(self, graph_type: str) -> Optional[Figure]: return self.summary_figure if graph_type == "summary" else self.residual_figure if graph_type == "residual" else None

    def copy_graph(self, graph_type: str):
        fig = self._figure_by_name(graph_type)
        if fig is None: QMessageBox.information(self, "No graph", "Run an analysis first."); return
        try:
            image = QImage()
            if not image.loadFromData(figure_to_png_bytes(fig), "PNG"): raise ValueError("Could not convert graph to clipboard image.")
            QApplication.clipboard().setImage(image); self.app_window.statusBar().showMessage("Graph copied to clipboard as image.")
        except Exception as exc: QMessageBox.critical(self, "Copy failed", str(exc))

    def export_png(self, graph_type: str):
        fig = self._figure_by_name(graph_type)
        if fig is None: QMessageBox.information(self, "No graph", "Run an analysis first."); return
        path, _ = QFileDialog.getSaveFileName(self, "Export PNG", "", "PNG Image (*.png)")
        if not path: return
        if not path.lower().endswith(".png"): path += ".png"
        fig.savefig(path, dpi=160, bbox_inches="tight"); self.app_window.statusBar().showMessage(f"Graph exported: {path}")

    def _journal_entry_payload(self) -> Dict[str, Any]:
        if self.current_analysis is None: raise ValueError("No analysis has been run.")
        summary_b64 = figure_to_png_base64(self.summary_figure) if self.summary_figure is not None else ""
        residual_b64 = figure_to_png_base64(self.residual_figure) if self.residual_figure is not None else ""
        return {"id": self.current_journal_entry_id or str(uuid.uuid4()), "name": "ANOVA Analysis", "type": "ANOVA Analysis", "module": "ANOVA Analysis", "worksheet": self.current_analysis.get("worksheet", ""), "status": self.current_analysis.get("means_result", {}).get("decision", "N/A") if self.current_analysis.get("means_result") else "N/A", "created_at": self.current_analysis.get("created_at", now_string()), "modified_at": now_string(), "summary": self.txt_summary.toPlainText(), "figure_png_base64": summary_b64, "residual_png_base64": residual_b64, "payload": serializable(self.current_analysis)}

    def save_to_journal(self):
        try:
            entry = self._journal_entry_payload()
            if self.current_journal_entry_id: self.app_window.update_journal_entry(self.current_journal_entry_id, entry)
            else:
                self.current_journal_entry_id = entry["id"]; self.app_window.add_journal_entry(entry)
            self.log("Journal entry saved.")
        except Exception as exc: self.log(f"Journal save failed: {exc}")

    def update_journal_entry(self):
        if not self.current_analysis: QMessageBox.information(self, "No analysis", "Run an analysis before updating the journal."); return
        self.save_to_journal(); self.app_window.statusBar().showMessage("ANOVA Analysis journal entry updated.")

    def load_from_journal_entry(self, entry: Dict[str, Any]):
        self.current_journal_entry_id = entry.get("id"); payload = entry.get("payload", {}); self.current_analysis = payload
        self.txt_summary.setPlainText(entry.get("summary", payload.get("summary", ""))); self._populate_numeric_results(payload); self._render_figures(payload); self.txt_residual.setPlainText(payload.get("residual_conclusion", ""))
