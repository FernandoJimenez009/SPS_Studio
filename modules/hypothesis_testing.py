
"""
modules/hypothesis_testing.py
SPS Studio — Hypothesis Test Builder

PySide6 module integrated with SPS Studio.

Study types:
- "one_sample": 1-Sample t
- "two_sample": 2-Sample t
- "paired": Paired t
- "proportions": Proportions

Visible UI text is English.
"""

from __future__ import annotations

import base64
import io
import math
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

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
    QTextEdit,
    QTabWidget,
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
# Helpers
# ─────────────────────────────────────────────

def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _fmt(value: Any, decimals: int = 4) -> str:
    if value is None:
        return "N/A"
    try:
        v = float(value)
        if np.isnan(v) or np.isinf(v):
            return "N/A"
        return f"{v:.{decimals}f}"
    except Exception:
        return str(value)


def _clean_numeric(series: pd.Series) -> np.ndarray:
    values = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
    return values[np.isfinite(values)]


def _decision(p_value: Any, alpha: float) -> str:
    try:
        return "Reject H0" if float(p_value) < alpha else "Accept H0"
    except Exception:
        return "N/A"


def _yes_no(decision: str) -> str:
    return "Yes" if decision == "Reject H0" else "No"


def _effect_label(value: float) -> str:
    try:
        a = abs(float(value))
    except Exception:
        return "N/A"
    if np.isnan(a):
        return "N/A"
    if a < 0.20:
        return "Negligible"
    if a < 0.50:
        return "Small"
    if a < 0.80:
        return "Medium"
    return "Large"


def _cohen_d_one_sample(x: np.ndarray, mu0: float) -> float:
    sd = np.std(x, ddof=1)
    return (np.mean(x) - mu0) / sd if sd > 0 else np.nan


def _cohen_d_independent(x: np.ndarray, y: np.ndarray) -> float:
    nx, ny = len(x), len(y)
    if nx + ny <= 2:
        return np.nan
    sx2, sy2 = np.var(x, ddof=1), np.var(y, ddof=1)
    sp = math.sqrt(((nx - 1) * sx2 + (ny - 1) * sy2) / (nx + ny - 2))
    return (np.mean(x) - np.mean(y)) / sp if sp > 0 else np.nan


def _cohen_d_paired(x: np.ndarray, y: np.ndarray) -> float:
    n = min(len(x), len(y))
    d = x[:n] - y[:n]
    sd = np.std(d, ddof=1)
    return np.mean(d) / sd if sd > 0 else np.nan


def _serializable(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, float)):
        return None if np.isnan(obj) or np.isinf(obj) else float(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, dict):
        return {str(k): _serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serializable(v) for v in obj]
    return obj


def figure_to_png_bytes(fig: Figure) -> bytes:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
    buffer.seek(0)
    return buffer.getvalue()


def figure_to_png_base64(fig: Figure) -> str:
    return base64.b64encode(figure_to_png_bytes(fig)).decode("ascii")


@dataclass
class HypothesisOptions:
    study_type: str = "two_sample"
    alpha: float = 0.05
    normality_alpha: float = 0.05
    std_alpha: float = 0.05
    mean_alternative: str = "two-sided"
    std_alternative: str = "two-sided"
    decimals: int = 4
    run_normality: bool = True
    run_std: bool = True
    run_mean: bool = True
    run_proportion: bool = True
    auto_journal_save: bool = True


# ─────────────────────────────────────────────
# Statistical engine
# ─────────────────────────────────────────────

def descriptive_stats(values: np.ndarray) -> Dict[str, Any]:
    n = len(values)
    return {
        "N": int(n),
        "Mean": float(np.mean(values)) if n else np.nan,
        "Std Dev": float(np.std(values, ddof=1)) if n > 1 else np.nan,
        "Variance": float(np.var(values, ddof=1)) if n > 1 else np.nan,
        "Median": float(np.median(values)) if n else np.nan,
        "Q1": float(np.percentile(values, 25)) if n else np.nan,
        "Q3": float(np.percentile(values, 75)) if n else np.nan,
        "IQR": float(np.percentile(values, 75) - np.percentile(values, 25)) if n else np.nan,
        "Min": float(np.min(values)) if n else np.nan,
        "Max": float(np.max(values)) if n else np.nan,
        "Skewness": float(stats.skew(values)) if n >= 3 else np.nan,
        "Kurtosis": float(stats.kurtosis(values)) if n >= 4 else np.nan,
    }


def normality_test(values: np.ndarray, alpha: float) -> Dict[str, Any]:
    if len(values) < 3:
        return {
            "section": "Normality",
            "test": "Shapiro-Wilk",
            "statistic_name": "W",
            "statistic": np.nan,
            "p_value": np.nan,
            "decision": "N/A",
            "normal": False,
            "conclusion": "Statistically normal? No",
        }
    stat, p = stats.shapiro(values)
    decision = _decision(p, alpha)
    normal = p >= alpha
    return {
        "section": "Normality",
        "test": "Shapiro-Wilk",
        "statistic_name": "W",
        "statistic": float(stat),
        "p_value": float(p),
        "decision": decision,
        "normal": bool(normal),
        "conclusion": "Statistically normal? Yes" if normal else "Statistically normal? No",
    }


def one_sample_mean_test(x: np.ndarray, mu0: float, alpha: float, alternative: str) -> Dict[str, Any]:
    alt = {
        "two-sided": "two-sided",
        "mean > value": "greater",
        "mean < value": "less",
    }.get(alternative, "two-sided")

    stat, p = stats.ttest_1samp(x, popmean=mu0, alternative=alt)
    n = len(x)
    mean = float(np.mean(x))
    sd = float(np.std(x, ddof=1))
    se = sd / math.sqrt(n)
    tcrit = float(stats.t.ppf(1 - alpha / 2, n - 1))
    ci = (mean - tcrit * se, mean + tcrit * se)
    decision = _decision(p, alpha)

    if alternative == "mean > value":
        conclusion = f"Statistically greater? {_yes_no(decision)}"
        h0, h1 = "H0: μ ≤ μ0", "H1: μ > μ0"
    elif alternative == "mean < value":
        conclusion = f"Statistically less? {_yes_no(decision)}"
        h0, h1 = "H0: μ ≥ μ0", "H1: μ < μ0"
    else:
        conclusion = f"Statistically different? {_yes_no(decision)}"
        h0, h1 = "H0: μ = μ0", "H1: μ ≠ μ0"

    d = _cohen_d_one_sample(x, mu0)
    return {
        "section": "Mean",
        "test": "1-Sample t",
        "h0": h0,
        "h1": h1,
        "statistic_name": "t",
        "statistic": float(stat),
        "p_value": float(p),
        "decision": decision,
        "conclusion": conclusion,
        "sample_mean": mean,
        "comparison_value": float(mu0),
        "difference": mean - float(mu0),
        "confidence_interval": ci,
        "effect_size": d,
        "effect_magnitude": _effect_label(d),
    }


def one_sample_std_test(x: np.ndarray, sigma0: float, alpha: float, alternative: str) -> Dict[str, Any]:
    if sigma0 <= 0:
        raise ValueError("Standard deviation comparison value must be greater than zero.")

    n = len(x)
    sample_var = float(np.var(x, ddof=1))
    sample_std = float(np.std(x, ddof=1))
    sigma2_0 = float(sigma0) ** 2
    chi_stat = (n - 1) * sample_var / sigma2_0
    df = n - 1

    if alternative == "std > value":
        p = stats.chi2.sf(chi_stat, df)
        conclusion = f"Statistically greater? {_yes_no(_decision(p, alpha))}"
        h0, h1 = "H0: σ ≤ σ0", "H1: σ > σ0"
    elif alternative == "std < value":
        p = stats.chi2.cdf(chi_stat, df)
        conclusion = f"Statistically less? {_yes_no(_decision(p, alpha))}"
        h0, h1 = "H0: σ ≥ σ0", "H1: σ < σ0"
    else:
        p = 2 * min(stats.chi2.cdf(chi_stat, df), stats.chi2.sf(chi_stat, df))
        p = min(float(p), 1.0)
        conclusion = f"Statistically different? {_yes_no(_decision(p, alpha))}"
        h0, h1 = "H0: σ = σ0", "H1: σ ≠ σ0"

    decision = _decision(p, alpha)
    lo_var = df * sample_var / stats.chi2.ppf(1 - alpha / 2, df)
    hi_var = df * sample_var / stats.chi2.ppf(alpha / 2, df)

    return {
        "section": "Standard Deviation",
        "test": "Chi-Square Standard Deviation Test",
        "h0": h0,
        "h1": h1,
        "statistic_name": "χ²",
        "statistic": float(chi_stat),
        "p_value": float(p),
        "decision": decision,
        "conclusion": conclusion,
        "sample_std_dev": sample_std,
        "comparison_value": float(sigma0),
        "difference": sample_std - float(sigma0),
        "confidence_interval": (math.sqrt(float(lo_var)), math.sqrt(float(hi_var))),
    }


def std_test_two_samples(x: np.ndarray, y: np.ndarray, alpha: float, alternative: str, both_normal: bool) -> Dict[str, Any]:
    sx = float(np.std(x, ddof=1))
    sy = float(np.std(y, ddof=1))
    vx = sx ** 2
    vy = sy ** 2

    if vx <= 0 or vy <= 0:
        raise ValueError("Standard deviation test requires positive sample variation in both groups.")

    # For directional alternatives, use F test. For two-sided normal data, use F test.
    # For two-sided non-normal data, use Levene/Brown-Forsythe as a robust path.
    if alternative == "std > group2":
        f_stat = vx / vy
        p = stats.f.sf(f_stat, len(x) - 1, len(y) - 1)
        test_name = "F Test for Standard Deviations"
        h0, h1 = "H0: σ1 ≤ σ2", "H1: σ1 > σ2"
        conclusion = f"Statistically greater? {_yes_no(_decision(p, alpha))}"
    elif alternative == "std < group2":
        f_stat = vx / vy
        p = stats.f.cdf(f_stat, len(x) - 1, len(y) - 1)
        test_name = "F Test for Standard Deviations"
        h0, h1 = "H0: σ1 ≥ σ2", "H1: σ1 < σ2"
        conclusion = f"Statistically less? {_yes_no(_decision(p, alpha))}"
    else:
        h0, h1 = "H0: σ1 = σ2", "H1: σ1 ≠ σ2"
        conclusion_prefix = "Statistically different? "
        if both_normal:
            f_stat = vx / vy
            cdf = stats.f.cdf(f_stat, len(x) - 1, len(y) - 1)
            p = 2 * min(cdf, 1 - cdf)
            p = min(float(p), 1.0)
            test_name = "F Test for Standard Deviations"
        else:
            f_stat, p = stats.levene(x, y, center="median")
            test_name = "Brown-Forsythe Standard Deviation Test"
        conclusion = conclusion_prefix + _yes_no(_decision(p, alpha))

    decision = _decision(p, alpha)
    return {
        "section": "Standard Deviation",
        "test": test_name,
        "h0": h0,
        "h1": h1,
        "statistic_name": "F" if "F Test" in test_name else "W",
        "statistic": float(f_stat),
        "p_value": float(p),
        "decision": decision,
        "conclusion": conclusion,
        "std_1": sx,
        "std_2": sy,
        "equal_std": decision == "Accept H0",
    }


def two_sample_mean_test(x: np.ndarray, y: np.ndarray, alpha: float, alternative: str, equal_std: bool, both_normal: bool) -> Dict[str, Any]:
    alt = {
        "two-sided": "two-sided",
        "group1 > group2": "greater",
        "group1 < group2": "less",
    }.get(alternative, "two-sided")

    if both_normal:
        stat, p = stats.ttest_ind(x, y, equal_var=equal_std, alternative=alt)
        test_name = "2-Sample t (Pooled)" if equal_std else "2-Sample t (Welch)"
        stat_name = "t"
    else:
        stat, p = stats.mannwhitneyu(x, y, alternative=alt, method="auto")
        test_name = "Mann-Whitney U"
        stat_name = "U"

    decision = _decision(p, alpha)

    if alternative == "group1 > group2":
        conclusion = f"Statistically greater? {_yes_no(decision)}"
        h0, h1 = "H0: μ1 ≤ μ2", "H1: μ1 > μ2"
    elif alternative == "group1 < group2":
        conclusion = f"Statistically less? {_yes_no(decision)}"
        h0, h1 = "H0: μ1 ≥ μ2", "H1: μ1 < μ2"
    else:
        conclusion = f"Statistically different? {_yes_no(decision)}"
        h0, h1 = "H0: μ1 = μ2", "H1: μ1 ≠ μ2"

    nx, ny = len(x), len(y)
    mean_diff = float(np.mean(x) - np.mean(y))
    se = math.sqrt(np.var(x, ddof=1) / nx + np.var(y, ddof=1) / ny)
    df_num = (np.var(x, ddof=1) / nx + np.var(y, ddof=1) / ny) ** 2
    df_den = ((np.var(x, ddof=1) / nx) ** 2 / (nx - 1)) + ((np.var(y, ddof=1) / ny) ** 2 / (ny - 1))
    df = df_num / df_den if df_den > 0 else nx + ny - 2
    tcrit = float(stats.t.ppf(1 - alpha / 2, df))
    ci = (mean_diff - tcrit * se, mean_diff + tcrit * se)
    d = _cohen_d_independent(x, y)

    return {
        "section": "Mean",
        "test": test_name,
        "h0": h0,
        "h1": h1,
        "statistic_name": stat_name,
        "statistic": float(stat),
        "p_value": float(p),
        "decision": decision,
        "conclusion": conclusion,
        "mean_difference": mean_diff,
        "confidence_interval": ci,
        "effect_size": d,
        "effect_magnitude": _effect_label(d),
    }


def paired_mean_test(x: np.ndarray, y: np.ndarray, alpha: float, alternative: str, diff_normal: bool) -> Dict[str, Any]:
    n = min(len(x), len(y))
    x = x[:n]
    y = y[:n]
    diff = x - y
    alt = {
        "two-sided": "two-sided",
        "group1 > group2": "greater",
        "group1 < group2": "less",
    }.get(alternative, "two-sided")

    if diff_normal:
        stat, p = stats.ttest_rel(x, y, alternative=alt)
        test_name = "Paired t"
        stat_name = "t"
    else:
        stat, p = stats.wilcoxon(diff, alternative=alt)
        test_name = "Wilcoxon Signed-Rank"
        stat_name = "W"

    decision = _decision(p, alpha)

    if alternative == "group1 > group2":
        conclusion = f"Statistically greater? {_yes_no(decision)}"
        h0, h1 = "H0: μd ≤ 0", "H1: μd > 0"
    elif alternative == "group1 < group2":
        conclusion = f"Statistically less? {_yes_no(decision)}"
        h0, h1 = "H0: μd ≥ 0", "H1: μd < 0"
    else:
        conclusion = f"Statistically different? {_yes_no(decision)}"
        h0, h1 = "H0: μd = 0", "H1: μd ≠ 0"

    mean_diff = float(np.mean(diff))
    sd = float(np.std(diff, ddof=1))
    se = sd / math.sqrt(n)
    tcrit = float(stats.t.ppf(1 - alpha / 2, n - 1))
    ci = (mean_diff - tcrit * se, mean_diff + tcrit * se)
    d = _cohen_d_paired(x, y)

    return {
        "section": "Mean",
        "test": test_name,
        "h0": h0,
        "h1": h1,
        "statistic_name": stat_name,
        "statistic": float(stat),
        "p_value": float(p),
        "decision": decision,
        "conclusion": conclusion,
        "mean_difference": mean_diff,
        "confidence_interval": ci,
        "effect_size": d,
        "effect_magnitude": _effect_label(d),
    }


def one_proportion_test(successes: int, trials: int, p0: float, alpha: float, alternative: str) -> Dict[str, Any]:
    if trials <= 0 or successes < 0 or successes > trials:
        raise ValueError("Successes and trials are invalid.")

    phat = successes / trials
    se0 = math.sqrt(p0 * (1 - p0) / trials)
    z = (phat - p0) / se0 if se0 > 0 else np.nan
    alt = {"p > value": "greater", "p < value": "less"}.get(alternative, "two-sided")

    if alt == "greater":
        p = stats.norm.sf(z)
        conclusion = f"Statistically greater? {_yes_no(_decision(p, alpha))}"
        h0, h1 = "H0: p ≤ p0", "H1: p > p0"
    elif alt == "less":
        p = stats.norm.cdf(z)
        conclusion = f"Statistically less? {_yes_no(_decision(p, alpha))}"
        h0, h1 = "H0: p ≥ p0", "H1: p < p0"
    else:
        p = 2 * stats.norm.sf(abs(z))
        conclusion = f"Statistically different? {_yes_no(_decision(p, alpha))}"
        h0, h1 = "H0: p = p0", "H1: p ≠ p0"

    return {
        "section": "Proportion",
        "test": "1-Proportion Z",
        "h0": h0,
        "h1": h1,
        "statistic_name": "Z",
        "statistic": float(z),
        "p_value": float(p),
        "decision": _decision(p, alpha),
        "conclusion": conclusion,
        "proportion": float(phat),
        "comparison_value": float(p0),
        "difference": float(phat - p0),
    }


def two_proportion_test(s1: int, n1: int, s2: int, n2: int, alpha: float, alternative: str) -> Dict[str, Any]:
    if n1 <= 0 or n2 <= 0 or not (0 <= s1 <= n1) or not (0 <= s2 <= n2):
        raise ValueError("Successes and trials are invalid.")

    p1, p2 = s1 / n1, s2 / n2
    pooled = (s1 + s2) / (n1 + n2)
    se0 = math.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
    z = (p1 - p2) / se0 if se0 > 0 else np.nan
    alt = {"p1 > p2": "greater", "p1 < p2": "less"}.get(alternative, "two-sided")

    if alt == "greater":
        p = stats.norm.sf(z)
        conclusion = f"Statistically greater? {_yes_no(_decision(p, alpha))}"
        h0, h1 = "H0: p1 ≤ p2", "H1: p1 > p2"
    elif alt == "less":
        p = stats.norm.cdf(z)
        conclusion = f"Statistically less? {_yes_no(_decision(p, alpha))}"
        h0, h1 = "H0: p1 ≥ p2", "H1: p1 < p2"
    else:
        p = 2 * stats.norm.sf(abs(z))
        conclusion = f"Statistically different? {_yes_no(_decision(p, alpha))}"
        h0, h1 = "H0: p1 = p2", "H1: p1 ≠ p2"

    return {
        "section": "Proportion",
        "test": "2-Proportion Z",
        "h0": h0,
        "h1": h1,
        "statistic_name": "Z",
        "statistic": float(z),
        "p_value": float(p),
        "decision": _decision(p, alpha),
        "conclusion": conclusion,
        "p1": float(p1),
        "p2": float(p2),
        "difference": float(p1 - p2),
    }


def build_step_by_step_text(analysis: Dict[str, Any], decimals: int = 4) -> str:
    lines: List[str] = []
    lines.append("=" * 92)
    lines.append("STEP-BY-STEP HYPOTHESIS TESTING WORKFLOW — SPS STUDIO")
    lines.append("=" * 92)
    lines.append(f"Study type          : {analysis.get('study_label', '')}")
    lines.append(f"Worksheet           : {analysis.get('worksheet', '')}")
    lines.append(f"Mean alternative    : {analysis.get('options', {}).get('mean_alternative', '')}")
    lines.append(f"Std Dev alternative : {analysis.get('options', {}).get('std_alternative', '')}")
    lines.append(f"Alpha               : {analysis.get('options', {}).get('alpha', '')}")
    lines.append("")

    lines.append("1) DESCRIPTIVE STATISTICS")
    lines.append("-" * 92)
    for group, metrics in analysis.get("descriptives", {}).items():
        lines.append(group)
        for key in ["N", "Mean", "Std Dev", "Variance", "Median", "Q1", "Q3", "IQR", "Min", "Max", "Skewness", "Kurtosis"]:
            if key in metrics:
                lines.append(f"  {key:<18}{_fmt(metrics[key], decimals)}")
        lines.append("")

    lines.append("2) NORMALITY ASSESSMENT")
    lines.append("-" * 92)
    normality = analysis.get("normality", {})
    if normality:
        for group, row in normality.items():
            lines.append(f"{group}")
            lines.append("  H0: Data follow a normal distribution.")
            lines.append("  H1: Data do not follow a normal distribution.")
            lines.append(f"  Test      : {row.get('test', '')}")
            lines.append(f"  Statistic : {_fmt(row.get('statistic'), decimals)}")
            lines.append(f"  p-value   : {_fmt(row.get('p_value'), decimals)}")
            lines.append(f"  Decision  : {row.get('decision', '')}")
            lines.append(f"  Conclusion: {row.get('conclusion', '')}")
            lines.append("")
    else:
        lines.append("Normality test was not selected.")
        lines.append("")

    lines.append("3) STANDARD DEVIATION ASSESSMENT")
    lines.append("-" * 92)
    std_tests = [t for t in analysis.get("tests", []) if t.get("section") == "Standard Deviation"]
    if std_tests:
        for row in std_tests:
            lines.append(f"Test      : {row.get('test', '')}")
            lines.append(row.get("h0", ""))
            lines.append(row.get("h1", ""))
            lines.append(f"{row.get('statistic_name', 'Statistic'):<10}: {_fmt(row.get('statistic'), decimals)}")
            lines.append(f"p-value   : {_fmt(row.get('p_value'), decimals)}")
            lines.append(f"Decision  : {row.get('decision', '')}")
            lines.append(f"Conclusion: {row.get('conclusion', '')}")
            lines.append("")
    else:
        lines.append("Standard deviation test was not selected or not applicable.")
        lines.append("")

    lines.append("4) MEAN / PROPORTION HYPOTHESIS TESTS")
    lines.append("-" * 92)
    main_tests = [t for t in analysis.get("tests", []) if t.get("section") != "Standard Deviation"]
    if not main_tests:
        lines.append("No mean or proportion test was selected.")
    for row in main_tests:
        lines.append(f"Test      : {row.get('test', '')}")
        lines.append(row.get("h0", ""))
        lines.append(row.get("h1", ""))
        lines.append(f"{row.get('statistic_name', 'Statistic'):<10}: {_fmt(row.get('statistic'), decimals)}")
        lines.append(f"p-value   : {_fmt(row.get('p_value'), decimals)}")
        lines.append(f"Decision  : {row.get('decision', '')}")
        lines.append(f"Conclusion: {row.get('conclusion', '')}")
        if "confidence_interval" in row:
            lo, hi = row.get("confidence_interval", [None, None])
            lines.append(f"95% CI    : [{_fmt(lo, decimals)}, {_fmt(hi, decimals)}]")
        if "effect_size" in row:
            lines.append(f"Effect    : {_fmt(row.get('effect_size'), decimals)} ({row.get('effect_magnitude', '')})")
        lines.append("")

    lines.append("5) DECISION SUMMARY")
    lines.append("-" * 92)
    header = f"{'Hypothesis':<30}{'p-value':<16}{'H0':<14}{'Conclusion':<30}"
    lines.append(header)
    lines.append("-" * len(header))
    for row in analysis.get("decision_rows", []):
        lines.append(f"{row.get('test',''):<30}{row.get('result',''):<16}{row.get('h0',''):<14}{row.get('conclusion',''):<30}")
    lines.append("=" * 92)
    return "\n".join(lines)


def build_report_text(analysis: Dict[str, Any], decimals: int = 4) -> str:
    return build_step_by_step_text(analysis, decimals)


# ─────────────────────────────────────────────
# Figures
# ─────────────────────────────────────────────

def _draw_combined_qq(ax, series: List[tuple]):
    try:
        for values, label in series:
            values = np.asarray(values, dtype=float)
            if len(values) < 3:
                continue
            (osm, osr), (slope, intercept, r) = stats.probplot(values, dist="norm", fit=True)
            ax.scatter(osm, osr, s=14, alpha=0.70, label=str(label))
            ax.plot(osm, slope * osm + intercept, linewidth=1.0, alpha=0.85)
        ax.set_title("Q-Q Plot", fontsize=10, fontweight="bold")
        ax.set_xlabel("Theoretical Quantiles")
        ax.set_ylabel("Sample Quantiles")
        ax.legend(fontsize=8, frameon=False)
        ax.grid(True, alpha=0.2)
    except Exception as exc:
        ax.text(0.5, 0.5, f"Q-Q plot unavailable\n{exc}", ha="center", va="center")


def _draw_decision_summary(ax, rows: List[Dict[str, Any]]):
    ax.axis("off")
    ax.set_title("Decision Summary", fontsize=11, fontweight="bold", pad=8)

    col_labels = ["Hypothesis", "p-value", "H0", "Conclusion"]
    table_rows = [
        [
            str(r.get("test", "")),
            str(r.get("result", "")),
            str(r.get("h0", "")),
            str(r.get("conclusion", "")),
        ]
        for r in rows
    ]
    if not table_rows:
        table_rows = [["No test selected", "", "", ""]]

    tbl = ax.table(cellText=table_rows, colLabels=col_labels, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.0)
    tbl.scale(1.08, 2.05)

    widths = [0.27, 0.16, 0.17, 0.40]
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
                if c == 3:
                    cell.set_text_props(fontweight="bold")


def build_hypothesis_figure(analysis: Dict[str, Any], decimals: int = 4) -> Figure:
    study_type = analysis["study_type"]
    fig = Figure(figsize=(15.5, 8.8), dpi=120)
    fig.patch.set_facecolor("white")

    title_ax = fig.add_axes([0.04, 0.875, 0.92, 0.055])
    title_ax.axis("off")
    title_ax.set_title(
        f"Hypothesis Test Report — {analysis.get('study_label', '')}",
        fontsize=16,
        fontweight="bold",
        pad=2,
    )

    if study_type == "proportions":
        gs = fig.add_gridspec(2, 3, left=0.055, right=0.985, top=0.81, bottom=0.08, wspace=0.30, hspace=0.42)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[0, 2])
        ax4 = fig.add_subplot(gs[1, :])
        mt = analysis.get("tests", [{}])[0]
        if "p1" in mt:
            labels = ["Group 1", "Group 2"]
            values = [mt.get("p1", 0), mt.get("p2", 0)]
        else:
            labels = ["Observed", "Hypothesized"]
            values = [mt.get("proportion", 0), mt.get("comparison_value", 0)]
        ax1.bar(labels, values, alpha=0.75)
        ax1.set_ylim(0, max(1.0, max(values) * 1.25))
        ax1.set_title("Proportion Comparison", fontsize=10, fontweight="bold")
        ax1.grid(True, axis="y", alpha=0.2)
        for ax in [ax2, ax3]:
            ax.axis("off")
        _draw_decision_summary(ax4, analysis.get("decision_rows", []))
        return fig

    gs = fig.add_gridspec(
        2, 3,
        left=0.055, right=0.985, top=0.81, bottom=0.08,
        wspace=0.30, hspace=0.42,
        width_ratios=[1.0, 1.0, 1.0],
    )

    ax_qq = fig.add_subplot(gs[0, 0])
    ax_hist = fig.add_subplot(gs[0, 1])
    ax_box = fig.add_subplot(gs[0, 2])
    ax_violin = fig.add_subplot(gs[1, 0])
    ax_summary = fig.add_subplot(gs[1, 1:])

    x = np.asarray(analysis.get("x_values", []), dtype=float)
    y = np.asarray(analysis.get("y_values", []), dtype=float) if "y_values" in analysis else None
    x_name = analysis.get("x_name", "Sample")
    y_name = analysis.get("y_name", "Second Sample")

    if study_type == "two_sample":
        _draw_combined_qq(ax_qq, [(x, x_name), (y, y_name)])

        bins = min(max(6, int(np.sqrt(max(len(x), len(y))))), 20)
        ax_hist.hist(x, bins=bins, alpha=0.60, edgecolor="white", label=x_name)
        ax_hist.hist(y, bins=bins, alpha=0.60, edgecolor="white", label=y_name)
        ax_hist.set_title("Overlay Histogram", fontsize=10, fontweight="bold")
        ax_hist.set_xlabel("Value")
        ax_hist.set_ylabel("Frequency")
        ax_hist.legend(fontsize=8, frameon=False)
        ax_hist.grid(True, alpha=0.2)

        ax_box.boxplot([x, y], labels=[x_name, y_name], patch_artist=True)
        means_xy = [float(np.mean(x)), float(np.mean(y))]
        ax_box.plot([1, 2], means_xy, marker="o", linewidth=1.8, color="#17324D", label="Mean")
        ax_box.legend(fontsize=8, frameon=False)
        ax_box.set_title("Boxplot Comparison", fontsize=10, fontweight="bold")
        ax_box.set_ylabel("Value")
        ax_box.grid(True, alpha=0.2)

        vp = ax_violin.violinplot([x, y], showmedians=True)
        for body in vp["bodies"]:
            body.set_alpha(0.65)
        ax_violin.plot([1, 2], means_xy, marker="o", linewidth=1.8, color="#17324D", label="Mean")
        ax_violin.legend(fontsize=8, frameon=False)
        ax_violin.set_xticks([1, 2])
        ax_violin.set_xticklabels([x_name, y_name])
        ax_violin.set_title("Violin Plot", fontsize=10, fontweight="bold")
        ax_violin.set_ylabel("Value")
        ax_violin.grid(True, alpha=0.2)

    elif study_type == "paired":
        n = min(len(x), len(y))
        x2 = x[:n]
        y2 = y[:n]
        diff = x2 - y2

        # Requested: Q-Q, histogram and boxplot show before/after values.
        # Only violin plot shows the paired difference.
        _draw_combined_qq(ax_qq, [(x2, x_name), (y2, y_name)])

        bins = min(max(6, int(np.sqrt(n))), 20)
        ax_hist.hist(x2, bins=bins, alpha=0.60, edgecolor="white", label=x_name)
        ax_hist.hist(y2, bins=bins, alpha=0.60, edgecolor="white", label=y_name)
        ax_hist.set_title("Overlay Histogram", fontsize=10, fontweight="bold")
        ax_hist.set_xlabel("Value")
        ax_hist.set_ylabel("Frequency")
        ax_hist.legend(fontsize=8, frameon=False)
        ax_hist.grid(True, alpha=0.2)

        ax_box.boxplot([x2, y2], labels=[x_name, y_name], patch_artist=True)
        means_xy = [float(np.mean(x2)), float(np.mean(y2))]
        ax_box.plot([1, 2], means_xy, marker="o", linewidth=1.8, color="#17324D", label="Mean")
        ax_box.legend(fontsize=8, frameon=False)
        ax_box.set_title("Boxplot Comparison", fontsize=10, fontweight="bold")
        ax_box.grid(True, alpha=0.2)

        vp = ax_violin.violinplot([diff], showmedians=True)
        for body in vp["bodies"]:
            body.set_alpha(0.65)
        ax_violin.axhline(0, linestyle="--", linewidth=1.1)
        ax_violin.set_xticks([1])
        ax_violin.set_xticklabels(["Difference"])
        ax_violin.set_title("Violin Plot — Difference", fontsize=10, fontweight="bold")
        ax_violin.grid(True, alpha=0.2)

    else:
        _draw_combined_qq(ax_qq, [(x, x_name)])
        bins = min(max(6, int(np.sqrt(len(x)))), 20)
        ax_hist.hist(x, bins=bins, alpha=0.70, edgecolor="white")
        if "hypothesized_mean" in analysis:
            ax_hist.axvline(analysis["hypothesized_mean"], linestyle="--", linewidth=1.2, label="Mean target")
        ax_hist.set_title("Histogram", fontsize=10, fontweight="bold")
        ax_hist.set_xlabel("Value")
        ax_hist.set_ylabel("Frequency")
        ax_hist.legend(fontsize=8, frameon=False)
        ax_hist.grid(True, alpha=0.2)

        ax_box.boxplot([x], labels=[x_name], patch_artist=True)
        mean_x = float(np.mean(x))
        ax_box.plot([1], [mean_x], marker="o", color="#17324D", label="Mean")
        if "hypothesized_mean" in analysis:
            ax_box.axhline(analysis["hypothesized_mean"], linestyle="--", linewidth=1.2)
        ax_box.legend(fontsize=8, frameon=False)
        ax_box.set_title("Boxplot", fontsize=10, fontweight="bold")
        ax_box.grid(True, alpha=0.2)

        vp = ax_violin.violinplot([x], showmedians=True)
        for body in vp["bodies"]:
            body.set_alpha(0.65)
        ax_violin.plot([1], [mean_x], marker="o", color="#17324D", label="Mean")
        ax_violin.set_xticks([1])
        ax_violin.set_xticklabels([x_name])
        ax_violin.legend(fontsize=8, frameon=False)
        ax_violin.set_title("Violin Plot", fontsize=10, fontweight="bold")
        ax_violin.grid(True, alpha=0.2)

    _draw_decision_summary(ax_summary, analysis.get("decision_rows", []))
    return fig


# ─────────────────────────────────────────────
# PySide6 Builder
# ─────────────────────────────────────────────

class HypothesisTestBuilder(QWidget):
    def __init__(self, app_window, study_type: str = "two_sample"):
        super().__init__()
        self.app_window = app_window
        self.study_type = study_type
        self.current_df = pd.DataFrame()
        self.current_analysis: Optional[Dict[str, Any]] = None
        self.current_figure: Optional[Figure] = None
        self.current_canvas: Optional[FigureCanvas] = None
        self.interaction_manager = None
        self.current_journal_entry_id: Optional[str] = None

        self._build_ui()
        self.refresh_columns()

    def _study_label(self) -> str:
        return {
            "one_sample": "1-Sample t",
            "two_sample": "2-Sample t",
            "paired": "Paired t",
            "proportions": "Proportions",
        }.get(self.study_type, "2-Sample t")

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)

        header = QHBoxLayout()
        title = QLabel(f"Hypothesis Test Builder — {self._study_label()}")
        title.setStyleSheet("font-size:20px; font-weight:bold; color:#17324D;")
        header.addWidget(title)
        header.addStretch()
        self.lbl_ws = QLabel("Worksheet: -")
        header.addWidget(self.lbl_ws)
        root.addLayout(header)

        self.tabs = QTabWidget()
        root.addWidget(self.tabs)

        self._build_setup_tab()
        self._build_results_tab()
        self._build_graphs_tab()
        self._build_summary_tab()
        self._build_debug_tab()

    def _build_setup_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(8, 8, 8, 8)

        top = QHBoxLayout()

        variables = QGroupBox("1. Variables")
        vgrid = QGridLayout(variables)

        self.cmb_x = QComboBox()
        self.cmb_y = QComboBox()
        self.lbl_x = QLabel("Sample column")
        self.lbl_y = QLabel("Second sample column")

        vgrid.addWidget(self.lbl_x, 0, 0)
        vgrid.addWidget(self.cmb_x, 0, 1)

        if self.study_type in {"two_sample", "paired"}:
            vgrid.addWidget(self.lbl_y, 1, 0)
            vgrid.addWidget(self.cmb_y, 1, 1)

        if self.study_type == "one_sample":
            self.spn_hyp_value = QDoubleSpinBox()
            self.spn_hyp_value.setRange(-1_000_000_000, 1_000_000_000)
            self.spn_hyp_value.setDecimals(6)
            self.spn_hyp_value.setValue(0.0)

            self.spn_hyp_std = QDoubleSpinBox()
            self.spn_hyp_std.setRange(0.000001, 1_000_000_000)
            self.spn_hyp_std.setDecimals(6)
            self.spn_hyp_std.setValue(1.0)

            vgrid.addWidget(QLabel("Mean comparison value"), 1, 0)
            vgrid.addWidget(self.spn_hyp_value, 1, 1)
            vgrid.addWidget(QLabel("Std Dev comparison value"), 2, 0)
            vgrid.addWidget(self.spn_hyp_std, 2, 1)

        if self.study_type == "proportions":
            self.cmb_prop_mode = QComboBox()
            self.cmb_prop_mode.addItems(["One proportion", "Two proportions"])
            self.spn_s1 = QSpinBox(); self.spn_s1.setRange(0, 1_000_000)
            self.spn_n1 = QSpinBox(); self.spn_n1.setRange(1, 1_000_000); self.spn_n1.setValue(100)
            self.spn_s2 = QSpinBox(); self.spn_s2.setRange(0, 1_000_000)
            self.spn_n2 = QSpinBox(); self.spn_n2.setRange(1, 1_000_000); self.spn_n2.setValue(100)
            self.spn_p0 = QDoubleSpinBox(); self.spn_p0.setRange(0.000001, 0.999999); self.spn_p0.setDecimals(6); self.spn_p0.setValue(0.5)

            for r, (lbl, w) in enumerate([
                ("Proportion mode", self.cmb_prop_mode),
                ("Successes 1", self.spn_s1),
                ("Trials 1", self.spn_n1),
                ("Successes 2", self.spn_s2),
                ("Trials 2", self.spn_n2),
                ("Hypothesized p", self.spn_p0),
            ]):
                vgrid.addWidget(QLabel(lbl), r, 0)
                vgrid.addWidget(w, r, 1)

        options = QGroupBox("2. Analysis Options")
        ogrid = QGridLayout(options)

        self.spn_alpha = QDoubleSpinBox()
        self.spn_alpha.setRange(0.001, 0.999)
        self.spn_alpha.setDecimals(3)
        self.spn_alpha.setSingleStep(0.005)
        self.spn_alpha.setValue(0.05)

        self.spn_alpha_n = QDoubleSpinBox()
        self.spn_alpha_n.setRange(0.001, 0.999)
        self.spn_alpha_n.setDecimals(3)
        self.spn_alpha_n.setSingleStep(0.005)
        self.spn_alpha_n.setValue(0.05)

        self.spn_alpha_std = QDoubleSpinBox()
        self.spn_alpha_std.setRange(0.001, 0.999)
        self.spn_alpha_std.setDecimals(3)
        self.spn_alpha_std.setSingleStep(0.005)
        self.spn_alpha_std.setValue(0.05)

        self.cmb_mean_alt = QComboBox()
        if self.study_type == "one_sample":
            self.cmb_mean_alt.addItems(["two-sided", "mean > value", "mean < value"])
        elif self.study_type == "proportions":
            self.cmb_mean_alt.addItems(["two-sided", "p1 > p2", "p1 < p2", "p > value", "p < value"])
        else:
            self.cmb_mean_alt.addItems(["two-sided", "group1 > group2", "group1 < group2"])

        self.cmb_std_alt = QComboBox()
        if self.study_type == "one_sample":
            self.cmb_std_alt.addItems(["two-sided", "std > value", "std < value"])
        elif self.study_type in {"two_sample", "paired"}:
            self.cmb_std_alt.addItems(["two-sided", "std > group2", "std < group2"])
        else:
            self.cmb_std_alt.addItems(["two-sided"])

        self.spn_decimals = QSpinBox()
        self.spn_decimals.setRange(0, 8)
        self.spn_decimals.setValue(4)

        self.chk_normality = QCheckBox("Run normality test")
        self.chk_std = QCheckBox("Run standard deviation test")
        self.chk_mean = QCheckBox("Run mean test")
        self.chk_proportion = QCheckBox("Run proportion test")
        self.chk_journal = QCheckBox("Auto journal save")

        for chk in [self.chk_normality, self.chk_std, self.chk_mean, self.chk_journal]:
            chk.setChecked(True)
        self.chk_proportion.setChecked(self.study_type == "proportions")

        if self.study_type == "proportions":
            self.chk_normality.setVisible(False)
            self.chk_std.setVisible(False)
            self.chk_mean.setVisible(False)
            self.chk_proportion.setVisible(True)
            self.cmb_std_alt.setVisible(False)
            self.spn_alpha_n.setVisible(False)
            self.spn_alpha_std.setVisible(False)
        else:
            self.chk_proportion.setVisible(False)

        rows = [
            ("Alpha", self.spn_alpha),
            ("Normality alpha", self.spn_alpha_n),
            ("Std Dev alpha", self.spn_alpha_std),
            ("Mean alternative", self.cmb_mean_alt),
            ("Std Dev alternative", self.cmb_std_alt),
            ("Decimals", self.spn_decimals),
            ("", self.chk_normality),
            ("", self.chk_std),
            ("", self.chk_mean),
            ("", self.chk_proportion),
            ("", self.chk_journal),
        ]
        for r, (lbl, widget) in enumerate(rows):
            if lbl:
                label_widget = QLabel(lbl)
                if self.study_type == "proportions" and lbl in {"Normality alpha", "Std Dev alpha", "Std Dev alternative"}:
                    label_widget.setVisible(False)
                ogrid.addWidget(label_widget, r, 0)
            ogrid.addWidget(widget, r, 1)

        top.addWidget(variables, 2)
        top.addWidget(options, 1)
        layout.addLayout(top)

        run_group = QGroupBox("3. Run")
        run_layout = QVBoxLayout(run_group)
        buttons = QHBoxLayout()
        self.btn_refresh = QPushButton("Refresh Columns")
        self.btn_run = QPushButton("Run Hypothesis Test")
        self.btn_run.setObjectName("PrimaryButton")
        self.btn_run.setStyleSheet("background:#2F5D87; color:white; font-weight:bold; padding:8px; border-radius:6px;")
        self.btn_refresh.clicked.connect(self.refresh_columns)
        self.btn_run.clicked.connect(self.run_analysis)
        buttons.addWidget(self.btn_refresh)
        buttons.addWidget(self.btn_run)
        run_layout.addLayout(buttons)

        note = QLabel("Default behavior runs all applicable checks. Use checkboxes to select only the hypotheses required.")
        note.setWordWrap(True)
        note.setStyleSheet("color:#4F6680;")
        run_layout.addWidget(note)

        layout.addWidget(run_group)
        layout.addStretch()
        self.tabs.addTab(tab, "Setup")

    def _build_results_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        self.txt_steps = QTextEdit()
        self.txt_steps.setReadOnly(True)
        self.txt_steps.setPlainText(self._default_steps_text())
        layout.addWidget(self.txt_steps)
        self.tabs.addTab(tab, "Step-by-Step")

    def _build_graphs_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        toolbar = QHBoxLayout()
        self.btn_copy_graph = QPushButton("Copy Graph")
        self.btn_export_png = QPushButton("Export PNG")
        self.btn_rerun = QPushButton("Re-run")
        self.btn_update_journal = QPushButton("Update Journal Entry")
        self.btn_copy_graph.clicked.connect(self.copy_graph)
        self.btn_export_png.clicked.connect(self.export_png)
        self.btn_rerun.clicked.connect(self.run_analysis)
        self.btn_update_journal.clicked.connect(self.update_journal_entry)
        toolbar.addStretch()
        toolbar.addWidget(self.btn_copy_graph)
        toolbar.addWidget(self.btn_export_png)
        toolbar.addWidget(self.btn_rerun)
        toolbar.addWidget(self.btn_update_journal)
        layout.addLayout(toolbar)

        self.graph_container = QWidget()
        self.graph_layout = QVBoxLayout(self.graph_container)
        layout.addWidget(self.graph_container)
        self.tabs.addTab(tab, "Graphs")

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

    def _default_steps_text(self) -> str:
        return (
            "Run the analysis to see the full step-by-step workflow.\n\n"
            "The workflow will show:\n"
            "1. Descriptive statistics\n"
            "2. Normality assessment\n"
            "3. Standard deviation assessment\n"
            "4. Mean or proportion hypothesis test(s)\n"
            "5. Decision Summary by selected hypothesis\n"
        )

    def _log(self, msg: str):
        self.txt_debug.append(f"[{_now()}] {msg}")

    def refresh_columns(self):
        self.current_df = self.app_window.get_active_dataframe()
        self.lbl_ws.setText(f"Worksheet: {self.app_window.active_worksheet_name()}")
        cols = list(self.current_df.columns) if not self.current_df.empty else []
        numeric_cols = [c for c in cols if pd.api.types.is_numeric_dtype(self.current_df[c])]

        for cmb in [self.cmb_x, self.cmb_y]:
            cmb.clear()
            cmb.addItems(numeric_cols)

        self._log(f"Columns refreshed. Numeric columns: {numeric_cols}")

    def _options(self) -> HypothesisOptions:
        return HypothesisOptions(
            study_type=self.study_type,
            alpha=float(self.spn_alpha.value()),
            normality_alpha=float(self.spn_alpha_n.value()),
            std_alpha=float(self.spn_alpha_std.value()),
            mean_alternative=self.cmb_mean_alt.currentText(),
            std_alternative=self.cmb_std_alt.currentText(),
            decimals=int(self.spn_decimals.value()),
            run_normality=self.chk_normality.isChecked(),
            run_std=self.chk_std.isChecked(),
            run_mean=self.chk_mean.isChecked(),
            run_proportion=self.chk_proportion.isChecked(),
            auto_journal_save=self.chk_journal.isChecked(),
        )

    def run_analysis(self):
        try:
            opts = self._options()
            analysis: Dict[str, Any] = {
                "id": str(uuid.uuid4()),
                "module": "Hypothesis Test",
                "type": "Hypothesis Test",
                "study_type": self.study_type,
                "study_label": self._study_label(),
                "worksheet": self.app_window.active_worksheet_name(),
                "options": asdict(opts),
                "created_at": _now(),
            }

            if self.study_type == "proportions":
                self._run_proportion_analysis(analysis, opts)
            else:
                self._run_numeric_analysis(analysis, opts)

            analysis["interpretation"] = self._build_interpretation(analysis)
            analysis["summary"] = build_report_text(analysis, opts.decimals)

            self.current_analysis = analysis
            self.txt_steps.setPlainText(analysis["summary"])
            self.txt_summary.setPlainText(analysis["summary"])
            self._render_figure(analysis)

            if opts.auto_journal_save:
                self.save_to_journal()

            self.tabs.setCurrentIndex(1)
            self._log("Analysis completed successfully.")

        except Exception as exc:
            self._log(f"ERROR: {exc}")
            QMessageBox.critical(self, "Analysis Error", str(exc))

    def _run_numeric_analysis(self, analysis: Dict[str, Any], opts: HypothesisOptions):
        if self.current_df.empty:
            raise ValueError("No active worksheet data found.")

        x_col = self.cmb_x.currentText()
        if not x_col:
            raise ValueError("Select a sample column.")

        x = _clean_numeric(self.current_df[x_col])
        if len(x) < 3:
            raise ValueError("The first sample needs at least 3 numeric observations.")

        analysis["x_name"] = x_col
        analysis["x_values"] = x
        analysis["descriptives"] = {x_col: descriptive_stats(x)}
        tests: List[Dict[str, Any]] = []

        if self.study_type in {"two_sample", "paired"}:
            y_col = self.cmb_y.currentText()
            if not y_col:
                raise ValueError("Select a second sample column.")
            if y_col == x_col:
                raise ValueError("Select two different columns.")

            y = _clean_numeric(self.current_df[y_col])
            if len(y) < 3:
                raise ValueError("The second sample needs at least 3 numeric observations.")
            if self.study_type == "paired" and len(x) != len(y):
                raise ValueError("Paired t requires both columns to have the same number of valid observations.")

            analysis["y_name"] = y_col
            analysis["y_values"] = y
            analysis["descriptives"][y_col] = descriptive_stats(y)

        if self.study_type == "one_sample" and not opts.run_mean and not opts.run_std:
            raise ValueError("For 1-Sample t, select at least one comparison: mean and/or standard deviation.")

        normality: Dict[str, Any] = {}
        if opts.run_normality:
            if self.study_type == "paired":
                # Decision summary remains aligned to the 2-sample structure, while the mean test uses paired differences.
                normality[x_col] = normality_test(x, opts.normality_alpha)
                normality[analysis["y_name"]] = normality_test(analysis["y_values"], opts.normality_alpha)
            else:
                normality[x_col] = normality_test(x, opts.normality_alpha)
                if self.study_type == "two_sample":
                    normality[analysis["y_name"]] = normality_test(analysis["y_values"], opts.normality_alpha)

        analysis["normality"] = normality
        both_normal = all(row.get("normal", False) for row in normality.values()) if normality else True

        if self.study_type == "one_sample":
            if opts.run_std:
                sigma0 = float(self.spn_hyp_std.value())
                analysis["hypothesized_std"] = sigma0
                tests.append(one_sample_std_test(x, sigma0, opts.std_alpha, opts.std_alternative))
            if opts.run_mean:
                mu0 = float(self.spn_hyp_value.value())
                analysis["hypothesized_mean"] = mu0
                tests.append(one_sample_mean_test(x, mu0, opts.alpha, opts.mean_alternative))

        elif self.study_type == "two_sample":
            std_result = None
            if opts.run_std:
                std_result = std_test_two_samples(x, analysis["y_values"], opts.std_alpha, opts.std_alternative, both_normal)
                tests.append(std_result)
            if opts.run_mean:
                equal_std = bool(std_result.get("equal_std", False)) if std_result else False
                tests.append(two_sample_mean_test(x, analysis["y_values"], opts.alpha, opts.mean_alternative, equal_std, both_normal))

        elif self.study_type == "paired":
            y = analysis["y_values"]
            diff = x - y
            diff_normal = normality_test(diff, opts.normality_alpha).get("normal", False)

            std_result = None
            if opts.run_std:
                std_result = std_test_two_samples(x, y, opts.std_alpha, opts.std_alternative, both_normal)
                tests.append(std_result)
            if opts.run_mean:
                tests.append(paired_mean_test(x, y, opts.alpha, opts.mean_alternative, diff_normal))

        analysis["tests"] = tests
        analysis["main_test"] = tests[-1] if tests else {"test": "Not selected", "decision": "N/A", "p_value": np.nan}
        analysis["decision_rows"] = self._build_decision_rows(analysis)

    def _run_proportion_analysis(self, analysis: Dict[str, Any], opts: HypothesisOptions):
        if not opts.run_proportion:
            raise ValueError("Proportion test is disabled.")

        mode = self.cmb_prop_mode.currentText()
        s1, n1 = int(self.spn_s1.value()), int(self.spn_n1.value())
        s2, n2 = int(self.spn_s2.value()), int(self.spn_n2.value())
        p0 = float(self.spn_p0.value())

        analysis["proportion_mode"] = mode
        analysis["counts"] = {"successes_1": s1, "trials_1": n1, "successes_2": s2, "trials_2": n2, "p0": p0}

        if mode == "One proportion":
            test = one_proportion_test(s1, n1, p0, opts.alpha, opts.mean_alternative)
        else:
            test = two_proportion_test(s1, n1, s2, n2, opts.alpha, opts.mean_alternative)

        analysis["tests"] = [test]
        analysis["main_test"] = test
        analysis["descriptives"] = {}
        analysis["normality"] = {}
        analysis["decision_rows"] = self._build_decision_rows(analysis)

    def _build_decision_rows(self, analysis: Dict[str, Any]) -> List[Dict[str, str]]:
        rows: List[Dict[str, str]] = []
        d = int(analysis.get("options", {}).get("decimals", 4))

        # 1) Normality
        for group, nrow in analysis.get("normality", {}).items():
            rows.append({
                "test": f"Normality - {group}",
                "result": f"p={_fmt(nrow.get('p_value'), d)}",
                "h0": nrow.get("decision", ""),
                "conclusion": nrow.get("conclusion", ""),
            })

        # 2) Standard deviation
        for t in analysis.get("tests", []):
            if t.get("section") == "Standard Deviation":
                rows.append({
                    "test": "Standard Deviation",
                    "result": f"p={_fmt(t.get('p_value'), d)}",
                    "h0": t.get("decision", ""),
                    "conclusion": t.get("conclusion", ""),
                })

        # 3) Mean / proportion
        for t in analysis.get("tests", []):
            if t.get("section") != "Standard Deviation":
                rows.append({
                    "test": t.get("section", t.get("test", "")),
                    "result": f"p={_fmt(t.get('p_value'), d)}",
                    "h0": t.get("decision", ""),
                    "conclusion": t.get("conclusion", ""),
                })

        return rows

    def _build_interpretation(self, analysis: Dict[str, Any]) -> str:
        tests = analysis.get("tests", [])
        if not tests:
            return "No main hypothesis test was selected."
        return " | ".join([t.get("conclusion", "") for t in tests])

    def _render_figure(self, analysis: Dict[str, Any]):
        while self.graph_layout.count():
            child = self.graph_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        d = int(analysis.get("options", {}).get("decimals", 4))
        self.current_figure = build_hypothesis_figure(analysis, d)
        self.current_canvas = FigureCanvas(self.current_figure)
        self.graph_layout.addWidget(self.current_canvas)
        self.current_canvas.draw()

        if enable_interaction is not None:
            try:
                self.interaction_manager = enable_interaction(
                    host=self,
                    figure_getter=lambda: self.current_figure,
                    canvas_getter=lambda: self.current_canvas,
                    decimals_getter=lambda: int(self.spn_decimals.value()),
                    on_status=lambda msg: self.app_window.statusBar().showMessage(msg),
                )
                self.interaction_manager.register_axes()
            except Exception as exc:
                self._log(f"Plot interaction was not enabled: {exc}")

    def copy_graph(self):
        if self.current_figure is None:
            QMessageBox.information(self, "No graph", "Run an analysis first.")
            return

        try:
            raw = figure_to_png_bytes(self.current_figure)
            image = QImage()
            if not image.loadFromData(raw, "PNG"):
                raise ValueError("Could not convert graph to clipboard image.")
            QApplication.clipboard().setImage(image)
            self.app_window.statusBar().showMessage("Graph copied to clipboard as image.")
        except Exception as exc:
            QMessageBox.critical(self, "Copy failed", str(exc))

    def export_png(self):
        if self.current_figure is None:
            QMessageBox.information(self, "No graph", "Run an analysis first.")
            return

        path, _ = QFileDialog.getSaveFileName(self, "Export PNG", "", "PNG Image (*.png)")
        if not path:
            return
        if not path.lower().endswith(".png"):
            path += ".png"
        self.current_figure.savefig(path, dpi=160, bbox_inches="tight")
        self.app_window.statusBar().showMessage(f"Graph exported: {path}")

    def _journal_entry_payload(self) -> Dict[str, Any]:
        if self.current_analysis is None:
            raise ValueError("No analysis has been run.")

        image_b64 = figure_to_png_base64(self.current_figure) if self.current_figure is not None else ""
        name = f"Hypothesis Test - {self.current_analysis.get('study_label', '')} - {self.current_analysis.get('worksheet', '')}"

        return {
            "id": self.current_journal_entry_id or str(uuid.uuid4()),
            "name": name,
            "type": "Hypothesis Test",
            "module": "Hypothesis Test",
            "worksheet": self.current_analysis.get("worksheet", ""),
            "status": self.current_analysis.get("main_test", {}).get("decision", "N/A"),
            "created_at": self.current_analysis.get("created_at", _now()),
            "modified_at": _now(),
            "summary": self.txt_summary.toPlainText(),
            "figure_png_base64": image_b64,
            "payload": _serializable(self.current_analysis),
        }

    def save_to_journal(self):
        try:
            entry = self._journal_entry_payload()
            if self.current_journal_entry_id:
                self.app_window.update_journal_entry(self.current_journal_entry_id, entry)
            else:
                self.current_journal_entry_id = entry["id"]
                self.app_window.add_journal_entry(entry)
            self._log("Journal entry saved.")
        except Exception as exc:
            self._log(f"Journal save failed: {exc}")

    def update_journal_entry(self):
        if not self.current_analysis:
            QMessageBox.information(self, "No analysis", "Run an analysis before updating the journal.")
            return
        self.save_to_journal()
        self.app_window.statusBar().showMessage("Hypothesis Test journal entry updated.")

    def load_from_journal_entry(self, entry: Dict[str, Any]):
        self.current_journal_entry_id = entry.get("id")
        payload = entry.get("payload", {})
        self.study_type = payload.get("study_type", self.study_type)
        self.current_analysis = payload
        self.txt_summary.setPlainText(entry.get("summary", payload.get("summary", "")))
        self.txt_steps.setPlainText(entry.get("summary", payload.get("summary", "")))
        try:
            self._render_figure(payload)
        except Exception as exc:
            self._log(f"Could not rebuild figure from journal payload: {exc}")

    def _log_no_status(self, msg: str):
        self._log(msg)
