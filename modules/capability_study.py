
"""
modules/capability_study.py
SPS Studio — Capability Study Builder
Phase 1: worksheet connection, configurable statistical engine, tables, text report and debug logs.

Notes:
- No Tkinter dependency.
- Visible UI text is English.
- Designed to be opened from ui/main_window.py.
"""

from __future__ import annotations

import math
import re
import io
import base64
import warnings
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import (
    anderson,
    chi2,
    kurtosis,
    norm,
    normaltest,
    shapiro,
    skew,
    t as t_dist,
    trim_mean,
    probplot,
)

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QFileDialog,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.backends.backend_pdf import PdfPages

try:
    from core.plot_interaction_tools import enable_interaction
except Exception:
    enable_interaction = None


D2_CONSTANTS = {
    2: 1.128,
    3: 1.693,
    4: 2.059,
    5: 2.326,
    6: 2.534,
    7: 2.704,
    8: 2.847,
    9: 2.970,
    10: 3.078,
    11: 3.173,
    12: 3.258,
    13: 3.336,
    14: 3.407,
    15: 3.472,
    16: 3.532,
    17: 3.588,
    18: 3.640,
    19: 3.689,
    20: 3.735,
    21: 3.778,
    22: 3.819,
    23: 3.858,
    24: 3.895,
    25: 3.931,
}

C4_CONSTANTS = {
    2: 0.7979,
    3: 0.8862,
    4: 0.9213,
    5: 0.9400,
    6: 0.9515,
    7: 0.9594,
    8: 0.9650,
    9: 0.9693,
    10: 0.9727,
    11: 0.9754,
    12: 0.9776,
    13: 0.9794,
    14: 0.9810,
    15: 0.9823,
    16: 0.9835,
    17: 0.9845,
    18: 0.9854,
    19: 0.9862,
    20: 0.9869,
    21: 0.9876,
    22: 0.9882,
    23: 0.9887,
    24: 0.9892,
    25: 0.9896,
}


# ─────────────────────────────────────────────
# Utility functions
# ─────────────────────────────────────────────

def safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    text = str(value).replace("\u00A0", " ").strip()
    if text == "":
        return None
    text = text.replace(",", ".")
    return float(text)


def parse_data_block(text: str) -> np.ndarray:
    if text is None:
        return np.array([], dtype=float)
    cleaned = text.replace("\u00A0", " ").strip()
    cleaned = cleaned.replace(";", " ").replace("\t", " ")
    rough_tokens = re.split(r"\s+", cleaned)
    data = []
    for token in rough_tokens:
        token = token.strip()
        if not token:
            continue
        if token.count(",") > 1:
            for sub in token.split(","):
                if sub.strip():
                    data.append(float(sub.strip().replace(",", ".")))
            continue
        if "," in token and "." not in token:
            data.append(float(token.replace(",", ".")))
            continue
        if "," in token and "." in token:
            try:
                data.append(float(token))
            except ValueError:
                for part in token.split(","):
                    if part.strip():
                        data.append(float(part.strip()))
            continue
        data.append(float(token))
    return np.array(data, dtype=float)


def is_na(value: Any) -> bool:
    return value is None or (isinstance(value, (float, np.floating)) and (np.isnan(value) or np.isinf(value)))


def fmt(value: Any, digits: int = 2) -> str:
    if is_na(value):
        return "N/A"
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return "N/A"


def fmt2(value: Any) -> str:
    return fmt(value, 2)


def fmt_int_or_2(value: Any) -> str:
    if is_na(value):
        return "N/A"
    try:
        v = float(value)
        if v.is_integer():
            return str(int(v))
        return f"{v:.2f}"
    except Exception:
        return "N/A"


def capability_status(value: Any) -> str:
    if is_na(value):
        return "N/A"
    value = float(value)
    if value >= 1.67:
        return "World Class"
    if value >= 1.33:
        return "Capable"
    if value >= 1.00:
        return "Marginal"
    return "Not Capable"


def status_color(status: str) -> str:
    return {
        "World Class": "#1a7f37",
        "Capable": "#2ea043",
        "Marginal": "#d4a017",
        "Not Capable": "#cf222e",
        "N/A": "#888888",
    }.get(status, "#888888")


def c4_unbiasing_constant(n: int) -> float:
    if n in C4_CONSTANTS:
        return C4_CONSTANTS[n]
    if n <= 1:
        return np.nan
    return math.sqrt(2.0 / (n - 1.0)) * math.gamma(n / 2.0) / math.gamma((n - 1.0) / 2.0)


def d2_constant(n: int) -> float:
    if n in D2_CONSTANTS:
        return D2_CONSTANTS[n]
    return D2_CONSTANTS[25]


# ─────────────────────────────────────────────
# Sigma estimation
# ─────────────────────────────────────────────

def moving_range_values(data: np.ndarray, length: int = 2) -> np.ndarray:
    length = max(2, int(length))
    if len(data) < length:
        return np.array([], dtype=float)
    ranges = []
    for i in range(0, len(data) - length + 1):
        window = data[i:i + length]
        ranges.append(float(np.max(window) - np.min(window)))
    return np.array(ranges, dtype=float)


def moving_range_sigma(data: np.ndarray, length: int = 2, method: str = "average") -> float:
    ranges = moving_range_values(data, length=length)
    if len(ranges) == 0:
        return np.nan
    d2 = d2_constant(length)
    if d2 <= 0:
        return np.nan
    if method == "median":
        center = float(np.median(ranges))
        # Minitab-like median moving range option normally needs a distribution-specific correction.
        # This implementation uses d2 as a practical first-phase approximation.
        return center / d2
    return float(np.mean(ranges)) / d2


def mssd_sigma(data: np.ndarray) -> float:
    if len(data) < 2:
        return np.nan
    diffs = np.diff(data)
    return math.sqrt(float(np.sum(diffs ** 2)) / (2.0 * (len(diffs))))


def subgroup_sigma_rbar(groups: List[np.ndarray], use_unbiasing: bool = True) -> float:
    ranges = []
    sizes = []
    for g in groups:
        if len(g) > 1:
            ranges.append(float(np.max(g) - np.min(g)))
            sizes.append(len(g))
    if not ranges:
        return np.nan

    if len(set(sizes)) == 1:
        n = sizes[0]
        divisor = d2_constant(n) if use_unbiasing else 1.0
        return float(np.mean(ranges)) / divisor if divisor > 0 else np.nan

    sigmas = []
    for r, n in zip(ranges, sizes):
        divisor = d2_constant(n) if use_unbiasing else 1.0
        if divisor > 0:
            sigmas.append(r / divisor)
    return float(np.mean(sigmas)) if sigmas else np.nan


def subgroup_sigma_sbar(groups: List[np.ndarray], use_unbiasing: bool = True) -> float:
    s_values = []
    sizes = []
    for g in groups:
        if len(g) > 1:
            s_values.append(float(np.std(g, ddof=1)))
            sizes.append(len(g))
    if not s_values:
        return np.nan

    if len(set(sizes)) == 1:
        n = sizes[0]
        divisor = c4_unbiasing_constant(n) if use_unbiasing else 1.0
        return float(np.mean(s_values)) / divisor if divisor > 0 else np.nan

    sigmas = []
    for s, n in zip(s_values, sizes):
        divisor = c4_unbiasing_constant(n) if use_unbiasing else 1.0
        if divisor > 0:
            sigmas.append(s / divisor)
    return float(np.mean(sigmas)) if sigmas else np.nan


def subgroup_sigma_pooled(groups: List[np.ndarray], use_unbiasing: bool = True) -> float:
    numerator = 0.0
    denominator = 0
    total_n = 0
    valid_groups = 0

    for g in groups:
        n = len(g)
        if n > 1:
            s = float(np.std(g, ddof=1))
            numerator += (n - 1) * (s ** 2)
            denominator += n - 1
            total_n += n
            valid_groups += 1

    if denominator <= 0:
        return np.nan

    pooled = math.sqrt(numerator / denominator)

    if use_unbiasing and total_n > valid_groups:
        # Effective degrees of freedom for pooled within sigma.
        df = total_n - valid_groups
        c4_eff = math.sqrt(2.0 / df) * math.gamma((df + 1.0) / 2.0) / math.gamma(df / 2.0)
        if c4_eff > 0:
            pooled = pooled / c4_eff

    return float(pooled)


def estimate_overall_sigma(data: np.ndarray, use_unbiasing: bool = False) -> float:
    if len(data) < 2:
        return np.nan
    sigma = float(np.std(data, ddof=1))
    if use_unbiasing:
        c4 = c4_unbiasing_constant(len(data))
        if c4 > 0:
            sigma = sigma / c4
    return sigma


def build_subgroups(
    data: np.ndarray,
    subgroup_mode: str,
    subgroup_size: int,
    subgroup_ids: Optional[pd.Series] = None,
) -> List[np.ndarray]:
    if subgroup_mode == "Constant":
        size = max(1, int(subgroup_size))
        groups = []
        for i in range(0, len(data), size):
            g = data[i:i + size]
            if len(g) > 0:
                groups.append(np.asarray(g, dtype=float))
        return groups

    if subgroup_mode == "Column":
        if subgroup_ids is None:
            return [np.asarray(data, dtype=float)]
        tmp = pd.DataFrame({"value": data, "subgroup": subgroup_ids.astype(str).values})
        groups = []
        for _, frame in tmp.groupby("subgroup", sort=False):
            values = pd.to_numeric(frame["value"], errors="coerce").dropna().to_numpy(dtype=float)
            if len(values) > 0:
                groups.append(values)
        return groups

    return [np.asarray(data, dtype=float)]


def estimate_within_sigma(
    data: np.ndarray,
    subgroup_mode: str = "Constant",
    subgroup_size: int = 1,
    subgroup_ids: Optional[pd.Series] = None,
    method_gt1: str = "Pooled standard deviation",
    method_eq1: str = "Average moving range",
    moving_range_length: int = 2,
    use_unbiasing_gt1: bool = True,
) -> Tuple[float, str, Dict[str, Any]]:
    groups = build_subgroups(data, subgroup_mode, subgroup_size, subgroup_ids)
    sizes = [len(g) for g in groups if len(g) > 0]
    max_size = max(sizes) if sizes else 1
    min_size = min(sizes) if sizes else 1

    details = {
        "subgroup_mode": subgroup_mode,
        "subgroup_size": subgroup_size,
        "subgroup_count": len(groups),
        "min_subgroup_size": min_size,
        "max_subgroup_size": max_size,
        "method_gt1": method_gt1,
        "method_eq1": method_eq1,
        "moving_range_length": moving_range_length,
        "use_unbiasing_gt1": use_unbiasing_gt1,
    }

    if max_size <= 1:
        if method_eq1 == "Median moving range":
            sigma = moving_range_sigma(data, length=moving_range_length, method="median")
            method_used = f"Median moving range (length={moving_range_length})"
        elif method_eq1 == "Square root of MSSD":
            sigma = mssd_sigma(data)
            method_used = "Square root of MSSD"
        else:
            sigma = moving_range_sigma(data, length=moving_range_length, method="average")
            method_used = f"Average moving range (length={moving_range_length})"
        return sigma, method_used, details

    if method_gt1 == "Rbar":
        sigma = subgroup_sigma_rbar(groups, use_unbiasing=use_unbiasing_gt1)
    elif method_gt1 == "Sbar":
        sigma = subgroup_sigma_sbar(groups, use_unbiasing=use_unbiasing_gt1)
    else:
        sigma = subgroup_sigma_pooled(groups, use_unbiasing=use_unbiasing_gt1)

    method_used = f"{method_gt1}" + (" with unbiasing constants" if use_unbiasing_gt1 else "")
    return sigma, method_used, details


# ─────────────────────────────────────────────
# Normality, outliers, stability
# ─────────────────────────────────────────────

def test_normality(data: np.ndarray) -> Dict[str, Any]:
    results: Dict[str, Any] = {}
    n = len(data)

    if n >= 3:
        try:
            stat, p = shapiro(data)
            results["Shapiro-Wilk"] = {
                "statistic": float(stat),
                "p_value": float(p),
                "normal": bool(p > 0.05),
                "label": f"W={stat:.4f}, p={p:.4f}",
            }
        except Exception as exc:
            results["Shapiro-Wilk"] = {"error": str(exc), "normal": False}

    if n >= 3:
        try:
            # SciPy 1.17+ emits a FutureWarning because the Anderson-Darling API
            # will move toward explicit p-value methods. For the current SPS logic,
            # we intentionally keep the classic 5% critical-value decision and
            # suppress the library migration warning in the desktop app.
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=FutureWarning, message=".*p-value calculation method.*")
                ad = anderson(data, dist="norm")
            crit = float(ad.critical_values[2])
            passed = bool(ad.statistic < crit)
            results["Anderson-Darling"] = {
                "statistic": float(ad.statistic),
                "critical_5pct": crit,
                "normal": passed,
                "label": f"A²={ad.statistic:.4f}, crit(5%)={crit:.4f}",
            }
        except Exception as exc:
            results["Anderson-Darling"] = {"error": str(exc), "normal": False}

    if n >= 8:
        try:
            stat, p = normaltest(data)
            results["D'Agostino-Pearson"] = {
                "statistic": float(stat),
                "p_value": float(p),
                "normal": bool(p > 0.05),
                "label": f"K²={stat:.4f}, p={p:.4f}",
            }
        except Exception as exc:
            results["D'Agostino-Pearson"] = {"error": str(exc), "normal": False}

    tests = [r for k, r in results.items() if k != "summary"]
    total = len(tests)
    passed_count = sum(1 for r in tests if r.get("normal", False))
    is_normal = bool(total > 0 and passed_count >= (total - 1))

    results["summary"] = {
        "is_normal": is_normal,
        "passed": int(passed_count),
        "total": int(total),
        "skewness": float(skew(data)) if n >= 3 else np.nan,
        "excess_kurtosis": float(kurtosis(data)) if n >= 4 else np.nan,
        "verdict": "Normal" if is_normal else "Non-Normal",
        "warning": (
            "" if is_normal else
            "Non-normal data detected. Cp/Cpk indices assume normality. Interpret with caution."
        ),
    }
    return results


def detect_outliers(data: np.ndarray, alpha: float = 0.05) -> Dict[str, Any]:
    n = len(data)
    mean = float(np.mean(data))
    std = float(np.std(data, ddof=1)) if n > 1 else np.nan

    if n > 2 and std > 0:
        G_stat = float(np.max(np.abs(data - mean)) / std)
        t_crit = float(t_dist.ppf(1 - alpha / (2 * n), df=n - 2))
        G_crit = float(((n - 1) / np.sqrt(n)) * np.sqrt(t_crit ** 2 / (n - 2 + t_crit ** 2)))
        grubbs_outlier = bool(G_stat > G_crit)
    else:
        G_stat = G_crit = np.nan
        grubbs_outlier = False

    q1, q3 = np.percentile(data, 25), np.percentile(data, 75)
    iqr = float(q3 - q1)
    lower_extreme = float(q1 - 3.0 * iqr)
    upper_extreme = float(q3 + 3.0 * iqr)
    lower_mild = float(q1 - 1.5 * iqr)
    upper_mild = float(q3 + 1.5 * iqr)

    iqr_extreme = data[(data < lower_extreme) | (data > upper_extreme)]
    iqr_mild = data[(data < lower_mild) | (data > upper_mild)]

    return {
        "grubbs": {
            "G_stat": G_stat,
            "G_critical": G_crit,
            "outlier_detected": grubbs_outlier,
            "label": f"G={fmt(G_stat, 4)}, crit={fmt(G_crit, 4)}",
        },
        "iqr_extreme": {
            "count": int(len(iqr_extreme)),
            "values": [float(x) for x in iqr_extreme.tolist()],
            "lower_fence": lower_extreme,
            "upper_fence": upper_extreme,
        },
        "iqr_mild": {
            "count": int(len(iqr_mild)),
            "values": [float(x) for x in iqr_mild.tolist()],
            "lower_fence": lower_mild,
            "upper_fence": upper_mild,
        },
        "recommendation": (
            "Extreme outliers detected — verify measurement validity."
            if len(iqr_extreme) > 0
            else "No extreme outliers detected."
        ),
    }


def _western_electric_rules(data: np.ndarray, ucl: float, lcl: float, cl: float, sigma: float) -> List[Tuple[int, str]]:
    signals: List[Tuple[int, str]] = []
    n = len(data)
    if n < 2 or sigma <= 0 or np.isnan(sigma):
        return signals

    z = [(float(d) - cl) / sigma for d in data]

    for i in range(n):
        if data[i] > ucl or data[i] < lcl:
            signals.append((i, "R1: Point beyond ±3σ"))

        if i >= 8:
            side = [1 if data[j] > cl else -1 for j in range(i - 8, i + 1)]
            if abs(sum(side)) == 9:
                signals.append((i, "R2: 9 points same side of CL"))

        if i >= 5:
            streak = data[i - 5:i + 1]
            if all(streak[k] < streak[k + 1] for k in range(5)) or all(streak[k] > streak[k + 1] for k in range(5)):
                signals.append((i, "R3: 6 points continuously trending"))

        if i >= 13:
            alt = all((data[j] > data[j + 1]) != (data[j + 1] > data[j + 2]) for j in range(i - 13, i - 1))
            if alt:
                signals.append((i, "R4: 14 points alternating up/down"))

        if i >= 2:
            seg = [z[j] for j in range(i - 2, i + 1)]
            if sum(1 for zv in seg if zv > 2) >= 2 or sum(1 for zv in seg if zv < -2) >= 2:
                signals.append((i, "R5: 2 of 3 points beyond ±2σ"))

        if i >= 4:
            seg = [z[j] for j in range(i - 4, i + 1)]
            if sum(1 for zv in seg if zv > 1) >= 4 or sum(1 for zv in seg if zv < -1) >= 4:
                signals.append((i, "R6: 4 of 5 points beyond ±1σ"))

    return signals


def calc_imr_control_limits(data: np.ndarray, moving_range_length: int = 2) -> Dict[str, Any]:
    if len(data) < 2:
        return {
            "X_chart": {"CL": np.nan, "UCL": np.nan, "LCL": np.nan},
            "MR_chart": {"CL": np.nan, "UCL": np.nan, "LCL": 0.0},
            "mr_values": np.array([], dtype=float),
            "signals": [],
            "in_control": True,
            "sigma_mr": np.nan,
        }

    mr = moving_range_values(data, length=max(2, moving_range_length))
    mr_bar = float(np.mean(mr)) if len(mr) > 0 else np.nan
    x_bar = float(np.mean(data))
    d2 = d2_constant(max(2, moving_range_length))
    sigma_mr = mr_bar / d2 if d2 > 0 else np.nan

    ucl_x = x_bar + 3 * sigma_mr if not np.isnan(sigma_mr) else np.nan
    lcl_x = x_bar - 3 * sigma_mr if not np.isnan(sigma_mr) else np.nan
    ucl_mr = 3.267 * mr_bar if not np.isnan(mr_bar) else np.nan
    lcl_mr = 0.0

    signals = _western_electric_rules(data, ucl_x, lcl_x, x_bar, sigma_mr)

    return {
        "X_chart": {"CL": x_bar, "UCL": ucl_x, "LCL": lcl_x},
        "MR_chart": {"CL": mr_bar, "UCL": ucl_mr, "LCL": lcl_mr},
        "mr_values": mr,
        "signals": signals,
        "in_control": len(signals) == 0,
        "sigma_mr": sigma_mr,
    }


# ─────────────────────────────────────────────
# Capability engine
# ─────────────────────────────────────────────

def ci_cp(cp: Any, n: int, alpha: float = 0.05) -> Tuple[Optional[float], Optional[float]]:
    if is_na(cp) or n < 3:
        return None, None
    df = n - 1
    lo = float(cp) * math.sqrt(float(chi2.ppf(alpha / 2, df)) / df)
    hi = float(cp) * math.sqrt(float(chi2.ppf(1 - alpha / 2, df)) / df)
    return lo, hi


def ci_cpk(cpk: Any, n: int, alpha: float = 0.05) -> Tuple[Optional[float], Optional[float]]:
    if is_na(cpk) or n < 4:
        return None, None
    cpk = float(cpk)
    z = float(norm.ppf(1 - alpha / 2))
    se = math.sqrt(1 / (9 * n) + cpk ** 2 / (2 * (n - 1)))
    return cpk - z * se, cpk + z * se


def minimum_n_for_cpk(target_cpk: float = 1.33, beta: float = 0.20, alpha: float = 0.05) -> int:
    z_a = float(norm.ppf(1 - alpha / 2))
    delta = target_cpk * 0.10
    n = int(math.ceil(0.5 * (z_a / delta) ** 2 + 1))
    return max(n, 30)


def sigma_level_from_cpk(cpk: Any, k_sigma: float = 6.0) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    if is_na(cpk):
        return None, None, None
    cpk = float(cpk)
    half_k = float(k_sigma) / 2.0 if k_sigma and k_sigma > 0 else 3.0
    sigma_st = cpk * half_k
    p_defect = float(norm.sf(half_k * cpk) * 2)
    dpmo = p_defect * 1_000_000
    yield_pct = (1 - p_defect) * 100
    return sigma_st, dpmo, yield_pct


def sigma_benchmark(sigma_st: Any) -> str:
    if is_na(sigma_st):
        return "N/A"
    sigma_st = float(sigma_st)
    if sigma_st >= 6.0:
        return "Six Sigma — World Class"
    if sigma_st >= 5.0:
        return "Excellent (5σ)"
    if sigma_st >= 4.0:
        return "Good Practice (4σ)"
    if sigma_st >= 3.0:
        return "Average Industry (3σ)"
    return "Below Average — Action Required"


def extended_descriptives(data: np.ndarray) -> Dict[str, Any]:
    n = len(data)
    mean = float(np.mean(data))
    std = float(np.std(data, ddof=1)) if n > 1 else np.nan
    return {
        "n": int(n),
        "mean": mean,
        "median": float(np.median(data)),
        "trimmed_mean": float(trim_mean(data, 0.05)) if n >= 3 else mean,
        "std": std,
        "variance": float(np.var(data, ddof=1)) if n > 1 else np.nan,
        "skewness": float(skew(data)) if n >= 3 else np.nan,
        "kurtosis_excess": float(kurtosis(data)) if n >= 4 else np.nan,
        "p0135": float(np.percentile(data, 0.135)),
        "p9865": float(np.percentile(data, 99.865)),
        "p25": float(np.percentile(data, 25)),
        "p75": float(np.percentile(data, 75)),
        "iqr": float(np.percentile(data, 75) - np.percentile(data, 25)),
        "cv_pct": float(std / abs(mean) * 100) if mean != 0 and not np.isnan(std) else None,
        "range": float(np.max(data) - np.min(data)),
        "min": float(np.min(data)),
        "max": float(np.max(data)),
    }


@dataclass
class CapabilityOptions:
    subgroup_mode: str = "Constant"  # Constant or Column
    subgroup_size: int = 1
    subgroup_column: str = ""
    method_gt1: str = "Pooled standard deviation"
    use_unbiasing_gt1: bool = True
    method_eq1: str = "Average moving range"
    moving_range_length: int = 2
    use_unbiasing_overall: bool = False
    run_within: bool = True
    run_overall: bool = True
    display_units: str = "Parts per million"
    capability_display: str = "Capability stats (Cp, Pp)"
    include_confidence_intervals: bool = False
    confidence_level: float = 95.0
    confidence_interval_type: str = "One-sided"
    study_k: float = 6.0
    title: str = ""


def calc_capability(
    data: np.ndarray,
    lsl: Optional[float] = None,
    usl: Optional[float] = None,
    target: Optional[float] = None,
    options: Optional[CapabilityOptions] = None,
    subgroup_ids: Optional[pd.Series] = None,
) -> Dict[str, Any]:
    options = options or CapabilityOptions()
    n = len(data)
    if n < 2:
        raise ValueError("At least 2 numeric data points are required.")

    mean = float(np.mean(data))
    std_overall = estimate_overall_sigma(data, use_unbiasing=options.use_unbiasing_overall)

    std_within, within_method, sigma_details = estimate_within_sigma(
        data=data,
        subgroup_mode=options.subgroup_mode,
        subgroup_size=options.subgroup_size,
        subgroup_ids=subgroup_ids,
        method_gt1=options.method_gt1,
        method_eq1=options.method_eq1,
        moving_range_length=options.moving_range_length,
        use_unbiasing_gt1=options.use_unbiasing_gt1,
    )

    obs_below = float(np.mean(data < lsl) * 100) if lsl is not None else None
    obs_above = float(np.mean(data > usl) * 100) if usl is not None else None
    if lsl is not None and usl is not None:
        obs_total = float(np.mean((data < lsl) | (data > usl)) * 100)
    elif lsl is not None:
        obs_total = obs_below
    elif usl is not None:
        obs_total = obs_above
    else:
        obs_total = None

    def exp_pct(lim, mu, sigma, tail="lower"):
        if lim is None or sigma is None or sigma <= 0 or np.isnan(sigma):
            return None
        if tail == "lower":
            return float(norm.cdf((lim - mu) / sigma) * 100)
        return float((1 - norm.cdf((lim - mu) / sigma)) * 100)

    def combine(a, b):
        if a is not None and b is not None:
            return float(a + b)
        return a if a is not None else b

    exp_ov_below = exp_pct(lsl, mean, std_overall, "lower")
    exp_ov_above = exp_pct(usl, mean, std_overall, "upper")
    exp_wi_below = exp_pct(lsl, mean, std_within, "lower")
    exp_wi_above = exp_pct(usl, mean, std_within, "upper")

    # Capability indices use the selected process spread multiplier K.
    # Default K=6 reproduces the classic formulas:
    # Cp/Pp = tolerance / (6*sigma), Cpk/Ppk side indices = distance / (3*sigma).
    k_sigma = float(options.study_k) if options.study_k and options.study_k > 0 else 6.0
    half_k = k_sigma / 2.0

    pp = ppl = ppu = ppk = cpm = None
    if options.run_overall:
        if lsl is not None and usl is not None and std_overall > 0:
            pp = (usl - lsl) / (k_sigma * std_overall)
        if lsl is not None and std_overall > 0:
            ppl = (mean - lsl) / (half_k * std_overall)
        if usl is not None and std_overall > 0:
            ppu = (usl - mean) / (half_k * std_overall)
        candidates = [x for x in [ppl, ppu] if x is not None]
        ppk = min(candidates) if candidates else None
        if lsl is not None and usl is not None and target is not None and std_overall > 0:
            denom = k_sigma * math.sqrt(std_overall ** 2 + (mean - target) ** 2)
            cpm = (usl - lsl) / denom if denom > 0 else None

    cp = cpl = cpu = cpk = None
    if options.run_within:
        if lsl is not None and usl is not None and std_within > 0:
            cp = (usl - lsl) / (k_sigma * std_within)
        if lsl is not None and std_within > 0:
            cpl = (mean - lsl) / (half_k * std_within)
        if usl is not None and std_within > 0:
            cpu = (usl - mean) / (half_k * std_within)
        wc = [x for x in [cpl, cpu] if x is not None]
        cpk = min(wc) if wc else None

    alpha = 1.0 - float(options.confidence_level) / 100.0
    cp_ci = ci_cp(cp, n, alpha) if options.include_confidence_intervals else (None, None)
    cpk_ci = ci_cpk(cpk, n, alpha) if options.include_confidence_intervals else (None, None)
    pp_ci = ci_cp(pp, n, alpha) if options.include_confidence_intervals else (None, None)
    ppk_ci = ci_cpk(ppk, n, alpha) if options.include_confidence_intervals else (None, None)

    sigma_st, dpmo_wi, yield_wi = sigma_level_from_cpk(cpk, k_sigma)
    sigma_lt, dpmo_ov, yield_ov = sigma_level_from_cpk(ppk, k_sigma)

    return {
        "n": int(n),
        "mean": mean,
        "std_overall": std_overall,
        "std_within": std_within,
        "within_sigma_method": within_method,
        "sigma_details": sigma_details,
        "min": float(np.min(data)),
        "max": float(np.max(data)),
        "lsl": lsl,
        "usl": usl,
        "target": target,
        "study_k": k_sigma,
        "observed": {"below": obs_below, "above": obs_above, "total": obs_total},
        "expected_overall": {
            "below": exp_ov_below,
            "above": exp_ov_above,
            "total": combine(exp_ov_below, exp_ov_above),
        },
        "expected_within": {
            "below": exp_wi_below,
            "above": exp_wi_above,
            "total": combine(exp_wi_below, exp_wi_above),
        },
        "overall_capability": {"Pp": pp, "PPL": ppl, "PPU": ppu, "Ppk": ppk, "Cpm": cpm},
        "within_capability": {"Cp": cp, "CPL": cpl, "CPU": cpu, "Cpk": cpk},
        "confidence_intervals": {
            "Cp": cp_ci,
            "Cpk": cpk_ci,
            "Pp": pp_ci,
            "Ppk": ppk_ci,
            "enabled": bool(options.include_confidence_intervals),
            "confidence_level": float(options.confidence_level),
            "type": options.confidence_interval_type,
        },
        "sigma_level": {
            "short_term": sigma_st,
            "long_term": sigma_lt,
            "dpmo_within": dpmo_wi,
            "dpmo_overall": dpmo_ov,
            "yield_within": yield_wi,
            "yield_overall": yield_ov,
            "benchmark": sigma_benchmark(sigma_st),
        },
        "summary": {
            "within_status": capability_status(cpk),
            "overall_status": capability_status(ppk),
            "minimum_n_for_cpk_133": minimum_n_for_cpk(1.33),
        },
    }


def generate_diagnosis(
    name: str,
    results: Dict[str, Any],
    normality: Dict[str, Any],
    control_data: Dict[str, Any],
    outliers: Dict[str, Any],
) -> List[str]:
    diagnosis = []
    cpk = results["within_capability"].get("Cpk")
    ppk = results["overall_capability"].get("Ppk")
    cp = results["within_capability"].get("Cp")
    cpm = results["overall_capability"].get("Cpm")

    diagnosis.append(f"Variable '{name}' was analyzed using normal capability assumptions.")

    if results["lsl"] is None and results["usl"] is None:
        diagnosis.append("No specification limits were provided. Capability indices cannot be interpreted as capability against customer requirements.")
    elif is_na(cpk):
        diagnosis.append("Cpk is not available because one or more required inputs are missing or within sigma could not be estimated.")
    elif cpk < 1.0:
        diagnosis.append("Cpk is below 1.00. The process is not capable and requires immediate centering and variation reduction actions.")
    elif cpk < 1.33:
        diagnosis.append("Cpk is between 1.00 and 1.33. The process is marginal and may not be robust for normal production variation.")
    elif cpk < 1.67:
        diagnosis.append("Cpk is at least 1.33. The process is generally capable, but improvement may still be needed for critical characteristics.")
    else:
        diagnosis.append("Cpk is at least 1.67. The process is performing at a strong capability level.")

    if not is_na(cp) and not is_na(cpk) and cp - cpk > 0.20:
        diagnosis.append("Cp is materially higher than Cpk. The process appears off-center relative to the specification limits.")

    if not is_na(cpk) and not is_na(ppk) and abs(cpk - ppk) > 0.20:
        diagnosis.append("Cpk and Ppk differ materially. Short-term and long-term variation are not aligned; check stability and drift.")

    if not normality.get("summary", {}).get("is_normal", False):
        diagnosis.append("Normality tests suggest non-normal data. Capability indices based on normality should be interpreted with caution.")

    if not control_data.get("in_control", True):
        diagnosis.append(f"I-MR stability checks detected {len(control_data.get('signals', []))} Western Electric signal(s). Investigate special causes before final capability approval.")

    if outliers.get("iqr_extreme", {}).get("count", 0) > 0:
        diagnosis.append("Extreme outliers were detected. Verify measurement validity, data collection and assignable causes.")

    if cpm is None and results["target"] is not None:
        diagnosis.append("Target was provided, but Cpm requires both LSL and USL. Cpm was reported as N/A.")

    return diagnosis


def build_text_report(
    name: str,
    results: Dict[str, Any],
    normality: Dict[str, Any],
    control_data: Dict[str, Any],
    outliers: Dict[str, Any],
    descriptives: Dict[str, Any],
    diagnosis: List[str],
    options: CapabilityOptions,
) -> str:
    d = 4
    s = []
    s.append("=" * 70)
    s.append("PROCESS CAPABILITY REPORT — SPS STUDIO")
    s.append(f"Variable: {name}")
    s.append("=" * 70)
    s.append("")

    s.append("PROCESS DATA")
    s.append("-" * 30)
    s.append(f"{'Sample N':<28}{results['n']}")
    s.append(f"{'Mean':<28}{fmt(results['mean'], d)}")
    s.append(f"{'StDev(Overall)':<28}{fmt(results['std_overall'], d)}")
    s.append(f"{'StDev(Within)':<28}{fmt(results['std_within'], d)}")
    s.append(f"{'Within sigma method':<28}{results.get('within_sigma_method', 'N/A')}")
    s.append(f"{'LSL':<28}{fmt(results['lsl'], d)}")
    s.append(f"{'Target':<28}{fmt(results['target'], d)}")
    s.append(f"{'USL':<28}{fmt(results['usl'], d)}")
    s.append("")

    s.append("OVERALL CAPABILITY")
    s.append("-" * 30)
    ov = results["overall_capability"]
    for key in ["Pp", "PPL", "PPU", "Ppk", "Cpm"]:
        s.append(f"{key:<28}{fmt(ov.get(key), d)}")
    s.append(f"{'Overall status':<28}{results['summary']['overall_status']}")
    s.append("")

    s.append("POTENTIAL WITHIN CAPABILITY")
    s.append("-" * 30)
    wi = results["within_capability"]
    for key in ["Cp", "CPL", "CPU", "Cpk"]:
        s.append(f"{key:<28}{fmt(wi.get(key), d)}")
    s.append(f"{'Within status':<28}{results['summary']['within_status']}")
    s.append("")

    s.append("PERFORMANCE")
    s.append("-" * 30)
    for block_name, block in [
        ("Observed", results["observed"]),
        ("Expected Overall", results["expected_overall"]),
        ("Expected Within", results["expected_within"]),
    ]:
        s.append(f"{block_name}:")
        s.append(f"  {'Below LSL':<22}{fmt(block.get('below'), d)} %")
        s.append(f"  {'Above USL':<22}{fmt(block.get('above'), d)} %")
        s.append(f"  {'Total':<22}{fmt(block.get('total'), d)} %")
    s.append("")

    s.append("SIGMA LEVEL")
    s.append("-" * 30)
    sl = results["sigma_level"]
    s.append(f"{'Short-term sigma':<28}{fmt(sl.get('short_term'), d)}")
    s.append(f"{'Long-term sigma':<28}{fmt(sl.get('long_term'), d)}")
    s.append(f"{'DPMO Within':<28}{fmt(sl.get('dpmo_within'), 2)}")
    s.append(f"{'Yield Within':<28}{fmt(sl.get('yield_within'), 4)} %")
    s.append(f"{'Benchmark':<28}{sl.get('benchmark', 'N/A')}")
    s.append("")

    s.append("NORMALITY")
    s.append("-" * 30)
    nsum = normality.get("summary", {})
    s.append(f"{'Verdict':<28}{nsum.get('verdict', 'N/A')}")
    s.append(f"{'Tests passed':<28}{nsum.get('passed', 0)} of {nsum.get('total', 0)}")
    s.append(f"{'Skewness':<28}{fmt(nsum.get('skewness'), d)}")
    s.append(f"{'Excess kurtosis':<28}{fmt(nsum.get('excess_kurtosis'), d)}")
    for test, row in normality.items():
        if test == "summary":
            continue
        s.append(f"{test:<28}{row.get('label', row.get('error', 'N/A'))}")
    s.append("")

    s.append("OUTLIER ANALYSIS")
    s.append("-" * 30)
    s.append(f"{'Grubbs':<28}{outliers['grubbs']['label']}")
    s.append(f"{'Grubbs outlier':<28}{'Yes' if outliers['grubbs']['outlier_detected'] else 'No'}")
    s.append(f"{'IQR mild count':<28}{outliers['iqr_mild']['count']}")
    s.append(f"{'IQR extreme count':<28}{outliers['iqr_extreme']['count']}")
    s.append(f"{'Recommendation':<28}{outliers['recommendation']}")
    s.append("")

    s.append("STABILITY / I-MR SIGNALS")
    s.append("-" * 30)
    s.append(f"{'In control':<28}{'Yes' if control_data.get('in_control', True) else 'No'}")
    s.append(f"{'Signal count':<28}{len(control_data.get('signals', []))}")
    for idx, msg in control_data.get("signals", [])[:25]:
        s.append(f"  Point {idx + 1}: {msg}")
    s.append("")

    s.append("DIAGNOSIS")
    s.append("-" * 30)
    for i, msg in enumerate(diagnosis, 1):
        s.append(f"{i}. {msg}")
    s.append("")

    s.append("OPTIONS")
    s.append("-" * 30)
    for key, value in asdict(options).items():
        s.append(f"{key:<28}{value}")

    return "\n".join(s)



# ─────────────────────────────────────────────
# Figure builders - Phase 3
# ─────────────────────────────────────────────

def _finite_or_none(value: Any) -> Optional[float]:
    return None if is_na(value) else float(value)


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


def figure_to_png_base64(fig: Figure) -> str:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
    buffer.seek(0)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _x_range_for_curves(data: np.ndarray, results: Dict[str, Any]) -> np.ndarray:
    points = [float(np.min(data)), float(np.max(data))]
    for k in ("lsl", "usl", "target"):
        if not is_na(results.get(k)):
            points.append(float(results[k]))
    lo, hi = min(points), max(points)
    span = hi - lo if hi > lo else 1.0
    return np.linspace(lo - 0.25 * span, hi + 0.25 * span, 500)



# ─────────────────────────────────────────────
# Figure option helpers
# ─────────────────────────────────────────────

def _options_dict(options: Any = None) -> Dict[str, Any]:
    if options is None:
        return {}
    if isinstance(options, CapabilityOptions):
        return asdict(options)
    if isinstance(options, dict):
        return options
    return {}


def _results_options(results: Dict[str, Any], options: Any = None) -> Dict[str, Any]:
    opts = _options_dict(options)
    if not opts and isinstance(results, dict):
        opts = _options_dict(results.get("options"))
    return opts


def _display_units(options: Any = None) -> str:
    opts = _options_dict(options)
    return str(opts.get("display_units", "Parts per million"))


def _capability_display(options: Any = None) -> str:
    opts = _options_dict(options)
    return str(opts.get("capability_display", "Capability stats (Cp, Pp)"))


def _ci_enabled(results: Dict[str, Any], options: Any = None) -> bool:
    opts = _options_dict(options)
    ci = results.get("confidence_intervals", {}) if isinstance(results, dict) else {}
    return bool(opts.get("include_confidence_intervals", ci.get("enabled", False)))


def _confidence_level(results: Dict[str, Any], options: Any = None) -> float:
    opts = _options_dict(options)
    ci = results.get("confidence_intervals", {}) if isinstance(results, dict) else {}
    try:
        return float(opts.get("confidence_level", ci.get("confidence_level", 95.0)))
    except Exception:
        return 95.0


def _is_ppm(options: Any = None) -> bool:
    return _display_units(options).lower().startswith("parts")


def _perf_multiplier(options: Any = None) -> float:
    return 10000.0 if _is_ppm(options) else 1.0


def _perf_prefix(options: Any = None) -> str:
    return "PPM" if _is_ppm(options) else "%"


def _perf_value(value: Any, options: Any = None, decimals: int = 2) -> str:
    if is_na(value):
        return "N/A"
    v = float(value) * _perf_multiplier(options)
    return fmt(v, decimals)


def _z_side(mean: Any, limit: Any, sigma: Any, side: str) -> Optional[float]:
    if is_na(mean) or is_na(limit) or is_na(sigma) or float(sigma) <= 0:
        return None
    if side == "lsl":
        return (float(mean) - float(limit)) / float(sigma)
    return (float(limit) - float(mean)) / float(sigma)


def _z_bench_from_expected_total(expected_total_pct: Any) -> Optional[float]:
    if is_na(expected_total_pct):
        return None
    p = max(0.0, min(0.999999999, float(expected_total_pct) / 100.0))
    if p <= 0:
        return np.inf
    try:
        return float(norm.ppf(1.0 - p))
    except Exception:
        return None


def _ci_lower(ci_tuple: Any) -> Optional[float]:
    try:
        lo, _hi = ci_tuple
        return None if is_na(lo) else float(lo)
    except Exception:
        return None


def build_capability_summary_figure(
    name: str,
    data: np.ndarray,
    results: Dict[str, Any],
    decimals: int = 2,
    options: Any = None,
) -> Figure:
    opts = _results_options(results, options)
    fig = Figure(figsize=(12.2, 8.2), dpi=120)
    fig.patch.set_facecolor("white")

    ci_on = _ci_enabled(results, opts)
    conf_level = _confidence_level(results, opts)
    cap_mode = _capability_display(opts)
    show_z = cap_mode.lower().startswith("benchmark")

    title = opts.get("title") or f"Process Capability Report for {name}"
    # Use a real Matplotlib Axes title instead of fig.text so the shared
    # plot interaction manager can edit this report title directly.
    ax_title = fig.add_axes([0.02, 0.860, 0.96, 0.080])
    ax_title.axis("off")
    ax_title.set_title(title, fontsize=18, fontweight="bold", pad=2)
    if ci_on:
        ax_title.text(0.5, 0.02, f"{fmt(conf_level, 0)}% Confidence", ha="center", va="bottom", fontsize=12, color="#555555")

    ax = fig.add_axes([0.27, 0.34, 0.43, 0.50])
    ax.set_facecolor("white")
    bins = min(max(8, int(np.sqrt(len(data)))), 18)
    ax.hist(data, bins=bins, density=True, alpha=0.65, edgecolor="black", color="#79A9DC")
    x = _x_range_for_curves(data, results)
    mu = results["mean"]
    s_ov = results["std_overall"]
    s_wi = results["std_within"]
    if not is_na(s_ov) and s_ov > 0:
        ax.plot(x, norm.pdf(x, mu, s_ov), linewidth=1.5, color="#A21B1B", label="Overall")
    if not is_na(s_wi) and s_wi > 0:
        ax.plot(x, norm.pdf(x, mu, s_wi), linestyle="--", linewidth=1.5, color="#666666", label="Within")
    ymax = ax.get_ylim()[1]
    for value, label in [(results.get("lsl"), "LSL"), (results.get("usl"), "USL")]:
        if not is_na(value):
            ax.axvline(value, linestyle="--", linewidth=1.0, color="red")
            ax.text(value, ymax * 1.01, label, ha="center", va="bottom", fontsize=8, fontweight="bold", color="darkred")
    if not is_na(results.get("target")):
        ax.axvline(results["target"], linestyle="--", linewidth=1.0, color="green")
        ax.text(results["target"], ymax * 1.01, "Target", ha="center", va="bottom", fontsize=8, fontweight="bold", color="darkgreen")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.02), frameon=False, fontsize=9)

    lx, lv, y, dy = 0.06, 0.18, 0.78, 0.028
    fig.text(lx, y, "Process Data", fontsize=11)
    rows = [("LSL", results.get("lsl")), ("Target", results.get("target")), ("USL", results.get("usl")),
            ("Sample Mean", results.get("mean")), ("Sample N", results.get("n")),
            ("StDev(Overall)", results.get("std_overall")), ("StDev(Within)", results.get("std_within"))]
    y -= 0.045
    for label, value in rows:
        fig.text(lx, y, label, fontsize=9)
        d = 0 if label == "Sample N" else max(decimals, 2)
        fig.text(lv, y, fmt(value, d), fontsize=9, family="monospace", ha="left")
        y -= dy

    rx, rv, y = 0.765, 0.965, 0.73
    ci = results.get("confidence_intervals", {})
    if show_z:
        fig.text(0.76, y, "Overall Capability", fontsize=9.2)
        y -= 0.034
        z_lsl_ov = _z_side(results.get("mean"), results.get("lsl"), results.get("std_overall"), "lsl")
        z_usl_ov = _z_side(results.get("mean"), results.get("usl"), results.get("std_overall"), "usl")
        z_bench_ov = _z_bench_from_expected_total(results.get("expected_overall", {}).get("total"))
        overall_rows = [("Z.Bench", z_bench_ov, "Ppk"), ("Z.LSL", z_lsl_ov, None), ("Z.USL", z_usl_ov, None), ("Ppk", results["overall_capability"].get("Ppk"), "Ppk"), ("Cpm", results["overall_capability"].get("Cpm"), None)]
        for key, value, ci_key in overall_rows:
            fig.text(rx, y, key, fontsize=9)
            fig.text(rv, y, fmt(value, decimals), fontsize=9, family="monospace", ha="right")
            y -= 0.024
            if ci_on and ci_key and _ci_lower(ci.get(ci_key)) is not None:
                lb = _ci_lower(ci.get(ci_key))
                if key.startswith("Z"):
                    lb = lb * (results.get("study_k", 6) / 2.0)
                fig.text(rx, y, f"LB for {key}", fontsize=8, color="#555555")
                fig.text(rv, y, fmt(lb, decimals), fontsize=8, family="monospace", ha="right", color="#555555")
                y -= 0.022
        y -= 0.014
        fig.text(0.72, y, "Potential (Within) Capability", fontsize=9.2)
        y -= 0.034
        z_lsl_wi = _z_side(results.get("mean"), results.get("lsl"), results.get("std_within"), "lsl")
        z_usl_wi = _z_side(results.get("mean"), results.get("usl"), results.get("std_within"), "usl")
        z_bench_wi = _z_bench_from_expected_total(results.get("expected_within", {}).get("total"))
        within_rows = [("Z.Bench", z_bench_wi, "Cpk"), ("Z.LSL", z_lsl_wi, None), ("Z.USL", z_usl_wi, None), ("Cpk", results["within_capability"].get("Cpk"), "Cpk")]
        for key, value, ci_key in within_rows:
            fig.text(rx, y, key, fontsize=9)
            fig.text(rv, y, fmt(value, decimals), fontsize=9, family="monospace", ha="right")
            y -= 0.024
            if ci_on and ci_key and _ci_lower(ci.get(ci_key)) is not None:
                lb = _ci_lower(ci.get(ci_key))
                if key.startswith("Z"):
                    lb = lb * (results.get("study_k", 6) / 2.0)
                fig.text(rx, y, f"LB for {key}", fontsize=8, color="#555555")
                fig.text(rv, y, fmt(lb, decimals), fontsize=8, family="monospace", ha="right", color="#555555")
                y -= 0.022
    else:
        fig.text(0.76, y, "Overall Capability", fontsize=9.2)
        y -= 0.034
        for key in ["Pp", "PPL", "PPU", "Ppk", "Cpm"]:
            fig.text(rx, y, key, fontsize=9)
            fig.text(rv, y, fmt(results["overall_capability"].get(key), decimals), fontsize=9, family="monospace", ha="right")
            y -= 0.024
            if ci_on and key in ("Pp", "Ppk") and _ci_lower(ci.get(key)) is not None:
                fig.text(rx, y, f"LB for {key}", fontsize=8, color="#555555")
                fig.text(rv, y, fmt(_ci_lower(ci.get(key)), decimals), fontsize=8, family="monospace", ha="right", color="#555555")
                y -= 0.022
        y -= 0.014
        fig.text(0.72, y, "Potential (Within) Capability", fontsize=9.2)
        y -= 0.034
        for key in ["Cp", "CPL", "CPU", "Cpk"]:
            fig.text(rx, y, key, fontsize=9)
            fig.text(rv, y, fmt(results["within_capability"].get(key), decimals), fontsize=9, family="monospace", ha="right")
            y -= 0.024
            if ci_on and key in ("Cp", "Cpk") and _ci_lower(ci.get(key)) is not None:
                fig.text(rx, y, f"LB for {key}", fontsize=8, color="#555555")
                fig.text(rv, y, fmt(_ci_lower(ci.get(key)), decimals), fontsize=8, family="monospace", ha="right", color="#555555")
                y -= 0.022

    perf_ax = fig.add_axes([0.09, 0.08, 0.70, 0.16])
    perf_ax.axis("off")
    perf_ax.text(0.38, 0.92, "Performance", fontsize=11, ha="center")
    headers = ["", "Observed", "Expected Overall", "Expected Within"]
    col_x = [0.02, 0.38, 0.68, 0.98]
    for xh, h in zip(col_x, headers):
        perf_ax.text(xh, 0.66, h, fontsize=8.5, family="monospace", ha="right" if h else "left")
    prefix = _perf_prefix(opts)
    rows = [(f"{prefix} < LSL", "below"), (f"{prefix} > USL", "above"), (f"{prefix} Total", "total")]
    for yy, (label, key) in zip([0.42, 0.22, 0.02], rows):
        perf_ax.text(col_x[0], yy, label, fontsize=9, ha="left")
        perf_ax.text(col_x[1], yy, _perf_value(results["observed"].get(key), opts, decimals), fontsize=9, family="monospace", ha="right")
        perf_ax.text(col_x[2], yy, _perf_value(results["expected_overall"].get(key), opts, decimals), fontsize=9, family="monospace", ha="right")
        perf_ax.text(col_x[3], yy, _perf_value(results["expected_within"].get(key), opts, decimals), fontsize=9, family="monospace", ha="right")
    spread_note = "The actual process spread is represented by " + fmt(results.get("study_k", 6), 2) + " sigma."
    fig.text(0.05, 0.035, spread_note, fontsize=10, style="italic")
    return fig

def build_capability_full_figure(name: str, data: np.ndarray, results: Dict[str, Any], normality: Dict[str, Any], control: Dict[str, Any], outliers: Dict[str, Any], descriptives: Dict[str, Any], decimals: int = 2, options: Any = None) -> Figure:
    opts = _results_options(results, options)
    fig = Figure(figsize=(16.8, 9.8), dpi=105)
    fig.patch.set_facecolor("white")
    # Left/middle area holds plots. Right column is manually allocated with
    # taller axes so text tables do not overlap when rendered inside SPS Studio.
    gs = fig.add_gridspec(3, 2, hspace=0.60, wspace=0.45, left=0.055, right=0.705, top=0.84, bottom=0.065)
    ax_hist = fig.add_subplot(gs[0, 0])
    ax_qq = fig.add_subplot(gs[0, 1])
    ax_i = fig.add_subplot(gs[1, :])
    ax_mr = fig.add_subplot(gs[2, :])
    ax_stats = fig.add_axes([0.735, 0.585, 0.235, 0.325])
    ax_right = fig.add_axes([0.735, 0.065, 0.235, 0.455])

    ax_title = fig.add_axes([0.02, 0.885, 0.96, 0.070])
    ax_title.axis("off")
    ax_title.set_title(f"Process Capability Report — {name}", fontsize=15, fontweight="bold", pad=2)
    ax_title.text(
        0.5, 0.04,
        f"Stability: {'IN CONTROL' if control.get('in_control') else 'OUT OF CONTROL'} | "
        f"Normality: {normality['summary']['verdict']} | n={results['n']} | Mean={fmt(results['mean'], decimals)}",
        ha="center", va="bottom", fontsize=9
    )

    # Histogram
    ax_hist.set_facecolor("white")
    bins = min(max(8, int(np.sqrt(len(data)))), 22)
    ax_hist.hist(data, bins=bins, density=True, alpha=0.65, edgecolor="black", linewidth=0.5, color="#79A9DC")
    x = _x_range_for_curves(data, results)
    mu, s_ov, s_wi = results["mean"], results["std_overall"], results["std_within"]
    if not is_na(s_ov) and s_ov > 0:
        ax_hist.plot(x, norm.pdf(x, mu, s_ov), lw=1.6, color="#A21B1B", label="Overall")
    if not is_na(s_wi) and s_wi > 0:
        ax_hist.plot(x, norm.pdf(x, mu, s_wi), lw=1.6, ls="--", color="#666666", label="Within")
    ymax = ax_hist.get_ylim()[1]
    for value, label in [(results.get("lsl"), "LSL"), (results.get("usl"), "USL")]:
        if not is_na(value):
            ax_hist.axvline(value, lw=1.1, ls="--", color="red")
            ax_hist.text(value, ymax * 1.01, label, ha="center", fontsize=8, fontweight="bold", color="darkred")
    if not is_na(results.get("target")):
        ax_hist.axvline(results["target"], lw=1.1, ls="--", color="green")
        ax_hist.text(results["target"], ymax * 1.01, "Target", ha="center", fontsize=8, fontweight="bold", color="darkgreen")
    ax_hist.set_title("Histogram + Fit Curves", fontsize=10, fontweight="bold")
    ax_hist.legend(fontsize=8, frameon=False)
    ax_hist.grid(True, alpha=0.2)

    # QQ
    ax_qq.set_facecolor("white")
    try:
        (osm, osr), (slope, intercept, r_val) = probplot(data, dist="norm", fit=True)
        ax_qq.scatter(osm, osr, s=14, alpha=0.7)
        line_y = slope * osm + intercept
        ax_qq.plot(osm, line_y, lw=1.5, label=f"R²={r_val**2:.4f}")
        ax_qq.legend(fontsize=8, frameon=False)
    except Exception as exc:
        ax_qq.text(0.5, 0.5, f"Q-Q plot unavailable\n{exc}", ha="center", va="center")
    ax_qq.set_title("Normal Q-Q Plot", fontsize=10, fontweight="bold")
    ax_qq.grid(True, alpha=0.2)

    # Capability table - written as fixed columns in a taller right-side axes.
    ax_stats.axis("off")
    ax_stats.set_xlim(0, 1)
    ax_stats.set_ylim(0, 1)
    ax_stats.set_title("Capability Indices", fontsize=10, fontweight="bold", pad=6)

    def cap_header(ypos, text):
        ax_stats.text(0.02, ypos, text, fontsize=8.8, fontweight="bold", va="center")

    def cap_row(ypos, label, value, status=None):
        ax_stats.text(0.05, ypos, label, fontsize=7.7, va="center")
        ax_stats.text(0.58, ypos, value, fontsize=7.7, family="monospace", ha="right", va="center")
        if status:
            ax_stats.text(0.98, ypos, status, fontsize=6.8, ha="right", va="center", fontweight="bold", color=status_color(status))

    y = 0.92
    cap_header(y, "Overall Capability"); y -= 0.050
    for k in ["Pp", "PPL", "PPU", "Ppk", "Cpm"]:
        val = results["overall_capability"].get(k)
        cap_row(y, k, fmt(val, decimals), capability_status(val) if k in ("Pp", "Ppk") else None)
        y -= 0.040

    y -= 0.025
    cap_header(y, "Within Capability"); y -= 0.050
    for k in ["Cp", "CPL", "CPU", "Cpk"]:
        val = results["within_capability"].get(k)
        cap_row(y, k, fmt(val, decimals), capability_status(val) if k in ("Cp", "Cpk") else None)
        y -= 0.040

    y -= 0.025
    sl = results["sigma_level"]
    cap_header(y, "Sigma / Performance"); y -= 0.050
    cap_row(y, "Short-term σ", fmt(sl.get("short_term"), decimals)); y -= 0.040
    cap_row(y, "DPMO", f"{sl.get('dpmo_within'):,.0f}" if not is_na(sl.get("dpmo_within")) else "N/A"); y -= 0.040
    cap_row(y, "Yield %", f"{fmt(sl.get('yield_within'), decimals)}%" if not is_na(sl.get("yield_within")) else "N/A")

    # I chart
    ax_i.set_facecolor("white")
    xidx = np.arange(len(data))
    signal_set = {idx for idx, _ in control.get("signals", [])}
    ax_i.plot(xidx, data, lw=0.9, alpha=0.7)
    ax_i.scatter(xidx, data, s=18)
    for val, label, ls in [(control["X_chart"].get("UCL"), "UCL", "--"), (control["X_chart"].get("CL"), "CL", "-"), (control["X_chart"].get("LCL"), "LCL", "--")]:
        if not is_na(val):
            ax_i.axhline(val, ls=ls, lw=1.1)
            ax_i.text(len(data) + 0.5, val, f"{label}={fmt(val, decimals)}", fontsize=7, va="center")
    for idx in signal_set:
        if 0 <= idx < len(data):
            ax_i.scatter([idx], [data[idx]], s=38, zorder=5)
    ax_i.set_title(f"Individuals (I) Chart ({len(signal_set)} signal points)", fontsize=10, fontweight="bold")
    ax_i.grid(True, alpha=0.2)

    # MR chart
    ax_mr.set_facecolor("white")
    mr = np.asarray(control.get("mr_values", []), dtype=float)
    if len(mr) > 0:
        ax_mr.plot(np.arange(1, len(mr) + 1), mr, lw=0.9, alpha=0.7)
        ax_mr.scatter(np.arange(1, len(mr) + 1), mr, s=14)
    for val, label, ls in [(control["MR_chart"].get("UCL"), "UCL", "--"), (control["MR_chart"].get("CL"), "MRbar", "-")]:
        if not is_na(val):
            ax_mr.axhline(val, ls=ls, lw=1.1)
            ax_mr.text(max(1, len(mr)) + 0.5, val, f"{label}={fmt(val, decimals)}", fontsize=7, va="center")
    ax_mr.set_ylim(bottom=0)
    ax_mr.set_title("Moving Range (MR) Chart", fontsize=10, fontweight="bold")
    ax_mr.grid(True, alpha=0.2)

    # Diagnostics panel
    ax_right.axis("off")
    ax_right.set_xlim(0, 1)
    ax_right.set_ylim(0, 1)
    ax_right.set_title("Performance & Diagnostics", fontsize=10, fontweight="bold", pad=6)
    y = 0.96

    def hdr(t):
        nonlocal y
        ax_right.text(0.04, y, t, fontsize=9.0, fontweight="bold", va="top")
        y -= 0.052

    def txt(l, v, fs=7.4):
        nonlocal y
        ax_right.text(0.04, y, l, fontsize=fs, va="center")
        ax_right.text(0.98, y, v, fontsize=fs, family="monospace", ha="right", va="center")
        y -= 0.038

    hdr("Performance")
    prefix = _perf_prefix(opts)
    txt("Metric", "Observed | Overall | Within", fs=7.1)
    for label, key in [(f"{prefix} < LSL", "below"), (f"{prefix} > USL", "above"), (f"{prefix} Total", "total")]:
        txt(label, f"{_perf_value(results['observed'].get(key), opts, decimals)} | {_perf_value(results['expected_overall'].get(key), opts, decimals)} | {_perf_value(results['expected_within'].get(key), opts, decimals)}", fs=7.0)

    y -= 0.020
    hdr("Normality Tests")
    for test, rowd in normality.items():
        if test == "summary":
            continue
        label = test.replace("D'Agostino-Pearson", "D'Agostino")
        txt(label, f"{'PASS' if rowd.get('normal') else 'FAIL'}", fs=7.2)
        lab = str(rowd.get("label", ""))[:38]
        if lab:
            txt("", lab, fs=6.8)
    txt("Verdict", normality["summary"]["verdict"], fs=7.3)

    y -= 0.020
    hdr("Outlier Analysis")
    txt("Grubbs", outliers["grubbs"].get("label", "N/A")[:34], fs=7.0)
    txt("Extreme outliers", str(outliers["iqr_extreme"].get("count", 0)), fs=7.2)
    txt("Mild outliers", str(outliers["iqr_mild"].get("count", 0)), fs=7.2)

    y -= 0.020
    hdr("Extended Descriptives")
    for k in ["median", "iqr", "cv_pct", "variance", "p0135", "p9865"]:
        txt(k, fmt(descriptives.get(k), decimals), fs=7.1)
    return fig


def build_dashboard_figure(reports: List[Dict[str, Any]], decimals: int = 2, options: Any = None) -> Figure:
    n_vars = max(1, len(reports))
    fig = Figure(figsize=(24, 3.2 + n_vars * 0.85), dpi=115)
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0.025, 0.08, 0.95, 0.80])
    ax.axis("off")
    ax.set_title("Executive Dashboard — Process Capability Summary", fontsize=15, fontweight="bold", pad=14)

    headers = [
        "CTQ / Variable", "n", "Mean", "StDev\n(Ov)", "StDev\n(Wi)",
        "Cp", "Cpk", "Pp", "Ppk", "Z σ", "DPMO", "Yield %",
        "Normality", "Stability", "Status"
    ]

    rows = []
    for item in reports:
        r = item["results"]
        sl = r["sigma_level"]
        status = r["summary"].get("within_status", "N/A")
        stability = "OK" if item.get("control", {}).get("in_control", True) else f"OOC ({len(item.get('control', {}).get('signals', []))})"
        rows.append([
            item.get("name", item.get("measurement_column", "Variable"))[:24],
            str(r.get("n", "")),
            fmt(r.get("mean"), decimals),
            fmt(r.get("std_overall"), decimals),
            fmt(r.get("std_within"), decimals),
            fmt(r["within_capability"].get("Cp"), decimals),
            fmt(r["within_capability"].get("Cpk"), decimals),
            fmt(r["overall_capability"].get("Pp"), decimals),
            fmt(r["overall_capability"].get("Ppk"), decimals),
            fmt(sl.get("short_term"), decimals),
            f"{sl.get('dpmo_within'):,.0f}" if not is_na(sl.get("dpmo_within")) else "N/A",
            f"{fmt(sl.get('yield_within'), decimals)}%" if not is_na(sl.get("yield_within")) else "N/A",
            item.get("normality", {}).get("summary", {}).get("verdict", "N/A"),
            stability,
            status,
        ])

    col_widths = [0.160, 0.035, 0.055, 0.060, 0.060, 0.043, 0.043, 0.043, 0.043, 0.050, 0.085, 0.075, 0.080, 0.080, 0.088]
    table = ax.table(
        cellText=rows,
        colLabels=headers,
        cellLoc="center",
        colLoc="center",
        colWidths=col_widths,
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(6.8)
    table.scale(1.0, 1.85)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#C8D2DD")
        cell.set_linewidth(0.6)
        if row == 0:
            cell.set_facecolor("#EEF3F8")
            cell.set_text_props(weight="bold", color="#1F2D3D")
        else:
            cell.set_facecolor("white")
            if col == 0:
                cell.set_text_props(ha="left")
            if col == 14:
                status = rows[row - 1][14]
                cell.set_text_props(weight="bold", color=status_color(status))
            if col == 12 and rows[row - 1][12] == "Non-Normal":
                cell.set_text_props(color="#B7791F", weight="bold")
            if col == 13 and rows[row - 1][13].startswith("OOC"):
                cell.set_text_props(color="#C53030", weight="bold")

    return fig

# ─────────────────────────────────────────────
# PySide6 builder
# ─────────────────────────────────────────────

class CapabilityStudyBuilder(QWidget):
    def __init__(self, app_window=None, parent=None):
        super().__init__(parent)
        self.app_window = app_window
        self.result: Optional[Dict[str, Any]] = None
        self.current_summary = ""
        self.current_data = pd.DataFrame()
        self.current_journal_entry_id = None
        self.current_figure: Optional[Figure] = None
        self.current_graph_type = "Capability Summary"
        self._plot_manager = None

        self.setWindowTitle("Capability Study Builder")
        self.resize(1200, 780)

        self._build_ui()
        self.refresh_columns()
        self.log("Capability Study Builder initialized.")

    # ─────────────────────────────────────────────
    # UI
    # ─────────────────────────────────────────────

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        header = QHBoxLayout()
        title = QLabel("Capability Study Builder")
        title.setStyleSheet("font-size: 18px; font-weight: bold;")
        header.addWidget(title)
        header.addStretch()
        self.lbl_worksheet = QLabel("Worksheet: N/A")
        header.addWidget(self.lbl_worksheet)
        root.addLayout(header)

        self.tabs = QTabWidget()
        root.addWidget(self.tabs)

        self.setup_tab = QWidget()
        self.results_tab = QWidget()
        self.graphs_tab = QWidget()
        self.summary_tab = QWidget()
        self.debug_tab = QWidget()

        self.tabs.addTab(self.setup_tab, "Setup")
        self.tabs.addTab(self.results_tab, "Results")
        self.tabs.addTab(self.graphs_tab, "Graphs")
        self.tabs.addTab(self.summary_tab, "Summary")
        self.tabs.addTab(self.debug_tab, "Debug")

        self._build_setup_tab()
        self._build_results_tab()
        self._build_graphs_tab()
        self._build_summary_tab()
        self._build_debug_tab()

    def _build_setup_tab(self):
        """Single-screen setup, aligned with the Gage Study workflow.

        Sections:
        1. Variables and specification limits
        2. Sigma estimation methods
        3. Analysis options
        4. Run capability analysis
        """
        layout = QVBoxLayout(self.setup_tab)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(12)

        top_grid = QGridLayout()
        top_grid.setHorizontalSpacing(14)
        top_grid.setVerticalSpacing(12)
        layout.addLayout(top_grid)

        # ─────────────────────────────────────────────
        # 1. Variables and limits
        # ─────────────────────────────────────────────
        variables_group = QGroupBox("1. Variables and Specification Limits")
        variables_grid = QGridLayout(variables_group)
        variables_grid.setColumnStretch(1, 1)

        self.cmb_measurement = QComboBox()
        self.cmb_subgroup_column = QComboBox()
        self.cmb_subgroup_column.addItem("")

        self.rad_single_column = QRadioButton("Single column")
        self.rad_single_column.setChecked(True)

        self.txt_arranged_columns = QLineEdit()
        self.txt_arranged_columns.setPlaceholderText("Example: Diameter_1 Diameter_2 Diameter_3")
        self.txt_arranged_columns.setEnabled(False)

        self.rad_subgroup_constant = QRadioButton("Use constant subgroup size")
        self.rad_subgroup_constant.setChecked(True)
        self.rad_subgroup_column = QRadioButton("Use subgroup ID column")

        self.subgroup_mode_group = QButtonGroup(self)
        self.subgroup_mode_group.addButton(self.rad_subgroup_constant)
        self.subgroup_mode_group.addButton(self.rad_subgroup_column)

        self.spn_subgroup_size = QSpinBox()
        self.spn_subgroup_size.setRange(1, 10000)
        self.spn_subgroup_size.setValue(1)

        self.txt_lsl = QLineEdit()
        self.txt_usl = QLineEdit()
        self.txt_target = QLineEdit()
        self.txt_lsl.setPlaceholderText("Optional")
        self.txt_usl.setPlaceholderText("Optional")
        self.txt_target.setPlaceholderText("Optional")

        self.spn_decimals = QSpinBox()
        self.spn_decimals.setRange(0, 8)
        self.spn_decimals.setValue(2)

        variables_grid.addWidget(QLabel("Measurement column"), 0, 0)
        variables_grid.addWidget(self.cmb_measurement, 0, 1)
        variables_grid.addWidget(QLabel("Data are arranged as"), 1, 0)
        variables_grid.addWidget(self.rad_single_column, 1, 1)
        variables_grid.addWidget(QLabel("Optional arranged columns"), 2, 0)
        variables_grid.addWidget(self.txt_arranged_columns, 2, 1)
        variables_grid.addWidget(QLabel("Subgrouping"), 3, 0)
        variables_grid.addWidget(self.rad_subgroup_constant, 3, 1)
        variables_grid.addWidget(QLabel("Subgroup size"), 4, 0)
        variables_grid.addWidget(self.spn_subgroup_size, 4, 1)
        variables_grid.addWidget(self.rad_subgroup_column, 5, 1)
        variables_grid.addWidget(QLabel("Subgroup column"), 6, 0)
        variables_grid.addWidget(self.cmb_subgroup_column, 6, 1)
        variables_grid.addWidget(QLabel("LSL"), 7, 0)
        variables_grid.addWidget(self.txt_lsl, 7, 1)
        variables_grid.addWidget(QLabel("USL"), 8, 0)
        variables_grid.addWidget(self.txt_usl, 8, 1)
        variables_grid.addWidget(QLabel("Target"), 9, 0)
        variables_grid.addWidget(self.txt_target, 9, 1)
        variables_grid.addWidget(QLabel("Decimal places"), 10, 0)
        variables_grid.addWidget(self.spn_decimals, 10, 1)

        # ─────────────────────────────────────────────
        # 2. Method options
        # ─────────────────────────────────────────────
        methods_group = QGroupBox("2. Method Options")
        methods_layout = QVBoxLayout(methods_group)

        grp_gt1 = QGroupBox("Within subgroup standard deviation — subgroup size > 1")
        gt1_grid = QGridLayout(grp_gt1)
        self.rad_rbar = QRadioButton("Rbar")
        self.rad_sbar = QRadioButton("Sbar")
        self.rad_pooled = QRadioButton("Pooled standard deviation")
        self.rad_pooled.setChecked(True)
        self.method_gt1_group = QButtonGroup(self)
        self.method_gt1_group.addButton(self.rad_rbar)
        self.method_gt1_group.addButton(self.rad_sbar)
        self.method_gt1_group.addButton(self.rad_pooled)
        self.chk_unbiasing_gt1 = QCheckBox("Use unbiasing constants")
        self.chk_unbiasing_gt1.setChecked(True)
        gt1_grid.addWidget(self.rad_rbar, 0, 0)
        gt1_grid.addWidget(self.chk_unbiasing_gt1, 0, 1)
        gt1_grid.addWidget(self.rad_sbar, 1, 0)
        gt1_grid.addWidget(self.rad_pooled, 2, 0)
        methods_layout.addWidget(grp_gt1)

        grp_eq1 = QGroupBox("Within subgroup standard deviation — subgroup size = 1")
        eq1_grid = QGridLayout(grp_eq1)
        self.rad_avg_mr = QRadioButton("Average moving range")
        self.rad_avg_mr.setChecked(True)
        self.rad_median_mr = QRadioButton("Median moving range")
        self.rad_mssd = QRadioButton("Square root of MSSD")
        self.method_eq1_group = QButtonGroup(self)
        self.method_eq1_group.addButton(self.rad_avg_mr)
        self.method_eq1_group.addButton(self.rad_median_mr)
        self.method_eq1_group.addButton(self.rad_mssd)
        self.spn_mr_length = QSpinBox()
        self.spn_mr_length.setRange(2, 100)
        self.spn_mr_length.setValue(2)
        eq1_grid.addWidget(self.rad_avg_mr, 0, 0)
        eq1_grid.addWidget(QLabel("Use moving range of length"), 0, 1)
        eq1_grid.addWidget(self.spn_mr_length, 0, 2)
        eq1_grid.addWidget(self.rad_median_mr, 1, 0)
        eq1_grid.addWidget(self.rad_mssd, 2, 0)
        methods_layout.addWidget(grp_eq1)

        self.chk_unbiasing_overall = QCheckBox("Use unbiasing constants to calculate overall standard deviation")
        self.chk_unbiasing_overall.setChecked(False)
        methods_layout.addWidget(self.chk_unbiasing_overall)
        methods_layout.addStretch()

        # ─────────────────────────────────────────────
        # 3. Analysis options
        # ─────────────────────────────────────────────
        analysis_group = QGroupBox("3. Analysis Options")
        analysis_grid = QGridLayout(analysis_group)
        analysis_grid.setColumnStretch(1, 1)
        analysis_grid.setColumnStretch(3, 1)

        self.txt_title = QLineEdit()
        self.txt_title.setPlaceholderText("Optional report title")

        self.spn_study_k = QDoubleSpinBox()
        self.spn_study_k.setRange(0.1, 100.0)
        self.spn_study_k.setDecimals(3)
        self.spn_study_k.setValue(6.0)

        self.chk_within_analysis = QCheckBox("Within subgroup analysis")
        self.chk_within_analysis.setChecked(True)
        self.chk_overall_analysis = QCheckBox("Overall analysis")
        self.chk_overall_analysis.setChecked(True)

        self.rad_ppm = QRadioButton("Parts per million")
        self.rad_percent = QRadioButton("Percents")
        self.rad_percent.setChecked(True)
        self.display_group = QButtonGroup(self)
        self.display_group.addButton(self.rad_ppm)
        self.display_group.addButton(self.rad_percent)

        self.rad_capability_stats = QRadioButton("Capability stats (Cp, Pp)")
        self.rad_capability_stats.setChecked(True)
        self.rad_benchmark_z = QRadioButton("Benchmark Z's (z level)")
        self.capability_display_group = QButtonGroup(self)
        self.capability_display_group.addButton(self.rad_capability_stats)
        self.capability_display_group.addButton(self.rad_benchmark_z)

        self.chk_ci = QCheckBox("Include confidence intervals")
        self.chk_ci.setChecked(False)
        self.spn_confidence = QDoubleSpinBox()
        self.spn_confidence.setRange(50.0, 99.99)
        self.spn_confidence.setDecimals(2)
        self.spn_confidence.setValue(95.0)
        self.spn_confidence.setEnabled(False)

        self.cmb_ci_type = QComboBox()
        self.cmb_ci_type.addItems(["One-sided", "Two-sided"])
        self.cmb_ci_type.setCurrentText("One-sided")
        self.cmb_ci_type.setEnabled(False)

        self.chk_auto_journal = QCheckBox("Auto journal save")
        self.chk_auto_journal.setChecked(True)
        self.chk_auto_journal.setEnabled(True)
        self.chk_auto_journal.setToolTip("Automatically save this analysis to the Analysis Journal after each run.")

        analysis_grid.addWidget(QLabel("Title"), 0, 0)
        analysis_grid.addWidget(self.txt_title, 0, 1, 1, 3)
        analysis_grid.addWidget(QLabel("Use tolerance of K × σ for capability statistics. K ="), 1, 0)
        analysis_grid.addWidget(self.spn_study_k, 1, 1)
        analysis_grid.addWidget(QLabel("Perform Analysis"), 2, 0)
        analysis_grid.addWidget(self.chk_within_analysis, 2, 1)
        analysis_grid.addWidget(self.chk_overall_analysis, 3, 1)
        analysis_grid.addWidget(QLabel("Display"), 2, 2)
        analysis_grid.addWidget(self.rad_ppm, 2, 3)
        analysis_grid.addWidget(self.rad_percent, 3, 3)
        analysis_grid.addWidget(self.rad_capability_stats, 4, 3)
        analysis_grid.addWidget(self.rad_benchmark_z, 5, 3)
        analysis_grid.addWidget(self.chk_ci, 6, 3)
        analysis_grid.addWidget(QLabel("Confidence level"), 7, 2)
        analysis_grid.addWidget(self.spn_confidence, 7, 3)
        analysis_grid.addWidget(QLabel("Confidence intervals"), 8, 2)
        analysis_grid.addWidget(self.cmb_ci_type, 8, 3)
        analysis_grid.addWidget(self.chk_auto_journal, 9, 3)

        # ─────────────────────────────────────────────
        # 4. Run
        # ─────────────────────────────────────────────
        run_group = QGroupBox("4. Run Capability Analysis")
        run_layout = QVBoxLayout(run_group)
        run_buttons = QHBoxLayout()
        self.btn_refresh = QPushButton("Refresh Columns")
        self.btn_run = QPushButton("Run Capability Analysis")
        self.btn_run.setMinimumHeight(38)
        self.btn_run.setStyleSheet("""
            QPushButton {
                background-color: #2F5D87;
                color: white;
                font-weight: bold;
                border: 1px solid #244B70;
                border-radius: 5px;
                padding: 8px 12px;
            }
            QPushButton:hover { background-color: #366D9D; }
            QPushButton:pressed { background-color: #244B70; }
        """)
        run_buttons.addWidget(self.btn_refresh)
        run_buttons.addWidget(self.btn_run, 1)
        run_layout.addLayout(run_buttons)
        note = QLabel(
            "Required: one numeric measurement column and at least one specification limit for capability interpretation. "
            "Subgroup size can be a constant value or a subgroup ID column."
        )
        note.setWordWrap(True)
        note.setStyleSheet("color:#5B6F8A;")
        run_layout.addWidget(note)
        run_layout.addStretch()

        top_grid.addWidget(variables_group, 0, 0)
        top_grid.addWidget(methods_group, 0, 1)
        top_grid.addWidget(analysis_group, 1, 0)
        top_grid.addWidget(run_group, 1, 1)
        top_grid.setColumnStretch(0, 1)
        top_grid.setColumnStretch(1, 1)
        top_grid.setRowStretch(0, 2)
        top_grid.setRowStretch(1, 1)

        layout.addStretch()

        self.btn_refresh.clicked.connect(self.refresh_columns)
        self.btn_run.clicked.connect(self.run_capability_study)
        self.rad_subgroup_constant.toggled.connect(self._sync_subgroup_controls)
        self.chk_ci.toggled.connect(self.spn_confidence.setEnabled)
        self.chk_ci.toggled.connect(self.cmb_ci_type.setEnabled)
        self._sync_subgroup_controls()

    def _sync_subgroup_controls(self):
        use_constant = self.rad_subgroup_constant.isChecked()
        self.spn_subgroup_size.setEnabled(use_constant)
        self.cmb_subgroup_column.setEnabled(not use_constant)

    def _build_results_tab(self):
        layout = QVBoxLayout(self.results_tab)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        self.results_tabs = QTabWidget()
        layout.addWidget(self.results_tabs)

        self.tbl_process = QTableWidget()
        self.tbl_overall = QTableWidget()
        self.tbl_within = QTableWidget()
        self.tbl_performance = QTableWidget()
        self.tbl_normality = QTableWidget()
        self.tbl_outliers = QTableWidget()
        self.tbl_stability = QTableWidget()

        self.results_tabs.addTab(self.tbl_process, "Process Data")
        self.results_tabs.addTab(self.tbl_overall, "Overall Capability")
        self.results_tabs.addTab(self.tbl_within, "Within Capability")
        self.results_tabs.addTab(self.tbl_performance, "Performance")
        self.results_tabs.addTab(self.tbl_normality, "Normality Tests")
        self.results_tabs.addTab(self.tbl_outliers, "Outlier Analysis")
        self.results_tabs.addTab(self.tbl_stability, "Stability / I-MR Signals")

        for table in [
            self.tbl_process,
            self.tbl_overall,
            self.tbl_within,
            self.tbl_performance,
            self.tbl_normality,
            self.tbl_outliers,
            self.tbl_stability,
        ]:
            table.setAlternatingRowColors(True)
            table.setWordWrap(True)
            table.setSortingEnabled(False)

    def _build_graphs_tab(self):
        layout = QVBoxLayout(self.graphs_tab)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        top = QHBoxLayout()
        top.addWidget(QLabel("Graph Type"))
        self.cmb_graph_type = QComboBox()
        self.cmb_graph_type.addItems(["Capability Summary", "Full Diagnostic Report", "Executive Dashboard"])
        top.addWidget(self.cmb_graph_type)
        top.addStretch()

        self.btn_copy_graph = QPushButton("Copy Graph")
        self.btn_export_png = QPushButton("Export PNG")
        self.btn_export_pdf = QPushButton("Export PDF")
        self.btn_rerun_graph = QPushButton("Re-run")
        self.btn_update_journal = QPushButton("Update Journal Entry")
        top.addWidget(self.btn_copy_graph)
        top.addWidget(self.btn_export_png)
        top.addWidget(self.btn_export_pdf)
        top.addWidget(self.btn_rerun_graph)
        top.addWidget(self.btn_update_journal)
        layout.addLayout(top)

        self.graph_figure = Figure(figsize=(11, 7), dpi=100)
        self.graph_canvas = FigureCanvas(self.graph_figure)
        self.graph_canvas_holder = QVBoxLayout()
        self.graph_canvas_holder.addWidget(self.graph_canvas)
        layout.addLayout(self.graph_canvas_holder)

        self.cmb_graph_type.currentTextChanged.connect(self._render_current_graph)
        self.btn_copy_graph.clicked.connect(self.copy_graph)
        self.btn_export_png.clicked.connect(self.export_graph_png)
        self.btn_export_pdf.clicked.connect(self.export_graph_pdf)
        self.btn_rerun_graph.clicked.connect(self.run_capability_study)
        self.btn_update_journal.clicked.connect(self.update_journal_entry)

    def _build_summary_tab(self):
        layout = QVBoxLayout(self.summary_tab)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        top = QHBoxLayout()
        top.addWidget(QLabel("Capability Study Report"))
        top.addStretch()
        self.btn_copy_summary = QPushButton("Copy Summary")
        top.addWidget(self.btn_copy_summary)
        layout.addLayout(top)

        self.summary_text = QTextEdit()
        self.summary_text.setReadOnly(True)
        self.summary_text.setPlaceholderText("Run a Capability Study to generate the report.")
        layout.addWidget(self.summary_text)

        self.btn_copy_summary.clicked.connect(self.copy_summary)

    def _build_debug_tab(self):
        layout = QVBoxLayout(self.debug_tab)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        top = QHBoxLayout()
        top.addWidget(QLabel("Debug Log"))
        top.addStretch()
        self.btn_clear_debug = QPushButton("Clear Debug Log")
        top.addWidget(self.btn_clear_debug)
        layout.addLayout(top)

        self.debug_text = QTextEdit()
        self.debug_text.setReadOnly(True)
        self.debug_text.setPlaceholderText("Debug messages will appear here.")
        layout.addWidget(self.debug_text)

        self.btn_clear_debug.clicked.connect(self.debug_text.clear)

    # ─────────────────────────────────────────────
    # Data access
    # ─────────────────────────────────────────────

    def log(self, message: str):
        ts = datetime.now().strftime("%H:%M:%S")
        self.debug_text.append(f"[{ts}] {message}")

    def get_df(self) -> pd.DataFrame:
        if self.app_window is None or not hasattr(self.app_window, "get_active_dataframe"):
            self.log("ERROR: app_window.get_active_dataframe() is not available.")
            return pd.DataFrame()
        try:
            df = self.app_window.get_active_dataframe()
            if df is None:
                return pd.DataFrame()
            return df.copy()
        except Exception as exc:
            self.log(f"ERROR while reading active worksheet: {exc}")
            return pd.DataFrame()

    def refresh_columns(self):
        df = self.get_df()
        self.current_data = df

        worksheet = "N/A"
        if self.app_window is not None and hasattr(self.app_window, "active_worksheet_name"):
            try:
                worksheet = self.app_window.active_worksheet_name()
            except Exception:
                worksheet = "N/A"

        self.lbl_worksheet.setText(f"Worksheet: {worksheet}")

        self.cmb_measurement.clear()
        self.cmb_subgroup_column.clear()
        self.cmb_subgroup_column.addItem("")

        if df.empty:
            self.log("No active worksheet data found.")
            return

        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) or pd.to_numeric(df[c], errors="coerce").notna().sum() > 0]
        all_cols = list(df.columns)

        self.cmb_measurement.addItems([str(c) for c in numeric_cols])
        self.cmb_subgroup_column.addItems([str(c) for c in all_cols])

        self.log(f"Worksheet detected: {worksheet}")
        self.log(f"Rows detected: {len(df)}")
        self.log(f"Columns detected: {', '.join(map(str, all_cols))}")
        self.log(f"Numeric columns detected: {', '.join(map(str, numeric_cols)) if numeric_cols else 'None'}")

    def _selected_options(self) -> CapabilityOptions:
        method_gt1 = "Pooled standard deviation"
        if self.rad_rbar.isChecked():
            method_gt1 = "Rbar"
        elif self.rad_sbar.isChecked():
            method_gt1 = "Sbar"

        method_eq1 = "Average moving range"
        if self.rad_median_mr.isChecked():
            method_eq1 = "Median moving range"
        elif self.rad_mssd.isChecked():
            method_eq1 = "Square root of MSSD"

        subgroup_mode = "Constant" if self.rad_subgroup_constant.isChecked() else "Column"

        display_units = "Parts per million" if self.rad_ppm.isChecked() else "Percents"
        capability_display = "Capability stats (Cp, Pp)" if self.rad_capability_stats.isChecked() else "Benchmark Z's (z level)"

        return CapabilityOptions(
            subgroup_mode=subgroup_mode,
            subgroup_size=int(self.spn_subgroup_size.value()),
            subgroup_column=self.cmb_subgroup_column.currentText().strip(),
            method_gt1=method_gt1,
            use_unbiasing_gt1=bool(self.chk_unbiasing_gt1.isChecked()),
            method_eq1=method_eq1,
            moving_range_length=int(self.spn_mr_length.value()),
            use_unbiasing_overall=bool(self.chk_unbiasing_overall.isChecked()),
            run_within=bool(self.chk_within_analysis.isChecked()),
            run_overall=bool(self.chk_overall_analysis.isChecked()),
            display_units=display_units,
            capability_display=capability_display,
            include_confidence_intervals=bool(self.chk_ci.isChecked()),
            confidence_level=float(self.spn_confidence.value()),
            confidence_interval_type=self.cmb_ci_type.currentText(),
            study_k=float(self.spn_study_k.value()),
            title=self.txt_title.text().strip(),
        )

    def _validate_inputs(self) -> Tuple[pd.DataFrame, str, np.ndarray, Optional[pd.Series], Optional[float], Optional[float], Optional[float], CapabilityOptions]:
        df = self.get_df()
        if df.empty:
            raise ValueError("No active worksheet data found.")

        measurement_col = self.cmb_measurement.currentText().strip()
        if not measurement_col:
            raise ValueError("Select a measurement column.")

        if measurement_col not in df.columns:
            raise ValueError("Selected measurement column was not found in the active worksheet.")

        data = pd.to_numeric(df[measurement_col], errors="coerce").dropna()
        if len(data) < 2:
            raise ValueError("At least 2 numeric data points are required.")

        options = self._selected_options()
        if not options.run_within and not options.run_overall:
            raise ValueError("Select at least one analysis: Within subgroup analysis or Overall analysis.")

        subgroup_ids = None
        if options.subgroup_mode == "Column":
            if not options.subgroup_column:
                raise ValueError("Select a subgroup column or use a constant subgroup size.")
            if options.subgroup_column not in df.columns:
                raise ValueError("Selected subgroup column was not found in the active worksheet.")
            aligned = df.loc[data.index, options.subgroup_column]
            if aligned.isna().all():
                raise ValueError("The selected subgroup column does not contain valid subgroup IDs.")
            subgroup_ids = aligned.fillna("Missing").astype(str)
        else:
            if options.subgroup_size < 1:
                raise ValueError("Subgroup size must be at least 1.")

        lsl = safe_float(self.txt_lsl.text())
        usl = safe_float(self.txt_usl.text())
        target = safe_float(self.txt_target.text())

        if lsl is None and usl is None:
            raise ValueError("Enter at least one specification limit: LSL or USL.")

        if lsl is not None and usl is not None and usl <= lsl:
            raise ValueError("USL must be greater than LSL.")

        return df, measurement_col, data.to_numpy(dtype=float), subgroup_ids, lsl, usl, target, options

    # ─────────────────────────────────────────────
    # Run
    # ─────────────────────────────────────────────

    def run_capability_study(self):
        try:
            df, name, data, subgroup_ids, lsl, usl, target, options = self._validate_inputs()

            self.log(f"Running Capability Study for: {name}")
            self.log(f"Valid numeric rows: {len(data)}")
            self.log(f"Subgroup mode: {options.subgroup_mode}")
            if options.subgroup_mode == "Constant":
                self.log(f"Subgroup size: {options.subgroup_size}")
            else:
                self.log(f"Subgroup column: {options.subgroup_column}")

            results = calc_capability(
                data=data,
                lsl=lsl,
                usl=usl,
                target=target,
                options=options,
                subgroup_ids=subgroup_ids,
            )
            normality = test_normality(data)
            outliers = detect_outliers(data)
            control = calc_imr_control_limits(data, moving_range_length=options.moving_range_length)
            descriptives = extended_descriptives(data)
            diagnosis = generate_diagnosis(name, results, normality, control, outliers)

            self.result = {
                "module": "Capability Study",
                "measurement_column": name,
                "worksheet": self.lbl_worksheet.text().replace("Worksheet: ", ""),
                "data": data,
                "results": results,
                "normality": normality,
                "outliers": outliers,
                "control": control,
                "descriptives": descriptives,
                "diagnosis": diagnosis,
                "options": asdict(options),
            }

            self.current_summary = build_text_report(
                name=name,
                results=results,
                normality=normality,
                control_data=control,
                outliers=outliers,
                descriptives=descriptives,
                diagnosis=diagnosis,
                options=options,
            )

            self._show_results()
            self.summary_text.setPlainText(self.current_summary)
            self._render_current_graph()
            self._auto_save_journal_if_needed()
            self.tabs.setCurrentWidget(self.graphs_tab)

            if self.app_window is not None and hasattr(self.app_window, "statusBar"):
                self.app_window.statusBar().showMessage("Capability Study completed.")

            self.log("Capability Study completed successfully.")

        except Exception as exc:
            self.log(f"ERROR: {exc}")
            QMessageBox.critical(self, "Capability Study failed", str(exc))

    # ─────────────────────────────────────────────
    # Tables
    # ─────────────────────────────────────────────

    def _populate_kv_table(self, table: QTableWidget, rows: List[Tuple[str, Any]], digits: Optional[int] = None):
        d = int(self.spn_decimals.value()) if digits is None else digits
        table.clear()
        table.setColumnCount(2)
        table.setRowCount(len(rows))
        table.setHorizontalHeaderLabels(["Metric", "Value"])

        for r, (metric, value) in enumerate(rows):
            table.setItem(r, 0, QTableWidgetItem(str(metric)))
            if isinstance(value, (float, np.floating, int, np.integer)) or value is None:
                text = fmt(value, d)
            else:
                text = str(value)
            table.setItem(r, 1, QTableWidgetItem(text))

        table.resizeColumnsToContents()

    def _populate_dataframe_table(self, table: QTableWidget, df: pd.DataFrame):
        table.clear()
        table.setColumnCount(len(df.columns))
        table.setRowCount(len(df))
        table.setHorizontalHeaderLabels([str(c) for c in df.columns])

        d = int(self.spn_decimals.value())
        for r in range(len(df)):
            for c, col in enumerate(df.columns):
                value = df.iloc[r, c]
                if isinstance(value, (float, np.floating, int, np.integer)) or value is None:
                    text = fmt(value, d)
                else:
                    text = str(value)
                table.setItem(r, c, QTableWidgetItem(text))
        table.resizeColumnsToContents()

    def _show_results(self):
        if self.result is None:
            return

        r = self.result["results"]
        n = self.result["normality"]
        o = self.result["outliers"]
        c = self.result["control"]
        desc = self.result["descriptives"]
        options = self.result["options"]

        self._populate_kv_table(self.tbl_process, [
            ("Sample N", r["n"]),
            ("Mean", r["mean"]),
            ("Median", desc["median"]),
            ("StDev(Overall)", r["std_overall"]),
            ("StDev(Within)", r["std_within"]),
            ("Within sigma method", r["within_sigma_method"]),
            ("Minimum", r["min"]),
            ("Maximum", r["max"]),
            ("Range", desc["range"]),
            ("LSL", r["lsl"]),
            ("Target", r["target"]),
            ("USL", r["usl"]),
            ("Subgroup mode", options["subgroup_mode"]),
            ("Subgroup size", options["subgroup_size"]),
            ("Subgroup column", options["subgroup_column"]),
            ("K", options["study_k"]),
        ])

        ov = r["overall_capability"]
        self._populate_kv_table(self.tbl_overall, [
            ("Pp", ov["Pp"]),
            ("PPL", ov["PPL"]),
            ("PPU", ov["PPU"]),
            ("Ppk", ov["Ppk"]),
            ("Cpm", ov["Cpm"]),
            ("Overall status", r["summary"]["overall_status"]),
            ("Long-term sigma", r["sigma_level"]["long_term"]),
            ("DPMO Overall", r["sigma_level"]["dpmo_overall"]),
            ("Yield Overall %", r["sigma_level"]["yield_overall"]),
        ])

        wi = r["within_capability"]
        self._populate_kv_table(self.tbl_within, [
            ("Cp", wi["Cp"]),
            ("CPL", wi["CPL"]),
            ("CPU", wi["CPU"]),
            ("Cpk", wi["Cpk"]),
            ("Within status", r["summary"]["within_status"]),
            ("Short-term sigma", r["sigma_level"]["short_term"]),
            ("DPMO Within", r["sigma_level"]["dpmo_within"]),
            ("Yield Within %", r["sigma_level"]["yield_within"]),
            ("Benchmark", r["sigma_level"]["benchmark"]),
        ])

        perf_rows = []
        for label, block in [
            ("Observed", r["observed"]),
            ("Expected Overall", r["expected_overall"]),
            ("Expected Within", r["expected_within"]),
        ]:
            perf_rows.append({"Performance": label, "Below LSL %": block.get("below"), "Above USL %": block.get("above"), "Total %": block.get("total")})
        self._populate_dataframe_table(self.tbl_performance, pd.DataFrame(perf_rows))

        norm_rows = []
        for test, row in n.items():
            if test == "summary":
                continue
            norm_rows.append({
                "Test": test,
                "Statistic": row.get("statistic"),
                "P-Value": row.get("p_value"),
                "Critical 5%": row.get("critical_5pct"),
                "Result": "Normal" if row.get("normal", False) else "Non-Normal",
                "Details": row.get("label", row.get("error", "")),
            })
        norm_rows.append({
            "Test": "Summary",
            "Statistic": None,
            "P-Value": None,
            "Critical 5%": None,
            "Result": n["summary"]["verdict"],
            "Details": f"{n['summary']['passed']} of {n['summary']['total']} tests passed",
        })
        self._populate_dataframe_table(self.tbl_normality, pd.DataFrame(norm_rows))

        outlier_rows = [
            {"Method": "Grubbs", "Metric": "G statistic", "Value": o["grubbs"]["G_stat"], "Details": o["grubbs"]["label"]},
            {"Method": "Grubbs", "Metric": "G critical", "Value": o["grubbs"]["G_critical"], "Details": "Outlier detected" if o["grubbs"]["outlier_detected"] else "No outlier detected"},
            {"Method": "IQR mild fence", "Metric": "Count", "Value": o["iqr_mild"]["count"], "Details": str(o["iqr_mild"]["values"][:20])},
            {"Method": "IQR extreme fence", "Metric": "Count", "Value": o["iqr_extreme"]["count"], "Details": str(o["iqr_extreme"]["values"][:20])},
            {"Method": "Recommendation", "Metric": "", "Value": None, "Details": o["recommendation"]},
        ]
        self._populate_dataframe_table(self.tbl_outliers, pd.DataFrame(outlier_rows))

        stability_rows = [
            {"Area": "X Chart", "Metric": "CL", "Value": c["X_chart"]["CL"], "Details": ""},
            {"Area": "X Chart", "Metric": "UCL", "Value": c["X_chart"]["UCL"], "Details": ""},
            {"Area": "X Chart", "Metric": "LCL", "Value": c["X_chart"]["LCL"], "Details": ""},
            {"Area": "MR Chart", "Metric": "CL", "Value": c["MR_chart"]["CL"], "Details": ""},
            {"Area": "MR Chart", "Metric": "UCL", "Value": c["MR_chart"]["UCL"], "Details": ""},
            {"Area": "MR Chart", "Metric": "LCL", "Value": c["MR_chart"]["LCL"], "Details": ""},
            {"Area": "Stability", "Metric": "In control", "Value": None, "Details": "Yes" if c["in_control"] else "No"},
            {"Area": "Stability", "Metric": "Signal count", "Value": len(c["signals"]), "Details": ""},
        ]
        for idx, msg in c["signals"]:
            stability_rows.append({"Area": "Signal", "Metric": f"Point {idx + 1}", "Value": None, "Details": msg})
        self._populate_dataframe_table(self.tbl_stability, pd.DataFrame(stability_rows))


    # ─────────────────────────────────────────────
    # Graphs / Journal - Phase 3
    # ─────────────────────────────────────────────

    def _make_figure_for_type(self, graph_type: str) -> Optional[Figure]:
        if self.result is None:
            return None
        data = np.asarray(self.result.get("data", []), dtype=float)
        if len(data) < 2:
            return None
        name = self.result.get("measurement_column", "Variable")
        decimals = int(self.spn_decimals.value())
        if graph_type == "Full Diagnostic Report":
            return build_capability_full_figure(
                name=name,
                data=data,
                results=self.result["results"],
                normality=self.result["normality"],
                control=self.result["control"],
                outliers=self.result["outliers"],
                descriptives=self.result["descriptives"],
                decimals=decimals,
                options=self.result.get("options"),
            )
        if graph_type == "Executive Dashboard":
            return build_dashboard_figure([{"name": name, **self.result}], decimals=decimals, options=self.result.get("options"))
        return build_capability_summary_figure(name, data, self.result["results"], decimals=decimals, options=self.result.get("options"))

    def _render_current_graph(self):
        if self.result is None:
            self.graph_figure.clear()
            self.graph_figure.text(0.5, 0.5, "Run a Capability Study to generate graphs.", ha="center", va="center")
            self.graph_canvas.draw_idle()
            return
        graph_type = self.cmb_graph_type.currentText() if hasattr(self, "cmb_graph_type") else "Capability Summary"
        self.current_graph_type = graph_type
        fig = self._make_figure_for_type(graph_type)
        if fig is None:
            return
        self.current_figure = fig
        old_canvas = self.graph_canvas
        self.graph_canvas = FigureCanvas(fig)
        self.graph_canvas_holder.replaceWidget(old_canvas, self.graph_canvas)
        old_canvas.setParent(None)
        old_canvas.deleteLater()
        self.graph_canvas.draw_idle()

        if enable_interaction is not None:
            try:
                self._plot_manager = enable_interaction(
                    host=self,
                    figure_getter=lambda: self.graph_canvas.figure,
                    canvas_getter=lambda: self.graph_canvas,
                    decimals_getter=lambda: int(self.spn_decimals.value()),
                    on_status=lambda msg: self.app_window.statusBar().showMessage(msg) if self.app_window and hasattr(self.app_window, "statusBar") else None,
                )
            except TypeError:
                try:
                    self._plot_manager = enable_interaction(self, lambda: self.graph_canvas.figure, lambda: self.graph_canvas)
                except Exception as exc:
                    self.log(f"Graph interaction not enabled: {exc}")
            except Exception as exc:
                self.log(f"Graph interaction not enabled: {exc}")

    def copy_graph(self):
        if self.current_figure is None:
            QMessageBox.warning(self, "No graph", "Run a Capability Study before copying a graph.")
            return
        buffer = io.BytesIO()
        self.current_figure.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
        pixmap = None
        try:
            from PySide6.QtGui import QPixmap
            pixmap = QPixmap()
            pixmap.loadFromData(buffer.getvalue(), "PNG")
            QApplication.clipboard().setPixmap(pixmap)
            self.log("Graph copied to clipboard.")
        except Exception as exc:
            QMessageBox.warning(self, "Copy Graph failed", str(exc))

    def export_graph_png(self):
        if self.current_figure is None:
            QMessageBox.warning(self, "No graph", "Run a Capability Study before exporting a graph.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export PNG", "capability_study.png", "PNG Files (*.png)")
        if not path:
            return
        self.current_figure.savefig(path, format="png", dpi=200, bbox_inches="tight")
        self.log(f"Graph exported to PNG: {path}")

    def export_graph_pdf(self):
        if self.current_figure is None:
            QMessageBox.warning(self, "No graph", "Run a Capability Study before exporting a graph.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export PDF", "capability_study.pdf", "PDF Files (*.pdf)")
        if not path:
            return
        with PdfPages(path) as pdf:
            pdf.savefig(self.current_figure, bbox_inches="tight")
        self.log(f"Graph exported to PDF: {path}")

    def _now(self) -> str:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def _safe_current_graph_base64(self) -> str:
        try:
            fig = self.current_figure or self._make_figure_for_type(self.current_graph_type or "Capability Summary")
            return figure_to_png_base64(fig) if fig is not None else ""
        except Exception as exc:
            self.log(f"WARNING: Unable to encode graph image for journal: {exc}")
            return ""

    def _journal_updates(self, change_note: str = "Updated") -> Dict[str, Any]:
        payload = self._build_journal_payload()
        image_b64 = payload.get("image_png_base64", "") or self._safe_current_graph_base64()
        status = "N/A"
        try:
            status = self.result.get("results", {}).get("summary", {}).get("within_status", "N/A")
        except Exception:
            pass
        return {
            "name": f"Capability Study - {payload.get('measurement_column', 'Variable')}",
            "type": "Capability Study",
            "worksheet": payload.get("worksheet", ""),
            "status": status,
            "summary": self.current_summary,
            "figure_png_base64": image_b64,
            "report_png_base64": image_b64,
            "image": image_b64,
            "report_image": image_b64,
            "payload": payload,
            "last_change": change_note,
        }

    def _build_journal_payload(self) -> Dict[str, Any]:
        if self.result is None:
            return {}
        graph_type = self.cmb_graph_type.currentText() if hasattr(self, "cmb_graph_type") else self.current_graph_type
        fig = self.current_figure or self._make_figure_for_type(graph_type)
        image_b64 = figure_to_png_base64(fig) if fig is not None else ""
        payload = {
            "module": "Capability Study",
            "graph_report_type": graph_type,
            "worksheet": self.result.get("worksheet", ""),
            "measurement_column": self.result.get("measurement_column", ""),
            "lsl": self.result.get("results", {}).get("lsl"),
            "usl": self.result.get("results", {}).get("usl"),
            "target": self.result.get("results", {}).get("target"),
            "decimals": int(self.spn_decimals.value()),
            "results": self.result.get("results", {}),
            "normality": self.result.get("normality", {}),
            "outliers": self.result.get("outliers", {}),
            "control": self.result.get("control", {}),
            "descriptives": self.result.get("descriptives", {}),
            "diagnosis": self.result.get("diagnosis", []),
            "options": self.result.get("options", {}),
            "summary_text": self.current_summary,
            "image_png_base64": image_b64,
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
        return _serializable(payload)

    def _auto_save_journal_if_needed(self):
        if not hasattr(self, "chk_auto_journal") or not self.chk_auto_journal.isChecked():
            return
        if self.app_window is None or not hasattr(self.app_window, "add_journal_entry"):
            self.log("Auto journal save skipped: app_window.add_journal_entry is not available.")
            return
        try:
            if self.current_journal_entry_id and hasattr(self.app_window, "update_journal_entry"):
                updated = self.app_window.update_journal_entry(
                    self.current_journal_entry_id,
                    self._journal_updates(change_note="Updated"),
                )
                if updated:
                    self.log("Capability Study journal entry updated.")
                    return

            entry_id = f"capability_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            now = self._now()
            entry = {
                "id": entry_id,
                "created_at": now,
                "modified_at": now,
                **self._journal_updates(change_note="Created"),
            }
            self.app_window.add_journal_entry(entry)
            self.current_journal_entry_id = entry_id
            self.log("Capability Study saved to Analysis Journal.")
        except Exception as exc:
            self.log(f"Journal save failed: {exc}")

    def load_from_journal_entry(self, entry: Dict[str, Any]):
        if not isinstance(entry, dict):
            QMessageBox.warning(self, "Invalid Journal Entry", "The selected Journal entry is not valid.")
            return
        payload = entry.get("payload", entry)
        if payload.get("module") != "Capability Study":
            QMessageBox.warning(self, "Invalid Journal Entry", "This Journal entry is not a Capability Study entry.")
            return
        self.current_journal_entry_id = entry.get("id") or payload.get("id") or self.current_journal_entry_id
        self.current_summary = payload.get("summary_text", "")
        self.summary_text.setPlainText(self.current_summary)
        self.log("Capability Study Journal entry loaded. Re-run is recommended to refresh live worksheet data.")
        # Restore visible setup fields where possible.
        meas = payload.get("measurement_column", "")
        idx = self.cmb_measurement.findText(meas)
        if idx >= 0:
            self.cmb_measurement.setCurrentIndex(idx)
        for widget, value in [(self.txt_lsl, payload.get("lsl")), (self.txt_usl, payload.get("usl")), (self.txt_target, payload.get("target"))]:
            widget.setText("" if value is None else str(value))
        options = payload.get("options", {}) or {}
        self.spn_decimals.setValue(int(payload.get("decimals", options.get("decimals", self.spn_decimals.value())) or self.spn_decimals.value()))
        self.tabs.setCurrentWidget(self.summary_tab)

    def update_journal_entry(self):
        if self.result is None:
            QMessageBox.warning(self, "No results", "Run a Capability Study before updating the Journal entry.")
            return
        if self.app_window is None:
            QMessageBox.information(self, "Journal integration", "No application window is available for Journal integration.")
            return
        try:
            if self.current_journal_entry_id and hasattr(self.app_window, "update_journal_entry"):
                updated = self.app_window.update_journal_entry(
                    self.current_journal_entry_id,
                    self._journal_updates(change_note="Updated"),
                )
                if updated:
                    self.log("Analysis Journal entry updated.")
                    return

            if not hasattr(self.app_window, "add_journal_entry"):
                QMessageBox.information(self, "Journal integration", "app_window.add_journal_entry is not available in this build.")
                return

            entry_id = f"capability_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            now = self._now()
            entry = {
                "id": entry_id,
                "created_at": now,
                "modified_at": now,
                **self._journal_updates(change_note="Created"),
            }
            self.app_window.add_journal_entry(entry)
            self.current_journal_entry_id = entry_id
            self.log("Analysis Journal entry created.")
        except Exception as exc:
            QMessageBox.warning(self, "Update Journal Entry failed", str(exc))
            self.log(f"Journal update failed: {exc}")


    def copy_summary(self):
        text = self.summary_text.toPlainText()
        if not text.strip():
            return
        QApplication.clipboard().setText(text)
        if self.app_window is not None and hasattr(self.app_window, "statusBar"):
            self.app_window.statusBar().showMessage("Summary copied to clipboard.")

