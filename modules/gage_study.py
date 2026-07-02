"""
modules/gage_study.py
SPS Studio — Gage Study Builder
Stage 3: Crossed Gage R&R integration with tables, summary, interactive graphs and report engine.

Features:
- Reads active worksheet from main window
- Detects worksheet columns
- Runs Gage R&R using ANOVA or Xbar and R
- Automatically handles single-operator studies using One-Way ANOVA
- Shows key metrics, ANOVA/method table, variance components, summary and interactive graphs
- Exports/copies graph and graph + summary reports
- Saves graph image payload to Analysis Journal
"""

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Dict, Any
import io
import base64


import numpy as np
import pandas as pd
from scipy.stats import f

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QGroupBox,
    QLabel, QComboBox, QPushButton, QTextEdit, QTabWidget,
    QMessageBox, QTableWidget, QTableWidgetItem, QDoubleSpinBox,
    QCheckBox, QSpinBox, QFileDialog, QApplication
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.image as mpimg

from core.plot_interaction_tools import enable_interaction


# ─────────────────────────────────────────────
# SPC constants
# ─────────────────────────────────────────────

CONTROL_CONSTANTS = {
    2: {"A2": 1.880, "D3": 0.000, "D4": 3.267},
    3: {"A2": 1.023, "D3": 0.000, "D4": 2.574},
    4: {"A2": 0.729, "D3": 0.000, "D4": 2.282},
    5: {"A2": 0.577, "D3": 0.000, "D4": 2.114},
    6: {"A2": 0.483, "D3": 0.000, "D4": 2.004},
    7: {"A2": 0.419, "D3": 0.076, "D4": 1.924},
    8: {"A2": 0.373, "D3": 0.136, "D4": 1.864},
    9: {"A2": 0.337, "D3": 0.184, "D4": 1.816},
    10: {"A2": 0.308, "D3": 0.223, "D4": 1.777},
}

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
}


@dataclass
class GRRResult:
    anova_table: pd.DataFrame
    variance_components: pd.DataFrame
    settings: Dict[str, Any]
    interpretation: Dict[str, Any]
    data: pd.DataFrame
    operator_label: str
    part_label: str
    value_label: str
    one_operator_mode: bool = False


# ─────────────────────────────────────────────
# Statistical engine - validation
# ─────────────────────────────────────────────

def validate_and_prepare_base(
    df: pd.DataFrame,
    part_col: str,
    operator_col: str,
    value_col: str,
) -> pd.DataFrame:
    required = [part_col, operator_col, value_col]
    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    data = df[[part_col, operator_col, value_col]].copy()
    data = data.replace("", np.nan).dropna()

    if data.empty:
        raise ValueError("No valid data found.")

    data[part_col] = data[part_col].astype(str).str.strip()
    data[operator_col] = data[operator_col].astype(str).str.strip()
    data[value_col] = pd.to_numeric(data[value_col], errors="coerce")
    data = data.dropna(subset=[value_col])

    if data.empty:
        raise ValueError("No valid numeric measurement values found.")

    if data[part_col].nunique() < 2:
        raise ValueError("At least 2 parts are required.")

    return data


def validate_balanced(data: pd.DataFrame, part_col: str, operator_col: str) -> int:
    counts = data.groupby([part_col, operator_col]).size().reset_index(name="n")

    if counts["n"].nunique() != 1:
        pivot = counts.pivot(index=part_col, columns=operator_col, values="n")
        raise ValueError(
            "The study is not balanced.\n"
            "Each Part × Operator combination must have the same number of replicates.\n\n"
            f"Replicate count summary:\n{pivot.to_string()}"
        )

    n = int(counts["n"].iloc[0])

    if n < 2:
        raise ValueError("At least 2 replicates per Part × Operator are required.")

    return n


def validate_and_prepare(
    df: pd.DataFrame,
    part_col: str,
    operator_col: str,
    value_col: str,
) -> pd.DataFrame:
    data = validate_and_prepare_base(df, part_col, operator_col, value_col)

    if data[operator_col].nunique() < 2:
        raise ValueError("At least 2 operators are required for crossed Gage R&R.")

    validate_balanced(data, part_col, operator_col)
    return data


# ─────────────────────────────────────────────
# ANOVA - 2+ operators
# ─────────────────────────────────────────────

def anova_components(
    df: pd.DataFrame,
    part_col: str,
    operator_col: str,
    value_col: str,
    alpha_remove_interaction: float = 0.05,
) -> Dict[str, Any]:
    parts = sorted(df[part_col].unique())
    operators = sorted(df[operator_col].unique())

    a = len(parts)
    b = len(operators)
    n = int(df.groupby([part_col, operator_col]).size().iloc[0])

    grand_mean = df[value_col].mean()
    mean_part = df.groupby(part_col)[value_col].mean()
    mean_operator = df.groupby(operator_col)[value_col].mean()
    mean_cell = df.groupby([part_col, operator_col])[value_col].mean()

    ss_part = b * n * ((mean_part - grand_mean) ** 2).sum()
    ss_operator = a * n * ((mean_operator - grand_mean) ** 2).sum()

    temp = df.merge(
        mean_cell.rename("cell_mean"),
        left_on=[part_col, operator_col],
        right_index=True
    )

    ss_repeat = ((temp[value_col] - temp["cell_mean"]) ** 2).sum()

    ss_interaction = 0.0
    for p in parts:
        for o in operators:
            ss_interaction += (
                mean_cell.loc[(p, o)]
                - mean_part.loc[p]
                - mean_operator.loc[o]
                + grand_mean
            ) ** 2

    ss_interaction *= n
    ss_total = ((df[value_col] - grand_mean) ** 2).sum()

    df_part = a - 1
    df_operator = b - 1
    df_interaction = (a - 1) * (b - 1)
    df_repeat = a * b * (n - 1)
    df_total = len(df) - 1

    ms_part = ss_part / df_part
    ms_operator = ss_operator / df_operator
    ms_interaction = ss_interaction / df_interaction
    ms_repeat = ss_repeat / df_repeat

    f_part = ms_part / ms_interaction if ms_interaction > 0 else np.nan
    f_operator = ms_operator / ms_interaction if ms_interaction > 0 else np.nan
    f_interaction = ms_interaction / ms_repeat if ms_repeat > 0 else np.nan

    p_part = 1 - f.cdf(f_part, df_part, df_interaction) if np.isfinite(f_part) else np.nan
    p_operator = 1 - f.cdf(f_operator, df_operator, df_interaction) if np.isfinite(f_operator) else np.nan
    p_interaction = 1 - f.cdf(f_interaction, df_interaction, df_repeat) if np.isfinite(f_interaction) else np.nan

    interaction_kept = bool(p_interaction < alpha_remove_interaction)

    if interaction_kept:
        var_repeat = max(ms_repeat, 0.0)
        var_operator = max((ms_operator - ms_interaction) / (a * n), 0.0)
        var_interaction = max((ms_interaction - ms_repeat) / n, 0.0)
        var_reproducibility = var_operator + var_interaction
        var_part = max((ms_part - ms_interaction) / (b * n), 0.0)

        anova_rows = [
            [part_col, df_part, ss_part, ms_part, f_part, p_part],
            [operator_col, df_operator, ss_operator, ms_operator, f_operator, p_operator],
            [f"{part_col}*{operator_col}", df_interaction, ss_interaction, ms_interaction, f_interaction, p_interaction],
            ["Repeatability", df_repeat, ss_repeat, ms_repeat, np.nan, np.nan],
            ["Total", df_total, ss_total, np.nan, np.nan, np.nan],
        ]

    else:
        ss_error_pooled = ss_repeat + ss_interaction
        df_error_pooled = df_repeat + df_interaction
        ms_error_pooled = ss_error_pooled / df_error_pooled

        var_repeat = max(ms_error_pooled, 0.0)
        var_operator = max((ms_operator - ms_error_pooled) / (a * n), 0.0)
        var_interaction = 0.0
        var_reproducibility = var_operator
        var_part = max((ms_part - ms_error_pooled) / (b * n), 0.0)

        f_part = ms_part / ms_error_pooled if ms_error_pooled > 0 else np.nan
        f_operator = ms_operator / ms_error_pooled if ms_error_pooled > 0 else np.nan

        p_part = 1 - f.cdf(f_part, df_part, df_error_pooled) if np.isfinite(f_part) else np.nan
        p_operator = 1 - f.cdf(f_operator, df_operator, df_error_pooled) if np.isfinite(f_operator) else np.nan

        anova_rows = [
            [part_col, df_part, ss_part, ms_part, f_part, p_part],
            [operator_col, df_operator, ss_operator, ms_operator, f_operator, p_operator],
            ["Repeatability", df_error_pooled, ss_error_pooled, ms_error_pooled, np.nan, np.nan],
            ["Total", df_total, ss_total, np.nan, np.nan, np.nan],
        ]

    var_grr = var_repeat + var_reproducibility
    var_total = var_grr + var_part

    anova_table = pd.DataFrame(
        anova_rows,
        columns=["Source", "DF", "SS", "MS", "F", "P"]
    )

    return {
        "anova_table": anova_table,
        "parts": a,
        "operators": b,
        "replicates": n,
        "interaction_kept": interaction_kept,
        "p_interaction": p_interaction,
        "var_repeat": var_repeat,
        "var_operator": var_operator,
        "var_interaction": var_interaction,
        "var_reproducibility": var_reproducibility,
        "var_part": var_part,
        "var_grr": var_grr,
        "var_total": var_total,
        "grand_mean": grand_mean,
    }


def gage_rr_crossed_anova(
    df: pd.DataFrame,
    part_col: str,
    operator_col: str,
    value_col: str,
    usl: Optional[float] = None,
    lsl: Optional[float] = None,
    alpha_remove_interaction: float = 0.05,
    study_k: float = 6.0,
) -> GRRResult:
    data = validate_and_prepare(df, part_col, operator_col, value_col)

    res = anova_components(
        data,
        part_col=part_col,
        operator_col=operator_col,
        value_col=value_col,
        alpha_remove_interaction=alpha_remove_interaction,
    )

    tolerance = None
    if usl is not None and lsl is not None:
        tolerance = float(usl) - float(lsl)
        if tolerance <= 0:
            raise ValueError("USL must be greater than LSL.")

    rows = [
        ("Total Gage R&R", res["var_grr"]),
        ("Repeatability", res["var_repeat"]),
        ("Reproducibility", res["var_reproducibility"]),
        (f"  {operator_col}", res["var_operator"]),
        (f"  {part_col}*{operator_col}", res["var_interaction"]),
        ("Part-To-Part", res["var_part"]),
        ("Total Variation", res["var_total"]),
    ]

    output = []
    for source, var_comp in rows:
        std_dev = math.sqrt(max(var_comp, 0.0))
        study_var = study_k * std_dev

        output.append({
            "Source": source,
            "VarComp": var_comp,
            "StdDev": std_dev,
            "StudyVar": study_var,
        })

    vc = pd.DataFrame(output)

    total_var = float(res["var_total"])
    vc["%Contribution"] = np.where(total_var > 0, 100 * vc["VarComp"] / total_var, np.nan)

    total_sd = float(vc.loc[vc["Source"] == "Total Variation", "StdDev"].iloc[0])
    vc["%StudyVar"] = np.where(total_sd > 0, 100 * vc["StdDev"] / total_sd, np.nan)

    if tolerance is not None:
        vc["%Tolerance"] = 100 * vc["StudyVar"] / tolerance
    else:
        vc["%Tolerance"] = np.nan

    sd_grr = float(vc.loc[vc["Source"] == "Total Gage R&R", "StdDev"].iloc[0])
    sd_part = float(vc.loc[vc["Source"] == "Part-To-Part", "StdDev"].iloc[0])

    if sd_grr > 0:
        ndc = max(1, math.floor(1.41 * sd_part / sd_grr))
    else:
        ndc = float("inf")

    grr_percent_study_var = float(
        vc.loc[vc["Source"] == "Total Gage R&R", "%StudyVar"].iloc[0]
    )

    if grr_percent_study_var < 10:
        aiag_judgment = "Acceptable"
    elif grr_percent_study_var <= 30:
        aiag_judgment = "Marginal"
    else:
        aiag_judgment = "Not acceptable"

    interpretation = {
        "interaction_kept": res["interaction_kept"],
        "p_interaction": res["p_interaction"],
        "ndc": ndc,
        "grr_percent_study_var": grr_percent_study_var,
        "aiag_judgment": aiag_judgment,
        "ndc_judgment": "Acceptable discrimination" if ndc == float("inf") or ndc >= 5 else "Poor discrimination",
        "method": "ANOVA",
    }

    return GRRResult(
        anova_table=res["anova_table"].copy(),
        variance_components=vc.copy(),
        settings={
            "parts": res["parts"],
            "operators": res["operators"],
            "replicates": res["replicates"],
            "alpha_remove_interaction": alpha_remove_interaction,
            "study_k": study_k,
            "tolerance": tolerance,
            "grand_mean": res["grand_mean"],
            "method": "ANOVA",
        },
        interpretation=interpretation,
        data=data.copy(),
        operator_label=operator_col,
        part_label=part_col,
        value_label=value_col,
        one_operator_mode=False,
    )


# ─────────────────────────────────────────────
# One-Way ANOVA - 1 operator
# ─────────────────────────────────────────────

def validate_and_prepare_one_operator(
    df: pd.DataFrame,
    part_col: str,
    operator_col: str,
    value_col: str,
) -> pd.DataFrame:
    data = validate_and_prepare_base(df, part_col, operator_col, value_col)

    counts = data.groupby(part_col)[value_col].count()

    if counts.nunique() != 1:
        raise ValueError(
            "The study is not balanced.\n"
            "Each part must have the same number of replicates.\n\n"
            f"Counts per part:\n{counts.to_string()}"
        )

    n = int(counts.iloc[0])

    if n < 2:
        raise ValueError("At least 2 replicates per part are required.")

    return data


def one_way_anova_components(
    df: pd.DataFrame,
    part_col: str,
    value_col: str,
) -> Dict[str, Any]:
    parts = sorted(df[part_col].unique())
    a = len(parts)
    n = int(df.groupby(part_col)[value_col].count().iloc[0])

    grand_mean = df[value_col].mean()
    mean_part = df.groupby(part_col)[value_col].mean()

    ss_part = n * ((mean_part - grand_mean) ** 2).sum()

    temp = df.merge(
        mean_part.rename("part_mean"),
        left_on=part_col,
        right_index=True
    )

    ss_repeat = ((temp[value_col] - temp["part_mean"]) ** 2).sum()
    ss_total = ((df[value_col] - grand_mean) ** 2).sum()

    df_part = a - 1
    df_repeat = a * (n - 1)
    df_total = a * n - 1

    ms_part = ss_part / df_part
    ms_repeat = ss_repeat / df_repeat

    f_part = ms_part / ms_repeat if ms_repeat > 0 else np.nan
    p_part = 1 - f.cdf(f_part, df_part, df_repeat) if np.isfinite(f_part) else np.nan

    var_repeat = max(ms_repeat, 0.0)
    var_part = max((ms_part - ms_repeat) / n, 0.0)
    var_grr = var_repeat
    var_total = var_repeat + var_part

    anova_table = pd.DataFrame(
        [
            [part_col, df_part, ss_part, ms_part, f_part, p_part],
            ["Repeatability", df_repeat, ss_repeat, ms_repeat, np.nan, np.nan],
            ["Total", df_total, ss_total, np.nan, np.nan, np.nan],
        ],
        columns=["Source", "DF", "SS", "MS", "F", "P"]
    )

    return {
        "anova_table": anova_table,
        "parts": a,
        "replicates": n,
        "var_repeat": var_repeat,
        "var_part": var_part,
        "var_grr": var_grr,
        "var_total": var_total,
        "grand_mean": grand_mean,
    }


def gage_rr_one_operator(
    df: pd.DataFrame,
    part_col: str,
    operator_col: str,
    value_col: str,
    usl: Optional[float] = None,
    lsl: Optional[float] = None,
    study_k: float = 6.0,
) -> GRRResult:
    data = validate_and_prepare_one_operator(df, part_col, operator_col, value_col)

    res = one_way_anova_components(
        data,
        part_col=part_col,
        value_col=value_col,
    )

    tolerance = None

    if usl is not None and lsl is not None:
        tolerance = float(usl) - float(lsl)

        if tolerance <= 0:
            raise ValueError("USL must be greater than LSL.")

    rows = [
        ("Total Gage R&R", res["var_grr"]),
        ("Repeatability", res["var_repeat"]),
        ("Part-To-Part", res["var_part"]),
        ("Total Variation", res["var_total"]),
    ]

    output = []

    for source, var_comp in rows:
        std_dev = math.sqrt(max(var_comp, 0.0))
        study_var = study_k * std_dev

        output.append({
            "Source": source,
            "VarComp": var_comp,
            "StdDev": std_dev,
            "StudyVar": study_var,
        })

    vc = pd.DataFrame(output)

    total_var = float(res["var_total"])

    vc["%Contribution"] = np.where(
        total_var > 0,
        100 * vc["VarComp"] / total_var,
        np.nan
    )

    total_sd = float(vc.loc[vc["Source"] == "Total Variation", "StdDev"].iloc[0])

    vc["%StudyVar"] = np.where(
        total_sd > 0,
        100 * vc["StdDev"] / total_sd,
        np.nan
    )

    if tolerance is not None:
        vc["%Tolerance"] = 100 * vc["StudyVar"] / tolerance
    else:
        vc["%Tolerance"] = np.nan

    sd_grr = float(vc.loc[vc["Source"] == "Total Gage R&R", "StdDev"].iloc[0])
    sd_part = float(vc.loc[vc["Source"] == "Part-To-Part", "StdDev"].iloc[0])

    if sd_grr > 0:
        ndc = max(1, math.floor(1.41 * sd_part / sd_grr))
    else:
        ndc = float("inf")

    grr_percent_study_var = float(
        vc.loc[vc["Source"] == "Total Gage R&R", "%StudyVar"].iloc[0]
    )

    if grr_percent_study_var < 10:
        aiag_judgment = "Acceptable"
    elif grr_percent_study_var <= 30:
        aiag_judgment = "Marginal"
    else:
        aiag_judgment = "Not acceptable"

    operator_name = str(data[operator_col].iloc[0])

    interpretation = {
        "interaction_kept": False,
        "p_interaction": np.nan,
        "ndc": ndc,
        "grr_percent_study_var": grr_percent_study_var,
        "aiag_judgment": aiag_judgment,
        "ndc_judgment": "Acceptable discrimination" if ndc == float("inf") or ndc >= 5 else "Poor discrimination",
        "operator_name": operator_name,
        "method": "One-Way ANOVA",
    }

    return GRRResult(
        anova_table=res["anova_table"].copy(),
        variance_components=vc.copy(),
        settings={
            "parts": res["parts"],
            "operators": 1,
            "replicates": res["replicates"],
            "alpha_remove_interaction": None,
            "study_k": study_k,
            "tolerance": tolerance,
            "grand_mean": res["grand_mean"],
            "method": "One-Way ANOVA",
        },
        interpretation=interpretation,
        data=data.copy(),
        operator_label=operator_col,
        part_label=part_col,
        value_label=value_col,
        one_operator_mode=True,
    )


# ─────────────────────────────────────────────
# Xbar and R - 2+ operators
# ─────────────────────────────────────────────

def d2_value(n: int) -> float:
    if n in D2_CONSTANTS:
        return D2_CONSTANTS[n]

    if n < 2:
        raise ValueError("d2 requires n >= 2.")

    return 3.078 + 0.118 * (n - 10) ** 0.5


def gage_rr_crossed_xbar_r(
    df: pd.DataFrame,
    part_col: str,
    operator_col: str,
    value_col: str,
    usl: Optional[float] = None,
    lsl: Optional[float] = None,
    study_k: float = 6.0,
) -> GRRResult:
    data = validate_and_prepare(df, part_col, operator_col, value_col)

    parts = sorted(data[part_col].unique())
    operators = sorted(data[operator_col].unique())

    n_parts = len(parts)
    n_operators = len(operators)
    n_replicates = int(data.groupby([part_col, operator_col]).size().iloc[0])

    tolerance = None
    if usl is not None and lsl is not None:
        tolerance = float(usl) - float(lsl)
        if tolerance <= 0:
            raise ValueError("USL must be greater than LSL.")

    d2_replicates = d2_value(n_replicates)
    d2_operators = d2_value(n_operators)
    d2_parts = d2_value(n_parts)

    cell_ranges = (
        data.groupby([part_col, operator_col])[value_col]
        .agg(lambda x: x.max() - x.min())
    )
    rbar = float(cell_ranges.mean())
    ev = rbar / d2_replicates

    operator_means = data.groupby(operator_col)[value_col].mean()
    xbar_diff = float(operator_means.max() - operator_means.min())

    av_raw = xbar_diff / d2_operators
    av = math.sqrt(max((av_raw ** 2) - (ev ** 2 / (n_parts * n_replicates)), 0.0))

    part_means = data.groupby(part_col)[value_col].mean()
    rp = float(part_means.max() - part_means.min())
    pv = rp / d2_parts

    grr_sd = math.sqrt(ev ** 2 + av ** 2)
    total_sd = math.sqrt(grr_sd ** 2 + pv ** 2)

    rows = [
        ("Total Gage R&R", grr_sd ** 2, grr_sd),
        ("Repeatability", ev ** 2, ev),
        ("Reproducibility", av ** 2, av),
        (f"  {operator_col}", av ** 2, av),
        (f"  {part_col}*{operator_col}", 0.0, 0.0),
        ("Part-To-Part", pv ** 2, pv),
        ("Total Variation", total_sd ** 2, total_sd),
    ]

    output = []
    for source, var_comp, std_dev in rows:
        output.append({
            "Source": source,
            "VarComp": var_comp,
            "StdDev": std_dev,
            "StudyVar": study_k * std_dev,
        })

    vc = pd.DataFrame(output)

    total_var = float(vc.loc[vc["Source"] == "Total Variation", "VarComp"].iloc[0])
    total_sd_value = float(vc.loc[vc["Source"] == "Total Variation", "StdDev"].iloc[0])

    vc["%Contribution"] = np.where(total_var > 0, 100 * vc["VarComp"] / total_var, np.nan)
    vc["%StudyVar"] = np.where(total_sd_value > 0, 100 * vc["StdDev"] / total_sd_value, np.nan)

    if tolerance is not None:
        vc["%Tolerance"] = 100 * vc["StudyVar"] / tolerance
    else:
        vc["%Tolerance"] = np.nan

    if grr_sd > 0:
        ndc = max(1, math.floor(1.41 * pv / grr_sd))
    else:
        ndc = float("inf")

    grr_percent_study_var = float(vc.loc[vc["Source"] == "Total Gage R&R", "%StudyVar"].iloc[0])

    if grr_percent_study_var < 10:
        aiag_judgment = "Acceptable"
    elif grr_percent_study_var <= 30:
        aiag_judgment = "Marginal"
    else:
        aiag_judgment = "Not acceptable"

    method_table = pd.DataFrame([
        ["Average Range", rbar, "Average range across Part × Operator cells"],
        ["Xbar Difference", xbar_diff, "Range of operator averages"],
        ["Part Average Range", rp, "Range of part averages"],
        ["d2 Replicates", d2_replicates, "d2 constant for replicates"],
        ["d2 Operators", d2_operators, "d2 constant for operators"],
        ["d2 Parts", d2_parts, "d2 constant for parts"],
    ], columns=["Source", "Value", "Notes"])

    interpretation = {
        "interaction_kept": False,
        "p_interaction": np.nan,
        "ndc": ndc,
        "grr_percent_study_var": grr_percent_study_var,
        "aiag_judgment": aiag_judgment,
        "ndc_judgment": "Acceptable discrimination" if ndc == float("inf") or ndc >= 5 else "Poor discrimination",
        "method": "Xbar and R",
    }

    return GRRResult(
        anova_table=method_table.copy(),
        variance_components=vc.copy(),
        settings={
            "parts": n_parts,
            "operators": n_operators,
            "replicates": n_replicates,
            "alpha_remove_interaction": None,
            "study_k": study_k,
            "tolerance": tolerance,
            "grand_mean": data[value_col].mean(),
            "method": "Xbar and R",
        },
        interpretation=interpretation,
        data=data.copy(),
        operator_label=operator_col,
        part_label=part_col,
        value_label=value_col,
        one_operator_mode=False,
    )


# ─────────────────────────────────────────────
# Unified engine
# ─────────────────────────────────────────────

def run_grr_study(
    df: pd.DataFrame,
    part_col: str,
    operator_col: str,
    value_col: str,
    method: str,
    usl: Optional[float] = None,
    lsl: Optional[float] = None,
    alpha_remove_interaction: float = 0.05,
    study_k: float = 6.0,
) -> GRRResult:
    temp = df[[operator_col, value_col]].copy()
    temp[operator_col] = temp[operator_col].astype(str).str.strip()
    temp[value_col] = pd.to_numeric(temp[value_col], errors="coerce")
    temp = temp.replace("", np.nan).dropna()

    n_operators = temp[operator_col].nunique()

    if n_operators == 1:
        return gage_rr_one_operator(
            df=df,
            part_col=part_col,
            operator_col=operator_col,
            value_col=value_col,
            usl=usl,
            lsl=lsl,
            study_k=study_k,
        )

    if method == "ANOVA":
        return gage_rr_crossed_anova(
            df=df,
            part_col=part_col,
            operator_col=operator_col,
            value_col=value_col,
            usl=usl,
            lsl=lsl,
            alpha_remove_interaction=alpha_remove_interaction,
            study_k=study_k,
        )

    if method == "Xbar and R":
        return gage_rr_crossed_xbar_r(
            df=df,
            part_col=part_col,
            operator_col=operator_col,
            value_col=value_col,
            usl=usl,
            lsl=lsl,
            study_k=study_k,
        )

    raise ValueError("Invalid Gage R&R method selected.")


# ─────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────

class GageStudyBuilder(QWidget):
    def __init__(self, app_window, study_type="Crossed", parent=None):
        super().__init__(parent)

        self.app_window = app_window
        self.study_type = study_type
        self.result = None
        self.current_summary = ""
        self.current_journal_entry_id = None
        self.decimals = 2
        self.graph_interactions = None

        self._build_ui()
        self._setup_graph_interactions()
        self.refresh_columns()
        self.debug_connection()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)

        header = QHBoxLayout()

        title_block = QVBoxLayout()
        title = QLabel("Gage R&R Study (Crossed)")
        title.setObjectName("TitleLabel")

        subtitle = QLabel("Analyze repeatability, reproducibility, part-to-part variation and ndc.")
        subtitle.setObjectName("SubtitleLabel")

        title_block.addWidget(title)
        title_block.addWidget(subtitle)

        header.addLayout(title_block)
        header.addStretch()

        self.btn_refresh = QPushButton("Refresh Columns")
        self.btn_refresh.clicked.connect(self.refresh_columns)
        header.addWidget(self.btn_refresh)

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
        layout = QGridLayout(self.setup_tab)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        self.card_variables = QGroupBox("1. Variables")
        self.card_options = QGroupBox("2. Options")
        self.card_run = QGroupBox("3. Run")

        layout.addWidget(self.card_variables, 0, 0)
        layout.addWidget(self.card_options, 0, 1)
        layout.addWidget(self.card_run, 1, 0, 1, 2)

        layout.setColumnStretch(0, 6)
        layout.setColumnStretch(1, 4)

        self._build_variables_card()
        self._build_options_card()
        self._build_run_card()

    def _build_variables_card(self):
        layout = QGridLayout(self.card_variables)

        self.cmb_part = QComboBox()
        self.cmb_operator = QComboBox()
        self.cmb_value = QComboBox()

        layout.addWidget(QLabel("Part column"), 0, 0)
        layout.addWidget(self.cmb_part, 0, 1)

        layout.addWidget(QLabel("Operator column"), 1, 0)
        layout.addWidget(self.cmb_operator, 1, 1)

        layout.addWidget(QLabel("Measurement column"), 2, 0)
        layout.addWidget(self.cmb_value, 2, 1)

    def _build_options_card(self):
        layout = QGridLayout(self.card_options)
        layout.setHorizontalSpacing(14)
        layout.setVerticalSpacing(10)
        layout.setColumnMinimumWidth(0, 180)
        layout.setColumnStretch(1, 1)

        self.cmb_method = QComboBox()
        self.cmb_method.addItems(["ANOVA", "Xbar and R"])
        self.cmb_method.setCurrentText("ANOVA")

        self.spn_decimals = QSpinBox()
        self.spn_decimals.setRange(0, 6)
        self.spn_decimals.setSingleStep(1)
        self.spn_decimals.setValue(2)

        self.chk_use_tolerance = QCheckBox("Use specification limits")
        self.chk_use_tolerance.setChecked(False)

        self.spn_lsl = QDoubleSpinBox()
        self.spn_lsl.setRange(-1_000_000_000, 1_000_000_000)
        self.spn_lsl.setDecimals(6)
        self.spn_lsl.setValue(0.0)

        self.spn_usl = QDoubleSpinBox()
        self.spn_usl.setRange(-1_000_000_000, 1_000_000_000)
        self.spn_usl.setDecimals(6)
        self.spn_usl.setValue(0.0)

        self.spn_alpha = QDoubleSpinBox()
        self.spn_alpha.setRange(0.001, 0.50)
        self.spn_alpha.setDecimals(3)
        self.spn_alpha.setSingleStep(0.005)
        self.spn_alpha.setValue(0.05)

        self.spn_study_k = QDoubleSpinBox()
        self.spn_study_k.setRange(1.0, 10.0)
        self.spn_study_k.setDecimals(2)
        self.spn_study_k.setSingleStep(0.5)
        self.spn_study_k.setValue(6.0)

        self.chk_auto_journal = QCheckBox("Auto journal save")
        self.chk_auto_journal.setChecked(True)

        layout.addWidget(QLabel("Method"), 0, 0)
        layout.addWidget(self.cmb_method, 0, 1)

        layout.addWidget(QLabel("Decimal places"), 1, 0)
        layout.addWidget(self.spn_decimals, 1, 1)

        layout.addWidget(self.chk_use_tolerance, 2, 0, 1, 2)

        layout.addWidget(QLabel("LSL"), 3, 0)
        layout.addWidget(self.spn_lsl, 3, 1)

        layout.addWidget(QLabel("USL"), 4, 0)
        layout.addWidget(self.spn_usl, 4, 1)

        layout.addWidget(QLabel("Alpha for interaction"), 5, 0)
        layout.addWidget(self.spn_alpha, 5, 1)

        layout.addWidget(QLabel("Study variation multiplier"), 6, 0)
        layout.addWidget(self.spn_study_k, 6, 1)

        layout.addWidget(self.chk_auto_journal, 7, 0, 1, 2)

        self.cmb_method.currentTextChanged.connect(self._method_changed)
        self._method_changed(self.cmb_method.currentText())

    def _method_changed(self, method):
        is_anova = method == "ANOVA"
        self.spn_alpha.setEnabled(is_anova)

        if is_anova:
            self.spn_alpha.setToolTip("Alpha used to decide whether Part × Operator interaction is retained.")
        else:
            self.spn_alpha.setToolTip("Not used by Xbar and R method.")

    def _build_run_card(self):
        layout = QVBoxLayout(self.card_run)

        self.btn_run = QPushButton("Run Gage R&R Study")
        self.btn_run.setObjectName("PrimaryButton")
        self.btn_run.clicked.connect(self.run_study)

        note = QLabel("Required structure: balanced crossed design with Part, Operator and Measurement columns.")
        note.setObjectName("SubtitleLabel")
        note.setWordWrap(True)

        layout.addWidget(self.btn_run)
        layout.addWidget(note)

    def _build_results_tab(self):
        layout = QVBoxLayout(self.results_tab)

        metrics_box = QGroupBox("Key Metrics")
        metrics_layout = QGridLayout(metrics_box)

        self.lbl_grr = QLabel("-")
        self.lbl_ndc = QLabel("-")
        self.lbl_aiag = QLabel("-")
        self.lbl_interaction = QLabel("-")

        metrics_layout.addWidget(QLabel("Total Gage R&R %StudyVar"), 0, 0)
        metrics_layout.addWidget(self.lbl_grr, 0, 1)

        metrics_layout.addWidget(QLabel("Number of distinct categories"), 1, 0)
        metrics_layout.addWidget(self.lbl_ndc, 1, 1)

        metrics_layout.addWidget(QLabel("AIAG judgment"), 2, 0)
        metrics_layout.addWidget(self.lbl_aiag, 2, 1)

        metrics_layout.addWidget(QLabel("Interaction decision"), 3, 0)
        metrics_layout.addWidget(self.lbl_interaction, 3, 1)

        layout.addWidget(metrics_box)

        self.tbl_anova = QTableWidget()
        self.tbl_vc = QTableWidget()

        layout.addWidget(QLabel("ANOVA / Method Table"))
        layout.addWidget(self.tbl_anova)

        layout.addWidget(QLabel("Variance Components"))
        layout.addWidget(self.tbl_vc)

    def _build_graphs_tab(self):
        layout = QVBoxLayout(self.graphs_tab)

        toolbar = QHBoxLayout()

        self.btn_refresh_graphs = QPushButton("Refresh Graphs")
        self.btn_refresh_graphs.clicked.connect(self.refresh_graphs)

        self.btn_copy_graph = QPushButton("Copy Graph")
        self.btn_copy_graph.clicked.connect(self.copy_graph_to_clipboard)

        self.btn_export_graph = QPushButton("Export PNG")
        self.btn_export_graph.clicked.connect(self.export_graph_png)

        self.btn_copy_report = QPushButton("Copy Graph + Summary")
        self.btn_copy_report.clicked.connect(self.copy_graph_summary_to_clipboard)

        self.btn_export_report = QPushButton("Export Graph + Summary")
        self.btn_export_report.clicked.connect(self.export_graph_summary_png)

        toolbar.addWidget(self.btn_refresh_graphs)
        toolbar.addWidget(self.btn_copy_graph)
        toolbar.addWidget(self.btn_export_graph)
        toolbar.addWidget(self.btn_copy_report)
        toolbar.addWidget(self.btn_export_report)
        toolbar.addStretch()

        layout.addLayout(toolbar)

        self.figure = Figure(figsize=(12, 8), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

    def _build_summary_tab(self):
        layout = QVBoxLayout(self.summary_tab)

        toolbar = QHBoxLayout()
        self.btn_copy_summary = QPushButton("Copy Summary")
        self.btn_copy_summary.clicked.connect(self.copy_summary)
        toolbar.addWidget(self.btn_copy_summary)
        toolbar.addStretch()
        layout.addLayout(toolbar)

        self.summary_text = QTextEdit()
        self.summary_text.setReadOnly(True)
        layout.addWidget(self.summary_text)

    def _build_debug_tab(self):
        layout = QVBoxLayout(self.debug_tab)

        toolbar = QHBoxLayout()
        btn_refresh = QPushButton("Refresh Debug")
        btn_refresh.clicked.connect(self.debug_connection)

        btn_clear = QPushButton("Clear Debug")
        btn_clear.clicked.connect(lambda: self.debug_text.clear())

        toolbar.addWidget(btn_refresh)
        toolbar.addWidget(btn_clear)
        toolbar.addStretch()

        layout.addLayout(toolbar)

        self.debug_text = QTextEdit()
        self.debug_text.setReadOnly(True)
        layout.addWidget(self.debug_text)

    # ─────────────────────────────────────────────
    # Graph interaction integration
    # ─────────────────────────────────────────────

    def _setup_graph_interactions(self):
        self.graph_interactions = enable_interaction(
            host=self,
            figure_getter=lambda: self.figure,
            canvas_getter=lambda: self.canvas,
            decimals_getter=lambda: int(self.decimals),
            on_changed=self._on_graph_interaction_changed,
            on_status=self._set_graph_status,
        )

    def _on_graph_interaction_changed(self):
        if self.result is not None:
            self._sync_current_graph_payload_to_journal()

    def _set_graph_status(self, message: str):
        try:
            self.app_window.statusBar().showMessage(message)
        except Exception:
            pass

    # ─────────────────────────────────────────────
    # Data helpers
    # ─────────────────────────────────────────────

    def log(self, text):
        stamp = datetime.now().strftime("%H:%M:%S")
        if hasattr(self, "debug_text"):
            self.debug_text.append(f"[{stamp}] {text}")

    def get_df(self):
        df = self.app_window.get_active_dataframe()
        return df if df is not None else pd.DataFrame()

    def all_columns(self):
        df = self.get_df()
        return list(df.columns) if not df.empty else []

    def numeric_columns(self):
        df = self.get_df()
        if df.empty:
            return []

        numeric_cols = []
        for col in df.columns:
            if pd.to_numeric(df[col], errors="coerce").notna().sum() > 0:
                numeric_cols.append(col)

        return numeric_cols

    def refresh_columns(self):
        df = self.get_df()
        all_cols = self.all_columns()
        numeric_cols = self.numeric_columns()

        for cmb in [self.cmb_part, self.cmb_operator, self.cmb_value]:
            cmb.clear()

        self.cmb_part.addItems(all_cols)
        self.cmb_operator.addItems(all_cols)
        self.cmb_value.addItems(numeric_cols)

        self._auto_select_columns(all_cols, numeric_cols)

        if df.empty:
            self.log("refresh_columns: DataFrame is empty.")
        else:
            self.log(f"Columns refreshed. all={all_cols} | numeric={numeric_cols}")

    def _auto_select_columns(self, all_cols, numeric_cols):
        def find_match(candidates):
            for c in all_cols:
                low = str(c).lower()
                if any(token in low for token in candidates):
                    return c
            return None

        part_guess = find_match(["part", "pieza", "sample", "unit", "serial", "serialno"])
        operator_guess = find_match(["operator", "operador", "appraiser", "inspector", "fixture"])
        value_guess = find_match(["value", "measurement", "result", "resultado", "medicion", "medición", "error"])

        if part_guess:
            self.cmb_part.setCurrentText(part_guess)

        if operator_guess:
            self.cmb_operator.setCurrentText(operator_guess)

        if value_guess and value_guess in numeric_cols:
            self.cmb_value.setCurrentText(value_guess)
        elif numeric_cols:
            self.cmb_value.setCurrentText(numeric_cols[0])

    def debug_connection(self):
        self.debug_text.clear()
        self.log("Checking connection to main window...")
        self.log(f"get_active_dataframe  : {hasattr(self.app_window, 'get_active_dataframe')}")
        self.log(f"active_worksheet_name : {hasattr(self.app_window, 'active_worksheet_name')}")
        self.log(f"add_journal_entry     : {hasattr(self.app_window, 'add_journal_entry')}")

        df = self.get_df()
        self.log(f"Active worksheet      : {self.app_window.active_worksheet_name() if hasattr(self.app_window, 'active_worksheet_name') else 'N/A'}")
        self.log(f"DataFrame shape       : {df.shape if df is not None else None}")

        if df is None or df.empty:
            self.log("WARNING: DataFrame is empty. Paste data into Worksheet first.")
            return

        for col in df.columns:
            self.log(f"  - {col} | non-null={df[col].notna().sum()}")

        self.log(f"Numeric columns       : {self.numeric_columns()}")

    # ─────────────────────────────────────────────
    # Run study
    # ─────────────────────────────────────────────

    def run_study(self):
        df = self.get_df()

        if df.empty:
            QMessageBox.warning(self, "No data", "Paste data into the active worksheet first.")
            return

        method = self.cmb_method.currentText()
        self.decimals = int(self.spn_decimals.value())

        part_col = self.cmb_part.currentText()
        operator_col = self.cmb_operator.currentText()
        value_col = self.cmb_value.currentText()

        if not part_col or not operator_col or not value_col:
            QMessageBox.warning(self, "Missing columns", "Select Part, Operator and Measurement columns.")
            return

        if part_col == operator_col or part_col == value_col or operator_col == value_col:
            QMessageBox.warning(self, "Invalid selection", "Part, Operator and Measurement columns must be different.")
            return

        lsl = None
        usl = None

        if self.chk_use_tolerance.isChecked():
            lsl = self.spn_lsl.value()
            usl = self.spn_usl.value()

            if usl <= lsl:
                QMessageBox.warning(self, "Invalid specification limits", "USL must be greater than LSL.")
                return

        try:
            self.result = run_grr_study(
                df=df,
                part_col=part_col,
                operator_col=operator_col,
                value_col=value_col,
                method=method,
                usl=usl,
                lsl=lsl,
                alpha_remove_interaction=self.spn_alpha.value(),
                study_k=self.spn_study_k.value(),
            )

            self._show_result(self.result)
            self.refresh_graphs()

            self.current_summary = self._build_summary(self.result)
            self.summary_text.setPlainText(self.current_summary)

            if self.chk_auto_journal.isChecked():
                self._save_or_update_journal_entry()

            self.tabs.setCurrentWidget(self.results_tab)
            self.app_window.statusBar().showMessage("Gage R&R Study completed.")

        except Exception as exc:
            self.log(f"ERROR while running Gage R&R Study: {exc}")
            QMessageBox.critical(self, "Gage R&R Study failed", str(exc))

    def _show_result(self, result: GRRResult):
        d = int(self.decimals)
        interp = result.interpretation

        self.lbl_grr.setText(f"{interp['grr_percent_study_var']:.{d}f}%")
        self.lbl_ndc.setText(str(interp["ndc"]))
        self.lbl_aiag.setText(interp["aiag_judgment"])

        if result.one_operator_mode:
            operator_name = interp.get("operator_name", "N/A")
            self.lbl_interaction.setText(
                f"Single operator detected: {operator_name}. Reproducibility cannot be estimated."
            )
        else:
            p_interaction = interp.get("p_interaction", float("nan"))
            method = interp.get("method", self.cmb_method.currentText())

            if method == "ANOVA":
                if interp["interaction_kept"]:
                    self.lbl_interaction.setText(f"Interaction retained (P = {p_interaction:.{d}f})")
                else:
                    self.lbl_interaction.setText(f"Interaction pooled into repeatability (P = {p_interaction:.{d}f})")
            else:
                self.lbl_interaction.setText("Not evaluated by Xbar and R method")

        self._populate_table(self.tbl_anova, result.anova_table)
        self._populate_table(self.tbl_vc, result.variance_components)

    # ─────────────────────────────────────────────
    # Graphs
    # ─────────────────────────────────────────────

    def refresh_graphs(self):
        if self.result is None:
            QMessageBox.information(self, "No study results", "Run a Gage R&R Study before refreshing graphs.")
            return

        self.figure.clear()

        if self.result.one_operator_mode:
            self._plot_one_operator_four_pack()
        else:
            self._plot_multi_operator_six_pack()

        self.figure.tight_layout(rect=[0, 0, 1, 0.96])

        if self.graph_interactions is not None:
            self.graph_interactions.register_axes()
            self.graph_interactions.apply_reference_lines()

        self.canvas.draw()

    def _plot_multi_operator_six_pack(self):
        result = self.result
        df = result.data.copy()
        vc = result.variance_components.copy()

        op_col = result.operator_label
        part_col = result.part_label
        value_col = result.value_label
        d = int(self.decimals)

        axes = self.figure.subplots(3, 2)
        self.figure.suptitle(f"Gage R&R Study Report for {value_col}", fontsize=14, fontweight="bold")

        ax = axes[0, 0]
        plot_vc = vc[vc["Source"].isin(["Total Gage R&R", "Repeatability", "Reproducibility", "Part-To-Part"])].copy()
        x = np.arange(len(plot_vc))
        width = 0.35
        bars_contribution = ax.bar(x - width / 2, plot_vc["%Contribution"].fillna(0), width, label="% Contribution")
        bars_study_var = ax.bar(x + width / 2, plot_vc["%StudyVar"].fillna(0), width, label="% Study Var")
        for bar in bars_contribution:
            bar._sps_legend_label = "% Contribution"
            bar._sps_series_key = "Components:% Contribution"
        for bar in bars_study_var:
            bar._sps_legend_label = "% Study Var"
            bar._sps_series_key = "Components:% Study Var"
        ax.axhline(10, linestyle="--", linewidth=0.8, alpha=0.6)
        ax.axhline(30, linestyle="--", linewidth=0.8, alpha=0.6)
        ax.set_title("Components of Variation")
        ax.set_ylabel("Percent")
        ax.set_xticks(x)
        ax.set_xticklabels(plot_vc["Source"], rotation=15, ha="right")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.25)

        ax = axes[0, 1]
        part_means = df.groupby(part_col)[value_col].mean().reset_index()
        ax.plot(part_means[part_col].astype(str), part_means[value_col], marker="o")
        ax.set_title(f"{value_col} by {part_col}")
        ax.set_xlabel(part_col)
        ax.set_ylabel(f"Mean {value_col}")
        ax.tick_params(axis="x", rotation=45)
        ax.grid(True, alpha=0.25)

        subgroup = df.groupby([op_col, part_col])[value_col].agg(["mean", "max", "min", "count"]).reset_index()
        subgroup["range"] = subgroup["max"] - subgroup["min"]
        subgroup["label"] = subgroup[op_col].astype(str) + "-" + subgroup[part_col].astype(str)

        n = int(result.settings["replicates"])
        const = CONTROL_CONSTANTS.get(n)

        ax = axes[1, 0]
        rbar = subgroup["range"].mean()
        ax.plot(range(len(subgroup)), subgroup["range"], marker="o")
        ax.axhline(rbar, linestyle="--", label=f"Rbar = {rbar:.{d}f}")
        if const:
            ucl_r = const["D4"] * rbar
            lcl_r = const["D3"] * rbar
            ax.axhline(ucl_r, linestyle=":", label=f"UCL = {ucl_r:.{d}f}")
            ax.axhline(lcl_r, linestyle=":", label=f"LCL = {lcl_r:.{d}f}")
        self._add_operator_separators(ax, subgroup, op_col)
        ax.set_title(f"R Chart by {op_col}")
        ax.set_xlabel(f"Subgroup ({op_col}-{part_col})")
        ax.set_ylabel("Sample Range")
        ax.set_xticks(range(len(subgroup)))
        ax.set_xticklabels(subgroup["label"], rotation=60, fontsize=7)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.25)

        ax = axes[1, 1]
        operators = sorted(df[op_col].unique())
        data_box = [df.loc[df[op_col] == op, value_col].dropna().values for op in operators]
        ax.boxplot(data_box, labels=[str(op) for op in operators], showmeans=True)
        ax.set_title(f"{value_col} by {op_col}")
        ax.set_xlabel(op_col)
        ax.set_ylabel(value_col)
        ax.grid(True, alpha=0.25)

        ax = axes[2, 0]
        xbarbar = subgroup["mean"].mean()
        ax.plot(range(len(subgroup)), subgroup["mean"], marker="o")
        ax.axhline(xbarbar, linestyle="--", label=f"CL = {xbarbar:.{d}f}")
        if const:
            ucl_x = xbarbar + const["A2"] * rbar
            lcl_x = xbarbar - const["A2"] * rbar
            ax.axhline(ucl_x, linestyle=":", label=f"UCL = {ucl_x:.{d}f}")
            ax.axhline(lcl_x, linestyle=":", label=f"LCL = {lcl_x:.{d}f}")
        self._add_operator_separators(ax, subgroup, op_col)
        ax.set_title(f"Xbar Chart by {op_col}")
        ax.set_xlabel(f"Subgroup ({op_col}-{part_col})")
        ax.set_ylabel("Sample Mean")
        ax.set_xticks(range(len(subgroup)))
        ax.set_xticklabels(subgroup["label"], rotation=60, fontsize=7)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.25)

        ax = axes[2, 1]
        interaction = df.groupby([part_col, op_col])[value_col].mean().reset_index()
        for op in sorted(interaction[op_col].unique()):
            temp = interaction[interaction[op_col] == op].sort_values(part_col)
            line, = ax.plot(temp[part_col].astype(str), temp[value_col], marker="o", label=str(op))
            line._sps_legend_label = str(op)
            line._sps_series_key = f"{op_col}:{op}"
        ax.set_title(f"{part_col} * {op_col} Interaction")
        ax.set_xlabel(part_col)
        ax.set_ylabel(f"Average {value_col}")
        ax.tick_params(axis="x", rotation=45)
        ax.legend(title=op_col, fontsize=8)
        ax.grid(True, alpha=0.25)

    def _plot_one_operator_four_pack(self):
        result = self.result
        df = result.data.copy()
        vc = result.variance_components.copy()

        part_col = result.part_label
        value_col = result.value_label
        d = int(self.decimals)

        axes = self.figure.subplots(2, 2)
        self.figure.suptitle(
            f"Gage R&R Study Report for {value_col}\nSingle Operator - Repeatability Only",
            fontsize=14,
            fontweight="bold"
        )

        ax = axes[0, 0]
        plot_vc = vc[vc["Source"].isin(["Total Gage R&R", "Repeatability", "Part-To-Part"])].copy()
        x = np.arange(len(plot_vc))
        width = 0.35
        bars_contribution = ax.bar(x - width / 2, plot_vc["%Contribution"].fillna(0), width, label="% Contribution")
        bars_study_var = ax.bar(x + width / 2, plot_vc["%StudyVar"].fillna(0), width, label="% Study Var")
        for bar in bars_contribution:
            bar._sps_legend_label = "% Contribution"
            bar._sps_series_key = "Components:% Contribution"
        for bar in bars_study_var:
            bar._sps_legend_label = "% Study Var"
            bar._sps_series_key = "Components:% Study Var"
        ax.axhline(10, linestyle="--", linewidth=0.8, alpha=0.6)
        ax.axhline(30, linestyle="--", linewidth=0.8, alpha=0.6)
        ax.set_title("Components of Variation")
        ax.set_ylabel("Percent")
        ax.set_xticks(x)
        ax.set_xticklabels(plot_vc["Source"], rotation=15, ha="right")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.25)

        ax = axes[0, 1]
        part_means = df.groupby(part_col)[value_col].mean().reset_index()
        ax.plot(part_means[part_col].astype(str), part_means[value_col], marker="o")
        ax.set_title(f"{value_col} by {part_col}")
        ax.set_xlabel(part_col)
        ax.set_ylabel(f"Mean {value_col}")
        ax.tick_params(axis="x", rotation=45)
        ax.grid(True, alpha=0.25)

        subgroup = df.groupby(part_col)[value_col].agg(["mean", "max", "min", "count"]).reset_index()
        subgroup["range"] = subgroup["max"] - subgroup["min"]

        n = int(result.settings["replicates"])
        const = CONTROL_CONSTANTS.get(n)
        rbar = subgroup["range"].mean()

        ax = axes[1, 0]
        ax.plot(range(len(subgroup)), subgroup["range"], marker="o")
        ax.axhline(rbar, linestyle="--", label=f"Rbar = {rbar:.{d}f}")
        if const:
            ucl_r = const["D4"] * rbar
            lcl_r = const["D3"] * rbar
            ax.axhline(ucl_r, linestyle=":", label=f"UCL = {ucl_r:.{d}f}")
            ax.axhline(lcl_r, linestyle=":", label=f"LCL = {lcl_r:.{d}f}")
        ax.set_title("R Chart")
        ax.set_xlabel(part_col)
        ax.set_ylabel("Sample Range")
        ax.set_xticks(range(len(subgroup)))
        ax.set_xticklabels(subgroup[part_col].astype(str), rotation=45, fontsize=7)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.25)

        ax = axes[1, 1]
        xbarbar = subgroup["mean"].mean()
        ax.plot(range(len(subgroup)), subgroup["mean"], marker="o")
        ax.axhline(xbarbar, linestyle="--", label=f"CL = {xbarbar:.{d}f}")
        if const:
            ucl_x = xbarbar + const["A2"] * rbar
            lcl_x = xbarbar - const["A2"] * rbar
            ax.axhline(ucl_x, linestyle=":", label=f"UCL = {ucl_x:.{d}f}")
            ax.axhline(lcl_x, linestyle=":", label=f"LCL = {lcl_x:.{d}f}")
        ax.set_title("Xbar Chart")
        ax.set_xlabel(part_col)
        ax.set_ylabel("Sample Mean")
        ax.set_xticks(range(len(subgroup)))
        ax.set_xticklabels(subgroup[part_col].astype(str), rotation=45, fontsize=7)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.25)

    def _add_operator_separators(self, ax, subgroup: pd.DataFrame, op_col: str):
        operators = list(subgroup[op_col].unique())
        start = 0
        for op in operators[:-1]:
            qty = len(subgroup[subgroup[op_col] == op])
            start += qty
            ax.axvline(start - 0.5, alpha=0.4, linewidth=0.8)

    # ─────────────────────────────────────────────
    # Report Engine
    # ─────────────────────────────────────────────

    def _figure_png_bytes(self, figure=None, dpi=180) -> bytes:
        fig = figure or self.figure
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", dpi=dpi, bbox_inches="tight", facecolor="white")
        data = buffer.getvalue()
        buffer.close()
        return data

    def _set_clipboard_png(self, png_bytes: bytes):
        pixmap = QPixmap()
        if not pixmap.loadFromData(png_bytes, "PNG"):
            raise RuntimeError("Unable to load generated image into clipboard.")
        QApplication.clipboard().setPixmap(pixmap)

    def copy_graph_to_clipboard(self):
        if self.result is None:
            QMessageBox.information(self, "No graph", "Run a Gage R&R Study before copying the graph.")
            return

        try:
            self._set_clipboard_png(self._figure_png_bytes(self.figure, dpi=180))
            self.app_window.statusBar().showMessage("Graph copied to clipboard.")
        except Exception as exc:
            QMessageBox.critical(self, "Copy Graph failed", str(exc))

    def export_graph_png(self):
        if self.result is None:
            QMessageBox.information(self, "No graph", "Run a Gage R&R Study before exporting the graph.")
            return

        default_name = f"Gage_RR_Graph_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        path, _ = QFileDialog.getSaveFileName(self, "Export Graph as PNG", default_name, "PNG Files (*.png)")
        if not path:
            return

        if not path.lower().endswith(".png"):
            path += ".png"

        try:
            self.figure.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
            self.app_window.statusBar().showMessage(f"Graph exported: {path}")
        except Exception as exc:
            QMessageBox.critical(self, "Export Graph failed", str(exc))

    def _build_graph_summary_figure(self) -> Figure:
        if self.result is None:
            raise RuntimeError("Run a Gage R&R Study before creating a report.")

        # Render the current graph into an image and compose it with the summary text.
        graph_png = self._figure_png_bytes(self.figure, dpi=160)
        graph_img = mpimg.imread(io.BytesIO(graph_png), format="png")

        report_fig = Figure(figsize=(12, 14), dpi=120)

        ax_graph = report_fig.add_subplot(2, 1, 1)
        ax_graph.imshow(graph_img)
        ax_graph.axis("off")
        ax_graph.set_title("Gage R&R Graph Report", fontweight="bold", pad=8)

        ax_summary = report_fig.add_subplot(2, 1, 2)
        summary = self.summary_text.toPlainText().strip() or self.current_summary or self._build_summary(self.result)
        ax_summary.text(
            0.01,
            0.99,
            summary,
            va="top",
            ha="left",
            fontsize=8,
            family="monospace",
            wrap=True,
        )
        ax_summary.axis("off")

        report_fig.tight_layout()
        return report_fig

    def copy_graph_summary_to_clipboard(self):
        if self.result is None:
            QMessageBox.information(self, "No report", "Run a Gage R&R Study before copying the report.")
            return

        try:
            report_fig = self._build_graph_summary_figure()
            self._set_clipboard_png(self._figure_png_bytes(report_fig, dpi=180))
            self.app_window.statusBar().showMessage("Graph + Summary copied to clipboard.")
        except Exception as exc:
            QMessageBox.critical(self, "Copy Report failed", str(exc))

    def export_graph_summary_png(self):
        if self.result is None:
            QMessageBox.information(self, "No report", "Run a Gage R&R Study before exporting the report.")
            return

        default_name = f"Gage_RR_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        path, _ = QFileDialog.getSaveFileName(self, "Export Graph + Summary as PNG", default_name, "PNG Files (*.png)")
        if not path:
            return

        if not path.lower().endswith(".png"):
            path += ".png"

        try:
            report_fig = self._build_graph_summary_figure()
            report_fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
            self.app_window.statusBar().showMessage(f"Report exported: {path}")
        except Exception as exc:
            QMessageBox.critical(self, "Export Report failed", str(exc))

    def _current_graph_base64(self) -> str:
        if self.result is None:
            return ""
        return base64.b64encode(self._figure_png_bytes(self.figure, dpi=140)).decode("utf-8")

    def _current_report_base64(self) -> str:
        if self.result is None:
            return ""
        report_fig = self._build_graph_summary_figure()
        return base64.b64encode(self._figure_png_bytes(report_fig, dpi=140)).decode("utf-8")

    def _sync_current_graph_payload_to_journal(self):
        """Update only image/summary fields when graph interactions change.

        This keeps the Analysis Journal snapshot current after title edits, axis edits,
        scale changes, or reference line changes without creating duplicate entries.
        """
        if not self.current_journal_entry_id:
            return

        if not hasattr(self.app_window, "update_journal_entry"):
            return

        if self.result is None:
            return

        try:
            self.current_summary = self._build_summary(self.result)
            if hasattr(self, "summary_text"):
                self.summary_text.setPlainText(self.current_summary)

            self.app_window.update_journal_entry(
                self.current_journal_entry_id,
                self._journal_updates(change_note="Graph updated"),
            )
        except Exception as exc:
            self.log(f"WARNING: Unable to sync journal graph snapshot: {exc}")

    # ─────────────────────────────────────────────
    # Results formatting
    # ─────────────────────────────────────────────

    def _populate_table(self, table: QTableWidget, df: pd.DataFrame):
        display = df.copy()
        decimals = int(self.decimals)

        for col in display.columns:
            if pd.api.types.is_numeric_dtype(display[col]):
                display[col] = display[col].map(lambda x: "" if pd.isna(x) else f"{x:.{decimals}f}")

        table.clear()
        table.setRowCount(len(display))
        table.setColumnCount(len(display.columns))
        table.setHorizontalHeaderLabels([str(c) for c in display.columns])

        for r in range(len(display)):
            for c, col in enumerate(display.columns):
                item = QTableWidgetItem(str(display.iloc[r, c]))
                table.setItem(r, c, item)

        table.resizeColumnsToContents()

    def _build_summary(self, result: GRRResult) -> str:
        d = int(self.decimals)

        interp = result.interpretation
        settings = result.settings
        method = settings.get("method", self.cmb_method.currentText())

        grr = interp["grr_percent_study_var"]
        ndc = interp["ndc"]
        interaction_p = interp.get("p_interaction", float("nan"))

        if grr < 10:
            recommendation = "The measurement system is acceptable. It can generally be used for process analysis and control."
        elif grr <= 30:
            recommendation = (
                "The measurement system is marginal. It may be acceptable depending on application risk, "
                "cost of measurement improvement and process criticality."
            )
        else:
            recommendation = "The measurement system is not acceptable. Measurement system improvement is recommended before relying on this data."

        ndc_text = "Infinite" if ndc == float("inf") else str(ndc)
        tolerance_text = f"{settings['tolerance']:.{d}f}" if settings.get("tolerance") is not None else "Not used"

        single_operator_note = ""
        if result.one_operator_mode:
            single_operator_note = """
Single Operator Notice
- Only one operator was detected.
- Reproducibility cannot be estimated.
- Gage R&R equals Repeatability.
- One-Way ANOVA was used automatically.
"""

        if result.one_operator_mode:
            method_block = """
Method Decision
- Single operator mode: Yes
- Operator × Part interaction: Not evaluated.
- Reproducibility: Not estimated.
"""
        elif method == "ANOVA":
            method_block = f"""
ANOVA Decision
- Operator × Part interaction retained: {"Yes" if interp["interaction_kept"] else "No"}
- Interaction P-value: {interaction_p:.{d}f}
- Alpha for interaction: {settings["alpha_remove_interaction"]:.{d}f}
"""
        else:
            method_block = """
Method Decision
- Operator × Part interaction: Not evaluated by Xbar and R method.
- Method note: Xbar and R estimates variation from average ranges and ranges of averages.
"""

        summary = f"""Gage R&R Study (Crossed) Summary

Study Design
- Worksheet: {self.app_window.active_worksheet_name()}
- Method: {method}
- Part column: {result.part_label}
- Operator column: {result.operator_label}
- Measurement column: {result.value_label}
- Parts: {settings["parts"]}
- Operators: {settings["operators"]}
- Replicates per Part × Operator: {settings["replicates"]}
- Study variation multiplier: {settings["study_k"]:.{d}f}
- Decimal places: {d}
- Tolerance: {tolerance_text}
{single_operator_note}
Key Results
- Total Gage R&R %StudyVar: {grr:.{d}f}%
- Number of distinct categories: {ndc_text}
- AIAG judgment: {interp["aiag_judgment"]}
- NDC judgment: {interp["ndc_judgment"]}
{method_block}
Interpretation
{recommendation}
"""
        return summary

    def copy_summary(self):
        text = self.summary_text.toPlainText()
        if not text.strip():
            return

        self.summary_text.selectAll()
        self.summary_text.copy()

    # ─────────────────────────────────────────────
    # Journal integration
    # ─────────────────────────────────────────────

    def _now(self) -> str:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def _safe_graph_base64(self) -> str:
        try:
            return self._current_graph_base64()
        except Exception as exc:
            self.log(f"WARNING: Unable to encode graph image for journal: {exc}")
            return ""

    def _safe_report_base64(self) -> str:
        try:
            return self._current_report_base64()
        except Exception as exc:
            self.log(f"WARNING: Unable to encode report image for journal: {exc}")
            return ""

    def _journal_payload(self) -> Dict[str, Any]:
        if self.result is None:
            return {}

        return {
            "study": "Gage R&R Study (Crossed)",
            "module": "GageStudyBuilder",
            "method": self.result.settings.get("method", self.cmb_method.currentText()),
            "decimals": int(self.decimals),
            "part_col": self.result.part_label,
            "operator_col": self.result.operator_label,
            "value_col": self.result.value_label,
            "use_tolerance": bool(self.chk_use_tolerance.isChecked()),
            "lsl": self.spn_lsl.value(),
            "usl": self.spn_usl.value(),
            "alpha": self.spn_alpha.value(),
            "study_k": self.spn_study_k.value(),
            "settings": self.result.settings,
            "interpretation": self.result.interpretation,
            "one_operator_mode": bool(self.result.one_operator_mode),
            "anova_table": self.result.anova_table.to_dict(orient="records"),
            "variance_components": self.result.variance_components.to_dict(orient="records"),
            "data": self.result.data.to_dict(orient="records"),
            "graph_reference_lines": (
                self.graph_interactions.to_dict()
                if self.graph_interactions is not None and hasattr(self.graph_interactions, "to_dict")
                else None
            ),
        }

    def _journal_updates(self, change_note: str = "Updated") -> Dict[str, Any]:
        if self.result is None:
            return {}

        graph_image = self._safe_graph_base64()
        report_image = self._safe_report_base64()

        status = self.result.interpretation.get("aiag_judgment", "")
        method = self.result.settings.get("method", self.cmb_method.currentText())

        return {
            "name": f"Gage R&R Study ({method})",
            "type": "Gage Study",
            "worksheet": self.app_window.active_worksheet_name(),
            "status": status,
            "summary": self.current_summary,
            # main_window.py journal viewer expects this key for the image preview.
            "figure_png_base64": graph_image,
            # Extra keys kept for future report viewer/export support.
            "report_png_base64": report_image,
            "image": graph_image,
            "report_image": report_image,
            "payload": self._journal_payload(),
            "last_change": change_note,
        }

    def _save_or_update_journal_entry(self):
        if self.result is None:
            return

        if not self.current_summary:
            self.current_summary = self._build_summary(self.result)

        # Update current entry when editing/re-running an existing Journal item.
        if self.current_journal_entry_id and hasattr(self.app_window, "update_journal_entry"):
            updated = self.app_window.update_journal_entry(
                self.current_journal_entry_id,
                self._journal_updates(change_note="Updated"),
            )
            if updated:
                self.app_window.statusBar().showMessage("Gage Study journal entry updated.")
                return

        self._save_journal_entry()

    def _save_journal_entry(self):
        if not hasattr(self.app_window, "add_journal_entry"):
            return

        if self.result is None:
            return

        entry_id = f"gage_rr_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        now = self._now()

        entry = {
            "id": entry_id,
            "created_at": now,
            "modified_at": now,
            **self._journal_updates(change_note="Created"),
        }

        self.app_window.add_journal_entry(entry)
        self.current_journal_entry_id = entry_id

        try:
            self.app_window.statusBar().showMessage("Gage Study journal entry created.")
        except Exception:
            pass

    def load_from_journal_entry(self, entry: Dict[str, Any]):
        """Load an existing Gage Study journal entry for review and re-run/update.

        This method is intended to be called from main_window.py through a method
        such as open_gage_study_from_journal(entry_id).
        """
        if not entry:
            QMessageBox.warning(self, "Journal entry not found", "The selected journal entry could not be found.")
            return

        self.current_journal_entry_id = entry.get("id")

        payload = entry.get("payload", {}) or {}

        # Restore controls.
        method = payload.get("method") or payload.get("settings", {}).get("method")
        if method in [self.cmb_method.itemText(i) for i in range(self.cmb_method.count())]:
            self.cmb_method.setCurrentText(method)

        decimals = payload.get("decimals")
        if decimals is not None:
            try:
                self.decimals = int(decimals)
                self.spn_decimals.setValue(self.decimals)
            except Exception:
                pass

        self.chk_use_tolerance.setChecked(bool(payload.get("use_tolerance", False)))

        for attr, key in [(self.spn_lsl, "lsl"), (self.spn_usl, "usl"), (self.spn_alpha, "alpha"), (self.spn_study_k, "study_k")]:
            if key in payload and payload.get(key) is not None:
                try:
                    attr.setValue(float(payload.get(key)))
                except Exception:
                    pass

        # Prefer stored data. If old entries do not have data, fall back to active worksheet.
        data_records = payload.get("data", [])
        data = pd.DataFrame(data_records) if data_records else self.get_df()

        part_col = payload.get("part_col", "")
        operator_col = payload.get("operator_col", "")
        value_col = payload.get("value_col", "")

        self.refresh_columns()
        if part_col:
            self.cmb_part.setCurrentText(part_col)
        if operator_col:
            self.cmb_operator.setCurrentText(operator_col)
        if value_col:
            self.cmb_value.setCurrentText(value_col)

        # Restore result object from payload when possible.
        anova_table = pd.DataFrame(payload.get("anova_table", []))
        variance_components = pd.DataFrame(payload.get("variance_components", []))
        settings = payload.get("settings", {}) or {}
        interpretation = payload.get("interpretation", {}) or {}

        if not data.empty and not variance_components.empty and settings and interpretation:
            self.result = GRRResult(
                anova_table=anova_table,
                variance_components=variance_components,
                settings=settings,
                interpretation=interpretation,
                data=data,
                operator_label=operator_col,
                part_label=part_col,
                value_label=value_col,
                one_operator_mode=bool(payload.get("one_operator_mode", settings.get("operators") == 1)),
            )

            self._show_result(self.result)
            self.current_summary = entry.get("summary", "") or self._build_summary(self.result)
            self.summary_text.setPlainText(self.current_summary)
            self.refresh_graphs()

            # Restore reference lines if the interaction manager supports it.
            refs = payload.get("graph_reference_lines")
            if refs and self.graph_interactions is not None and hasattr(self.graph_interactions, "from_dict"):
                try:
                    self.graph_interactions.from_dict(refs)
                    self.graph_interactions.register_axes()
                    self.graph_interactions.apply_reference_lines()
                    self.canvas.draw_idle()
                except Exception as exc:
                    self.log(f"WARNING: Unable to restore reference lines: {exc}")

        else:
            # Old/incomplete entry: show the saved summary and ask the user to re-run if needed.
            self.current_summary = entry.get("summary", "")
            self.summary_text.setPlainText(self.current_summary or "No summary available.")
            QMessageBox.information(
                self,
                "Partial journal entry",
                "This journal entry does not contain enough stored data to fully restore the study. "
                "Select the active worksheet and run the study again to update this entry.",
            )

        self.tabs.setCurrentWidget(self.summary_tab)

    def rerun_current_journal_entry(self):
        """Convenience wrapper for future toolbar buttons."""
        self.run_study()
