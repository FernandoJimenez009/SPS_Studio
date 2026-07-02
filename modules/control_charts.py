"""
modules/control_charts.py
SPS Studio — Control Charts Builder

PySide6 module integrated with SPS Studio.

Scope:
- Individuals (I) chart and Moving Range (MR) chart.
- Optional stages/phases with stage-specific limits.
- Original data order is preserved.
- 3-sigma limits are always displayed.
- Optional 1-sigma / 2-sigma zone lines.
- Optional extra sigma limits, e.g. 4 means calculated ±4 sigma limits.
- Separate control-test configuration for I Chart and MR Chart.
- Graphs tab contains only charts.
- Results tab contains stage limit tables and control-test summaries.
- Copy/export graph and Analysis Journal integration.

Visible UI text is English.
"""

from __future__ import annotations

import base64
import io
import math
import re
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from PySide6.QtCore import Qt, QEvent
from PySide6.QtGui import QImage
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
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
# Utilities
# ─────────────────────────────────────────────

def now_string() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def fmt(value: Any, decimals: int = 3) -> str:
    try:
        v = float(value)
        if np.isnan(v) or np.isinf(v):
            return "N/A"
        return f"{v:.{decimals}f}"
    except Exception:
        return str(value)


def figure_to_png_bytes(fig: Figure) -> bytes:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=160, bbox_inches="tight")
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


def parse_extra_sigma_limits(text: str) -> List[float]:
    """
    Accepts sigma multipliers separated by comma, semicolon, spaces or new lines.
    Example: 4, 4.5, 5
    These are calculated limits, not reference horizontal values.
    """
    if not text or not text.strip():
        return []
    parts = re.split(r"[,;\s]+", text.strip())
    values = []
    for p in parts:
        if not p:
            continue
        try:
            k = float(p)
            if k <= 0:
                raise ValueError
            values.append(k)
        except Exception:
            raise ValueError(f"Invalid sigma multiplier: {p}")

    out = []
    for v in values:
        # Avoid duplicating standard 1, 2, 3 sigma references.
        if (
            not math.isclose(v, 1.0)
            and not math.isclose(v, 2.0)
            and not math.isclose(v, 3.0)
            and v not in out
        ):
            out.append(v)
    return out


def unique_nonempty_columns(df: pd.DataFrame) -> List[str]:
    cols = []
    for col in df.columns:
        s = df[col]
        not_empty = s.notna() & (s.astype(str).str.strip() != "")
        if int(not_empty.sum()) > 0:
            cols.append(str(col))
    return cols


def test_suffix(key: str) -> str:
    return key.replace("test_", "")

def ensure_dataframe(value: Any) -> pd.DataFrame:
    """Rebuild DataFrame objects after Analysis Journal JSON serialization."""
    if isinstance(value, pd.DataFrame):
        return value
    if isinstance(value, list):
        return pd.DataFrame(value)
    if isinstance(value, dict):
        return pd.DataFrame(value)
    return pd.DataFrame()


# ─────────────────────────────────────────────
# Scroll / zoom
# ─────────────────────────────────────────────

class ReportScrollArea(QScrollArea):
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

        self.container = QWidget()
        self.container_layout = QVBoxLayout(self.container)
        self.container_layout.setContentsMargins(8, 8, 8, 8)
        self.container_layout.addWidget(self.canvas, alignment=Qt.AlignCenter)
        self.container_layout.addStretch()
        self.setWidget(self.container)

        self.viewport().installEventFilter(self)
        self.container.installEventFilter(self)
        self.canvas.installEventFilter(self)

        self.apply_zoom(self.zoom, center=False)

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Wheel:
            self._handle_wheel_event(event)
            return True
        return super().eventFilter(obj, event)

    def _wheel_delta(self, event):
        delta = event.pixelDelta()
        if not delta.isNull():
            return delta.x(), delta.y()
        delta = event.angleDelta()
        return delta.x(), delta.y()

    def _handle_wheel_event(self, event):
        dx, dy = self._wheel_delta(event)
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

        self.zoom = max(0.62, min(1.85, float(zoom)))
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
# Options and engine
# ─────────────────────────────────────────────

@dataclass
class ControlChartOptions:
    decimals: int = 4
    show_grid: bool = True
    chart_i_enabled: bool = True
    chart_mr_enabled: bool = True
    show_1_sigma: bool = False
    show_2_sigma: bool = False
    extra_sigma_limits_text: str = ""
    auto_journal_save: bool = True

    # I Chart tests
    i_test_1_enabled: bool = True
    i_test_1_k: int = 3
    i_test_2_enabled: bool = False
    i_test_2_k: int = 9
    i_test_3_enabled: bool = False
    i_test_3_k: int = 6
    i_test_4_enabled: bool = False
    i_test_4_k: int = 14
    i_test_5_enabled: bool = False
    i_test_5_k: int = 2
    i_test_6_enabled: bool = False
    i_test_6_k: int = 4
    i_test_7_enabled: bool = False
    i_test_7_k: int = 15
    i_test_8_enabled: bool = False
    i_test_8_k: int = 8

    # MR Chart tests
    mr_test_1_enabled: bool = True
    mr_test_1_k: int = 3
    mr_test_2_enabled: bool = False
    mr_test_2_k: int = 9
    mr_test_3_enabled: bool = False
    mr_test_3_k: int = 6
    mr_test_4_enabled: bool = False
    mr_test_4_k: int = 14
    mr_test_5_enabled: bool = False
    mr_test_5_k: int = 2
    mr_test_6_enabled: bool = False
    mr_test_6_k: int = 4
    mr_test_7_enabled: bool = False
    mr_test_7_k: int = 15
    mr_test_8_enabled: bool = False
    mr_test_8_k: int = 8


def chart_test_settings(opts: ControlChartOptions, chart: str) -> Dict[str, Tuple[bool, int]]:
    prefix = "i" if chart == "I" else "mr"
    return {
        "test_1": (getattr(opts, f"{prefix}_test_1_enabled"), getattr(opts, f"{prefix}_test_1_k")),
        "test_2": (getattr(opts, f"{prefix}_test_2_enabled"), getattr(opts, f"{prefix}_test_2_k")),
        "test_3": (getattr(opts, f"{prefix}_test_3_enabled"), getattr(opts, f"{prefix}_test_3_k")),
        "test_4": (getattr(opts, f"{prefix}_test_4_enabled"), getattr(opts, f"{prefix}_test_4_k")),
        "test_5": (getattr(opts, f"{prefix}_test_5_enabled"), getattr(opts, f"{prefix}_test_5_k")),
        "test_6": (getattr(opts, f"{prefix}_test_6_enabled"), getattr(opts, f"{prefix}_test_6_k")),
        "test_7": (getattr(opts, f"{prefix}_test_7_enabled"), getattr(opts, f"{prefix}_test_7_k")),
        "test_8": (getattr(opts, f"{prefix}_test_8_enabled"), getattr(opts, f"{prefix}_test_8_k")),
    }


def moving_range_sigma(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) >= 2:
        mr = np.abs(np.diff(values))
        mr_bar = np.nanmean(mr)
        if pd.notna(mr_bar) and mr_bar > 0:
            return float(mr_bar / 1.128)  # d2 for MR(2)
    if len(values) >= 2:
        s = np.nanstd(values, ddof=1)
        if pd.notna(s) and s > 0:
            return float(s)
    return np.nan


def mr_sigma_from_mrbar(mr_bar: float) -> float:
    # sigma estimate for MR chart using MRbar/d2, d2=1.128 for moving range of 2.
    if pd.isna(mr_bar) or mr_bar <= 0:
        return np.nan
    return float(mr_bar / 1.128)


def mr_ucl_from_mrbar(mr_bar: float) -> float:
    # D4 for MR(2) = 3.267, D3 = 0.
    if pd.isna(mr_bar):
        return np.nan
    return float(3.267 * mr_bar)


def build_chart_data(df: pd.DataFrame, y_col: str, stage_col: Optional[str]) -> pd.DataFrame:
    data = df[[y_col] + ([stage_col] if stage_col else [])].copy()
    data[y_col] = pd.to_numeric(data[y_col], errors="coerce")
    data = data.dropna(subset=[y_col]).reset_index(drop=True)
    data["Obs"] = np.arange(1, len(data) + 1)

    if data.empty:
        raise ValueError("No valid numeric data found for the selected Y column.")

    if stage_col:
        data["_stage_"] = data[stage_col].astype(str)
    else:
        data["_stage_"] = "All Data"

    # Moving range within each stage. The first MR in each stage is NaN.
    data["MR"] = np.nan
    for stage in list(dict.fromkeys(data["_stage_"].tolist())):
        idx = data.index[data["_stage_"] == stage].tolist()
        vals = data.loc[idx, y_col].to_numpy(dtype=float)
        if len(vals) >= 2:
            mr = np.abs(np.diff(vals))
            data.loc[idx[1:], "MR"] = mr

    return data


def stage_limit_tables(data: pd.DataFrame, y_col: str, extra_sigmas: List[float]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    stage_order = list(dict.fromkeys(data["_stage_"].tolist()))

    i_rows = []
    mr_rows = []

    for stage in stage_order:
        mask = data["_stage_"] == stage
        vals = data.loc[mask, y_col].to_numpy(dtype=float)
        mr_vals = data.loc[mask, "MR"].dropna().to_numpy(dtype=float)

        mean = float(np.nanmean(vals))
        sigma_i = moving_range_sigma(vals)
        mr_bar = float(np.nanmean(mr_vals)) if len(mr_vals) else np.nan
        sigma_mr = mr_sigma_from_mrbar(mr_bar)

        i_row = {
            "Chart": "I",
            "Stage": stage,
            "Start Obs": int(data.loc[mask, "Obs"].min()),
            "End Obs": int(data.loc[mask, "Obs"].max()),
            "N": int(mask.sum()),
            "CL": mean,
            "Sigma": sigma_i,
            "LCL_1": mean - 1 * sigma_i if pd.notna(sigma_i) else np.nan,
            "UCL_1": mean + 1 * sigma_i if pd.notna(sigma_i) else np.nan,
            "LCL_2": mean - 2 * sigma_i if pd.notna(sigma_i) else np.nan,
            "UCL_2": mean + 2 * sigma_i if pd.notna(sigma_i) else np.nan,
            "LCL_3": mean - 3 * sigma_i if pd.notna(sigma_i) else np.nan,
            "UCL_3": mean + 3 * sigma_i if pd.notna(sigma_i) else np.nan,
        }
        for k in extra_sigmas:
            i_row[f"LCL_{k:g}"] = mean - k * sigma_i if pd.notna(sigma_i) else np.nan
            i_row[f"UCL_{k:g}"] = mean + k * sigma_i if pd.notna(sigma_i) else np.nan
        i_rows.append(i_row)

        mr_row = {
            "Chart": "MR",
            "Stage": stage,
            "Start Obs": int(data.loc[mask, "Obs"].min()),
            "End Obs": int(data.loc[mask, "Obs"].max()),
            "N": int(np.sum(~np.isnan(data.loc[mask, "MR"].to_numpy(dtype=float)))),
            "CL": mr_bar,
            "Sigma": sigma_mr,
            "LCL_3": 0.0 if pd.notna(mr_bar) else np.nan,
            "UCL_3": mr_ucl_from_mrbar(mr_bar),
        }
        for k in extra_sigmas:
            # For MR chart, add calculated ±K sigma limits around MRbar.
            # LCL is clipped at 0 because moving ranges cannot be negative.
            if pd.notna(mr_bar) and pd.notna(sigma_mr):
                mr_row[f"LCL_{k:g}"] = max(0.0, mr_bar - k * sigma_mr)
                mr_row[f"UCL_{k:g}"] = mr_bar + k * sigma_mr
            else:
                mr_row[f"LCL_{k:g}"] = np.nan
                mr_row[f"UCL_{k:g}"] = np.nan
        mr_rows.append(mr_row)

        # Store I limits on each row.
        for key, value in i_row.items():
            if key in ["CL", "Sigma", "LCL_1", "UCL_1", "LCL_2", "UCL_2", "LCL_3", "UCL_3"] or key.startswith(("LCL_", "UCL_")):
                data.loc[mask, f"I_{key}"] = value

        # Store MR limits on each row.
        for key, value in mr_row.items():
            if key in ["CL", "Sigma", "LCL_3", "UCL_3"] or key.startswith(("LCL_", "UCL_")):
                data.loc[mask, f"MR_{key}"] = value

    return data, pd.DataFrame(i_rows), pd.DataFrame(mr_rows)


def add_violation(violations: List[Dict[str, Any]], chart: str, obs: int, stage: str, test: str, y: float, detail: str):
    violations.append({
        "Chart": chart,
        "Obs": int(obs),
        "Stage": str(stage),
        "Test": test,
        "Y": float(y),
        "Detail": detail,
    })


def run_control_tests_for_series(
    data: pd.DataFrame,
    value_col: str,
    chart_prefix: str,
    chart_name: str,
    settings: Dict[str, Tuple[bool, int]],
) -> List[Dict[str, Any]]:
    violations: List[Dict[str, Any]] = []

    for stage in list(dict.fromkeys(data["_stage_"].tolist())):
        sd = data.loc[data["_stage_"] == stage].copy()
        sd = sd.dropna(subset=[value_col])
        if sd.empty:
            continue

        y = sd[value_col].to_numpy(dtype=float)
        obs = sd["Obs"].to_numpy(dtype=int)
        cl = sd[f"{chart_prefix}_CL"].to_numpy(dtype=float)
        sigma = sd[f"{chart_prefix}_Sigma"].to_numpy(dtype=float)

        if len(y) == 0 or np.all(pd.isna(sigma)):
            continue

        z = (y - cl) / sigma
        z[~np.isfinite(z)] = np.nan

        # Test 1: 1 point > K standard deviations from center line.
        enabled, k = settings["test_1"]
        if enabled:
            idxs = np.where(np.abs(z) > int(k))[0]
            for i in idxs:
                add_violation(violations, chart_name, obs[i], stage, f"Test 1: 1 point > {int(k)} sigma", y[i], f"z={fmt(z[i], 3)}")

        # Test 2: K points in a row on same side of center line.
        enabled, k = settings["test_2"]
        if enabled:
            k = int(k)
            signs = np.sign(y - cl)
            for i in range(k - 1, len(y)):
                window = signs[i - k + 1:i + 1]
                if np.all(window > 0) or np.all(window < 0):
                    add_violation(violations, chart_name, obs[i], stage, f"Test 2: {k} points same side", y[i], "Run on one side of CL")

        # Test 3: K points in a row, all increasing or all decreasing.
        enabled, k = settings["test_3"]
        if enabled:
            k = int(k)
            for i in range(k - 1, len(y)):
                window = y[i - k + 1:i + 1]
                diffs = np.diff(window)
                if np.all(diffs > 0) or np.all(diffs < 0):
                    add_violation(violations, chart_name, obs[i], stage, f"Test 3: {k} increasing/decreasing", y[i], "Trend detected")

        # Test 4: K points in a row, alternating up and down.
        enabled, k = settings["test_4"]
        if enabled:
            k = int(k)
            for i in range(k - 1, len(y)):
                diffs = np.diff(y[i - k + 1:i + 1])
                signs = np.sign(diffs)
                if len(signs) and np.all(signs[:-1] * signs[1:] < 0):
                    add_violation(violations, chart_name, obs[i], stage, f"Test 4: {k} alternating", y[i], "Alternating pattern")

        # Test 5: K out of K+1 points > 2 sigma from center line, same side.
        enabled, k = settings["test_5"]
        if enabled:
            k = int(k)
            nwin = k + 1
            for i in range(nwin - 1, len(y)):
                wz = z[i - nwin + 1:i + 1]
                if np.sum(wz > 2) >= k or np.sum(wz < -2) >= k:
                    add_violation(violations, chart_name, obs[i], stage, f"Test 5: {k} of {k+1} > 2 sigma", y[i], "Same side beyond 2 sigma")

        # Test 6: K out of K+1 points > 1 sigma from center line, same side.
        enabled, k = settings["test_6"]
        if enabled:
            k = int(k)
            nwin = k + 1
            for i in range(nwin - 1, len(y)):
                wz = z[i - nwin + 1:i + 1]
                if np.sum(wz > 1) >= k or np.sum(wz < -1) >= k:
                    add_violation(violations, chart_name, obs[i], stage, f"Test 6: {k} of {k+1} > 1 sigma", y[i], "Same side beyond 1 sigma")

        # Test 7: K points in a row within 1 sigma of center line, either side.
        enabled, k = settings["test_7"]
        if enabled:
            k = int(k)
            for i in range(k - 1, len(y)):
                wz = z[i - k + 1:i + 1]
                if np.all(np.abs(wz) < 1):
                    add_violation(violations, chart_name, obs[i], stage, f"Test 7: {k} points within 1 sigma", y[i], "Stratification pattern")

        # Test 8: K points in a row > 1 sigma from center line, either side.
        enabled, k = settings["test_8"]
        if enabled:
            k = int(k)
            for i in range(k - 1, len(y)):
                wz = z[i - k + 1:i + 1]
                if np.all(np.abs(wz) > 1):
                    add_violation(violations, chart_name, obs[i], stage, f"Test 8: {k} points > 1 sigma", y[i], "Mixture pattern")

    # Deduplicate exact duplicate flags.
    seen = set()
    unique = []
    for v in violations:
        key = (v["Chart"], v["Obs"], v["Stage"], v["Test"])
        if key not in seen:
            seen.add(key)
            unique.append(v)
    return unique


def control_chart_fit(
    df: pd.DataFrame,
    y_col: str,
    stage_col: Optional[str],
    opts: ControlChartOptions,
) -> Dict[str, Any]:
    if not y_col:
        raise ValueError("Select a numeric Y column.")

    if not opts.chart_i_enabled and not opts.chart_mr_enabled:
        raise ValueError("Select at least one chart: I Chart, MR Chart, or both.")

    if stage_col in ["", None, "(None)"]:
        stage_col = None

    extra_sigmas = parse_extra_sigma_limits(opts.extra_sigma_limits_text)
    data = build_chart_data(df, y_col, stage_col)
    data, i_limits, mr_limits = stage_limit_tables(data, y_col, extra_sigmas)

    violations: List[Dict[str, Any]] = []
    if opts.chart_i_enabled:
        violations += run_control_tests_for_series(data, y_col, "I", "I", chart_test_settings(opts, "I"))
    if opts.chart_mr_enabled:
        violations += run_control_tests_for_series(data, "MR", "MR", "MR", chart_test_settings(opts, "MR"))

    return {
        "y_name": y_col,
        "stage_name": stage_col or "",
        "n": int(len(data)),
        "stage_count": int(i_limits.shape[0]),
        "data": data,
        "i_limits": i_limits,
        "mr_limits": mr_limits,
        "extra_sigmas": extra_sigmas,
        "violations": violations,
        "chart_i_enabled": bool(opts.chart_i_enabled),
        "chart_mr_enabled": bool(opts.chart_mr_enabled),
    }


def build_summary_text(analysis: Dict[str, Any]) -> str:
    d = int(analysis["options"]["decimals"])
    fit = analysis["fit"]
    violations = fit["violations"]

    lines = [
        "=" * 86,
        "CONTROL CHART REPORT — SPS STUDIO",
        "=" * 86,
        f"Worksheet: {analysis.get('worksheet', '')}",
        f"Y variable: {fit.get('y_name', '')}",
        f"Stage column: {fit.get('stage_name', '') or 'None'}",
        f"Charts: {'I ' if fit.get('chart_i_enabled') else ''}{'MR' if fit.get('chart_mr_enabled') else ''}",
        f"N: {fit.get('n')}",
        f"Stages: {fit.get('stage_count')}",
        f"Extra sigma limits: {', '.join(str(x) for x in fit.get('extra_sigmas', [])) or 'None'}",
        f"Violations detected: {len(violations)}",
        "",
    ]

    if fit.get("chart_i_enabled"):
        lines += [
            "I CHART LIMITS BY STAGE",
            "-" * 86,
            f"{'Stage':<22}{'N':<8}{'CL':<14}{'Sigma':<14}{'LCL(3σ)':<14}{'UCL(3σ)':<14}",
        ]
        for _, r in fit["i_limits"].iterrows():
            lines.append(
                f"{str(r['Stage']):<22}{int(r['N']):<8}{fmt(r['CL'], d):<14}"
                f"{fmt(r['Sigma'], d):<14}{fmt(r['LCL_3'], d):<14}{fmt(r['UCL_3'], d):<14}"
            )
        lines.append("")

    if fit.get("chart_mr_enabled"):
        lines += [
            "MR CHART LIMITS BY STAGE",
            "-" * 86,
            f"{'Stage':<22}{'N':<8}{'MR-bar':<14}{'LCL':<14}{'UCL':<14}",
        ]
        for _, r in fit["mr_limits"].iterrows():
            lines.append(
                f"{str(r['Stage']):<22}{int(r['N']):<8}{fmt(r['CL'], d):<14}"
                f"{fmt(r['LCL_3'], d):<14}{fmt(r['UCL_3'], d):<14}"
            )
        lines.append("")

    lines += ["CONTROL TEST VIOLATIONS", "-" * 86]
    if not violations:
        lines.append("No active control test violations were detected.")
    else:
        lines.append(f"{'Chart':<8}{'Obs':<8}{'Stage':<22}{'Test':<34}{'Y':<14}{'Detail':<20}")
        for v in violations:
            lines.append(
                f"{v['Chart']:<8}{v['Obs']:<8}{v['Stage']:<22}{v['Test']:<34}"
                f"{fmt(v['Y'], d):<14}{v['Detail']:<20}"
            )

    return "\n".join(lines)


# ─────────────────────────────────────────────
# Figure
# ─────────────────────────────────────────────

def _right_limit_text(chart_name: str, last: pd.Series, extra_sigmas: List[float], decimals: int) -> str:
    lines = []

    # Extra sigma labels first, like the reference chart.
    for k in sorted(extra_sigmas, reverse=True):
        u = last.get(f"UCL_{k:g}", np.nan)
        if pd.notna(u):
            lines.append(f"+{k:g}SL={fmt(u, decimals)}")

    lines.append(f"UCL={fmt(last['UCL_3'], decimals)}")

    if chart_name == "I":
        lines.append(f"X\u0304={fmt(last['CL'], decimals)}")
    else:
        lines.append(f"MR\u0305={fmt(last['CL'], decimals)}")

    lines.append(f"LCL={fmt(last['LCL_3'], decimals)}")

    for k in sorted(extra_sigmas):
        l = last.get(f"LCL_{k:g}", np.nan)
        if pd.notna(l):
            lines.append(f"-{k:g}SL={fmt(l, decimals)}")

    return "\n".join(lines)


def _plot_stage_chart(
    ax,
    data: pd.DataFrame,
    limits: pd.DataFrame,
    value_col: str,
    prefix: str,
    y_label: str,
    opts: Dict[str, Any],
    extra_sigmas: List[float],
    violations: List[Dict[str, Any]],
    decimals: int,
):
    show_grid = bool(opts.get("show_grid", True))
    chart_name = "I" if prefix == "I" else "MR"

    plot_data = data.dropna(subset=[value_col]).copy()
    x = plot_data["Obs"].to_numpy(dtype=int)
    y = plot_data[value_col].to_numpy(dtype=float)

    ax.plot(x, y, marker="o", linewidth=1.0, markersize=3.8, color="#0057A8", alpha=0.90)

    violated_obs = {int(v["Obs"]) for v in violations if v.get("Chart") == chart_name}
    if violated_obs:
        bad = plot_data[plot_data["Obs"].isin(violated_obs)]
        ax.scatter(
            bad["Obs"], bad[value_col],
            s=48, color="#C62828", edgecolors="white", linewidths=0.8,
            zorder=5
        )
        for _, br in bad.iterrows():
            ax.text(
                br["Obs"], br[value_col],
                "1",
                ha="center", va="bottom",
                fontsize=7,
                color="#C62828",
                fontweight="bold",
                zorder=6,
            )

    for _, r in limits.iterrows():
        x0, x1 = int(r["Start Obs"]), int(r["End Obs"])
        stage = str(r["Stage"])
        cl = r["CL"]

        ax.hlines(cl, x0, x1, colors="#2E7D32", linestyles="-", linewidth=1.1)
        ax.hlines(r["UCL_3"], x0, x1, colors="#C62828", linestyles="-", linewidth=1.0)
        ax.hlines(r["LCL_3"], x0, x1, colors="#C62828", linestyles="-", linewidth=1.0)

        if prefix == "I":
            if bool(opts.get("show_2_sigma", False)):
                ax.hlines(r["UCL_2"], x0, x1, colors="#F57C00", linestyles=":", linewidth=0.9)
                ax.hlines(r["LCL_2"], x0, x1, colors="#F57C00", linestyles=":", linewidth=0.9)

            if bool(opts.get("show_1_sigma", False)):
                ax.hlines(r["UCL_1"], x0, x1, colors="#7A8793", linestyles=":", linewidth=0.8)
                ax.hlines(r["LCL_1"], x0, x1, colors="#7A8793", linestyles=":", linewidth=0.8)

        # Extra calculated sigma limits for I and MR.
        for k in extra_sigmas:
            u = r.get(f"UCL_{k:g}", np.nan)
            l = r.get(f"LCL_{k:g}", np.nan)
            if pd.notna(u):
                ax.hlines(u, x0, x1, colors="#6A1B9A", linestyles="-.", linewidth=0.85)
            if pd.notna(l):
                ax.hlines(l, x0, x1, colors="#6A1B9A", linestyles="-.", linewidth=0.85)

        # Stage boundary and label.
        ax.axvline(x0 - 0.5, color="#7E57C2", linestyle="--", linewidth=0.7, alpha=0.75)
        if len(limits) > 1:
            ax.text(
                (x0 + x1) / 2, 1.025, stage,
                transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=7, color="#111111"
            )

    if not limits.empty:
        last = limits.iloc[-1]
        txt = _right_limit_text(chart_name, last, extra_sigmas, decimals)
        ax.text(
            1.005, last["CL"], txt,
            transform=ax.get_yaxis_transform(),
            ha="left", va="center",
            fontsize=7,
            color="#111111",
            clip_on=False,
        )

        last_end = int(limits["End Obs"].max())
        ax.axvline(last_end + 0.5, color="#7E57C2", linestyle="--", linewidth=0.7, alpha=0.75)

    ax.set_ylabel(y_label, fontsize=8)
    ax.grid(show_grid, alpha=0.18)
    ax.tick_params(labelsize=8)


def build_control_chart_figure(analysis: Dict[str, Any]) -> Figure:
    fit = analysis["fit"]
    opts = analysis["options"]
    d = int(opts.get("decimals", 4))

    data = fit["data"]
    y_col = fit["y_name"]
    violations = fit.get("violations", [])
    extra_sigmas = fit.get("extra_sigmas", [])

    chart_i = bool(fit.get("chart_i_enabled", True))
    chart_mr = bool(fit.get("chart_mr_enabled", True))
    n_charts = int(chart_i) + int(chart_mr)

    fig_h = 6.0 if n_charts == 2 else 4.4
    fig = Figure(figsize=(11.4, fig_h), dpi=120)
    fig.patch.set_facecolor("white")

    title_ax = fig.add_axes([0.04, 0.925, 0.92, 0.045])
    title_ax.axis("off")
    has_stage = bool(fit.get("stage_name", ""))
    title_text = "I-MR Chart" if chart_i and chart_mr else ("Individuals Chart" if chart_i else "Moving Range Chart")
    suffix = " by Stage" if has_stage else ""
    title_ax.set_title(f"{title_text} of {y_col}{suffix}", fontsize=13, fontweight="bold", pad=1)

    top = 0.86
    bottom = 0.10
    hspace = 0.32

    if n_charts == 2:
        gs = fig.add_gridspec(2, 1, left=0.07, right=0.88, top=top, bottom=bottom, hspace=hspace)
    else:
        gs = fig.add_gridspec(1, 1, left=0.07, right=0.88, top=top, bottom=bottom)

    axes = []
    if chart_i:
        axes.append(("I", fig.add_subplot(gs[len(axes), 0] if n_charts == 2 else gs[0, 0])))
    if chart_mr:
        axes.append(("MR", fig.add_subplot(gs[len(axes), 0] if n_charts == 2 else gs[0, 0])))

    for chart, ax in axes:
        if chart == "I":
            _plot_stage_chart(
                ax=ax,
                data=data,
                limits=fit["i_limits"],
                value_col=y_col,
                prefix="I",
                y_label="Individual Value",
                opts=opts,
                extra_sigmas=extra_sigmas,
                violations=violations,
                decimals=d,
            )
        else:
            _plot_stage_chart(
                ax=ax,
                data=data,
                limits=fit["mr_limits"],
                value_col="MR",
                prefix="MR",
                y_label="Moving Range",
                opts=opts,
                extra_sigmas=extra_sigmas,
                violations=violations,
                decimals=d,
            )

    axes[-1][1].set_xlabel("Observation", fontsize=8)

    return fig


# ─────────────────────────────────────────────
# Main widget
# ─────────────────────────────────────────────

class ControlChartsBuilder(QWidget):
    def __init__(self, app_window):
        super().__init__()
        self.app_window = app_window
        self.current_df = pd.DataFrame()
        self.current_analysis: Optional[Dict[str, Any]] = None

        self.chart_figure: Optional[Figure] = None
        self.chart_canvas: Optional[FigureCanvas] = None
        self.chart_scroll_area: Optional[ReportScrollArea] = None
        self.current_journal_entry_id: Optional[str] = None
        self.interaction_managers: List[Any] = []

        self._build_ui()
        self.refresh_columns()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)

        header = QHBoxLayout()
        title = QLabel("Control Charts Builder")
        title.setStyleSheet("font-size:20px; font-weight:bold; color:#17324D;")
        header.addWidget(title)
        header.addStretch()
        self.lbl_ws = QLabel("Worksheet: -")
        header.addWidget(self.lbl_ws)
        root.addLayout(header)

        self.tabs = QTabWidget()
        root.addWidget(self.tabs)

        self._build_setup_tab()
        self._build_graph_tab()
        self._build_results_tab()
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
        self.cmb_stage = QComboBox()

        var_layout.addWidget(QLabel("Measurement Y"), 0, 0)
        var_layout.addWidget(self.cmb_y, 0, 1)
        var_layout.addWidget(QLabel("Stage column"), 1, 0)
        var_layout.addWidget(self.cmb_stage, 1, 1)

        opt_box = QGroupBox("2. Chart Options")
        opt = QGridLayout(opt_box)

        self.spn_decimals = QSpinBox()
        self.spn_decimals.setRange(0, 8)
        self.spn_decimals.setValue(4)

        self.chk_chart_i = QCheckBox("I Chart")
        self.chk_chart_i.setChecked(True)

        self.chk_chart_mr = QCheckBox("MR Chart")
        self.chk_chart_mr.setChecked(True)

        self.chk_grid = QCheckBox("Show grid")
        self.chk_grid.setChecked(True)

        self.chk_1sigma = QCheckBox("Show 1 sigma zone lines")
        self.chk_1sigma.setChecked(False)

        self.chk_2sigma = QCheckBox("Show 2 sigma zone lines")
        self.chk_2sigma.setChecked(False)

        self.txt_extra_sigmas = QTextEdit()
        self.txt_extra_sigmas.setMaximumHeight(55)
        self.txt_extra_sigmas.setPlaceholderText("Optional extra sigma limits, e.g. 4, 4.5, 5")

        self.chk_journal = QCheckBox("Auto journal save")
        self.chk_journal.setChecked(True)

        opt.addWidget(QLabel("Decimals"), 0, 0)
        opt.addWidget(self.spn_decimals, 0, 1)
        opt.addWidget(self.chk_chart_i, 1, 1)
        opt.addWidget(self.chk_chart_mr, 2, 1)
        opt.addWidget(self.chk_grid, 3, 1)
        opt.addWidget(self.chk_1sigma, 4, 1)
        opt.addWidget(self.chk_2sigma, 5, 1)
        opt.addWidget(QLabel("Extra sigma limits"), 6, 0)
        opt.addWidget(self.txt_extra_sigmas, 6, 1)
        opt.addWidget(self.chk_journal, 7, 1)

        top.addWidget(var_box, 1)
        top.addWidget(opt_box, 1)
        layout.addLayout(top)

        tests_tabs = QTabWidget()
        tests_tabs.addTab(self._build_tests_box("I Chart Control Tests", "i"), "I Chart Tests")
        tests_tabs.addTab(self._build_tests_box("MR Chart Control Tests", "mr"), "MR Chart Tests")
        layout.addWidget(tests_tabs)

        run_box = QGroupBox("4. Run")
        run_layout = QHBoxLayout(run_box)
        self.btn_refresh = QPushButton("Refresh Columns")
        self.btn_run = QPushButton("Run Control Chart")
        self.btn_run.setStyleSheet("background:#2F5D87; color:white; font-weight:bold; padding:8px; border-radius:6px;")

        self.btn_refresh.clicked.connect(self.refresh_columns)
        self.btn_run.clicked.connect(self.run_analysis)

        run_layout.addStretch()
        run_layout.addWidget(self.btn_refresh)
        run_layout.addWidget(self.btn_run)
        layout.addWidget(run_box)

        layout.addStretch()
        self.tabs.addTab(tab, "Setup")

    def _build_tests_box(self, title: str, prefix: str):
        box = QWidget()
        root = QVBoxLayout(box)
        group = QGroupBox(title)
        tests_layout = QGridLayout(group)

        if not hasattr(self, "test_controls"):
            self.test_controls = {"i": {}, "mr": {}}

        base_test_defs = [
            ("test_1", "1 point > K standard deviations from center line", 3, True),
            ("test_2", "K points in a row on same side of center line", 9, False),
            ("test_3", "K points in a row, all increasing or all decreasing", 6, False),
            ("test_4", "K points in a row, alternating up and down", 14, False),
            ("test_5", "K out of K+1 points > 2 standard deviations from center line (same side)", 2, False),
            ("test_6", "K out of K+1 points > 1 standard deviation from center line (same side)", 4, False),
            ("test_7", "K points in a row within 1 standard deviation from center line (either side)", 15, False),
            ("test_8", "K points in a row > 1 standard deviation from center line (either side)", 8, False),
        ]

        # MR Chart only uses the first 4 tests shown in the standard MR test setup.
        test_defs = base_test_defs[:4] if prefix == "mr" else base_test_defs

        for row, (key, label, default_k, default_enabled) in enumerate(test_defs):
            chk = QCheckBox(label)
            chk.setChecked(default_enabled)
            spn = QSpinBox()
            spn.setRange(1, 100)
            spn.setValue(default_k)
            spn.setMaximumWidth(70)
            tests_layout.addWidget(chk, row, 0)
            tests_layout.addWidget(spn, row, 1)
            self.test_controls[prefix][key] = (chk, spn)

        root.addWidget(group)
        return box

    def _build_graph_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        toolbar = QHBoxLayout()
        self.btn_zoom_out = QPushButton("Zoom -")
        self.btn_zoom_fit = QPushButton("Fit")
        self.btn_zoom_reset = QPushButton("100%")
        self.btn_zoom_in = QPushButton("Zoom +")
        self.btn_copy = QPushButton("Copy Graph")
        self.btn_export = QPushButton("Export PNG")
        self.btn_update_journal = QPushButton("Update Journal Entry")

        self.btn_zoom_out.clicked.connect(lambda: self.zoom_graph("out"))
        self.btn_zoom_fit.clicked.connect(lambda: self.zoom_graph("fit"))
        self.btn_zoom_reset.clicked.connect(lambda: self.zoom_graph("reset"))
        self.btn_zoom_in.clicked.connect(lambda: self.zoom_graph("in"))
        self.btn_copy.clicked.connect(self.copy_graph)
        self.btn_export.clicked.connect(self.export_png)
        self.btn_update_journal.clicked.connect(self.update_journal_entry)

        toolbar.addStretch()
        toolbar.addWidget(self.btn_zoom_out)
        toolbar.addWidget(self.btn_zoom_fit)
        toolbar.addWidget(self.btn_zoom_reset)
        toolbar.addWidget(self.btn_zoom_in)
        toolbar.addWidget(self.btn_copy)
        toolbar.addWidget(self.btn_export)
        toolbar.addWidget(self.btn_update_journal)

        layout.addLayout(toolbar)

        self.chart_container = QWidget()
        self.chart_layout = QVBoxLayout(self.chart_container)
        layout.addWidget(self.chart_container)

        self.tabs.addTab(tab, "Graphs")

    def _build_results_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        self.tbl_limits = QTableWidget()
        self.tbl_limits.setColumnCount(9)
        self.tbl_limits.setHorizontalHeaderLabels(["Chart", "Stage", "Start Obs", "End Obs", "N", "CL", "Sigma", "LCL 3σ", "UCL 3σ"])

        self.tbl_test_summary = QTableWidget()
        self.tbl_test_summary.setColumnCount(3)
        self.tbl_test_summary.setHorizontalHeaderLabels(["Chart", "Control Test", "Count"])

        self.tbl_violations = QTableWidget()
        self.tbl_violations.setColumnCount(6)
        self.tbl_violations.setHorizontalHeaderLabels(["Chart", "Obs", "Stage", "Test", "Y", "Detail"])

        layout.addWidget(QLabel("Control Limits by Stage"))
        layout.addWidget(self.tbl_limits)
        layout.addWidget(QLabel("Control Test Summary"))
        layout.addWidget(self.tbl_test_summary)
        layout.addWidget(QLabel("Control Test Violations"))
        layout.addWidget(self.tbl_violations)

        self.tabs.addTab(tab, "Results")

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

        self.cmb_y.clear()
        self.cmb_stage.clear()
        self.cmb_stage.addItem("(None)")

        if self.current_df.empty:
            return

        numeric_cols = []
        for col in self.current_df.columns:
            s = pd.to_numeric(self.current_df[col], errors="coerce")
            if s.notna().sum() >= 3:
                numeric_cols.append(str(col))

        # Stage selector: only columns with actual data, not blank-only columns.
        stage_cols = unique_nonempty_columns(self.current_df)

        self.cmb_y.addItems(numeric_cols)
        self.cmb_stage.addItems(stage_cols)

        self.log(f"Columns refreshed. Numeric={numeric_cols}; Stage candidates={stage_cols}")

    def _read_chart_tests(self, prefix: str) -> Dict[str, Any]:
        t = self.test_controls[prefix]
        defaults = {
            "test_1": (True, 3),
            "test_2": (False, 9),
            "test_3": (False, 6),
            "test_4": (False, 14),
            "test_5": (False, 2),
            "test_6": (False, 4),
            "test_7": (False, 15),
            "test_8": (False, 8),
        }

        out = {}
        for key, (default_enabled, default_k) in defaults.items():
            if key in t:
                out[f"{prefix}_{key}_enabled"] = t[key][0].isChecked()
                out[f"{prefix}_{key}_k"] = int(t[key][1].value())
            else:
                # Used for MR Chart tests 5-8, which are intentionally not available in the UI.
                out[f"{prefix}_{key}_enabled"] = default_enabled
                out[f"{prefix}_{key}_k"] = default_k

        return out

    def _options(self) -> ControlChartOptions:
        kwargs = {
            "decimals": int(self.spn_decimals.value()),
            "show_grid": self.chk_grid.isChecked(),
            "chart_i_enabled": self.chk_chart_i.isChecked(),
            "chart_mr_enabled": self.chk_chart_mr.isChecked(),
            "show_1_sigma": self.chk_1sigma.isChecked(),
            "show_2_sigma": self.chk_2sigma.isChecked(),
            "extra_sigma_limits_text": self.txt_extra_sigmas.toPlainText(),
            "auto_journal_save": self.chk_journal.isChecked(),
        }
        kwargs.update(self._read_chart_tests("i"))
        kwargs.update(self._read_chart_tests("mr"))
        return ControlChartOptions(**kwargs)

    def run_analysis(self):
        try:
            if self.current_df.empty:
                raise ValueError("No active worksheet data found.")

            opts = self._options()
            y_col = self.cmb_y.currentText()
            stage_col = self.cmb_stage.currentText()

            fit = control_chart_fit(self.current_df, y_col, stage_col, opts)

            analysis = {
                "id": str(uuid.uuid4()),
                "module": "Control Charts",
                "type": "Control Chart",
                "worksheet": self.app_window.active_worksheet_name(),
                "options": asdict(opts),
                "fit": fit,
                "created_at": now_string(),
            }
            analysis["summary"] = build_summary_text(analysis)

            self.current_analysis = analysis
            self._populate_results(analysis)
            self._render_figure(analysis)
            self.txt_summary.setPlainText(analysis["summary"])

            if opts.auto_journal_save:
                self.save_to_journal()

            self.tabs.setCurrentIndex(1)
            self.log("Control chart completed successfully.")

        except Exception as exc:
            self.log(f"ERROR: {exc}")
            QMessageBox.critical(self, "Control Chart Error", str(exc))

    def _populate_results(self, analysis: Dict[str, Any]):
        d = int(analysis["options"]["decimals"])
        fit = analysis["fit"]

        # Journal payload stores DataFrames as lists. Rebuild them before using iterrows().
        fit["data"] = ensure_dataframe(fit.get("data"))
        fit["i_limits"] = ensure_dataframe(fit.get("i_limits"))
        fit["mr_limits"] = ensure_dataframe(fit.get("mr_limits"))

        extra_sigmas = fit.get("extra_sigmas", []) or []

        # Rebuild the table structure dynamically to include extra sigma limits.
        headers = ["Chart", "Stage", "Start Obs", "End Obs", "N", "CL", "Sigma", "LCL 3σ", "UCL 3σ"]
        for k in extra_sigmas:
            headers.extend([f"LCL {k:g}σ", f"UCL {k:g}σ"])

        self.tbl_limits.setColumnCount(len(headers))
        self.tbl_limits.setHorizontalHeaderLabels(headers)

        rows = []

        if fit.get("chart_i_enabled"):
            for _, row in fit["i_limits"].iterrows():
                values = [
                    "I",
                    row.get("Stage", ""),
                    int(row.get("Start Obs", 0)),
                    int(row.get("End Obs", 0)),
                    int(row.get("N", 0)),
                    fmt(row.get("CL"), d),
                    fmt(row.get("Sigma"), d),
                    fmt(row.get("LCL_3"), d),
                    fmt(row.get("UCL_3"), d),
                ]
                for k in extra_sigmas:
                    values.extend([
                        fmt(row.get(f"LCL_{k:g}"), d),
                        fmt(row.get(f"UCL_{k:g}"), d),
                    ])
                rows.append(values)

        if fit.get("chart_mr_enabled"):
            for _, row in fit["mr_limits"].iterrows():
                values = [
                    "MR",
                    row.get("Stage", ""),
                    int(row.get("Start Obs", 0)),
                    int(row.get("End Obs", 0)),
                    int(row.get("N", 0)),
                    fmt(row.get("CL"), d),
                    fmt(row.get("Sigma"), d),
                    fmt(row.get("LCL_3"), d),
                    fmt(row.get("UCL_3"), d),
                ]
                for k in extra_sigmas:
                    values.extend([
                        fmt(row.get(f"LCL_{k:g}"), d),
                        fmt(row.get(f"UCL_{k:g}"), d),
                    ])
                rows.append(values)

        self.tbl_limits.setRowCount(len(rows))
        for r, values in enumerate(rows):
            for c, val in enumerate(values):
                self.tbl_limits.setItem(r, c, QTableWidgetItem(str(val)))
        self.tbl_limits.resizeColumnsToContents()

        summary_counts: Dict[Tuple[str, str], int] = {}
        for v in fit.get("violations", []):
            key = (v.get("Chart", ""), v.get("Test", ""))
            summary_counts[key] = summary_counts.get(key, 0) + 1

        summary_rows = [[chart, test, count] for (chart, test), count in summary_counts.items()]
        if not summary_rows:
            summary_rows = [["Active charts", "No violations", 0]]

        self.tbl_test_summary.setRowCount(len(summary_rows))
        for r, values in enumerate(summary_rows):
            for c, val in enumerate(values):
                self.tbl_test_summary.setItem(r, c, QTableWidgetItem(str(val)))
        self.tbl_test_summary.resizeColumnsToContents()

        violations = fit.get("violations", [])
        self.tbl_violations.setRowCount(len(violations))
        for r, row in enumerate(violations):
            values = [
                row.get("Chart", ""),
                row.get("Obs", ""),
                row.get("Stage", ""),
                row.get("Test", ""),
                fmt(row.get("Y"), d),
                row.get("Detail", ""),
            ]
            for c, val in enumerate(values):
                self.tbl_violations.setItem(r, c, QTableWidgetItem(str(val)))
        self.tbl_violations.resizeColumnsToContents()

    def _render_figure(self, analysis: Dict[str, Any]):
        self._clear_layout(self.chart_layout)

        # Journal payload stores DataFrames as lists. Rebuild before plotting.
        fit = analysis.get("fit", {})
        fit["data"] = ensure_dataframe(fit.get("data"))
        fit["i_limits"] = ensure_dataframe(fit.get("i_limits"))
        fit["mr_limits"] = ensure_dataframe(fit.get("mr_limits"))

        self.chart_figure = build_control_chart_figure(analysis)
        self.chart_canvas = FigureCanvas(self.chart_figure)
        self.chart_scroll_area = ReportScrollArea(self.chart_canvas, self.chart_figure, initial_zoom=0.92)
        self.chart_layout.addWidget(self.chart_scroll_area)
        self.chart_canvas.draw()

        self._enable_interactions()

    def _enable_interactions(self):
        self.interaction_managers.clear()
        if enable_interaction is None or self.chart_figure is None or self.chart_canvas is None:
            return

        try:
            manager = enable_interaction(
                host=self,
                figure_getter=lambda: self.chart_figure,
                canvas_getter=lambda: self.chart_canvas,
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

    def zoom_graph(self, action: str):
        if self.chart_scroll_area is None:
            return
        if action == "in":
            self.chart_scroll_area.zoom_in()
        elif action == "out":
            self.chart_scroll_area.zoom_out()
        elif action == "fit":
            self.chart_scroll_area.fit_to_view()
        elif action == "reset":
            self.chart_scroll_area.reset_zoom()

    def copy_graph(self):
        if self.chart_figure is None:
            QMessageBox.information(self, "No graph", "Run an analysis first.")
            return
        try:
            image = QImage()
            if not image.loadFromData(figure_to_png_bytes(self.chart_figure), "PNG"):
                raise ValueError("Could not convert graph to clipboard image.")
            QApplication.clipboard().setImage(image)
            self.app_window.statusBar().showMessage("Graph copied to clipboard as image.")
        except Exception as exc:
            QMessageBox.critical(self, "Copy failed", str(exc))

    def export_png(self):
        if self.chart_figure is None:
            QMessageBox.information(self, "No graph", "Run an analysis first.")
            return

        path, _ = QFileDialog.getSaveFileName(self, "Export PNG", "", "PNG Image (*.png)")
        if not path:
            return
        if not path.lower().endswith(".png"):
            path += ".png"
        self.chart_figure.savefig(path, dpi=160, bbox_inches="tight")
        self.app_window.statusBar().showMessage(f"Graph exported: {path}")

    def _journal_entry_payload(self) -> Dict[str, Any]:
        if self.current_analysis is None:
            raise ValueError("No analysis has been run.")

        fit = self.current_analysis.get("fit", {})
        y_name = fit.get("y_name", "")
        violations = fit.get("violations", [])
        name = f"Control Chart - {y_name}"

        return {
            "id": self.current_journal_entry_id or str(uuid.uuid4()),
            "name": name,
            "type": "Control Chart",
            "module": "Control Charts",
            "worksheet": self.current_analysis.get("worksheet", ""),
            "status": "Review" if violations else "In Control",
            "created_at": self.current_analysis.get("created_at", now_string()),
            "modified_at": now_string(),
            "summary": self.txt_summary.toPlainText(),
            "figure_png_base64": figure_to_png_base64(self.chart_figure) if self.chart_figure is not None else "",
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
        self.app_window.statusBar().showMessage("Control Chart journal entry updated.")

    def load_from_journal_entry(self, entry: Dict[str, Any]):
        self.current_journal_entry_id = entry.get("id")
        payload = entry.get("payload", {})
        self.current_analysis = payload
        self._populate_results(payload)
        self._render_figure(payload)
        self.txt_summary.setPlainText(entry.get("summary", payload.get("summary", "")))
