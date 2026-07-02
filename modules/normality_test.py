"""
modules/normality_test.py
SPS Studio — Normality Test v1.0

Features:
- Integrates with app_window exactly like GraphBuilder (journal, worksheet, statusBar).
- Supports 1 sample (single variable) or 2+ samples (multi-variable) analysis.
- Normality tests: Shapiro-Wilk, Kolmogorov-Smirnov, Anderson-Darling, D'Agostino-Pearson.
- Visualizations: Histogram + normal curve, Q-Q Plot (per sample).
- Results table with traffic-light coloring (green/red/orange).
- Summary tab with full descriptive stats, auto-saved to Analysis Journal.
- load_from_journal_entry() lets journal entries be reopened and re-edited.
- Debug tab mirrors graph_builder.py style.
"""

import base64
import io
import traceback
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QGroupBox,
    QLabel, QComboBox, QPushButton, QTextEdit, QTabWidget,
    QMessageBox, QFileDialog, QListWidget, QAbstractItemView,
    QCheckBox, QFrame, QDoubleSpinBox, QApplication,
    QTableWidget, QTableWidgetItem, QHeaderView, QScrollArea,
    QSizePolicy,
)
from PySide6.QtGui import QColor, QFont, QImage

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.gridspec as gridspec

try:
    from core.plot_interaction_tools import enable_interaction
except Exception:
    enable_interaction = None

# ─────────────────────────────────────────────────────────────────────────────
# Style constants — matching the main app palette
# ─────────────────────────────────────────────────────────────────────────────
APP_PRIMARY   = "#17324D"
APP_SECONDARY = "#2F5D87"
APP_ACCENT    = "#E8633A"

COLOR_OK      = "#2E7D32"   # green  — normal
COLOR_FAIL    = "#C62828"   # red    — not normal
COLOR_MIXED   = "#E65100"   # orange — mixed verdict

SAMPLE_COLORS = [
    "#2F5D87", "#E8633A", "#4CAF50", "#9C27B0",
    "#FF9800", "#009688", "#E91E63", "#795548",
]

ALPHA_OPTIONS = ["0.01", "0.05", "0.10"]

TESTS_ALL = [
    "Shapiro-Wilk",
    "Kolmogorov-Smirnov",
    "Anderson-Darling",
    "D'Agostino-Pearson",
]

ANALYSIS_MODES = [
    "1 Sample — Single variable",
    "2+ Samples — Multiple variables",
]


# ─────────────────────────────────────────────────────────────────────────────
# Statistical core
# ─────────────────────────────────────────────────────────────────────────────

class NormalityResult:
    """
    Runs all applicable normality tests on a single numeric sample
    and stores results in a unified dict.
    """

    def __init__(self, name: str, data: np.ndarray, alpha: float):
        self.name   = name
        self.data   = data.astype(float)
        self.n      = len(data)
        self.alpha  = alpha
        self.mean   = float(np.mean(data))
        self.std    = float(np.std(data, ddof=1)) if self.n > 1 else 0.0
        self.skewness = float(stats.skew(data))
        self.kurtosis = float(stats.kurtosis(data))   # excess kurtosis
        self.tests: dict[str, dict] = {}
        self._run_tests()

    def _run_tests(self):
        d, n, a = self.data, self.n, self.alpha

        # ── Shapiro-Wilk  (3 ≤ n ≤ 5000) ────────────────────────────────────
        if 3 <= n <= 5000:
            stat, pval = stats.shapiro(d)
            self.tests["Shapiro-Wilk"] = {
                "statistic": stat,
                "p_value": pval,
                "normal": pval >= a,
                "note": "Best for n < 50",
            }

        # ── Kolmogorov-Smirnov (parameters estimated from sample) ────────────
        stat, pval = stats.kstest(d, "norm", args=(self.mean, max(self.std, 1e-12)))
        self.tests["Kolmogorov-Smirnov"] = {
            "statistic": stat,
            "p_value": pval,
            "normal": pval >= a,
            "note": "Estimated μ, σ from sample",
        }

        # ── Anderson-Darling ─────────────────────────────────────────────────
        # SciPy 1.17+ warns that users should explicitly choose a p-value method.
        # Keep the current critical-value workflow but suppress that warning so the
        # desktop app does not show non-actionable terminal noise. If a future SciPy
        # version returns a p-value-only object, fall back to that p-value safely.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=".*users must choose a p-value calculation method.*",
                category=FutureWarning,
            )
            result = stats.anderson(d, dist="norm")

        stat_ad = float(getattr(result, "statistic", np.nan))
        critical_values = getattr(result, "critical_values", None)
        significance_levels = getattr(result, "significance_level", None)
        pvalue = getattr(result, "pvalue", None)

        if critical_values is not None and significance_levels is not None:
            # significance levels: [15%, 10%, 5%, 2.5%, 1%]
            _alpha_idx = {0.15: 0, 0.10: 1, 0.05: 2, 0.025: 3, 0.01: 4}
            a_idx = _alpha_idx.get(a, 2)
            critical = float(critical_values[a_idx])
            sig_level = float(significance_levels[a_idx])
            normal_ad = stat_ad < critical
            self.tests["Anderson-Darling"] = {
                "statistic":      stat_ad,
                "critical_value": critical,
                "sig_level":      sig_level,
                "p_value":        None,
                "normal":         normal_ad,
                "note":           f"Critical value at {sig_level}%",
            }
        else:
            pval = float(pvalue) if pvalue is not None else np.nan
            normal_ad = bool(pval >= a) if not np.isnan(pval) else False
            self.tests["Anderson-Darling"] = {
                "statistic":      stat_ad,
                "critical_value": None,
                "sig_level":      None,
                "p_value":        pval,
                "normal":         normal_ad,
                "note":           "p-value method used",
            }

        # ── D'Agostino-Pearson  (n ≥ 8) ─────────────────────────────────────
        if n >= 8:
            stat, pval = stats.normaltest(d)
            self.tests["D'Agostino-Pearson"] = {
                "statistic": stat,
                "p_value": pval,
                "normal": pval >= a,
                "note": "Combines skewness + kurtosis",
            }

    # ── Derived properties ────────────────────────────────────────────────────

    @property
    def count_normal(self) -> int:
        return sum(v["normal"] for v in self.tests.values())

    @property
    def overall_normal(self) -> bool:
        decisions = [v["normal"] for v in self.tests.values()]
        return sum(decisions) > len(decisions) / 2

    @property
    def verdict(self) -> str:
        c, t = self.count_normal, len(self.tests)
        if c == t:   return "Normal ✔"
        if c == 0:   return "Not Normal ✘"
        return f"Mixed ({c}/{t}) ⚠"

    @property
    def verdict_color(self) -> str:
        c, t = self.count_normal, len(self.tests)
        if c == t:   return COLOR_OK
        if c == 0:   return COLOR_FAIL
        return COLOR_MIXED


def run_normality_analysis(
    samples: dict[str, list],
    alpha: float = 0.05,
) -> dict[str, NormalityResult]:
    results: dict[str, NormalityResult] = {}
    for name, data in samples.items():
        arr = pd.to_numeric(pd.Series(data), errors="coerce").dropna().to_numpy()
        if len(arr) < 3:
            continue
        results[name] = NormalityResult(name, arr, alpha)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Chart builder
# ─────────────────────────────────────────────────────────────────────────────

def build_normality_figure(
    results: dict[str, NormalityResult],
    palette: list[str],
) -> Figure:
    """
    Returns a Matplotlib Figure with:
        Row 0 — Histogram + normal curve overlay (one per sample)
        Row 1 — Q-Q Plot                         (one per sample)
    """
    n = len(results)
    fig = Figure(figsize=(max(6, 5.5 * n), 8), dpi=100)
    fig.patch.set_facecolor("white")

    gs = gridspec.GridSpec(
        2, n, figure=fig,
        hspace=0.50, wspace=0.38,
        left=0.07, right=0.97,
        top=0.88, bottom=0.07,
    )

    for col, (name, res) in enumerate(results.items()):
        color = palette[col % len(palette)]
        x     = res.data

        # ── Histogram + normal PDF ────────────────────────────────────────────
        ax_h = fig.add_subplot(gs[0, col])
        ax_h.set_facecolor("#FAFAFA")

        n_bins = max(5, min(30, int(np.sqrt(res.n))))
        ax_h.hist(
            x, bins=n_bins, density=True,
            color=color, alpha=0.65, edgecolor="white",
        )
        xr = np.linspace(
            x.min() - 0.5 * max(res.std, 1e-9),
            x.max() + 0.5 * max(res.std, 1e-9),
            300,
        )
        ax_h.plot(xr, stats.norm.pdf(xr, res.mean, max(res.std, 1e-12)),
                  color="#17324D", linewidth=1.8, label="Normal PDF")

        ax_h.set_title(f"{name}\nHistogram (n={res.n})",
                       fontsize=9, fontweight="bold", color="#17324D")
        ax_h.set_xlabel("Value", fontsize=8)
        ax_h.set_ylabel("Density", fontsize=8)
        ax_h.tick_params(labelsize=7)
        ax_h.legend(fontsize=7, framealpha=0.7)

        # μ/σ annotation
        ax_h.text(
            0.97, 0.97,
            f"μ = {res.mean:.3f}\nσ = {res.std:.3f}",
            transform=ax_h.transAxes,
            ha="right", va="top", fontsize=7.5,
            bbox=dict(boxstyle="round,pad=0.3",
                      facecolor="white", edgecolor="#CCCCCC", alpha=0.85),
        )

        # verdict badge
        badge_color = res.verdict_color
        ax_h.text(
            0.03, 0.97, res.verdict,
            transform=ax_h.transAxes,
            ha="left", va="top", fontsize=7.5, fontweight="bold",
            color="white",
            bbox=dict(boxstyle="round,pad=0.3",
                      facecolor=badge_color, edgecolor="none", alpha=0.90),
        )

        # ── Q-Q Plot ──────────────────────────────────────────────────────────
        ax_q = fig.add_subplot(gs[1, col])
        ax_q.set_facecolor("#FAFAFA")

        (osm, osr), (slope, intercept, r) = stats.probplot(x, dist="norm")
        ax_q.scatter(osm, osr, s=20, color=color, alpha=0.80, zorder=3, edgecolors="none")
        ref_x = np.array([min(osm), max(osm)])
        ax_q.plot(ref_x, slope * ref_x + intercept,
                  color="#17324D", linewidth=1.5, linestyle="--",
                  label=f"R² = {r**2:.4f}")

        ax_q.set_title(f"{name}\nQ-Q Plot", fontsize=9, fontweight="bold", color="#17324D")
        ax_q.set_xlabel("Theoretical quantiles", fontsize=8)
        ax_q.set_ylabel("Sample quantiles", fontsize=8)
        ax_q.tick_params(labelsize=7)
        ax_q.legend(fontsize=7, framealpha=0.7)

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Main widget
# ─────────────────────────────────────────────────────────────────────────────

class NormalityTestWidget(QWidget):
    """
    Drop-in module for SPS Studio.
    Follows the same conventions as GraphBuilder:
      • app_window.get_active_dataframe()
      • app_window.active_worksheet_name()
      • app_window.statusBar().showMessage()
      • app_window.add_journal_entry() / update_journal_entry()
    """

    def __init__(self, app_window, parent=None):
        super().__init__(parent)
        self.app_window = app_window

        # ── State ─────────────────────────────────────────────────────────────
        self.current_summary             = ""
        self.current_journal_entry_id    = None
        self._results: dict[str, NormalityResult] = {}
        self._loading_from_journal       = False
        self._last_figure                = None
        self._last_canvas                = None
        self._interaction_manager        = None

        self._build_ui()
        self.refresh_variables()
        self.debug_connection()

    # ═════════════════════════════════════════════════════════════════════════
    # UI construction  (mirrors GraphBuilder tab/card structure)
    # ═════════════════════════════════════════════════════════════════════════

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)

        # ── Header ────────────────────────────────────────────────────────────
        header = QHBoxLayout()
        title_block = QVBoxLayout()
        title = QLabel("Normality Test")
        title.setObjectName("TitleLabel")
        subtitle = QLabel(
            "Shapiro-Wilk · Kolmogorov-Smirnov · Anderson-Darling · "
            "D'Agostino-Pearson — single or multiple samples."
        )
        subtitle.setObjectName("SubtitleLabel")
        title_block.addWidget(title)
        title_block.addWidget(subtitle)
        header.addLayout(title_block)
        header.addStretch()

        self.btn_debug = QPushButton("Debug Worksheet Connection")
        self.btn_debug.clicked.connect(self.debug_connection)
        header.addWidget(self.btn_debug)
        root.addLayout(header)

        # ── Tabs ──────────────────────────────────────────────────────────────
        self.tabs = QTabWidget()
        root.addWidget(self.tabs)

        self.setup_tab   = QWidget()
        self.results_tab = QWidget()
        self.summary_tab = QWidget()
        self.debug_tab   = QWidget()

        self.tabs.addTab(self.setup_tab,   "Setup")
        self.tabs.addTab(self.results_tab, "Results")
        self.tabs.addTab(self.summary_tab, "Summary")
        self.tabs.addTab(self.debug_tab,   "Debug")

        self._build_setup_tab()
        self._build_results_tab()
        self._build_summary_tab()
        self._build_debug_tab()

    # ── Setup tab ─────────────────────────────────────────────────────────────

    def _build_setup_tab(self):
        layout = QGridLayout(self.setup_tab)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        self.card_mode    = QGroupBox("1. Analysis Mode")
        self.card_vars    = QGroupBox("2. Variables")
        self.card_options = QGroupBox("3. Options")
        self.card_run     = QGroupBox("4. Run")

        layout.addWidget(self.card_mode,    0, 0)
        layout.addWidget(self.card_vars,    0, 1)
        layout.addWidget(self.card_options, 1, 0)
        layout.addWidget(self.card_run,     1, 1)
        layout.setColumnStretch(0, 5)
        layout.setColumnStretch(1, 5)
        layout.setRowStretch(0, 6)
        layout.setRowStretch(1, 4)

        self._build_mode_card()
        self._build_variables_card()
        self._build_options_card()
        self._build_run_card()

    def _build_mode_card(self):
        layout = QVBoxLayout(self.card_mode)

        layout.addWidget(QLabel("Analysis mode"))
        self.cmb_mode = QComboBox()
        self.cmb_mode.addItems(ANALYSIS_MODES)
        self.cmb_mode.currentTextChanged.connect(self._on_mode_changed)
        layout.addWidget(self.cmb_mode)

        layout.addSpacing(8)
        self.lbl_mode_hint = QLabel(
            "Single variable: test one column for normality.\n"
            "Multiple variables: compare 2 or more columns side by side."
        )
        self.lbl_mode_hint.setObjectName("SubtitleLabel")
        self.lbl_mode_hint.setWordWrap(True)
        layout.addWidget(self.lbl_mode_hint)
        layout.addStretch()

    def _build_variables_card(self):
        layout = QGridLayout(self.card_vars)
        layout.setColumnStretch(1, 1)

        # ── Single-variable mode ──────────────────────────────────────────────
        self.lbl_single_y = QLabel("Variable (Y)")
        self.cmb_single_y = QComboBox()
        layout.addWidget(self.lbl_single_y, 0, 0)
        layout.addWidget(self.cmb_single_y, 0, 1)

        # ── Multi-variable mode ───────────────────────────────────────────────
        self.lbl_multi_y = QLabel("Variables (select ≥ 2)")
        self.lst_multi_y = QListWidget()
        self.lst_multi_y.setSelectionMode(QAbstractItemView.MultiSelection)
        self.lst_multi_y.setMinimumHeight(110)
        layout.addWidget(self.lbl_multi_y, 1, 0, Qt.AlignTop)
        layout.addWidget(self.lst_multi_y, 1, 1)

        self.btn_refresh_vars = QPushButton("Refresh Variables")
        self.btn_refresh_vars.clicked.connect(self.refresh_variables)
        layout.addWidget(self.btn_refresh_vars, 2, 0, 1, 2)

        self._on_mode_changed(self.cmb_mode.currentText())

    def _build_options_card(self):
        layout = QVBoxLayout(self.card_options)

        # Alpha
        row_alpha = QHBoxLayout()
        row_alpha.addWidget(QLabel("Significance level (α)"))
        self.cmb_alpha = QComboBox()
        self.cmb_alpha.addItems(ALPHA_OPTIONS)
        self.cmb_alpha.setCurrentText("0.05")
        row_alpha.addWidget(self.cmb_alpha)
        row_alpha.addStretch()
        layout.addLayout(row_alpha)

        layout.addSpacing(6)

        self.chk_auto_journal = QCheckBox("Auto journal save")
        self.chk_auto_journal.setChecked(True)
        layout.addWidget(self.chk_auto_journal)

        self.chk_show_charts = QCheckBox("Show charts (Histogram + Q-Q)")
        self.chk_show_charts.setChecked(True)
        layout.addWidget(self.chk_show_charts)

        layout.addSpacing(6)

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setFrameShadow(QFrame.Sunken)
        layout.addWidget(sep)

        note = QLabel(
            "Tests applied per sample:\n"
            "• Shapiro-Wilk (n ≤ 5000)\n"
            "• Kolmogorov-Smirnov\n"
            "• Anderson-Darling\n"
            "• D'Agostino-Pearson (n ≥ 8)"
        )
        note.setObjectName("SubtitleLabel")
        layout.addWidget(note)
        layout.addStretch()

    def _build_run_card(self):
        layout = QVBoxLayout(self.card_run)

        self.btn_run = QPushButton("Run Normality Test")
        self.btn_run.setObjectName("PrimaryButton")
        self.btn_run.clicked.connect(self.run_analysis)
        self.btn_run.setMinimumHeight(36)

        self.btn_export_summary = QPushButton("Export Summary")
        self.btn_export_summary.clicked.connect(self.export_summary)

        note = QLabel(
            "Run auto-saves or updates the same Analysis Journal "
            "entry when enabled."
        )
        note.setObjectName("SubtitleLabel")
        note.setWordWrap(True)

        layout.addStretch()
        layout.addWidget(self.btn_run)
        layout.addSpacing(4)
        layout.addWidget(self.btn_export_summary)
        layout.addSpacing(8)
        layout.addWidget(note)
        layout.addStretch()

    # ── Results tab ───────────────────────────────────────────────────────────

    def _build_results_tab(self):
        layout = QVBoxLayout(self.results_tab)

        # Toolbar
        toolbar = QHBoxLayout()
        self.btn_copy_graph = QPushButton("Copy Chart")
        self.btn_copy_graph.clicked.connect(self.copy_chart_to_clipboard)
        self.btn_export_chart = QPushButton("Export Chart")
        self.btn_export_chart.clicked.connect(self.export_chart)
        self.btn_rerun = QPushButton("Re-run")
        self.btn_rerun.clicked.connect(self.run_analysis)
        toolbar.addWidget(self.btn_copy_graph)
        toolbar.addWidget(self.btn_export_chart)
        toolbar.addWidget(self.btn_rerun)
        toolbar.addStretch()
        layout.addLayout(toolbar)

        # Inner tabs: Table | Charts
        self.results_inner_tabs = QTabWidget()
        layout.addWidget(self.results_inner_tabs)

        # ── Table sub-tab ─────────────────────────────────────────────────────
        table_widget = QWidget()
        table_layout = QVBoxLayout(table_widget)

        self.results_table = QTableWidget()
        self.results_table.setAlternatingRowColors(True)
        self.results_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.results_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.results_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.results_table.verticalHeader().setVisible(False)
        table_layout.addWidget(self.results_table)
        self.results_inner_tabs.addTab(table_widget, "📋  Results Table")

        # ── Charts sub-tab ────────────────────────────────────────────────────
        charts_widget = QWidget()
        charts_layout = QVBoxLayout(charts_widget)

        self._chart_scroll = QScrollArea()
        self._chart_scroll.setWidgetResizable(True)
        self._chart_scroll.setFrameShape(QFrame.NoFrame)
        self._chart_container = QWidget()
        self._chart_container_layout = QVBoxLayout(self._chart_container)
        self._chart_scroll.setWidget(self._chart_container)
        charts_layout.addWidget(self._chart_scroll)

        self.results_inner_tabs.addTab(charts_widget, "📈  Charts")

    # ── Summary tab ───────────────────────────────────────────────────────────

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

    # ── Debug tab ─────────────────────────────────────────────────────────────

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

    # ═════════════════════════════════════════════════════════════════════════
    # Debug / data helpers  (mirrors graph_builder.py exactly)
    # ═════════════════════════════════════════════════════════════════════════

    def log(self, text: str):
        stamp = datetime.now().strftime("%H:%M:%S")
        if hasattr(self, "debug_text"):
            self.debug_text.append(f"[{stamp}] {text}")

    def debug_connection(self):
        self.debug_text.clear()
        self.log("Checking connection to main window...")
        self.log(f"get_active_dataframe  : {hasattr(self.app_window, 'get_active_dataframe')}")
        self.log(f"active_worksheet_name : {hasattr(self.app_window, 'active_worksheet_name')}")
        self.log(f"add_journal_entry     : {hasattr(self.app_window, 'add_journal_entry')}")
        self.log(f"update_journal_entry  : {hasattr(self.app_window, 'update_journal_entry')}")

        df = self.get_df()
        ws = self.app_window.active_worksheet_name() if hasattr(self.app_window, "active_worksheet_name") else "N/A"
        self.log(f"Active worksheet      : {ws}")
        self.log(f"DataFrame shape       : {df.shape if df is not None else None}")

        if df is None or df.empty:
            self.log("WARNING: DataFrame is empty. Paste data into Worksheet first.")
            return

        for col in df.columns:
            self.log(f"  - {col} | non-null={df[col].notna().sum()}")
        self.log(f"Numeric columns       : {self.numeric_columns()}")
        self.refresh_variables()

    def get_df(self):
        df = self.app_window.get_active_dataframe()
        return df if df is not None else pd.DataFrame()

    def numeric_columns(self):
        df = self.get_df()
        if df.empty:
            return []
        return [c for c in df.columns
                if pd.to_numeric(df[c], errors="coerce").notna().sum() > 0]

    def refresh_variables(self):
        df = self.get_df()
        numeric_cols = self.numeric_columns()

        prev_single = self.cmb_single_y.currentText()
        prev_multi  = self._selected_multi_y()

        self.cmb_single_y.clear()
        self.lst_multi_y.clear()

        if df.empty:
            self.log("refresh_variables: DataFrame is empty.")
            return

        self.cmb_single_y.addItems(numeric_cols)
        for col in numeric_cols:
            self.lst_multi_y.addItem(col)

        # Restore previous selections when possible
        if prev_single in numeric_cols:
            self.cmb_single_y.setCurrentText(prev_single)

        for i in range(self.lst_multi_y.count()):
            item = self.lst_multi_y.item(i)
            if item.text() in prev_multi or (not prev_multi and i < 2):
                item.setSelected(True)

        self.log(f"Variables refreshed — numeric cols: {numeric_cols}")

    def _selected_multi_y(self) -> list[str]:
        return [self.lst_multi_y.item(i).text()
                for i in range(self.lst_multi_y.count())
                if self.lst_multi_y.item(i).isSelected()]

    def _on_mode_changed(self, mode: str):
        single = mode.startswith("1 Sample")
        self.lbl_single_y.setVisible(single)
        self.cmb_single_y.setVisible(single)
        self.lbl_multi_y.setVisible(not single)
        self.lst_multi_y.setVisible(not single)

    def _active_samples(self) -> dict[str, list]:
        """Return {name: raw_data_list} based on current mode."""
        mode = self.cmb_mode.currentText()
        df   = self.get_df()

        if mode.startswith("1 Sample"):
            col = self.cmb_single_y.currentText()
            if not col:
                return {}
            vals = pd.to_numeric(df[col], errors="coerce").dropna().tolist()
            return {col: vals}

        # Multi-variable
        cols = self._selected_multi_y()
        samples = {}
        for col in cols:
            vals = pd.to_numeric(df[col], errors="coerce").dropna().tolist()
            if vals:
                samples[col] = vals
        return samples

    # ═════════════════════════════════════════════════════════════════════════
    # Validation
    # ═════════════════════════════════════════════════════════════════════════

    def validate_inputs(self) -> bool:
        df = self.get_df()
        if df.empty:
            QMessageBox.warning(self, "No data",
                                "Paste data into the active worksheet first.")
            return False

        samples = self._active_samples()
        if not samples:
            if self.cmb_mode.currentText().startswith("1 Sample"):
                QMessageBox.warning(self, "No variable",
                                    "Select a Y variable.")
            else:
                QMessageBox.warning(self, "No variables",
                                    "Select at least 2 numeric variables in the list.")
            return False

        # Multi-mode: need ≥ 2
        if not self.cmb_mode.currentText().startswith("1 Sample") and len(samples) < 2:
            QMessageBox.warning(self, "Too few variables",
                                "Select at least 2 variables for multi-sample mode.")
            return False

        # Minimum n per sample
        for name, vals in samples.items():
            arr = pd.to_numeric(pd.Series(vals), errors="coerce").dropna()
            if len(arr) < 3:
                QMessageBox.warning(self, "Insufficient data",
                                    f"'{name}' has fewer than 3 valid values (n={len(arr)}).")
                return False

        return True

    # ═════════════════════════════════════════════════════════════════════════
    # Main analysis
    # ═════════════════════════════════════════════════════════════════════════

    def run_analysis(self):
        if not self.validate_inputs():
            return

        alpha   = float(self.cmb_alpha.currentText())
        samples = self._active_samples()

        try:
            self._results = run_normality_analysis(samples, alpha)
        except Exception as exc:
            self.log(f"ERROR during analysis: {exc}\n{traceback.format_exc()}")
            QMessageBox.critical(self, "Analysis failed", str(exc))
            return

        # ── Populate UI ───────────────────────────────────────────────────────
        self._populate_results_table()
        self._populate_charts()

        self.current_summary = self._build_summary()
        self.summary_text.setPlainText(self.current_summary)

        if self.chk_auto_journal.isChecked() and not self._loading_from_journal:
            self._save_or_update_journal_entry()

        self.tabs.setCurrentWidget(self.results_tab)
        self.results_inner_tabs.setCurrentIndex(0)
        self.app_window.statusBar().showMessage(
            f"Normality Test completed — {len(self._results)} sample(s) | α = {alpha}"
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Results table
    # ─────────────────────────────────────────────────────────────────────────

    def _populate_results_table(self):
        alpha = float(self.cmb_alpha.currentText())
        cols = ["Sample", "n", "Test", "Statistic",
                "p-value / Critical", "Normal?", "Overall verdict"]
        rows = []

        for name, res in self._results.items():
            first = True
            for test_name, info in res.tests.items():
                stat = f"{info['statistic']:.5f}"

                if info.get("p_value") is not None:
                    pv_str = f"{info['p_value']:.5f}"
                else:
                    pv_str = f"CV = {info['critical_value']:.5f}"

                normal_str = "Yes ✔" if info["normal"] else "No ✘"
                verdict    = res.verdict if first else ""
                n_str      = str(res.n)   if first else ""
                name_str   = name         if first else ""
                rows.append((name_str, n_str, test_name, stat,
                             pv_str, normal_str, verdict, res))
                first = False

        self.results_table.setRowCount(len(rows))
        self.results_table.setColumnCount(len(cols))
        self.results_table.setHorizontalHeaderLabels(cols)

        for r_idx, row in enumerate(rows):
            res_obj: NormalityResult = row[-1]
            for c_idx, cell in enumerate(row[:-1]):   # skip the result object
                item = QTableWidgetItem(str(cell))
                item.setTextAlignment(Qt.AlignCenter)

                # Traffic-light colours
                if c_idx == 5:   # Normal? column
                    item.setForeground(
                        QColor(COLOR_OK if "Yes" in cell else COLOR_FAIL)
                    )
                    font = item.font()
                    font.setBold(True)
                    item.setFont(font)
                if c_idx == 6 and cell:   # Overall verdict column
                    item.setForeground(QColor(res_obj.verdict_color))
                    font = item.font()
                    font.setBold(True)
                    item.setFont(font)

                self.results_table.setItem(r_idx, c_idx, item)

        self.results_table.resizeColumnsToContents()
        self.results_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeToContents
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Charts
    # ─────────────────────────────────────────────────────────────────────────

    def _populate_charts(self):
        # Clear previous canvas(es)
        while self._chart_container_layout.count():
            item = self._chart_container_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        if not self._results or not self.chk_show_charts.isChecked():
            return

        try:
            fig    = build_normality_figure(self._results, SAMPLE_COLORS)
            canvas = FigureCanvas(fig)
            n      = len(self._results)
            canvas.setMinimumHeight(560)
            canvas.setMinimumWidth(max(640, 520 * n))
            canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            self._chart_container_layout.addWidget(canvas)
            canvas.draw()
            self._last_figure = fig   # kept for export / clipboard
            self._last_canvas = canvas
            self._enable_plot_interaction()
        except Exception:
            err_label = QLabel(
                f"Chart generation failed:\n{traceback.format_exc()}"
            )
            err_label.setWordWrap(True)
            self._chart_container_layout.addWidget(err_label)
            self._last_figure = None

    def _enable_plot_interaction(self):
        """Enable reusable SPS Matplotlib interactions for the current chart canvas."""
        self._interaction_manager = None

        if enable_interaction is None:
            self.log("Plot interaction tools not available: enable_interaction import failed.")
            return

        if self._last_figure is None or self._last_canvas is None:
            self.log("Plot interaction skipped: no active figure/canvas.")
            return

        try:
            self._interaction_manager = enable_interaction(
                host=self,
                figure_getter=lambda: self._last_figure,
                canvas_getter=lambda: self._last_canvas,
                decimals_getter=lambda: 4,
                on_changed=lambda: self._last_canvas.draw_idle() if self._last_canvas is not None else None,
                on_status=lambda msg: self.app_window.statusBar().showMessage(msg),
            )
            self._interaction_manager.register_axes()
            self.log("Plot interaction tools enabled.")
        except Exception as exc:
            self._interaction_manager = None
            self.log(f"Plot interaction tools could not be enabled: {exc}")

    # ═════════════════════════════════════════════════════════════════════════
    # Summary builder  (follows graph_builder._build_summary() style)
    # ═════════════════════════════════════════════════════════════════════════

    def _build_summary(self) -> str:
        alpha = float(self.cmb_alpha.currentText())
        ws    = self.app_window.active_worksheet_name()
        W     = 72
        lines = [
            "NORMALITY TEST", "=" * W,
            f"Worksheet  : {ws}",
            f"Created    : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Mode       : {self.cmb_mode.currentText()}",
            f"Alpha (α)  : {alpha}",
            "",
            "H0 : The data follow a normal distribution.",
            "H1 : The data do NOT follow a normal distribution.",
            f"Decision rule: Reject H0 when p-value < α ({alpha}).",
            f"               For Anderson-Darling: Reject H0 when statistic > critical value.",
            "",
        ]

        for name, res in self._results.items():
            lines += [
                "─" * W,
                f"  SAMPLE: {name}   (n = {res.n})",
                "─" * W,
                f"  Mean          : {res.mean:.6f}",
                f"  Std Dev (s)   : {res.std:.6f}",
                f"  Skewness      : {res.skewness:.6f}",
                f"  Kurtosis (ex) : {res.kurtosis:.6f}",
                "",
                f"  {'Test':<28} {'Statistic':>12} {'p-value / CV':>18} {'Normal?':>10}",
                f"  {'─'*28} {'─'*12} {'─'*18} {'─'*10}",
            ]

            for test_name, info in res.tests.items():
                stat_s = f"{info['statistic']:.6f}"
                if info.get("p_value") is not None:
                    pv_s  = f"{info['p_value']:.6f}"
                    flag  = "≥ α  → Normal" if info["normal"] else "< α  → Not Normal"
                else:
                    pv_s  = f"CV={info['critical_value']:.5f}"
                    flag  = "< CV → Normal" if info["normal"] else "≥ CV → Not Normal"
                decision = "Yes" if info["normal"] else "No"
                lines.append(f"  {test_name:<28} {stat_s:>12} {pv_s:>18} {decision:>10}   [{flag}]")
                if info.get("note"):
                    lines.append(f"  {'':>28} {'Note:':>12} {info['note']}")

            lines += [
                "",
                f"  Overall verdict : {res.verdict}",
                f"  ({res.count_normal}/{len(res.tests)} tests do not reject H0)",
                "",
            ]

        lines += [
            "=" * W,
            "INTERPRETATION GUIDE",
            "─" * W,
            "  Normal  ✔  : All tests agree data are normally distributed.",
            "  Mixed   ⚠  : Tests disagree. Inspect the Q-Q plot carefully.",
            "  Not Normal ✘: All tests reject normality.",
            "  Shapiro-Wilk is most reliable for n < 50.",
            "  For large n, even small deviations from normality become significant.",
            "=" * W,
        ]

        return "\n".join(lines)

    # ═════════════════════════════════════════════════════════════════════════
    # Journal  (same interface as graph_builder.py)
    # ═════════════════════════════════════════════════════════════════════════

    def _build_payload(self) -> dict:
        return {
            "mode":      self.cmb_mode.currentText(),
            "alpha":     self.cmb_alpha.currentText(),
            "worksheet": self.app_window.active_worksheet_name(),
            "variables": {
                "single_y": self.cmb_single_y.currentText(),
                "multi_y":  self._selected_multi_y(),
            },
            "options": {
                "auto_journal": self.chk_auto_journal.isChecked(),
                "show_charts":  self.chk_show_charts.isChecked(),
            },
        }

    def _figure_to_base64(self) -> str:
        fig = getattr(self, "_last_figure", None)
        if fig is None:
            return ""
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=140, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        return base64.b64encode(buf.getvalue()).decode("ascii")

    def _journal_updates(self, status="Updated") -> dict:
        return {
            "status":            status,
            "summary":           self.current_summary,
            "payload":           self._build_payload(),
            "figure_png_base64": self._figure_to_base64(),
            "module":            "Normality Test",
            "type":              "Normality Test",
        }

    def _save_or_update_journal_entry(self):
        if (self.current_journal_entry_id
                and hasattr(self.app_window, "update_journal_entry")):
            updated = self.app_window.update_journal_entry(
                self.current_journal_entry_id,
                self._journal_updates("Updated"),
            )
            if updated:
                return
        self._auto_save_to_journal()

    def _auto_save_to_journal(self):
        ts       = datetime.now()
        entry_id = ts.strftime("normality_%Y%m%d_%H%M%S_%f")
        self.current_journal_entry_id = entry_id
        ws = self.app_window.active_worksheet_name()
        entry = {
            "id":          entry_id,
            "name":        f"Normality Test — {ws} — {ts.strftime('%H:%M:%S')}",
            "type":        "Normality Test",
            "module":      "Normality Test",
            "worksheet":   ws,
            "created_at":  ts.strftime("%Y-%m-%d %H:%M:%S"),
            "modified_at": ts.strftime("%Y-%m-%d %H:%M:%S"),
            **self._journal_updates("Auto Saved"),
        }
        self.app_window.add_journal_entry(entry)

    def load_from_journal_entry(self, entry: dict):
        """
        Reopens a previously saved journal entry and re-runs the analysis.
        Mirrors graph_builder.load_from_journal_entry() exactly.
        """
        if not entry:
            return

        self._loading_from_journal = True
        try:
            self.current_journal_entry_id = entry.get("id")
            payload   = entry.get("payload", {}) or {}
            variables = payload.get("variables", {}) or {}
            options   = payload.get("options",   {}) or {}

            mode = payload.get("mode", ANALYSIS_MODES[0])
            idx  = self.cmb_mode.findText(mode)
            if idx >= 0:
                self.cmb_mode.setCurrentIndex(idx)
            self._on_mode_changed(self.cmb_mode.currentText())

            self.refresh_variables()

            alpha = payload.get("alpha", "0.05")
            if self.cmb_alpha.findText(alpha) >= 0:
                self.cmb_alpha.setCurrentText(alpha)

            # Single-variable
            single_y = variables.get("single_y", "")
            if self.cmb_single_y.findText(single_y) >= 0:
                self.cmb_single_y.setCurrentText(single_y)

            # Multi-variable
            multi_y = set(variables.get("multi_y", []))
            for i in range(self.lst_multi_y.count()):
                item = self.lst_multi_y.item(i)
                item.setSelected(item.text() in multi_y)

            self.chk_auto_journal.setChecked(bool(options.get("auto_journal", True)))
            self.chk_show_charts.setChecked(bool(options.get("show_charts",  True)))

        finally:
            self._loading_from_journal = False

        self.run_analysis()

    # ═════════════════════════════════════════════════════════════════════════
    # Utilities
    # ═════════════════════════════════════════════════════════════════════════

    def copy_summary(self):
        self.summary_text.selectAll()
        self.summary_text.copy()
        self.app_window.statusBar().showMessage("Summary copied to clipboard.")

    def export_summary(self):
        if not self.current_summary:
            QMessageBox.information(self, "Nothing to export",
                                    "Run the analysis first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Summary",
            "normality_results.txt",
            "Text files (*.txt);;All files (*.*)",
        )
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(self.current_summary)
            self.app_window.statusBar().showMessage(f"Summary saved: {path}")
        except Exception as exc:
            QMessageBox.critical(self, "Export failed", str(exc))

    def copy_chart_to_clipboard(self):
        fig = getattr(self, "_last_figure", None)
        if fig is None:
            QMessageBox.information(self, "No chart",
                                    "Run the analysis first.")
            return
        try:
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=220, bbox_inches="tight",
                        facecolor="white")
            buf.seek(0)
            image = QImage.fromData(buf.getvalue(), "PNG")
            if image.isNull():
                raise RuntimeError("Could not convert chart to clipboard image.")
            QApplication.clipboard().setImage(image)
            self.app_window.statusBar().showMessage("Chart copied to clipboard.")
        except Exception as exc:
            QMessageBox.critical(self, "Copy failed", str(exc))

    def export_chart(self):
        fig = getattr(self, "_last_figure", None)
        if fig is None:
            QMessageBox.information(self, "No chart",
                                    "Run the analysis first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Chart", "",
            "PNG Image (*.png);;PDF File (*.pdf);;SVG File (*.svg)",
        )
        if not path:
            return
        try:
            fig.savefig(path, dpi=160, bbox_inches="tight")
            self.app_window.statusBar().showMessage(f"Chart exported: {path}")
        except Exception as exc:
            QMessageBox.critical(self, "Export failed", str(exc))

    def reset_setup(self):
        """Called by the main window's Reset action, if any."""
        self.cmb_mode.setCurrentIndex(0)
        self._on_mode_changed(self.cmb_mode.currentText())
        self.cmb_alpha.setCurrentText("0.05")
        self.chk_auto_journal.setChecked(True)
        self.chk_show_charts.setChecked(True)
        self.refresh_variables()
        self._results = {}
        self.current_summary = ""
        self.current_journal_entry_id = None
        self.summary_text.clear()
        self.results_table.setRowCount(0)

# Backward-compatible alias for main_window imports.
NormalityTestBuilder = NormalityTestWidget
