"""
modules/graph_builder.py
SPS Studio — Graph Builder v3.3

Fixes included:
- Graph Type is populated immediately when Graph Builder opens.
- Context-sensitive variables/options are shown/hidden by graph type.
- Journal entries can be reopened and edited; edits update the same entry.
- Time Series Plot supports optional X/stamp, default observation index, and groups without restarting X at 1.
- Scatterplot allows X and Y to use the same variable, even when only one numeric variable exists.
- One Y / Multiple Y's arrangements for the graph types where it applies.
- Draggable reference lines are preserved and synced to Analysis Journal.
- v3.11: reference lines can be edited by right-clicking them; color/style/width/value/delete are supported.
"""

import base64
import io
import math
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

from PySide6.QtCore import Qt
from PySide6.QtGui import QImage
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QGroupBox,
    QLabel, QComboBox, QCheckBox, QPushButton, QTextEdit,
    QTabWidget, QMessageBox, QFileDialog, QMenu, QInputDialog,
    QSpinBox, QFrame, QListWidget, QAbstractItemView, QColorDialog,
    QDialog, QDialogButtonBox, QLineEdit, QDoubleSpinBox, QApplication,
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from core.plot_interaction_tools import enable_interaction

try:
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    HAS_3D = True
except ImportError:
    HAS_3D = False

try:
    from scipy.interpolate import griddata, CloughTocher2DInterpolator
    HAS_SCIPY = True
except Exception:
    griddata = None
    CloughTocher2DInterpolator = None
    HAS_SCIPY = False


GRAPH_FAMILIES = {
    "Distribution / Exploratory": ["Histogram", "Dotplot", "Empirical CDF", "Symmetry Plot"],
    "Relationship": ["Scatterplot", "Matrix Plot", "Bubble Plot"],
    "Comparison": ["Boxplot", "Interval Plot", "Individual Value Plot", "Line Plot", "Multi-Vari Chart", "Variability Chart"],
    "Categorical / Others": ["Bar Chart", "Pie Chart"],
    "Time Based": ["Time Series Plot", "Area Graph"],
    "Advanced": ["Contour Plot", "3D Scatterplot", "3D Surface Plot"],
}

IMPLEMENTED_GRAPHS = {g for values in GRAPH_FAMILIES.values() for g in values}

REFERENCE_LINE_COLOR = "#C56A2D"
APP_PRIMARY = "#17324D"
APP_SECONDARY = "#2F5D87"
APP_ACCENT = "#E8633A"

COLOR_PALETTES = {
    "SPS Blue": ["#2F5D87", "#E8633A", "#4CAF50", "#9C27B0", "#FF9800", "#009688"],
    "Vibrant": ["#E91E63", "#2196F3", "#4CAF50", "#FF5722", "#9C27B0", "#00BCD4"],
    "Muted": ["#6C8EBF", "#D6A35C", "#7BBF7B", "#BF7B9C", "#A08FBF", "#7BA8BF"],
    "Monochrome": ["#212121", "#424242", "#616161", "#757575", "#9E9E9E", "#BDBDBD"],
    "Warm": ["#C0392B", "#E67E22", "#F1C40F", "#E74C3C", "#D35400", "#F39C12"],
    "Cool": ["#1ABC9C", "#3498DB", "#9B59B6", "#2ECC71", "#34495E", "#16A085"],
}

# X is required for relationship/advanced graphs, optional for time/index plots.
REQUIRES_X = {"Scatterplot", "Bubble Plot", "Contour Plot", "3D Scatterplot", "3D Surface Plot"}
OPTIONAL_X = {"Line Plot", "Time Series Plot", "Area Graph", "Multi-Vari Chart", "Variability Chart"}
REQUIRES_Z = {"Bubble Plot", "Contour Plot", "3D Scatterplot", "3D Surface Plot"}
SUPPORTS_GROUP = {
    "Histogram", "Dotplot", "Empirical CDF", "Boxplot", "Interval Plot",
    "Individual Value Plot", "Bar Chart", "Line Plot", "Time Series Plot", "Area Graph",
    "Multi-Vari Chart", "Variability Chart",
}
SUPPORTS_MULTI_Y = {
    "Histogram", "Dotplot", "Empirical CDF", "Boxplot", "Interval Plot",
    "Individual Value Plot", "Line Plot", "Time Series Plot", "Area Graph", "Bar Chart",
    "Multi-Vari Chart", "Variability Chart",
}
SHOWS_BINS = {"Histogram"}
SHOWS_TREND = {"Scatterplot", "Line Plot", "Time Series Plot", "Area Graph"}
SHOWS_JITTER = {"Individual Value Plot", "Dotplot", "Multi-Vari Chart", "Variability Chart"}
SHOWS_CONTOUR_OPTIONS = {"Contour Plot"}

ARRANGEMENTS = [
    "One Y - Simple",
    "One Y - With Groups",
    "Multiple Y's - Simple",
    "Multiple Y's - With Groups",
]


class AxisScaleDialog(QDialog):
    def __init__(self, parent, axis_name, current_min, current_max, current_auto_min=True, current_auto_max=True):
        super().__init__(parent)

        self.setWindowTitle(f"{axis_name} Axis Scale Range")
        self.setModal(True)

        root = QVBoxLayout(self)

        title = QLabel("Scale Range")
        title.setStyleSheet("font-weight: bold;")
        root.addWidget(title)

        grid = QGridLayout()

        self.chk_auto_min = QCheckBox("Auto")
        self.chk_auto_min.setChecked(current_auto_min)

        self.txt_min = QLineEdit()
        self.txt_min.setText("" if current_min is None else str(current_min))
        self.txt_min.setEnabled(not current_auto_min)

        self.chk_auto_max = QCheckBox("Auto")
        self.chk_auto_max.setChecked(current_auto_max)

        self.txt_max = QLineEdit()
        self.txt_max.setText("" if current_max is None else str(current_max))
        self.txt_max.setEnabled(not current_auto_max)

        grid.addWidget(QLabel("Minimum:"), 0, 0)
        grid.addWidget(self.chk_auto_min, 0, 1)
        grid.addWidget(self.txt_min, 0, 2)

        grid.addWidget(QLabel("Maximum:"), 1, 0)
        grid.addWidget(self.chk_auto_max, 1, 1)
        grid.addWidget(self.txt_max, 1, 2)

        root.addLayout(grid)

        self.chk_auto_min.toggled.connect(lambda checked: self.txt_min.setEnabled(not checked))
        self.chk_auto_max.toggled.connect(lambda checked: self.txt_max.setEnabled(not checked))

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

    def values(self):
        min_text = "" if self.chk_auto_min.isChecked() else self.txt_min.text().strip()
        max_text = "" if self.chk_auto_max.isChecked() else self.txt_max.text().strip()

        return {
            "auto_min": self.chk_auto_min.isChecked(),
            "auto_max": self.chk_auto_max.isChecked(),
            "min_text": min_text,
            "max_text": max_text,
        }

class GraphBuilder(QWidget):
    def __init__(self, app_window, initial_graph="Histogram", parent=None):
        super().__init__(parent)
        self.app_window = app_window
        self.initial_graph = initial_graph

        self.current_summary = ""
        self.current_payload = None
        self.current_journal_entry_id = None

        self.reference_lines = {"x": [], "y": []}
        self._ref_artists = []
        self._ref_artist_map = {}
        self._dragging_ref = None
        self._loading_from_journal = False
        self._last_auto_title = ""
        self._axis_label_overrides = {"x": "", "y": ""}
        self._axis_limits = {
            "x": [None, None],
            "y": [None, None],
        }
        self._contour_boundary_z = ""
        self.graph_interactions = None

        self._build_ui()
        self.refresh_variables()
        self.set_graph_type(initial_graph)
        self.debug_connection()

    # ─────────────────────────────────────────────
    # UI
    # ─────────────────────────────────────────────

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)

        header = QHBoxLayout()
        title_block = QVBoxLayout()
        title = QLabel("SPS Graph Builder")
        title.setObjectName("TitleLabel")
        subtitle = QLabel("Create Minitab-style graphs, reference lines and journaled analysis snapshots.")
        subtitle.setObjectName("SubtitleLabel")
        title_block.addWidget(title)
        title_block.addWidget(subtitle)
        header.addLayout(title_block)
        header.addStretch()

        self.btn_debug = QPushButton("Debug Worksheet Connection")
        self.btn_debug.clicked.connect(self.debug_connection)
        header.addWidget(self.btn_debug)
        root.addLayout(header)

        self.tabs = QTabWidget()
        root.addWidget(self.tabs)

        self.setup_tab = QWidget()
        self.graph_tab = QWidget()
        self.summary_tab = QWidget()
        self.debug_tab = QWidget()

        self.tabs.addTab(self.setup_tab, "Setup")
        self.tabs.addTab(self.graph_tab, "Graph")
        self.tabs.addTab(self.summary_tab, "Summary")
        self.tabs.addTab(self.debug_tab, "Debug")

        self._build_setup_tab()
        self._build_graph_tab()
        self._build_summary_tab()
        self._build_debug_tab()

    def _build_setup_tab(self):
        layout = QGridLayout(self.setup_tab)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        self.card_graph = QGroupBox("1. Choose Graph")
        self.card_vars = QGroupBox("2. Variables")
        self.card_options = QGroupBox("3. Options")
        self.card_run = QGroupBox("4. Run")

        layout.addWidget(self.card_graph, 0, 0)
        layout.addWidget(self.card_vars, 0, 1)
        layout.addWidget(self.card_options, 1, 0)
        layout.addWidget(self.card_run, 1, 1)
        layout.setColumnStretch(0, 6)
        layout.setColumnStretch(1, 4)
        layout.setRowStretch(0, 6)
        layout.setRowStretch(1, 4)

        self._build_graph_card()
        self._build_variables_card()
        self._build_options_card()
        self._build_run_card()

        # Populate Graph Type after all dependent widgets exist.
        self._family_changed(self.cmb_family.currentText())

    def _build_graph_card(self):
        layout = QVBoxLayout(self.card_graph)

        layout.addWidget(QLabel("Graph Family"))
        self.cmb_family = QComboBox()
        self.cmb_family.addItems(GRAPH_FAMILIES.keys())
        layout.addWidget(self.cmb_family)

        layout.addWidget(QLabel("Graph Type"))
        self.cmb_graph = QComboBox()
        layout.addWidget(self.cmb_graph)

        quick = QHBoxLayout()
        for graph in ["Histogram", "Scatterplot", "Boxplot", "Time Series Plot"]:
            btn = QPushButton(graph)
            btn.clicked.connect(lambda checked=False, g=graph: self.set_graph_type(g))
            quick.addWidget(btn)
        layout.addLayout(quick)
        layout.addStretch()

        self.cmb_family.currentTextChanged.connect(self._family_changed)
        self.cmb_graph.currentTextChanged.connect(self._graph_changed)


    def _build_variables_card(self):
        layout = QGridLayout(self.card_vars)

        self.lbl_arrangement = QLabel("Data arrangement")
        self.cmb_arrangement = QComboBox()
        self.cmb_arrangement.addItems(ARRANGEMENTS)
        self.cmb_arrangement.currentTextChanged.connect(lambda _: self._update_requirements(self.cmb_graph.currentText()))
        layout.addWidget(self.lbl_arrangement, 0, 0)
        layout.addWidget(self.cmb_arrangement, 0, 1)

        self.lbl_y = QLabel("Y / Response")
        self.cmb_y = QComboBox()
        layout.addWidget(self.lbl_y, 1, 0)
        layout.addWidget(self.cmb_y, 1, 1)

        self.lbl_multi_y = QLabel("Multiple Y's")
        self.lst_y = QListWidget()
        self.lst_y.setSelectionMode(QAbstractItemView.MultiSelection)
        self.lst_y.setMaximumHeight(95)
        layout.addWidget(self.lbl_multi_y, 2, 0)
        layout.addWidget(self.lst_y, 2, 1)

        self.lbl_x = QLabel("X / Stamp / Predictor")
        self.cmb_x = QComboBox()
        layout.addWidget(self.lbl_x, 3, 0)
        layout.addWidget(self.cmb_x, 3, 1)

        self.lbl_z = QLabel("Z / Size")
        self.cmb_z = QComboBox()
        layout.addWidget(self.lbl_z, 4, 0)
        layout.addWidget(self.cmb_z, 4, 1)

        self.lbl_group = QLabel("Group / Category")
        self.cmb_group = QComboBox()
        layout.addWidget(self.lbl_group, 5, 0)
        layout.addWidget(self.cmb_group, 5, 1)

        layout.addWidget(QLabel("Chart Title"), 6, 0)
        self.cmb_title = QComboBox()
        self.cmb_title.setEditable(True)
        layout.addWidget(self.cmb_title, 6, 1)

        # Auto-title refresh. Manual user edits are preserved.
        self.cmb_y.currentTextChanged.connect(lambda _: self._update_auto_title())
        self.cmb_x.currentTextChanged.connect(lambda _: self._update_auto_title())
        self.cmb_group.currentTextChanged.connect(lambda _: self._update_auto_title())
        self.cmb_arrangement.currentTextChanged.connect(lambda _: self._update_auto_title())
        self.lst_y.itemSelectionChanged.connect(self._update_auto_title)

        self.btn_refresh_vars = QPushButton("Refresh Variables")
        self.btn_refresh_vars.clicked.connect(self.refresh_variables)
        layout.addWidget(self.btn_refresh_vars, 7, 0, 1, 2)
        layout.setColumnStretch(1, 1)

    def _build_options_card(self):
        layout = QVBoxLayout(self.card_options)

        self.chk_grid = QCheckBox("Show grid")
        self.chk_grid.setChecked(True)
        self.chk_mean = QCheckBox("Show mean")
        self.chk_mean.setChecked(True)
        self.chk_median = QCheckBox("Show median")
        self.chk_median.setChecked(False)
        self.chk_auto_conclusions = QCheckBox("Auto conclusions")
        self.chk_auto_conclusions.setChecked(True)
        self.chk_auto_journal = QCheckBox("Auto journal save")
        self.chk_auto_journal.setChecked(True)

        for w in [self.chk_grid, self.chk_mean, self.chk_median, self.chk_auto_conclusions, self.chk_auto_journal]:
            layout.addWidget(w)

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setFrameShadow(QFrame.Sunken)
        layout.addWidget(sep)

        row_bins = QHBoxLayout()
        self.lbl_bins = QLabel("Bins:")
        self.spn_bins = QSpinBox()
        self.spn_bins.setRange(2, 200)
        self.spn_bins.setValue(10)
        row_bins.addWidget(self.lbl_bins)
        row_bins.addWidget(self.spn_bins)
        row_bins.addStretch()
        self.row_bins_widget = QWidget()
        self.row_bins_widget.setLayout(row_bins)
        layout.addWidget(self.row_bins_widget)

        self.chk_trend = QCheckBox("Show regression/trend line")
        self.chk_trend.setChecked(False)
        layout.addWidget(self.chk_trend)

        self.chk_jitter = QCheckBox("Add jitter")
        self.chk_jitter.setChecked(True)
        layout.addWidget(self.chk_jitter)

        row_pal = QHBoxLayout()
        row_pal.addWidget(QLabel("Palette:"))
        self.cmb_palette = QComboBox()
        self.cmb_palette.addItems(COLOR_PALETTES.keys())
        row_pal.addWidget(self.cmb_palette)
        layout.addLayout(row_pal)

        # Contour Plot options
        self.contour_options_box = QGroupBox("Contour Interpolation Method")
        contour_layout = QGridLayout(self.contour_options_box)

        contour_layout.addWidget(QLabel("Interpolation Method:"), 0, 0)
        self.cmb_contour_method = QComboBox()
        self.cmb_contour_method.addItems([
            "Distance method",
            "Akima's polynomial method",
        ])
        contour_layout.addWidget(self.cmb_contour_method, 0, 1)

        contour_layout.addWidget(QLabel("Distance power:"), 1, 0)
        self.spn_distance_power = QDoubleSpinBox()
        self.spn_distance_power.setRange(0.1, 10.0)
        self.spn_distance_power.setDecimals(2)
        self.spn_distance_power.setSingleStep(0.5)
        self.spn_distance_power.setValue(2.0)
        contour_layout.addWidget(self.spn_distance_power, 1, 1)

        self.chk_standardize_xy = QCheckBox("Standardize x- and y-data")
        self.chk_standardize_xy.setChecked(True)
        contour_layout.addWidget(self.chk_standardize_xy, 2, 0, 1, 2)

        contour_layout.addWidget(QLabel("Boundary z-value:"), 3, 0)
        self.txt_boundary_z = QLineEdit()
        self.txt_boundary_z.setPlaceholderText("Optional")
        contour_layout.addWidget(self.txt_boundary_z, 3, 1)

        layout.addWidget(self.contour_options_box)

        self.cmb_contour_method.currentTextChanged.connect(self._update_contour_options_visibility)
        self._update_contour_options_visibility()

        layout.addStretch()

    def _build_run_card(self):
        layout = QVBoxLayout(self.card_run)

        self.btn_create = QPushButton("Create Graph")
        self.btn_create.setObjectName("PrimaryButton")
        self.btn_create.clicked.connect(self.create_graph)

        note = QLabel("Create Graph auto-saves or updates the same Analysis Journal entry when enabled.")
        note.setObjectName("SubtitleLabel")
        note.setWordWrap(True)

        layout.addStretch()
        layout.addWidget(self.btn_create)
        layout.addWidget(note)
        layout.addStretch()

    def _build_graph_tab(self):
        layout = QVBoxLayout(self.graph_tab)
        toolbar = QHBoxLayout()
        self.btn_export = QPushButton("Export Figure")
        self.btn_export.clicked.connect(self.export_figure)
        self.btn_copy_graph = QPushButton("Copy Graph")
        self.btn_copy_graph.clicked.connect(self.copy_graph_to_clipboard)
        self.btn_rerun = QPushButton("Re-run")
        self.btn_rerun.clicked.connect(self.create_graph)
        toolbar.addWidget(self.btn_export)
        toolbar.addWidget(self.btn_copy_graph)
        toolbar.addWidget(self.btn_rerun)
        toolbar.addStretch()
        layout.addLayout(toolbar)

        self.figure = Figure(figsize=(8, 5), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.graph_interactions = enable_interaction(
            host=self,
            figure_getter=lambda: self.figure,
            canvas_getter=lambda: self.canvas,
            decimals_getter=lambda: 2,
            on_changed=self._refresh_summary_after_reference_change,
            on_status=lambda msg: self.app_window.statusBar().showMessage(msg),
        )
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
    # Debug / data helpers
    # ─────────────────────────────────────────────

    def log(self, text):
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
        self.log(f"3D support            : {HAS_3D}")
        self.log(f"scipy/griddata support: {HAS_SCIPY}")

        df = self.get_df()
        self.log(f"Active worksheet      : {self.app_window.active_worksheet_name() if hasattr(self.app_window, 'active_worksheet_name') else 'N/A'}")
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
        return [c for c in df.columns if pd.to_numeric(df[c], errors="coerce").notna().sum() > 0]

    def all_columns(self):
        df = self.get_df()
        return list(df.columns) if not df.empty else []

    def refresh_variables(self):
        df = self.get_df()
        numeric_cols = self.numeric_columns()
        all_cols = self.all_columns()
        current = {
            "y": self.cmb_y.currentText(),
            "x": self.cmb_x.currentText(),
            "z": self.cmb_z.currentText(),
            "group": self.cmb_group.currentText(),
            "multi": self.selected_y_columns(),
        }

        for cmb in [self.cmb_y, self.cmb_x, self.cmb_z, self.cmb_group]:
            cmb.clear()
        self.lst_y.clear()

        if df.empty:
            self.log("refresh_variables: DataFrame is empty.")
            return

        self.cmb_y.addItems(numeric_cols)
        self.cmb_x.addItem("")  # Optional X for time/index plots.
        self.cmb_x.addItems(all_cols)  # Include all cols to support date stamps.
        self.cmb_z.addItems(numeric_cols)
        self.cmb_group.addItem("")
        self.cmb_group.addItems(all_cols)

        for col in numeric_cols:
            self.lst_y.addItem(col)

        if current["y"] in numeric_cols:
            self.cmb_y.setCurrentText(current["y"])
        elif numeric_cols:
            self.cmb_y.setCurrentText(numeric_cols[0])

        # X can be the same as Y. For optional-X plots (Time Series/Line/Area),
        # default to blank so quick analysis uses the observation index.
        current_graph = self.cmb_graph.currentText() if hasattr(self, "cmb_graph") else ""
        if current_graph in OPTIONAL_X and not self._loading_from_journal:
            self.cmb_x.setCurrentText("")
        elif current["x"] in all_cols:
            self.cmb_x.setCurrentText(current["x"])
        elif numeric_cols:
            self.cmb_x.setCurrentText(numeric_cols[0])

        if current["z"] in numeric_cols:
            self.cmb_z.setCurrentText(current["z"])
        elif len(numeric_cols) >= 2:
            self.cmb_z.setCurrentText(numeric_cols[1])

        if current["group"] in all_cols:
            self.cmb_group.setCurrentText(current["group"])

        for i in range(self.lst_y.count()):
            item = self.lst_y.item(i)
            if item.text() in current["multi"] or (not current["multi"] and i == 0):
                item.setSelected(True)

        self._update_requirements(self.cmb_graph.currentText())
        self.log(f"Variables refreshed — numeric={numeric_cols} | all={all_cols}")

    def selected_y_columns(self):
        return [i.text() for i in self.lst_y.selectedItems()]

    def active_y_columns(self):
        if self.is_multiple_y_mode():
            ys = self.selected_y_columns()
            return ys if ys else ([self.cmb_y.currentText()] if self.cmb_y.currentText() else [])
        return [self.cmb_y.currentText()] if self.cmb_y.currentText() else []

    def is_multiple_y_mode(self):
        return self.cmb_arrangement.currentText().startswith("Multiple")

    def is_group_mode(self):
        return self.cmb_arrangement.currentText().endswith("With Groups")

    def get_numeric_series(self, col_name):
        df = self.get_df()
        if df.empty or not col_name or col_name not in df.columns:
            return pd.Series(dtype=float)
        return pd.to_numeric(df[col_name], errors="coerce").dropna()

    def get_clean_xy(self, x_col, y_col, allow_datetime_x=False, optional_x=False):
        df = self.get_df()
        if df.empty or y_col not in df.columns:
            return pd.DataFrame(columns=["_x", y_col, "_row"])

        temp = pd.DataFrame({"_row": np.arange(1, len(df) + 1), y_col: pd.to_numeric(df[y_col], errors="coerce")})

        if optional_x and not x_col:
            temp["_x"] = temp["_row"]
            return temp.dropna(subset=["_x", y_col])

        if not x_col or x_col not in df.columns:
            return pd.DataFrame(columns=["_x", y_col, "_row"])

        if allow_datetime_x:
            parsed_x = pd.to_datetime(df[x_col], errors="coerce")
            if parsed_x.notna().sum() > 0:
                temp["_x"] = parsed_x
            else:
                temp["_x"] = pd.to_numeric(df[x_col], errors="coerce")
        else:
            temp["_x"] = pd.to_numeric(df[x_col], errors="coerce")

        return temp.dropna(subset=["_x", y_col])

    def get_clean_xyz(self, x_col, y_col, z_col):
        df = self.get_df()
        if df.empty or not all(c in df.columns for c in [x_col, y_col, z_col]):
            return pd.DataFrame(columns=[x_col, y_col, z_col])
        temp = df[[x_col, y_col, z_col]].copy()
        for c in [x_col, y_col, z_col]:
            temp[c] = pd.to_numeric(temp[c], errors="coerce")
        return temp.dropna(subset=[x_col, y_col, z_col])

    def get_clean_y_group(self, y_col, group_col=""):
        df = self.get_df()
        if df.empty or y_col not in df.columns:
            return pd.DataFrame(columns=[y_col])
        if group_col and group_col in df.columns:
            temp = df[[y_col, group_col]].copy()
            temp[y_col] = pd.to_numeric(temp[y_col], errors="coerce")
            return temp.dropna(subset=[y_col, group_col])
        temp = df[[y_col]].copy()
        temp[y_col] = pd.to_numeric(temp[y_col], errors="coerce")
        return temp.dropna(subset=[y_col])

    # ─────────────────────────────────────────────
    # Graph selection / options
    # ─────────────────────────────────────────────

    def set_graph_type(self, graph_name):
        for family, graphs in GRAPH_FAMILIES.items():
            if graph_name in graphs:
                self.cmb_family.blockSignals(True)
                self.cmb_family.setCurrentText(family)
                self.cmb_family.blockSignals(False)
                self._populate_graph_types(family, graph_name)
                self._graph_changed(graph_name)
                return

    def _family_changed(self, family):
        self._populate_graph_types(family)

    def _populate_graph_types(self, family, preferred=None):
        self.cmb_graph.blockSignals(True)
        self.cmb_graph.clear()
        self.cmb_graph.addItems(GRAPH_FAMILIES.get(family, []))
        if preferred and preferred in GRAPH_FAMILIES.get(family, []):
            self.cmb_graph.setCurrentText(preferred)
        self.cmb_graph.blockSignals(False)
        if self.cmb_graph.currentText():
            self._graph_changed(self.cmb_graph.currentText())

    def _graph_changed(self, graph_name):
        if not graph_name:
            return

        self._update_requirements(graph_name)

        if graph_name in OPTIONAL_X and not self._loading_from_journal:
            self.cmb_x.blockSignals(True)
            self.cmb_x.setCurrentText("")
            self.cmb_x.blockSignals(False)

        self._update_auto_title(force=True)

    def _set_row_visible(self, label, widget, visible):
        label.setVisible(visible)
        widget.setVisible(visible)

    def _update_contour_options_visibility(self):
        if not hasattr(self, "cmb_contour_method"):
            return

        method = self.cmb_contour_method.currentText()

        is_distance = method == "Distance method"
        is_akima = method == "Akima's polynomial method"

        self.spn_distance_power.setVisible(is_distance)
        self.chk_standardize_xy.setVisible(is_distance)

        # Labels inside grid stay visible unless manually handled.
        # Simpler: keep Boundary z-value visible only for Akima.
        self.txt_boundary_z.setVisible(is_akima)

    def _update_requirements(self, graph_name):
        supports_multi = graph_name in SUPPORTS_MULTI_Y
        supports_group = graph_name in SUPPORTS_GROUP
        requires_x = graph_name in REQUIRES_X
        optional_x = graph_name in OPTIONAL_X
        requires_z = graph_name in REQUIRES_Z

        self.lbl_arrangement.setVisible(supports_multi or supports_group)
        self.cmb_arrangement.setVisible(supports_multi or supports_group)

        if not supports_multi and self.is_multiple_y_mode():
            self.cmb_arrangement.setCurrentText("One Y - Simple")

        if not supports_group and self.is_group_mode():
            self.cmb_arrangement.setCurrentText("One Y - Simple" if not self.is_multiple_y_mode() else "Multiple Y's - Simple")

        multiple = self.is_multiple_y_mode() and supports_multi
        group_mode = self.is_group_mode() and supports_group

        self._set_row_visible(self.lbl_y, self.cmb_y, not multiple and graph_name != "Matrix Plot")
        self._set_row_visible(self.lbl_multi_y, self.lst_y, multiple)
        self._set_row_visible(self.lbl_x, self.cmb_x, requires_x or optional_x)
        self._set_row_visible(self.lbl_z, self.cmb_z, requires_z)
        self._set_row_visible(self.lbl_group, self.cmb_group, group_mode or graph_name == "Pie Chart")

        if graph_name in OPTIONAL_X:
            self.lbl_x.setText("X / Stamp (optional)")
        else:
            self.lbl_x.setText("X / Predictor")

        self.row_bins_widget.setVisible(graph_name in SHOWS_BINS)
        self.chk_trend.setVisible(graph_name in SHOWS_TREND)
        self.chk_jitter.setVisible(graph_name in SHOWS_JITTER)
        self.contour_options_box.setVisible(graph_name in SHOWS_CONTOUR_OPTIONS)
        self._update_contour_options_visibility()
        self.chk_mean.setVisible(graph_name not in {"Scatterplot", "Bubble Plot", "Contour Plot", "3D Scatterplot", "3D Surface Plot", "Pie Chart", "Matrix Plot"})
        self.chk_median.setVisible(graph_name in {"Histogram", "Boxplot", "Individual Value Plot", "Dotplot"})
        self._update_auto_title()

    def _auto_title_text(self):
        graph = self.cmb_graph.currentText()
        ys = self.active_y_columns() if hasattr(self, "lst_y") else []
        x_col = self.cmb_x.currentText() if hasattr(self, "cmb_x") else ""

        y_text = ", ".join(ys) if ys else ""

        if graph == "Scatterplot":
            y = self.cmb_y.currentText()
            x = x_col or "X"
            return f"Scatterplot of {y} vs {x}" if y else "Scatterplot"

        if graph in {"Time Series Plot", "Line Plot", "Area Graph"}:
            return f"{graph} of {y_text}" if y_text else graph

        if graph in {"Boxplot", "Histogram", "Dotplot", "Empirical CDF", "Individual Value Plot", "Interval Plot", "Bar Chart", "Symmetry Plot", "Multi-Vari Chart", "Variability Chart"}:
            return f"{graph} of {y_text}" if y_text else graph

        return graph

    def _update_auto_title(self, force=False):
        if not hasattr(self, "cmb_title"):
            return

        new_title = self._auto_title_text()
        current = self.cmb_title.currentText().strip()

        should_update = (
            force
            or not current
            or current == self._last_auto_title
            or current in IMPLEMENTED_GRAPHS
        )

        if should_update:
            self.cmb_title.setEditText(new_title)
            self._last_auto_title = new_title

    def _palette(self):
        return COLOR_PALETTES.get(self.cmb_palette.currentText(), COLOR_PALETTES["SPS Blue"])

    def _color(self, index=0):
        pal = self._palette()
        return pal[index % len(pal)]

    # ─────────────────────────────────────────────
    # Validation
    # ─────────────────────────────────────────────

    def validate_inputs(self):
        df = self.get_df()
        graph = self.cmb_graph.currentText()
        if df.empty:
            QMessageBox.warning(self, "No data", "Paste data into the active worksheet first.")
            return False
        if graph not in IMPLEMENTED_GRAPHS:
            QMessageBox.warning(self, "Graph not implemented", f"{graph} is not implemented.")
            return False

        y_cols = self.active_y_columns()
        x_col = self.cmb_x.currentText()
        z_col = self.cmb_z.currentText()
        group_col = self.cmb_group.currentText()

        if graph == "Matrix Plot":
            if len(self.numeric_columns()) < 2:
                QMessageBox.warning(self, "Too few variables", "Matrix Plot needs at least 2 numeric columns.")
                return False
            return True

        if graph == "Pie Chart":
            if not group_col:
                QMessageBox.warning(self, "Missing category", "Pie Chart requires a Group / Category variable.")
                return False
            return True

        if not y_cols:
            QMessageBox.warning(self, "Missing Y", "Select at least one Y / Response variable.")
            return False
        for y in y_cols:
            if len(self.get_numeric_series(y)) == 0:
                QMessageBox.warning(self, "Invalid Y", f"{y} has no numeric values.")
                return False

        if graph in REQUIRES_X:
            if not x_col:
                QMessageBox.warning(self, "Missing X", "Select an X / Predictor variable.")
                return False
            allow_dt = False
            if self.get_clean_xy(x_col, y_cols[0], allow_datetime_x=allow_dt).empty:
                QMessageBox.warning(self, "Invalid X/Y", "No valid paired X/Y rows found.")
                return False

        if graph in {"Multi-Vari Chart", "Variability Chart"}:
            if not x_col:
                QMessageBox.warning(self, "Missing X", "Select a categorical X variable.")
                return False
            if x_col not in df.columns:
                QMessageBox.warning(self, "Invalid X", "The selected X variable does not exist in the active worksheet.")
                return False

        if graph in REQUIRES_Z:
            if not z_col:
                QMessageBox.warning(self, "Missing Z", "Select a Z / Size variable.")
                return False
            if len(self.get_numeric_series(z_col)) == 0:
                QMessageBox.warning(self, "Invalid Z", "Z variable has no numeric values.")
                return False

        if self.is_group_mode() and graph in SUPPORTS_GROUP and not group_col:
            QMessageBox.warning(self, "Missing Group", "Select a Group / Category variable or use a Simple arrangement.")
            return False

        return True

    # ─────────────────────────────────────────────
    # Plot dispatcher
    # ─────────────────────────────────────────────

    def create_graph(self):
        if not self.validate_inputs():
            return
        graph = self.cmb_graph.currentText()
        try:
            self.figure.clear()
            is_3d = graph in {"3D Scatterplot", "3D Surface Plot"}
            if is_3d and HAS_3D:
                ax = self.figure.add_subplot(111, projection="3d")
            elif graph == "Matrix Plot":
                ax = None
            else:
                ax = self.figure.add_subplot(111)

            dispatch = {
                "Histogram": self._plot_histogram,
                "Dotplot": self._plot_dotplot,
                "Empirical CDF": self._plot_ecdf,
                "Scatterplot": self._plot_scatter,
                "Matrix Plot": self._plot_matrix,
                "Bubble Plot": self._plot_bubble,
                "Boxplot": self._plot_boxplot,
                "Interval Plot": self._plot_interval,
                "Individual Value Plot": self._plot_individual_value,
                "Line Plot": lambda a: self._plot_line(a, area=False, time_series=False),
                "Time Series Plot": lambda a: self._plot_line(a, area=False, time_series=True),
                "Area Graph": lambda a: self._plot_line(a, area=True, time_series=False),
                "Bar Chart": self._plot_bar,
                "Multi-Vari Chart": self._plot_multi_vari,
                "Variability Chart": self._plot_variability,
                "Symmetry Plot": self._plot_symmetry,
                "Pie Chart": self._plot_pie,
                "Contour Plot": self._plot_contour,
                "3D Scatterplot": self._plot_3d_scatter,
                "3D Surface Plot": self._plot_3d_surface,
            }
            dispatch[graph](ax)

            if ax is not None and graph != "Pie Chart":
                self._update_auto_title()
                title = self.cmb_title.currentText().strip() or self._auto_title_text()
                ax.set_title(title, fontweight="bold", pad=10)
                if self.graph_interactions is not None:
                    self.graph_interactions.apply_axis_labels(ax, self._axis_label_overrides)
                    self.graph_interactions.apply_axis_limits(ax, self._axis_limits)
                else:
                    self._apply_axis_label_overrides(ax)
                    self._apply_axis_limits(ax)

                if self.chk_grid.isChecked() and graph not in {"Contour Plot", "3D Scatterplot", "3D Surface Plot"}:
                    ax.grid(True, alpha=0.25, linestyle="--")   
                if graph not in {"3D Scatterplot", "3D Surface Plot", "Matrix Plot"}:
                    if self.graph_interactions is not None:
                        self.graph_interactions.register_axes()
                        self.graph_interactions.from_graph_builder_reference_dict(self.reference_lines)
                        self.graph_interactions.apply_reference_lines()
                    else:
                        self._draw_reference_lines(ax)

            self.figure.tight_layout()
            self.canvas.draw()
            self.current_summary = self._build_summary()
            self.summary_text.setPlainText(self.current_summary)

            if self.chk_auto_journal.isChecked() and not self._loading_from_journal:
                self._save_or_update_journal_entry()
            elif self.current_journal_entry_id:
                self._sync_current_journal_entry()

            self.tabs.setCurrentWidget(self.graph_tab)
            self.app_window.statusBar().showMessage(f"Graph created: {graph}")
        except Exception as exc:
            self.log(f"ERROR while creating graph: {exc}")
            QMessageBox.critical(self, "Graph failed", str(exc))
            raise

    # ─────────────────────────────────────────────
    # Plot implementations
    # ─────────────────────────────────────────────

    def _plot_histogram(self, ax):
        ys = self.active_y_columns()
        group_col = self.cmb_group.currentText() if self.is_group_mode() else ""
        bins = self.spn_bins.value()
        pal = self._palette()
        if group_col:
            df = self.get_df()
            y = ys[0]
            temp = df[[y, group_col]].copy()
            temp[y] = pd.to_numeric(temp[y], errors="coerce")
            temp = temp.dropna(subset=[y, group_col])
            for i, g in enumerate(pd.unique(temp[group_col])):
                vals = temp.loc[temp[group_col] == g, y]
                ax.hist(vals, bins=bins, alpha=0.55, label=str(g), color=pal[i % len(pal)], edgecolor="white")
            ax.legend(frameon=False, fontsize=8)
            ax.set_xlabel(y)
        else:
            for i, y in enumerate(ys):
                vals = self.get_numeric_series(y)
                ax.hist(vals, bins=bins, alpha=0.60 if len(ys) > 1 else 0.85, label=y, color=pal[i % len(pal)], edgecolor="white")
            if len(ys) > 1:
                ax.legend(frameon=False, fontsize=8)
            ax.set_xlabel(", ".join(ys))
        ax.set_ylabel("Frequency")

    def _plot_dotplot(self, ax):
        ys = self.active_y_columns()
        group_col = self.cmb_group.currentText() if self.is_group_mode() else ""
        pal = self._palette()
        if group_col:
            data = self.get_clean_y_group(ys[0], group_col)
            labels = []
            for i, g in enumerate(pd.unique(data[group_col]), start=1):
                vals = data.loc[data[group_col] == g, ys[0]].values
                jitter = np.random.uniform(-0.15, 0.15, len(vals)) if self.chk_jitter.isChecked() else 0
                ax.scatter(vals, np.full(len(vals), i) + jitter, color=pal[(i-1) % len(pal)], alpha=0.7)
                labels.append(str(g))
            ax.set_yticks(range(1, len(labels) + 1))
            ax.set_yticklabels(labels)
            ax.set_xlabel(ys[0])
        else:
            for i, y in enumerate(ys, start=1):
                vals = self.get_numeric_series(y).values
                jitter = np.random.uniform(-0.15, 0.15, len(vals)) if self.chk_jitter.isChecked() else 0
                ax.scatter(vals, np.full(len(vals), i) + jitter, color=pal[(i-1) % len(pal)], alpha=0.7)
            ax.set_yticks(range(1, len(ys) + 1))
            ax.set_yticklabels(ys)
            ax.set_xlabel("Value")

    def _plot_ecdf(self, ax):
        ys = self.active_y_columns()
        group_col = self.cmb_group.currentText() if self.is_group_mode() else ""
        pal = self._palette()
        if group_col:
            data = self.get_clean_y_group(ys[0], group_col)
            for i, g in enumerate(pd.unique(data[group_col])):
                vals = np.sort(data.loc[data[group_col] == g, ys[0]].values)
                ecdf = np.arange(1, len(vals) + 1) / len(vals)
                ax.step(vals, ecdf, where="post", label=str(g), color=pal[i % len(pal)])
        else:
            for i, y in enumerate(ys):
                vals = self.get_numeric_series(y).sort_values().values
                ecdf = np.arange(1, len(vals) + 1) / len(vals)
                ax.step(vals, ecdf, where="post", label=y, color=pal[i % len(pal)])
        ax.set_ylabel("Cumulative Probability")
        ax.set_ylim(0, 1.05)
        ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
        ax.legend(frameon=False, fontsize=8)

    def _plot_scatter(self, ax):
        x_col = self.cmb_x.currentText()
        y_col = self.cmb_y.currentText()
        data = self.get_clean_xy(x_col, y_col)
        ax.scatter(data["_x"], data[y_col], color=self._color(0), alpha=0.7, s=30, edgecolors="none")
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        if self.chk_trend.isChecked() and len(data) >= 2:
            x_num = pd.to_numeric(data["_x"], errors="coerce").values.astype(float)
            y_num = data[y_col].values.astype(float)
            coef = np.polyfit(x_num, y_num, 1)
            poly = np.poly1d(coef)
            xs = np.linspace(x_num.min(), x_num.max(), 200)
            ax.plot(xs, poly(xs), color=self._color(1), linestyle="--", linewidth=1.6, label="Regression")
            ax.legend(frameon=False, fontsize=8)

    def _plot_matrix(self, ax):
        numeric_cols = self.numeric_columns()[:6]
        df = self.get_df()[numeric_cols].apply(pd.to_numeric, errors="coerce").dropna()
        n = len(numeric_cols)
        self.figure.clear()
        pal = self._palette()
        for i, col_y in enumerate(numeric_cols):
            for j, col_x in enumerate(numeric_cols):
                sub = self.figure.add_subplot(n, n, i * n + j + 1)
                if i == j:
                    sub.hist(df[col_y], bins=12, color=pal[i % len(pal)], edgecolor="white", alpha=0.8)
                else:
                    sub.scatter(df[col_x], df[col_y], color=pal[0], alpha=0.45, s=8, edgecolors="none")
                sub.tick_params(labelsize=6)
                if i == n - 1:
                    sub.set_xlabel(col_x, fontsize=7)
                else:
                    sub.set_xticklabels([])
                if j == 0:
                    sub.set_ylabel(col_y, fontsize=7)
                else:
                    sub.set_yticklabels([])

    def _plot_bubble(self, ax):
        data = self.get_clean_xyz(self.cmb_x.currentText(), self.cmb_y.currentText(), self.cmb_z.currentText())
        z = data[self.cmb_z.currentText()].values
        sizes = 20 + 380 * ((z - z.min()) / max(z.max() - z.min(), 1e-9))
        sc = ax.scatter(data[self.cmb_x.currentText()], data[self.cmb_y.currentText()], s=sizes, c=z, cmap="viridis", alpha=0.75, edgecolors="white", linewidths=0.5)
        self.figure.colorbar(sc, ax=ax, fraction=0.046, pad=0.04).set_label(self.cmb_z.currentText(), fontsize=8)
        ax.set_xlabel(self.cmb_x.currentText())
        ax.set_ylabel(self.cmb_y.currentText())

    def _plot_boxplot(self, ax):
        ys = self.active_y_columns()
        group_col = self.cmb_group.currentText() if self.is_group_mode() else ""
        pal = self._palette()
        groups, labels, colors = [], [], []

        if group_col and len(ys) == 1:
            # One Y with groups: one box per group, legend title = group variable.
            y = ys[0]
            data = self.get_clean_y_group(y, group_col)
            for i, g in enumerate(pd.unique(data[group_col])):
                vals = data.loc[data[group_col] == g, y].dropna()
                if len(vals):
                    groups.append(vals)
                    labels.append(str(g))
                    colors.append(pal[i % len(pal)])
            ax.set_ylabel(y)
            self._add_group_legend(ax, group_col, labels, colors)

        elif group_col and len(ys) > 1:
            # Multiple Y's with groups: compact Minitab-style grouped boxes.
            df = self.get_df()
            tick_positions, tick_labels = [], []
            positions = []
            pos = 1
            group_values = list(pd.unique(df[group_col].dropna()))
            for yi, y in enumerate(ys):
                start_pos = pos
                for gi, g in enumerate(group_values):
                    temp = df.loc[df[group_col] == g, y]
                    vals = pd.to_numeric(temp, errors="coerce").dropna()
                    if len(vals):
                        groups.append(vals)
                        labels.append(str(g))
                        colors.append(pal[gi % len(pal)])
                        positions.append(pos)
                        pos += 1
                if pos > start_pos:
                    tick_positions.append((start_pos + pos - 1) / 2)
                    tick_labels.append(y)
                    pos += 1
            bp = ax.boxplot(groups, positions=positions, showmeans=self.chk_mean.isChecked(), patch_artist=True)
            for i, patch in enumerate(bp["boxes"]):
                legend_label = labels[i] if i < len(labels) else ""
                patch.set_facecolor(colors[i % len(colors)])
                patch.set_alpha(0.75)
                patch._sps_legend_label = str(legend_label)
                patch._sps_series_key = f"{group_col}:{legend_label}" if group_col and legend_label else str(legend_label)
            ax.set_xticks(tick_positions)
            ax.set_xticklabels(tick_labels, rotation=0, ha="center")
            ax.set_ylabel("Value")
            self._add_group_legend(ax, group_col, [str(g) for g in group_values], [pal[i % len(pal)] for i in range(len(group_values))])
            return

        else:
            # Multiple Y's Simple / One Y Simple: one box per Y variable.
            for i, y in enumerate(ys):
                vals = self.get_numeric_series(y)
                if len(vals):
                    groups.append(vals)
                    labels.append(y)
                    colors.append(pal[i % len(pal)])
            ax.set_ylabel("Value" if len(ys) > 1 else ys[0])

        bp = ax.boxplot(groups, tick_labels=labels, showmeans=self.chk_mean.isChecked(), patch_artist=True)
        for i, patch in enumerate(bp["boxes"]):
            legend_label = labels[i] if i < len(labels) else ""
            patch.set_facecolor(colors[i % len(colors)] if colors else pal[i % len(pal)])
            patch.set_alpha(0.75)
            patch._sps_legend_label = str(legend_label)
            patch._sps_series_key = f"{group_col}:{legend_label}" if group_col and legend_label else str(legend_label)
        ax.tick_params(axis="x", rotation=20)

    def _plot_interval(self, ax):
        ys = self.active_y_columns()
        group_col = self.cmb_group.currentText() if self.is_group_mode() else ""
        pal = self._palette()
        labels, means, cis = [], [], []
        if group_col:
            data = self.get_clean_y_group(ys[0], group_col)
            for g in pd.unique(data[group_col]):
                vals = data.loc[data[group_col] == g, ys[0]].dropna()
                if len(vals) >= 2:
                    labels.append(str(g)); means.append(vals.mean()); cis.append(1.96 * vals.sem())
        else:
            for y in ys:
                vals = self.get_numeric_series(y)
                if len(vals) >= 2:
                    labels.append(y); means.append(vals.mean()); cis.append(1.96 * vals.sem())
        for i, (m, ci) in enumerate(zip(means, cis), start=1):
            ax.errorbar(i, m, yerr=ci, fmt="o", color=pal[(i-1) % len(pal)], capsize=5, markersize=8, linewidth=1.8)
        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_ylabel("Mean ± 95% CI")

    def _plot_individual_value(self, ax):
        ys = self.active_y_columns()
        group_col = self.cmb_group.currentText() if self.is_group_mode() else ""
        pal = self._palette()
        labels = []
        if group_col:
            data = self.get_clean_y_group(ys[0], group_col)
            for i, g in enumerate(pd.unique(data[group_col]), start=1):
                vals = data.loc[data[group_col] == g, ys[0]].dropna().values
                jitter = np.random.uniform(-0.15, 0.15, len(vals)) if self.chk_jitter.isChecked() else 0
                ax.scatter(np.full(len(vals), i) + jitter, vals, color=pal[(i-1) % len(pal)], alpha=0.7, s=22, edgecolors="none")
                ax.hlines(np.nanmean(vals), i - 0.3, i + 0.3, colors=pal[(i-1) % len(pal)], linewidth=2)
                labels.append(str(g))
        else:
            for i, y in enumerate(ys, start=1):
                vals = self.get_numeric_series(y).values
                jitter = np.random.uniform(-0.15, 0.15, len(vals)) if self.chk_jitter.isChecked() else 0
                ax.scatter(np.full(len(vals), i) + jitter, vals, color=pal[(i-1) % len(pal)], alpha=0.7, s=22, edgecolors="none")
                if len(vals):
                    ax.hlines(np.nanmean(vals), i - 0.3, i + 0.3, colors=pal[(i-1) % len(pal)], linewidth=2)
                labels.append(y)
        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_ylabel("Value")

    def _plot_line(self, ax, area=False, time_series=False):
        ys = self.active_y_columns()
        x_col = self.cmb_x.currentText()
        group_col = self.cmb_group.currentText() if self.is_group_mode() else ""
        pal = self._palette()
        df = self.get_df().copy()

        for yi, y in enumerate(ys):
            if group_col:
                temp = df[[y, group_col]].copy()
                temp[y] = pd.to_numeric(temp[y], errors="coerce")
                if x_col:
                    temp[x_col] = df[x_col]
                temp["_row"] = np.arange(1, len(df) + 1)
                temp = temp.dropna(subset=[y, group_col])

                for gi, g in enumerate(pd.unique(temp[group_col])):
                    sub = temp[temp[group_col] == g].copy()
                    x = self._series_x_values(sub, x_col, time_series)
                    label = f"{g}" if len(ys) == 1 else f"{y} - {g}"
                    color = pal[gi % len(pal)] if len(ys) == 1 else pal[(yi + gi) % len(pal)]
                    ax.plot(x, sub[y].values, marker="o", markersize=4, linewidth=1.6, color=color, label=label)
                    if area:
                        ax.fill_between(x, sub[y].values, color=color, alpha=0.18)
            else:
                temp = df[[y]].copy()
                temp[y] = pd.to_numeric(temp[y], errors="coerce")
                if x_col:
                    temp[x_col] = df[x_col]
                temp["_row"] = np.arange(1, len(df) + 1)
                temp = temp.dropna(subset=[y])
                x = self._series_x_values(temp, x_col, time_series)
                ax.plot(x, temp[y].values, marker="o", markersize=4, linewidth=1.6, color=pal[yi % len(pal)], label=y)
                if area:
                    ax.fill_between(x, temp[y].values, color=pal[yi % len(pal)], alpha=0.18)

        if self.chk_trend.isChecked() and len(ys) == 1 and not group_col:
            temp = df[[ys[0]]].copy()
            temp[ys[0]] = pd.to_numeric(temp[ys[0]], errors="coerce")
            temp = temp.dropna(subset=[ys[0]])
            if len(temp) >= 2:
                x_num = np.arange(len(temp), dtype=float)
                coef = np.polyfit(x_num, temp[ys[0]].values, 1)
                poly = np.poly1d(coef)
                x_plot = self._series_x_values(temp.assign(_row=np.arange(1, len(temp)+1)), x_col, time_series)
                ax.plot(x_plot, poly(x_num), color=self._color(1), linestyle="--", linewidth=1.4, label="Trend")

        ax.set_xlabel(x_col if x_col else "Index")
        ax.set_ylabel("Value" if len(ys) > 1 else ys[0])
        if group_col:
            ax.legend(title=group_col, frameon=False, fontsize=8, title_fontsize=8)
        elif len(ys) > 1 or self.chk_trend.isChecked():
            ax.legend(frameon=False, fontsize=8)
        if time_series and x_col:
            self.figure.autofmt_xdate()

    def _series_x_values(self, data, x_col, time_series):
        if x_col and x_col in data.columns:
            if time_series:
                parsed = pd.to_datetime(data[x_col], errors="coerce")
                if parsed.notna().sum() > 0:
                    return parsed
            numeric = pd.to_numeric(data[x_col], errors="coerce")
            if numeric.notna().sum() > 0:
                return numeric
        # Default X uses original raw row order. Groups do not restart at 1.
        if "_row" in data.columns:
            return data["_row"]
        return np.arange(1, len(data) + 1)

    def _plot_multi_vari(self, ax):
        y_col = self.cmb_y.currentText()
        x_col = self.cmb_x.currentText()
        group_col = self.cmb_group.currentText() if self.is_group_mode() else ""
        df = self.get_df()
        pal = self._palette()

        required = [y_col, x_col] + ([group_col] if group_col else [])
        data = df[required].copy()
        data[y_col] = pd.to_numeric(data[y_col], errors="coerce")
        data = data.dropna(subset=[y_col, x_col])

        if data.empty:
            raise ValueError("Multi-Vari Chart requires valid Y and categorical X data.")

        x_order = list(pd.unique(data[x_col].astype(str)))
        x_positions = {label: idx + 1 for idx, label in enumerate(x_order)}

        if group_col:
            for gi, (group_name, sub) in enumerate(data.groupby(group_col, sort=False)):
                means = sub.groupby(x_col, sort=False)[y_col].mean().reindex(x_order)
                xs = [x_positions[str(lbl)] for lbl in x_order if not pd.isna(means.loc[lbl])]
                ys = [means.loc[lbl] for lbl in x_order if not pd.isna(means.loc[lbl])]
                color = pal[gi % len(pal)]
                line, = ax.plot(xs, ys, marker="o", linewidth=1.8, color=color, label=str(group_name))
                line._sps_legend_label = str(group_name)
                line._sps_series_key = f"{group_col}:{group_name}"

                if self.chk_jitter.isChecked():
                    for label in x_order:
                        vals = sub.loc[sub[x_col].astype(str) == str(label), y_col].dropna().values
                        if len(vals):
                            jitter = np.random.uniform(-0.08, 0.08, len(vals))
                            scatter = ax.scatter(
                                np.full(len(vals), x_positions[str(label)]) + jitter,
                                vals,
                                s=18,
                                alpha=0.35,
                                color=color,
                                edgecolors="none",
                            )
                            scatter._sps_legend_label = str(group_name)
                            scatter._sps_series_key = f"{group_col}:{group_name}"
            ax.legend(title=group_col, frameon=False, fontsize=8, title_fontsize=8)
        else:
            means = data.groupby(x_col, sort=False)[y_col].mean().reindex(x_order)
            xs = [x_positions[str(lbl)] for lbl in x_order if not pd.isna(means.loc[lbl])]
            ys = [means.loc[lbl] for lbl in x_order if not pd.isna(means.loc[lbl])]
            line, = ax.plot(xs, ys, marker="o", linewidth=1.8, color=pal[0], label="Mean")
            line._sps_legend_label = "Mean"
            line._sps_series_key = "Mean"
            if self.chk_jitter.isChecked():
                for label in x_order:
                    vals = data.loc[data[x_col].astype(str) == str(label), y_col].dropna().values
                    jitter = np.random.uniform(-0.08, 0.08, len(vals))
                    scatter = ax.scatter(
                        np.full(len(vals), x_positions[str(label)]) + jitter,
                        vals,
                        s=18,
                        alpha=0.40,
                        color=pal[1 % len(pal)],
                        edgecolors="none",
                    )
                    scatter._sps_legend_label = "Individual Data"
                    scatter._sps_series_key = "Individual Data"
            ax.legend(frameon=False, fontsize=8)

        ax.set_xticks(range(1, len(x_order) + 1))
        ax.set_xticklabels(x_order, rotation=20, ha="right")
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)

    def _plot_variability(self, ax):
        y_col = self.cmb_y.currentText()
        x_col = self.cmb_x.currentText()
        group_col = self.cmb_group.currentText() if self.is_group_mode() else ""
        df = self.get_df()
        pal = self._palette()

        required = [y_col, x_col] + ([group_col] if group_col else [])
        data = df[required].copy()
        data[y_col] = pd.to_numeric(data[y_col], errors="coerce")
        data = data.dropna(subset=[y_col, x_col])

        if data.empty:
            raise ValueError("Variability Chart requires valid Y and categorical X data.")

        labels = []
        groups = []
        colors = []

        if group_col:
            combined = data[x_col].astype(str) + "\n" + data[group_col].astype(str)
            data = data.assign(_combined=combined)
            for i, label in enumerate(pd.unique(data["_combined"])):
                vals = data.loc[data["_combined"] == label, y_col].dropna()
                if len(vals):
                    labels.append(str(label))
                    groups.append(vals)
                    colors.append(pal[i % len(pal)])
        else:
            for i, label in enumerate(pd.unique(data[x_col].astype(str))):
                vals = data.loc[data[x_col].astype(str) == str(label), y_col].dropna()
                if len(vals):
                    labels.append(str(label))
                    groups.append(vals)
                    colors.append(pal[i % len(pal)])

        positions = np.arange(1, len(groups) + 1)
        bp = ax.boxplot(groups, positions=positions, patch_artist=True, showmeans=self.chk_mean.isChecked())
        for i, patch in enumerate(bp["boxes"]):
            legend_label = labels[i] if i < len(labels) else ""
            patch.set_facecolor(colors[i % len(colors)])
            patch.set_alpha(0.55)
            patch._sps_legend_label = str(legend_label)
            patch._sps_series_key = f"Variability:{legend_label}" if legend_label else ""

        for i, vals in enumerate(groups, start=1):
            arr = np.asarray(vals, dtype=float)
            if len(arr):
                jitter = np.random.uniform(-0.10, 0.10, len(arr)) if self.chk_jitter.isChecked() else 0
                scatter = ax.scatter(
                    np.full(len(arr), i) + jitter,
                    arr,
                    s=18,
                    alpha=0.55,
                    color=colors[(i - 1) % len(colors)],
                    edgecolors="none",
                )
                legend_label = labels[i - 1] if (i - 1) < len(labels) else ""
                scatter._sps_legend_label = str(legend_label)
                scatter._sps_series_key = f"Variability:{legend_label}" if legend_label else ""
                ax.hlines(np.nanmean(arr), i - 0.28, i + 0.28, colors="black", linewidth=1.5)

        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.set_xlabel(x_col if not group_col else f"{x_col} / {group_col}")
        ax.set_ylabel(y_col)

    def _plot_symmetry(self, ax):
        y_col = self.cmb_y.currentText()
        vals = self.get_numeric_series(y_col).dropna().sort_values().to_numpy(dtype=float)

        if len(vals) < 4:
            raise ValueError("Symmetry Plot requires at least 4 valid numeric values.")

        median = float(np.median(vals))
        n_pairs = len(vals) // 2
        lower = vals[:n_pairs]
        upper = vals[-n_pairs:][::-1]

        lower_dist = median - lower
        upper_dist = upper - median

        ax.scatter(lower_dist, upper_dist, color=self._color(0), alpha=0.75, s=30, edgecolors="none")
        max_range = float(np.nanmax([lower_dist.max(), upper_dist.max()])) if n_pairs else 1.0
        ax.plot([0, max_range], [0, max_range], linestyle="--", color=self._color(1), linewidth=1.4, label="Perfect symmetry")
        ax.set_xlabel("Distance below median")
        ax.set_ylabel("Distance above median")
        ax.legend(frameon=False, fontsize=8)
        ax.set_aspect("equal", adjustable="box")

    def _plot_bar(self, ax):
        ys = self.active_y_columns()
        group_col = self.cmb_group.currentText() if self.is_group_mode() else ""
        pal = self._palette()
        if group_col:
            data = self.get_clean_y_group(ys[0], group_col)
            grouped = data.groupby(group_col, sort=False)[ys[0]].mean()
            bars = ax.bar(range(1, len(grouped) + 1), grouped.values, color=[pal[i % len(pal)] for i in range(len(grouped))], edgecolor="white")
            for i, bar in enumerate(bars):
                legend_label = str(grouped.index[i])
                bar._sps_legend_label = legend_label
                bar._sps_series_key = f"{group_col}:{legend_label}"
            ax.set_xticks(range(1, len(grouped) + 1))
            ax.set_xticklabels([str(v) for v in grouped.index], rotation=20, ha="right")
            ax.set_xlabel(group_col)
            ax.set_ylabel(f"Mean of {ys[0]}")
        else:
            means = [self.get_numeric_series(y).mean() for y in ys]
            bars = ax.bar(range(1, len(ys) + 1), means, color=[pal[i % len(pal)] for i in range(len(ys))], edgecolor="white")
            for i, bar in enumerate(bars):
                legend_label = str(ys[i]) if i < len(ys) else ""
                bar._sps_legend_label = legend_label
                bar._sps_series_key = f"Bar:{legend_label}" if legend_label else ""
            ax.set_xticks(range(1, len(ys) + 1))
            ax.set_xticklabels(ys, rotation=20, ha="right")
            ax.set_ylabel("Mean")

    def _plot_pie(self, ax):
        df = self.get_df()
        group_col = self.cmb_group.currentText()
        counts = df[group_col].astype(str).value_counts()
        pal = self._palette()
        ax.pie(counts.values, labels=counts.index, autopct="%1.1f%%", colors=[pal[i % len(pal)] for i in range(len(counts))], wedgeprops={"edgecolor": "white", "linewidth": 1.2}, startangle=140)
        ax.axis("equal")

    def _plot_contour(self, ax):
        x_col = self.cmb_x.currentText()
        y_col = self.cmb_y.currentText()
        z_col = self.cmb_z.currentText()

        data = self.get_clean_xyz(x_col, y_col, z_col)

        if len(data) < 6:
            raise ValueError("Contour Plot needs at least 6 valid data points.")

        if griddata is None:
            raise RuntimeError("Contour Plot requires scipy.")

        x = data[x_col].to_numpy(dtype=float)
        y = data[y_col].to_numpy(dtype=float)
        z = data[z_col].to_numpy(dtype=float)

        if np.unique(x).size < 3 or np.unique(y).size < 3:
            raise ValueError("Contour Plot needs at least 3 unique X values and 3 unique Y values.")

        method = self.cmb_contour_method.currentText()

        grid_size = 180
        xi = np.linspace(np.nanmin(x), np.nanmax(x), grid_size)
        yi = np.linspace(np.nanmin(y), np.nanmax(y), grid_size)
        Xi, Yi = np.meshgrid(xi, yi)

        if method == "Distance method":
            power = float(self.spn_distance_power.value())
            standardize = self.chk_standardize_xy.isChecked()
            Zi = self._interpolate_idw(x, y, z, Xi, Yi, power=power, standardize=standardize)

        elif method == "Akima's polynomial method":
            Zi = self._interpolate_akima_like(x, y, z, Xi, Yi)

            boundary_text = self.txt_boundary_z.text().strip()
            if boundary_text:
                try:
                    boundary_z = float(boundary_text)
                    Zi = np.where(np.isnan(Zi), boundary_z, Zi)
                except ValueError:
                    raise ValueError("Boundary z-value must be numeric.")

            if np.isnan(Zi).any():
                Zi_nearest = griddata((x, y), z, (Xi, Yi), method="nearest")
                Zi = np.where(np.isnan(Zi), Zi_nearest, Zi)

        else:
            Zi = griddata((x, y), z, (Xi, Yi), method="cubic")
            if np.isnan(Zi).all():
                Zi = griddata((x, y), z, (Xi, Yi), method="linear")
            Zi_nearest = griddata((x, y), z, (Xi, Yi), method="nearest")
            Zi = np.where(np.isnan(Zi), Zi_nearest, Zi)

        z_min = np.nanmin(z)
        z_max = np.nanmax(z)

        if z_min == z_max:
            raise ValueError("Contour Plot requires variation in Z values.")

        levels = np.linspace(z_min, z_max, 9)
        cmap = "YlGnBu_r"

        cs = ax.contourf(
            Xi,
            Yi,
            Zi,
            levels=levels,
            cmap=cmap,
            extend="both",
            antialiased=True,
        )

        ax.contour(
            Xi,
            Yi,
            Zi,
            levels=levels,
            colors="white",
            linewidths=0.45,
            alpha=0.65,
        )

        cbar = self.figure.colorbar(
            cs,
            ax=ax,
            fraction=0.046,
            pad=0.04,
        )
        cbar.set_label(z_col, fontsize=8)
        cbar.ax.tick_params(labelsize=8)

        ax.scatter(
            x,
            y,
            c="white",
            s=12,
            alpha=0.75,
            edgecolors="black",
            linewidths=0.35,
            zorder=5,
        )

        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)


    def _interpolate_idw(self, x, y, z, Xi, Yi, power=2.0, standardize=True):
        x_work = x.astype(float).copy()
        y_work = y.astype(float).copy()
        xi_work = Xi.astype(float).copy()
        yi_work = Yi.astype(float).copy()

        if standardize:
            x_mean = np.nanmean(x_work)
            x_std = np.nanstd(x_work) or 1.0
            y_mean = np.nanmean(y_work)
            y_std = np.nanstd(y_work) or 1.0

            x_work = (x_work - x_mean) / x_std
            y_work = (y_work - y_mean) / y_std
            xi_work = (xi_work - x_mean) / x_std
            yi_work = (yi_work - y_mean) / y_std

        Zi = np.empty_like(xi_work, dtype=float)

        points = np.column_stack([x_work, y_work])

        for idx in np.ndindex(xi_work.shape):
            gx = xi_work[idx]
            gy = yi_work[idx]

            dist = np.sqrt((points[:, 0] - gx) ** 2 + (points[:, 1] - gy) ** 2)

            if np.any(dist == 0):
                Zi[idx] = z[np.argmin(dist)]
            else:
                weights = 1.0 / np.power(dist, power)
                Zi[idx] = np.sum(weights * z) / np.sum(weights)

        return Zi


    def _interpolate_akima_like(self, x, y, z, Xi, Yi):
        if CloughTocher2DInterpolator is None:
            raise RuntimeError("Akima-like polynomial interpolation requires scipy.")

        points = np.column_stack([x, y])
        interpolator = CloughTocher2DInterpolator(points, z)

        Zi = interpolator(Xi, Yi)
        return Zi


    def _plot_3d_scatter(self, ax):
        if not HAS_3D:
            raise RuntimeError("mpl_toolkits.mplot3d is not available.")
        data = self.get_clean_xyz(self.cmb_x.currentText(), self.cmb_y.currentText(), self.cmb_z.currentText())
        ax.scatter(data[self.cmb_x.currentText()], data[self.cmb_y.currentText()], data[self.cmb_z.currentText()], c=data[self.cmb_z.currentText()], cmap="viridis", alpha=0.75, s=30, edgecolors="none")
        ax.set_xlabel(self.cmb_x.currentText(), fontsize=8)
        ax.set_ylabel(self.cmb_y.currentText(), fontsize=8)
        ax.set_zlabel(self.cmb_z.currentText(), fontsize=8)

    def _plot_3d_surface(self, ax):
        if not HAS_3D or griddata is None:
            raise RuntimeError("3D Surface Plot requires scipy.interpolate.griddata.")
        data = self.get_clean_xyz(self.cmb_x.currentText(), self.cmb_y.currentText(), self.cmb_z.currentText())
        if len(data) < 9:
            raise ValueError("3D Surface Plot needs at least 9 data points to interpolate.")
        x, y, z = data[self.cmb_x.currentText()].values, data[self.cmb_y.currentText()].values, data[self.cmb_z.currentText()].values
        n_grid = min(30, int(math.sqrt(len(data)) * 2))
        xi, yi = np.linspace(x.min(), x.max(), n_grid), np.linspace(y.min(), y.max(), n_grid)
        Xi, Yi = np.meshgrid(xi, yi)
        Zi = griddata((x, y), z, (Xi, Yi), method="cubic")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            surf = ax.plot_surface(Xi, Yi, Zi, cmap="viridis", alpha=0.85, linewidth=0, antialiased=True)
        self.figure.colorbar(surf, ax=ax, fraction=0.03, pad=0.08, shrink=0.6).set_label(self.cmb_z.currentText())
        ax.set_xlabel(self.cmb_x.currentText(), fontsize=8)
        ax.set_ylabel(self.cmb_y.currentText(), fontsize=8)
        ax.set_zlabel(self.cmb_z.currentText(), fontsize=8)

    # ─────────────────────────────────────────────
    # Graph label editing / legends
    # ─────────────────────────────────────────────

    def _add_group_legend(self, ax, group_name, labels, colors):
        if not labels:
            return
        handles = []
        for i, label in enumerate(labels):
            handle = Patch(facecolor=colors[i % len(colors)], edgecolor="black", alpha=0.75, label=str(label))
            handle._sps_legend_label = str(label)
            handle._sps_series_key = f"{group_name}:{label}"
            handles.append(handle)
        ax.legend(handles=handles, title=group_name, frameon=False, fontsize=8, title_fontsize=8)

    def _apply_axis_label_overrides(self, ax):
        if self._axis_label_overrides.get("x"):
            ax.set_xlabel(self._axis_label_overrides["x"])
        if self._axis_label_overrides.get("y"):
            ax.set_ylabel(self._axis_label_overrides["y"])

    def _on_graph_button_press(self, event):
        if getattr(event, "dblclick", False):
            self._handle_graph_double_click(event)
            return
        self._on_reference_press(event)

    def _handle_graph_double_click(self, event):
        if not self.figure.axes:
            return

        try:
            renderer = self.canvas.get_renderer()
        except Exception:
            self.canvas.draw()
            renderer = self.canvas.get_renderer()

        px = event.x
        py = event.y

        for ax in self.figure.axes:
            try:
                box = ax.title.get_window_extent(renderer=renderer).expanded(1.2, 1.8)
                if box.contains(px, py):
                    self._edit_graph_title(ax)
                    return
            except Exception:
                pass

            try:
                box = ax.xaxis.label.get_window_extent(renderer=renderer).expanded(1.4, 2.2)
                if box.contains(px, py):
                    self._edit_axis_label(ax, "x")
                    return
            except Exception:
                pass

            try:
                box = ax.yaxis.label.get_window_extent(renderer=renderer).expanded(1.8, 1.4)
                if box.contains(px, py):
                    self._edit_axis_label(ax, "y")
                    return
            except Exception:
                pass

            for tick in ax.get_xticklabels():
                try:
                    box = tick.get_window_extent(renderer=renderer).expanded(1.5, 2.0)
                    if box.contains(px, py):
                        self._edit_axis_scale("x")
                        return
                except Exception:
                    pass

            for tick in ax.get_yticklabels():
                try:
                    box = tick.get_window_extent(renderer=renderer).expanded(2.0, 1.5)
                    if box.contains(px, py):
                        self._edit_axis_scale("y")
                        return
                except Exception:
                    pass


    def _apply_axis_label_overrides(self, ax):
        if self._axis_label_overrides.get("x"):
            ax.set_xlabel(self._axis_label_overrides["x"])
        if self._axis_label_overrides.get("y"):
            ax.set_ylabel(self._axis_label_overrides["y"])


    def _on_graph_button_press(self, event):
        if getattr(event, "dblclick", False):
            self._handle_graph_double_click(event)
            return

        self._on_reference_press(event)


    def _handle_graph_double_click(self, event):
        if not self.figure.axes:
            return

        try:
            renderer = self.canvas.get_renderer()
        except Exception:
            self.canvas.draw()
            renderer = self.canvas.get_renderer()

        px = event.x
        py = event.y

        for ax in self.figure.axes:
            # 1) Title
            try:
                box = ax.title.get_window_extent(renderer=renderer).expanded(1.2, 1.8)
                if box.contains(px, py):
                    self._edit_graph_title(ax)
                    return
            except Exception:
                pass

            # 2) X axis label
            try:
                box = ax.xaxis.label.get_window_extent(renderer=renderer).expanded(1.4, 2.2)
                if box.contains(px, py):
                    self._edit_axis_label(ax, "x")
                    return
            except Exception:
                pass

            # 3) Y axis label
            try:
                box = ax.yaxis.label.get_window_extent(renderer=renderer).expanded(1.8, 1.4)
                if box.contains(px, py):
                    self._edit_axis_label(ax, "y")
                    return
            except Exception:
                pass

            # 4) X tick labels / scale values
            for tick in ax.get_xticklabels():
                try:
                    if not tick.get_visible():
                        continue

                    text = tick.get_text()
                    if text is None or str(text).strip() == "":
                        continue

                    box = tick.get_window_extent(renderer=renderer).expanded(1.8, 2.6)
                    if box.contains(px, py):
                        self._edit_axis_scale("x")
                        return
                except Exception:
                    pass

            # 5) Y tick labels / scale values
            for tick in ax.get_yticklabels():
                try:
                    if not tick.get_visible():
                        continue

                    text = tick.get_text()
                    if text is None or str(text).strip() == "":
                        continue

                    box = tick.get_window_extent(renderer=renderer).expanded(2.6, 1.8)
                    if box.contains(px, py):
                        self._edit_axis_scale("y")
                        return
                except Exception:
                    pass


    def _edit_graph_title(self, ax):
        current = ax.get_title() or self.cmb_title.currentText()
        text, ok = QInputDialog.getText(self, "Edit Graph Title", "Title:", text=current)

        if not ok:
            return

        new_text = text.strip()
        self.cmb_title.setEditText(new_text)
        self._last_auto_title = new_text

        ax.set_title(new_text, fontweight="bold", pad=10)
        self.canvas.draw_idle()
        self._refresh_summary_after_reference_change()


    def _edit_axis_label(self, ax, axis):
        if axis == "x":
            current = ax.get_xlabel()
            title = "Edit X Axis Label"
            prompt = "X axis label:"
        else:
            current = ax.get_ylabel()
            title = "Edit Y Axis Label"
            prompt = "Y axis label:"

        text, ok = QInputDialog.getText(self, title, prompt, text=current)

        if not ok:
            return

        new_text = text.strip()
        self._axis_label_overrides[axis] = new_text

        if axis == "x":
            ax.set_xlabel(new_text)
        else:
            ax.set_ylabel(new_text)

        self.canvas.draw_idle()
        self._refresh_summary_after_reference_change()


    def _parse_axis_limit_value(self, axis, text):
        text = str(text).strip()

        if not text or text.lower() in {"auto", "none", "na", "n/a"}:
            return None

        try:
            return float(text)
        except ValueError:
            pass

        if axis == "x" and self.cmb_graph.currentText() == "Time Series Plot":
            try:
                dt = pd.to_datetime(text, errors="raise")
                return mdates.date2num(dt.to_pydatetime())
            except Exception:
                pass

        raise ValueError(f"Invalid limit value: {text}")


    def _edit_axis_scale(self, axis):
        if not self.figure.axes:
            return

        ax = self.figure.axes[0]
        axis_name = "X" if axis == "x" else "Y"

        current_axis_limits = ax.get_xlim() if axis == "x" else ax.get_ylim()
        saved_min, saved_max = self._axis_limits.get(axis, [None, None])

        auto_min = saved_min is None
        auto_max = saved_max is None

        dialog = AxisScaleDialog(
            parent=self,
            axis_name=axis_name,
            current_min=current_axis_limits[0] if auto_min else saved_min,
            current_max=current_axis_limits[1] if auto_max else saved_max,
            current_auto_min=auto_min,
            current_auto_max=auto_max,
        )

        if dialog.exec() != QDialog.Accepted:
            return

        values = dialog.values()

        try:
            lower = None if values["auto_min"] else self._parse_axis_limit_value(axis, values["min_text"])
            upper = None if values["auto_max"] else self._parse_axis_limit_value(axis, values["max_text"])
        except ValueError as exc:
            QMessageBox.warning(self, "Invalid scale", str(exc))
            return

        if lower is not None and upper is not None and lower >= upper:
            QMessageBox.warning(self, "Invalid scale", "Minimum must be lower than maximum.")
            return

        self._axis_limits[axis] = [lower, upper]
        self._apply_axis_limits(ax)
        self.canvas.draw_idle()
        self._refresh_summary_after_reference_change()


    def _apply_axis_limits(self, ax):
        if ax is None:
            return

        x_lower, x_upper = self._axis_limits.get("x", [None, None])
        y_lower, y_upper = self._axis_limits.get("y", [None, None])

        if x_lower is not None or x_upper is not None:
            current_x = ax.get_xlim()
            left = x_lower if x_lower is not None else current_x[0]
            right = x_upper if x_upper is not None else current_x[1]
            if left < right:
                ax.set_xlim(left, right)

        if y_lower is not None or y_upper is not None:
            current_y = ax.get_ylim()
            bottom = y_lower if y_lower is not None else current_y[0]
            top = y_upper if y_upper is not None else current_y[1]
            if bottom < top:
                ax.set_ylim(bottom, top)


    def _reset_axis_scales(self):
        self._axis_limits = {
            "x": [None, None],
            "y": [None, None],
        }

        if self.figure.axes:
            ax = self.figure.axes[0]
            ax.autoscale(enable=True, axis="both")

        self.canvas.draw_idle()
        self._refresh_summary_after_reference_change()


    # ─────────────────────────────────────────────
    # Reference lines
    # ─────────────────────────────────────────────

    def _show_graph_context_menu(self, pos):
        """Right-click menu. If the click is near an existing reference line,
        show the line-specific editor; otherwise show the general reference menu.
        """
        if not self.figure.axes:
            return

        item = self._reference_item_at_pos(pos)
        if item:
            self._show_reference_line_menu(pos, item)
            return

        self._show_general_graph_menu(pos)

    def _show_general_graph_menu(self, pos):
        menu = QMenu(self)

        scale_x = menu.addAction("Set X axis scale...")
        scale_y = menu.addAction("Set Y axis scale...")
        reset_scale = menu.addAction("Reset axis scales")

        menu.addSeparator()

        add_x = menu.addAction("Add X reference line(s) — Vertical")
        add_y = menu.addAction("Add Y reference line(s) — Horizontal")

        menu.addSeparator()

        rem_x = menu.addAction("Remove X reference line(s) by value")
        rem_y = menu.addAction("Remove Y reference line(s) by value")

        menu.addSeparator()

        clear_x = menu.addAction("Clear X reference lines")
        clear_y = menu.addAction("Clear Y reference lines")
        clear_all = menu.addAction("Clear all reference lines")

        action = menu.exec(self.canvas.mapToGlobal(pos))

        if action == scale_x:
            self._edit_axis_scale("x")
        elif action == scale_y:
            self._edit_axis_scale("y")
        elif action == reset_scale:
            self._reset_axis_scales()
        elif action == add_x:
            self._add_reference_lines("x")
        elif action == add_y:
            self._add_reference_lines("y")
        elif action == rem_x:
            self._remove_reference_lines_by_value("x")
        elif action == rem_y:
            self._remove_reference_lines_by_value("y")
        elif action == clear_x:
            self._clear_reference_lines("x")
        elif action == clear_y:
            self._clear_reference_lines("y")
        elif action == clear_all:
            self._clear_reference_lines("all")

    def _show_reference_line_menu(self, pos, item):
        axis = item.get("axis", "x")
        current_value = self._format_reference_value(axis, item.get("value"))

        menu = QMenu(self)
        edit_value = menu.addAction(f"Edit value ({axis.upper()} = {current_value})")
        change_color = menu.addAction("Change color...")

        width_menu = menu.addMenu("Line width")
        width_1 = width_menu.addAction("Thin")
        width_2 = width_menu.addAction("Normal")
        width_3 = width_menu.addAction("Thick")

        style_menu = menu.addMenu("Line style")
        solid = style_menu.addAction("Solid")
        dashed = style_menu.addAction("Dashed")
        dashdot = style_menu.addAction("Dash-dot")
        dotted = style_menu.addAction("Dotted")

        menu.addSeparator()
        delete_line = menu.addAction("Delete this reference line")

        action = menu.exec(self.canvas.mapToGlobal(pos))

        if action == edit_value:
            self._edit_reference_line_value(item)
        elif action == change_color:
            self._edit_reference_line_color(item)
        elif action == width_1:
            item["linewidth"] = 1.0
            self._redraw_reference_lines()
            self._refresh_summary_after_reference_change()
        elif action == width_2:
            item["linewidth"] = 1.4
            self._redraw_reference_lines()
            self._refresh_summary_after_reference_change()
        elif action == width_3:
            item["linewidth"] = 2.2
            self._redraw_reference_lines()
            self._refresh_summary_after_reference_change()
        elif action == solid:
            item["linestyle"] = "-"
            self._redraw_reference_lines()
            self._refresh_summary_after_reference_change()
        elif action == dashed:
            item["linestyle"] = "--"
            self._redraw_reference_lines()
            self._refresh_summary_after_reference_change()
        elif action == dashdot:
            item["linestyle"] = "-."
            self._redraw_reference_lines()
            self._refresh_summary_after_reference_change()
        elif action == dotted:
            item["linestyle"] = ":"
            self._redraw_reference_lines()
            self._refresh_summary_after_reference_change()
        elif action == delete_line:
            self._delete_reference_line_item(item)
            self._refresh_summary_after_reference_change()

    def _reference_item_at_pos(self, pos):
        """Detect whether a Qt right-click position is close to a reference line.
        Uses pixel distance, not data value comparison, so it works after dragging.
        """
        if not self.figure.axes:
            return None

        ax = self.figure.axes[0]
        mouse_x = pos.x()
        mouse_y = self.canvas.height() - pos.y()
        tolerance_px = 10

        for item in self.reference_lines["x"]:
            value = item.get("value")
            try:
                px = ax.transData.transform((float(value), ax.get_ylim()[0]))[0]
                if abs(mouse_x - px) <= tolerance_px:
                    return item
            except Exception:
                pass

        for item in self.reference_lines["y"]:
            value = item.get("value")
            try:
                py = ax.transData.transform((ax.get_xlim()[0], float(value)))[1]
                if abs(mouse_y - py) <= tolerance_px:
                    return item
            except Exception:
                pass

        return None

    def _make_reference_line_item(self, axis, value, color=None, linestyle=None, linewidth=None):
        return {
            "axis": axis,
            "value": value,
            "color": color or REFERENCE_LINE_COLOR,
            "linestyle": linestyle or "--",
            "linewidth": float(linewidth if linewidth is not None else 1.4),
        }

    def _add_reference_lines(self, axis):
        text, ok = QInputDialog.getText(
            self,
            f"Add {axis.upper()} Reference Lines",
            "Enter one or more values separated by spaces:",
        )
        if not ok or not text.strip():
            return

        values, invalid = self._parse_reference_values(axis, text)
        if invalid:
            QMessageBox.warning(self, "Invalid values", f"Ignored: {', '.join(invalid)}")

        for value in values:
            self.reference_lines[axis].append(self._make_reference_line_item(axis, value))

        self._redraw_reference_lines()
        self._refresh_summary_after_reference_change()

    def _parse_reference_values(self, axis, text):
        values, invalid = [], []
        allow_date_x = axis == "x" and self.cmb_graph.currentText() == "Time Series Plot"

        for raw in text.replace(",", " ").split():
            parsed = None
            try:
                parsed = float(raw)
            except ValueError:
                if allow_date_x:
                    try:
                        parsed = mdates.date2num(pd.to_datetime(raw).to_pydatetime())
                    except Exception:
                        pass
            if parsed is None:
                invalid.append(raw)
            else:
                values.append(parsed)

        return values, invalid

    def _draw_reference_lines(self, ax):
        self._ref_artists = []
        self._ref_artist_map = {}

        for item in self.reference_lines["x"]:
            value = item.get("value")
            color = item.get("color", REFERENCE_LINE_COLOR)
            linestyle = item.get("linestyle", "--")
            linewidth = item.get("linewidth", 1.4)

            line = ax.axvline(
                value,
                color=color,
                linestyle=linestyle,
                linewidth=linewidth,
                alpha=0.95,
                picker=8,
                zorder=10,
            )
            text = ax.text(
                value,
                0.98,
                self._format_reference_value("x", value),
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=8,
                color=color,
                bbox={
                    "facecolor": "white",
                    "edgecolor": color,
                    "boxstyle": "round,pad=0.2",
                    "alpha": 0.85,
                },
                zorder=11,
            )
            item["line"] = line
            item["text"] = text
            self._ref_artists += [line, text]
            self._ref_artist_map[line] = item
            self._ref_artist_map[text] = item

        for item in self.reference_lines["y"]:
            value = item.get("value")
            color = item.get("color", REFERENCE_LINE_COLOR)
            linestyle = item.get("linestyle", "--")
            linewidth = item.get("linewidth", 1.4)

            line = ax.axhline(
                value,
                color=color,
                linestyle=linestyle,
                linewidth=linewidth,
                alpha=0.95,
                picker=8,
                zorder=10,
            )
            text = ax.text(
                0.01,
                value,
                self._format_reference_value("y", value),
                transform=ax.get_yaxis_transform(),
                ha="left",
                va="bottom",
                fontsize=8,
                color=color,
                bbox={
                    "facecolor": "white",
                    "edgecolor": color,
                    "boxstyle": "round,pad=0.2",
                    "alpha": 0.85,
                },
                zorder=11,
            )
            item["line"] = line
            item["text"] = text
            self._ref_artists += [line, text]
            self._ref_artist_map[line] = item
            self._ref_artist_map[text] = item

    def _redraw_reference_lines(self):
        if not self.figure.axes:
            return
        ax = self.figure.axes[0]
        for artist in self._ref_artists:
            try:
                artist.remove()
            except Exception:
                pass
        self._draw_reference_lines(ax)
        self.canvas.draw_idle()

    def _clear_reference_lines(self, axis="all"):
        if axis == "all":
            self.reference_lines["x"].clear()
            self.reference_lines["y"].clear()
        else:
            self.reference_lines[axis].clear()
        self._redraw_reference_lines()
        self._refresh_summary_after_reference_change()

    def _remove_reference_lines_by_value(self, axis):
        if not self.reference_lines[axis]:
            return

        current = " ".join(
            self._format_reference_value(axis, item["value"])
            for item in self.reference_lines[axis]
        )
        text, ok = QInputDialog.getText(
            self,
            f"Remove {axis.upper()} Reference Lines",
            f"Current values:\n{current}\n\nEnter values to remove:",
        )
        if not ok or not text.strip():
            return

        values, invalid = self._parse_reference_values(axis, text)
        if invalid:
            QMessageBox.warning(self, "Invalid values", f"Ignored: {', '.join(invalid)}")

        self.reference_lines[axis] = [
            item for item in self.reference_lines[axis]
            if not any(self._reference_values_match(item["value"], value, axis) for value in values)
        ]
        self._redraw_reference_lines()
        self._refresh_summary_after_reference_change()

    def _reference_values_match(self, a, b, axis=None):
        """Fuzzy match displayed reference-line values against stored values.
        This fixes the common case where a line is dragged to 11.718943...
        but the label displays 11.72 and the user enters 11.72 to remove it.
        """
        try:
            a = float(a)
            b = float(b)

            if round(a, 4) == round(b, 4):
                return True
            if round(a, 3) == round(b, 3):
                return True
            if round(a, 2) == round(b, 2):
                return True

            if self.figure.axes:
                ax = self.figure.axes[0]
                lo, hi = ax.get_xlim() if axis == "x" else ax.get_ylim()
                span = abs(hi - lo)
                tol = max(span * 0.003, 1e-6)
                return abs(a - b) <= tol

            return abs(a - b) <= max(1e-6, abs(a) * 1e-4)

        except Exception:
            return str(a) == str(b)

    def _format_reference_value(self, axis, value):
        if axis == "x" and self.cmb_graph.currentText() == "Time Series Plot":
            try:
                return mdates.num2date(float(value)).strftime("%Y-%m-%d")
            except Exception:
                pass
        try:
            value = float(value)
            if abs(value) >= 1000:
                return f"{value:,.2f}"
            if abs(value) >= 10:
                return f"{value:.2f}"
            return f"{value:.4f}"
        except Exception:
            return str(value)

    def _edit_reference_line_value(self, item):
        axis = item.get("axis", "x")
        current = self._format_reference_value(axis, item.get("value"))
        text, ok = QInputDialog.getText(
            self,
            "Edit Reference Line Value",
            f"New {axis.upper()} value:",
            text=current,
        )
        if not ok or not text.strip():
            return

        values, invalid = self._parse_reference_values(axis, text)
        if invalid or len(values) != 1:
            QMessageBox.warning(self, "Invalid value", "Enter exactly one valid reference value.")
            return

        item["value"] = values[0]
        self._redraw_reference_lines()
        self._refresh_summary_after_reference_change()

    def _edit_reference_line_color(self, item):
        color = QColorDialog.getColor(parent=self)
        if not color.isValid():
            return
        item["color"] = color.name()
        self._redraw_reference_lines()
        self._refresh_summary_after_reference_change()

    def _delete_reference_line_item(self, item):
        axis = item.get("axis")
        if axis not in {"x", "y"}:
            return
        try:
            self.reference_lines[axis].remove(item)
        except ValueError:
            pass
        self._redraw_reference_lines()

    def _on_reference_press(self, event):
        if event.button != 1 or event.inaxes is None:
            return
        for artist, item in self._ref_artist_map.items():
            try:
                contains, _ = artist.contains(event)
            except Exception:
                contains = False
            if contains:
                self._dragging_ref = item
                return

    def _on_reference_motion(self, event):
        if self._dragging_ref is None or event.inaxes is None:
            return

        axis = self._dragging_ref["axis"]
        coord = event.xdata if axis == "x" else event.ydata
        if coord is None:
            return

        self._dragging_ref["value"] = coord
        line = self._dragging_ref.get("line")
        text = self._dragging_ref.get("text")

        if axis == "x":
            if line:
                line.set_xdata([coord, coord])
            if text:
                text.set_position((coord, 0.98))
                text.set_text(self._format_reference_value("x", coord))
        else:
            if line:
                line.set_ydata([coord, coord])
            if text:
                text.set_position((0.01, coord))
                text.set_text(self._format_reference_value("y", coord))

        self.canvas.draw_idle()

    def _on_reference_release(self, event):
        if self._dragging_ref is None:
            return
        self._dragging_ref = None
        self._refresh_summary_after_reference_change()

    # ─────────────────────────────────────────────
    # Summary / Journal
    # ─────────────────────────────────────────────

    def _build_summary(self):
        graph = self.cmb_graph.currentText()
        ws = self.app_window.active_worksheet_name()
        ys = self.active_y_columns()
        W = 70
        lines = [
            graph.upper(), "=" * W,
            f"Worksheet : {ws}",
            f"Created   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "", "SETTINGS", "-" * W,
            f"Graph          : {graph}",
            f"Arrangement    : {self.cmb_arrangement.currentText() if self.cmb_arrangement.isVisible() else 'N/A'}",
            f"Y variable(s)  : {', '.join(ys) if ys else 'N/A'}",
            f"X / Stamp      : {self.cmb_x.currentText() or 'Observation index / N/A'}",
            f"Z / Size       : {self.cmb_z.currentText() if self.cmb_z.isVisible() else 'N/A'}",
            f"Group          : {self.cmb_group.currentText() if self.cmb_group.isVisible() else 'N/A'}",
            f"Palette        : {self.cmb_palette.currentText()}",
            "", "REFERENCE LINES", "-" * W,
            f"X lines : {self._reference_summary_text('x')}",
            f"Y lines : {self._reference_summary_text('y')}",
            "", "DESCRIPTIVE STATISTICS", "-" * W,
        ]
        for y_col in ys:
            y = self.get_numeric_series(y_col)
            if len(y) == 0:
                continue
            q1, q3 = y.quantile(0.25), y.quantile(0.75)
            cv = (y.std(ddof=1) / y.mean() * 100) if y.mean() != 0 else float("nan")
            lines += [
                f"Variable       : {y_col}",
                f"n              : {len(y)}",
                f"Mean           : {y.mean():.4f}",
                f"Std dev (s)    : {y.std(ddof=1):.4f}",
                f"CV (%)         : {cv:.2f}",
                f"Median         : {y.median():.4f}",
                f"Q1             : {q1:.4f}",
                f"Q3             : {q3:.4f}",
                f"IQR            : {q3 - q1:.4f}",
                f"Min            : {y.min():.4f}",
                f"Max            : {y.max():.4f}",
                f"Skewness       : {y.skew():.4f}",
                "",
            ]
        if graph == "Scatterplot" and self.cmb_x.currentText() and ys:
            data = self.get_clean_xy(self.cmb_x.currentText(), ys[0])
            if len(data) >= 2:
                r = pd.to_numeric(data["_x"], errors="coerce").corr(data[ys[0]])
                lines += ["RELATIONSHIP", "-" * W, f"Valid pairs    : {len(data)}", f"Pearson r      : {r:.4f}", f"R²             : {r*r:.4f}"]
        return "\n".join(lines)

    def _reference_summary_text(self, axis):
        if self.graph_interactions is not None:
            self.reference_lines = self.graph_interactions.to_graph_builder_reference_dict()
        if not self.reference_lines[axis]:
            return "N/A"
        return ", ".join(self._format_reference_value(axis, i["value"]) for i in self.reference_lines[axis])

    def preview_summary(self):
        self.current_summary = self._build_summary()
        self.summary_text.setPlainText(self.current_summary)
        self.tabs.setCurrentWidget(self.summary_tab)

    def _figure_to_base64(self):
        buf = io.BytesIO()
        self.figure.savefig(buf, format="png", dpi=140, bbox_inches="tight")
        return base64.b64encode(buf.getvalue()).decode("ascii")

    def _save_or_update_journal_entry(self):
        if self.current_journal_entry_id and hasattr(self.app_window, "update_journal_entry"):
            updated = self.app_window.update_journal_entry(self.current_journal_entry_id, self._journal_updates("Updated"))
            if updated:
                return
        self._auto_save_to_journal()

    def _auto_save_to_journal(self):
        ts = datetime.now()
        entry_id = ts.strftime("graph_%Y%m%d_%H%M%S_%f")
        self.current_journal_entry_id = entry_id
        entry = {
            "id": entry_id,
            "name": f"{self.cmb_graph.currentText()} — {self.app_window.active_worksheet_name()} — {ts.strftime('%H:%M:%S')}",
            "type": "Graph",
            "graph": self.cmb_graph.currentText(),
            "worksheet": self.app_window.active_worksheet_name(),
            "created_at": ts.strftime("%Y-%m-%d %H:%M:%S"),
            "modified_at": ts.strftime("%Y-%m-%d %H:%M:%S"),
            **self._journal_updates("Auto Saved"),
        }
        self.app_window.add_journal_entry(entry)

    def _journal_updates(self, status="Updated"):
        return {
            "status": status,
            "summary": self.current_summary,
            "reference_lines": self._reference_lines_payload(raw=False),
            "reference_lines_raw": self._reference_lines_payload(raw=True),
            "reference_lines_full": self._reference_lines_full_payload(),
            "payload": self._build_graph_payload(),
            "figure_png_base64": self._figure_to_base64(),
        }

    def _build_graph_payload(self):
        return {
            "graph": self.cmb_graph.currentText(),
            "worksheet": self.app_window.active_worksheet_name(),
            "arrangement": self.cmb_arrangement.currentText(),
            "variables": {
                "y": self.cmb_y.currentText(),
                "ys": self.active_y_columns(),
                "x": self.cmb_x.currentText(),
                "z": self.cmb_z.currentText(),
                "group": self.cmb_group.currentText(),
                "title": self.cmb_title.currentText(),
                "xlabel": self._axis_label_overrides.get("x", ""),
                "ylabel": self._axis_label_overrides.get("y", ""),
            },
            "options": {
                "grid": self.chk_grid.isChecked(),
                "mean": self.chk_mean.isChecked(),
                "median": self.chk_median.isChecked(),
                "auto_conclusions": self.chk_auto_conclusions.isChecked(),
                "auto_journal": self.chk_auto_journal.isChecked(),
                "trend": self.chk_trend.isChecked(),
                "jitter": self.chk_jitter.isChecked(),
                "bins": self.spn_bins.value(),
                "palette": self.cmb_palette.currentText(),
                "axis_limits": self._axis_limits,
                "contour_method": self.cmb_contour_method.currentText(),
                "distance_power": self.spn_distance_power.value(),
                "standardize_xy": self.chk_standardize_xy.isChecked(),
                "boundary_z": self.txt_boundary_z.text(),
            },
            "reference_lines_raw": self._reference_lines_payload(raw=True),
            "reference_lines_full": self._reference_lines_full_payload(),
        }

    def _reference_lines_payload(self, raw=False):
        if self.graph_interactions is not None:
            self.reference_lines = self.graph_interactions.to_graph_builder_reference_dict()
        if raw:
            return {
                "x": [item.get("value") for item in self.reference_lines["x"]],
                "y": [item.get("value") for item in self.reference_lines["y"]],
            }
        return {
            "x": [self._format_reference_value("x", item["value"]) for item in self.reference_lines["x"]],
            "y": [self._format_reference_value("y", item["value"]) for item in self.reference_lines["y"]],
        }

    def _reference_lines_full_payload(self):
        if self.graph_interactions is not None:
            self.reference_lines = self.graph_interactions.to_graph_builder_reference_dict()
        return {
            "x": [
                {
                    "value": item.get("value"),
                    "color": item.get("color", REFERENCE_LINE_COLOR),
                    "linestyle": item.get("linestyle", "--"),
                    "linewidth": item.get("linewidth", 1.4),
                }
                for item in self.reference_lines["x"]
            ],
            "y": [
                {
                    "value": item.get("value"),
                    "color": item.get("color", REFERENCE_LINE_COLOR),
                    "linestyle": item.get("linestyle", "--"),
                    "linewidth": item.get("linewidth", 1.4),
                }
                for item in self.reference_lines["y"]
            ],
        }

    def _sync_current_journal_entry(self):
        if self.current_journal_entry_id and hasattr(self.app_window, "update_journal_entry"):
            self.current_summary = self._build_summary()
            self.app_window.update_journal_entry(self.current_journal_entry_id, self._journal_updates("Updated"))

    def _refresh_summary_after_reference_change(self):
        self.current_summary = self._build_summary()
        self.summary_text.setPlainText(self.current_summary)
        self._sync_current_journal_entry()

    def load_from_journal_entry(self, entry):
        if not entry:
            return
        self._loading_from_journal = True
        try:
            self.current_journal_entry_id = entry.get("id")
            payload = entry.get("payload", {}) or {}
            graph = payload.get("graph") or entry.get("graph") or self.initial_graph
            self.set_graph_type(graph)
            self.refresh_variables()

            arrangement = payload.get("arrangement") or "One Y - Simple"
            if arrangement in ARRANGEMENTS:
                self.cmb_arrangement.setCurrentText(arrangement)

            variables = payload.get("variables", {}) or {}
            self._set_combo_value(self.cmb_y, variables.get("y", ""))
            self._set_combo_value(self.cmb_x, variables.get("x", ""))
            self._set_combo_value(self.cmb_z, variables.get("z", ""))
            self._set_combo_value(self.cmb_group, variables.get("group", ""))
            self.cmb_title.setEditText(variables.get("title", "") or graph)
            self._last_auto_title = self.cmb_title.currentText()
            self._axis_label_overrides = {
                "x": variables.get("xlabel", "") or "",
                "y": variables.get("ylabel", "") or "",
            }

            ys = variables.get("ys", []) or ([variables.get("y")] if variables.get("y") else [])
            self._select_multi_y_values(ys)

            options = payload.get("options", {}) or {}
            self.chk_grid.setChecked(bool(options.get("grid", True)))
            self.chk_mean.setChecked(bool(options.get("mean", True)))
            self.chk_median.setChecked(bool(options.get("median", False)))
            self.chk_auto_conclusions.setChecked(bool(options.get("auto_conclusions", True)))
            self.chk_auto_journal.setChecked(bool(options.get("auto_journal", True)))
            self.chk_trend.setChecked(bool(options.get("trend", False)))
            self.chk_jitter.setChecked(bool(options.get("jitter", True)))
            self.spn_bins.setValue(int(options.get("bins", 10)))
            self._set_combo_value(self.cmb_palette, options.get("palette", "SPS Blue"))
            self._set_combo_value(self.cmb_contour_method, options.get("contour_method", "Distance method"))
            self.spn_distance_power.setValue(float(options.get("distance_power", 2.0)))
            self.chk_standardize_xy.setChecked(bool(options.get("standardize_xy", True)))
            self.txt_boundary_z.setText(str(options.get("boundary_z", "") or ""))
            self._update_contour_options_visibility()

            axis_limits = options.get("axis_limits") or {}
            self._axis_limits = {
                "x": axis_limits.get("x", [None, None]),
                "y": axis_limits.get("y", [None, None]),
            }

            full_refs = payload.get("reference_lines_full") or entry.get("reference_lines_full") or {}
            raw_refs = payload.get("reference_lines_raw") or entry.get("reference_lines_raw") or {}
            self.reference_lines = {"x": [], "y": []}

            if full_refs:
                for axis in ["x", "y"]:
                    for ref in full_refs.get(axis, []):
                        try:
                            self.reference_lines[axis].append(self._make_reference_line_item(
                                axis=axis,
                                value=float(ref.get("value")),
                                color=ref.get("color", REFERENCE_LINE_COLOR),
                                linestyle=ref.get("linestyle", "--"),
                                linewidth=ref.get("linewidth", 1.4),
                            ))
                        except Exception:
                            pass
            else:
                # Backward compatibility with older journal entries.
                for axis in ["x", "y"]:
                    for value in raw_refs.get(axis, []):
                        try:
                            self.reference_lines[axis].append(self._make_reference_line_item(axis, float(value)))
                        except Exception:
                            pass

            self._update_requirements(graph)
            if self.graph_interactions is not None:
                self.graph_interactions.from_graph_builder_reference_dict(self.reference_lines)
        finally:
            self._loading_from_journal = False
        self.create_graph()

    def _set_combo_value(self, combo, value):
        if value is None:
            value = ""
        idx = combo.findText(str(value))
        if idx >= 0:
            combo.setCurrentIndex(idx)

    def _select_multi_y_values(self, values):
        values = set(v for v in values if v)
        for i in range(self.lst_y.count()):
            item = self.lst_y.item(i)
            item.setSelected(item.text() in values)

    # ─────────────────────────────────────────────
    # Utilities
    # ─────────────────────────────────────────────

    def copy_graph_to_clipboard(self):
        if not self.figure.axes:
            QMessageBox.information(self, "No graph", "Create a graph before copying it.")
            return

        try:
            buf = io.BytesIO()
            self.figure.savefig(buf, format="png", dpi=220, bbox_inches="tight", facecolor="white")
            buf.seek(0)
            image = QImage.fromData(buf.getvalue(), "PNG")
            if image.isNull():
                raise RuntimeError("Unable to convert graph to clipboard image.")
            QApplication.clipboard().setImage(image)
            self.app_window.statusBar().showMessage("Graph copied to clipboard.")
        except Exception as exc:
            QMessageBox.critical(self, "Copy Graph failed", str(exc))

    def export_figure(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export Figure", "", "PNG Image (*.png);;PDF File (*.pdf);;SVG File (*.svg)")
        if not path:
            return
        self.figure.savefig(path, dpi=160, bbox_inches="tight")
        self.app_window.statusBar().showMessage(f"Figure exported: {path}")

    def copy_summary(self):
        self.summary_text.selectAll()
        self.summary_text.copy()

    def reset_setup(self):
        self.cmb_arrangement.setCurrentText("One Y - Simple")
        self.chk_grid.setChecked(True)
        self.chk_mean.setChecked(True)
        self.chk_median.setChecked(False)
        self.chk_trend.setChecked(False)
        self.chk_jitter.setChecked(True)
        self.chk_auto_conclusions.setChecked(True)
        self.chk_auto_journal.setChecked(True)
        self.spn_bins.setValue(10)
        self.cmb_palette.setCurrentIndex(0)
        self.cmb_contour_method.setCurrentText("Distance method")
        self.spn_distance_power.setValue(2.0)
        self.chk_standardize_xy.setChecked(True)
        self.txt_boundary_z.setText("")
        self._update_contour_options_visibility()
        self._axis_label_overrides = {"x": "", "y": ""}
        self._axis_limits = {
            "x": [None, None],
            "y": [None, None],
        }
        self.refresh_variables()
        self._update_requirements(self.cmb_graph.currentText())
        self._update_auto_title(force=True)
