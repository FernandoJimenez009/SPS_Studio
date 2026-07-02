"""
core/plot_interaction_tools.py
SPS Studio — reusable Matplotlib interaction tools for PySide6 modules.

Purpose:
- Edit panel title by double-click or context menu.
- Edit X/Y axis labels.
- Edit X/Y axis scale.
- Add, edit and delete reference lines.
- Draw reference lines using the Graph Builder visual style.
- Drag reference lines with the mouse after they are created.
- Edit plot element styles from the graph context menu.
- Keep graph-interaction code centralized for reuse across SPS modules.

Reusable for future modules:
- Capability Study
- Hypothesis Testing
- Control Charts
- Regression
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from PySide6.QtCore import Qt
from matplotlib.backend_bases import MouseEvent
from PySide6.QtWidgets import (
    QColorDialog,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
    QInputDialog,
)

try:
    import pandas as pd
    import matplotlib.dates as mdates
except Exception:  # pragma: no cover - optional only for date parsing/formatting
    pd = None
    mdates = None

try:
    from matplotlib.colors import to_hex, to_rgba
except Exception:  # pragma: no cover - matplotlib is expected in the app runtime
    to_hex = None
    to_rgba = None

REFERENCE_LINE_COLOR = "#C56A2D"
DEFAULT_PLOT_COLOR = "#2F5D87"


# ─────────────────────────────────────────────
# REUSABLE GRAPH INTERACTION BLOCK - DATA MODELS / DIALOGS - START
# ─────────────────────────────────────────────

@dataclass
class ReferenceLineSpec:
    axis_index: int = 0
    axis: str = "y"
    value: float = 0.0
    label: str = ""
    color: str = REFERENCE_LINE_COLOR
    linestyle: str = "--"
    linewidth: float = 1.4
    show_value: bool = True
    artist: Any = field(default=None, repr=False)
    text_artist: Any = field(default=None, repr=False)


class AxisScaleDialog(QDialog):
    def __init__(self, parent, axis_name, current_min, current_max, current_auto_min=True, current_auto_max=True):
        super().__init__(parent)
        self.setWindowTitle(f"{axis_name} Axis Scale Range")
        self.setModal(True)

        from PySide6.QtWidgets import QCheckBox

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
        return {
            "auto_min": self.chk_auto_min.isChecked(),
            "auto_max": self.chk_auto_max.isChecked(),
            "min_text": "" if self.chk_auto_min.isChecked() else self.txt_min.text().strip(),
            "max_text": "" if self.chk_auto_max.isChecked() else self.txt_max.text().strip(),
        }


class ReferenceLineDialog(QDialog):
    LINESTYLE_ITEMS = [
        ("Solid", "-"),
        ("Dashed", "--"),
        ("Dotted", ":"),
        ("Dash-dot", "-."),
    ]

    def __init__(self, parent, spec: ReferenceLineSpec, decimals: int = 2, title: str = "Reference Line"):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(True)
        self._color = spec.color or REFERENCE_LINE_COLOR
        self._decimals = int(decimals)

        from PySide6.QtWidgets import QCheckBox

        root = QVBoxLayout(self)
        grid = QGridLayout()

        self.cmb_axis = QComboBox()
        self.cmb_axis.addItems(["x", "y"])
        self.cmb_axis.setCurrentText(spec.axis if spec.axis in {"x", "y"} else "y")

        self.spn_value = QDoubleSpinBox()
        self.spn_value.setRange(-1_000_000_000, 1_000_000_000)
        self.spn_value.setDecimals(max(0, min(8, self._decimals)))
        self.spn_value.setValue(float(spec.value))

        self.txt_label = QLineEdit()
        self.txt_label.setText(spec.label or "")

        self.chk_show_value = QCheckBox("Show value label")
        self.chk_show_value.setChecked(bool(spec.show_value))

        self.cmb_style = QComboBox()
        for label, code in self.LINESTYLE_ITEMS:
            self.cmb_style.addItem(label, code)
        idx = self.cmb_style.findData(spec.linestyle)
        self.cmb_style.setCurrentIndex(idx if idx >= 0 else 1)

        self.spn_width = QDoubleSpinBox()
        self.spn_width.setRange(0.5, 8.0)
        self.spn_width.setDecimals(1)
        self.spn_width.setSingleStep(0.5)
        self.spn_width.setValue(float(spec.linewidth or 1.4))

        self.btn_color = QPushButton("Select Color")
        self.btn_color.clicked.connect(self._select_color)
        self._update_color_button()

        grid.addWidget(QLabel("Axis"), 0, 0)
        grid.addWidget(self.cmb_axis, 0, 1)
        grid.addWidget(QLabel("Value"), 1, 0)
        grid.addWidget(self.spn_value, 1, 1)
        grid.addWidget(QLabel("Label"), 2, 0)
        grid.addWidget(self.txt_label, 2, 1)
        grid.addWidget(QLabel("Line style"), 3, 0)
        grid.addWidget(self.cmb_style, 3, 1)
        grid.addWidget(QLabel("Line width"), 4, 0)
        grid.addWidget(self.spn_width, 4, 1)
        grid.addWidget(QLabel("Color"), 5, 0)
        grid.addWidget(self.btn_color, 5, 1)
        grid.addWidget(self.chk_show_value, 6, 0, 1, 2)
        root.addLayout(grid)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

    def _select_color(self):
        color = QColorDialog.getColor(parent=self, title="Select Reference Line Color")
        if color.isValid():
            self._color = color.name()
            self._update_color_button()

    def _update_color_button(self):
        self.btn_color.setStyleSheet(f"background:{self._color}; color:white; font-weight:bold;")
        self.btn_color.setText(self._color)

    def to_spec(self, axis_index: int) -> ReferenceLineSpec:
        return ReferenceLineSpec(
            axis_index=axis_index,
            axis=self.cmb_axis.currentText(),
            value=float(self.spn_value.value()),
            label=self.txt_label.text().strip(),
            color=self._color,
            linestyle=self.cmb_style.currentData(),
            linewidth=float(self.spn_width.value()),
            show_value=self.chk_show_value.isChecked(),
        )


class PlotStyleDialog(QDialog):
    """Generic plot style editor for Matplotlib artists.

    This dialog is intentionally generic so the same interaction manager can style
    lines, scatter collections, bars, boxplot patches and future SPS plots without
    duplicating UI code in each module.
    """

    LINESTYLE_ITEMS = [
        ("Keep current", "__keep__"),
        ("Solid", "-"),
        ("Dashed", "--"),
        ("Dotted", ":"),
        ("Dash-dot", "-."),
        ("None", "None"),
    ]

    MARKER_ITEMS = [
        ("Keep current", "__keep__"),
        ("None", "None"),
        ("Circle", "o"),
        ("Square", "s"),
        ("Triangle", "^"),
        ("Diamond", "D"),
        ("X", "x"),
        ("Plus", "+"),
        ("Point", "."),
    ]

    def __init__(self, parent, title: str, current_color: str = DEFAULT_PLOT_COLOR,
                 current_alpha: float = 1.0, current_linewidth: float = 1.5):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(True)
        self._color = current_color or DEFAULT_PLOT_COLOR

        from PySide6.QtWidgets import QCheckBox

        root = QVBoxLayout(self)
        grid = QGridLayout()

        self.chk_color = QCheckBox("Apply color")
        self.chk_color.setChecked(True)
        self.btn_color = QPushButton("Select Color")
        self.btn_color.clicked.connect(self._select_color)
        self._update_color_button()

        self.chk_alpha = QCheckBox("Apply transparency")
        self.chk_alpha.setChecked(True)
        self.spn_alpha = QDoubleSpinBox()
        self.spn_alpha.setRange(0.05, 1.00)
        self.spn_alpha.setDecimals(2)
        self.spn_alpha.setSingleStep(0.05)
        self.spn_alpha.setValue(float(current_alpha if current_alpha is not None else 1.0))

        self.chk_linewidth = QCheckBox("Apply line / edge width")
        self.chk_linewidth.setChecked(False)
        self.spn_linewidth = QDoubleSpinBox()
        self.spn_linewidth.setRange(0.1, 12.0)
        self.spn_linewidth.setDecimals(1)
        self.spn_linewidth.setSingleStep(0.5)
        self.spn_linewidth.setValue(float(current_linewidth if current_linewidth is not None else 1.5))

        self.cmb_linestyle = QComboBox()
        for label, code in self.LINESTYLE_ITEMS:
            self.cmb_linestyle.addItem(label, code)

        self.cmb_marker = QComboBox()
        for label, code in self.MARKER_ITEMS:
            self.cmb_marker.addItem(label, code)

        grid.addWidget(self.chk_color, 0, 0)
        grid.addWidget(self.btn_color, 0, 1)

        grid.addWidget(self.chk_alpha, 1, 0)
        grid.addWidget(self.spn_alpha, 1, 1)

        grid.addWidget(self.chk_linewidth, 2, 0)
        grid.addWidget(self.spn_linewidth, 2, 1)

        grid.addWidget(QLabel("Line style"), 3, 0)
        grid.addWidget(self.cmb_linestyle, 3, 1)

        grid.addWidget(QLabel("Marker"), 4, 0)
        grid.addWidget(self.cmb_marker, 4, 1)

        root.addLayout(grid)

        note = QLabel("Only supported properties will be applied to the selected plot element.")
        note.setWordWrap(True)
        note.setStyleSheet("color:#555555;")
        root.addWidget(note)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

    def _select_color(self):
        color = QColorDialog.getColor(parent=self, title="Select Plot Color")
        if color.isValid():
            self._color = color.name()
            self._update_color_button()

    def _update_color_button(self):
        self.btn_color.setStyleSheet(f"background:{self._color}; color:white; font-weight:bold;")
        self.btn_color.setText(self._color)

    def values(self) -> dict:
        return {
            "apply_color": self.chk_color.isChecked(),
            "color": self._color,
            "apply_alpha": self.chk_alpha.isChecked(),
            "alpha": float(self.spn_alpha.value()),
            "apply_linewidth": self.chk_linewidth.isChecked(),
            "linewidth": float(self.spn_linewidth.value()),
            "linestyle": self.cmb_linestyle.currentData(),
            "marker": self.cmb_marker.currentData(),
        }


# ─────────────────────────────────────────────
# REUSABLE GRAPH INTERACTION BLOCK - MANAGER - START
# ─────────────────────────────────────────────

class PlotInteractionManager:
    """
    Reusable manager for Matplotlib FigureCanvasQTAgg interactions.

    Notes for future modules:
    - Use enable_interaction(...) from each module after creating FigureCanvas.
    - Call register_axes() after each full redraw.
    - Call apply_reference_lines() after plotting data and applying axis limits.
    - Use to_dict()/from_dict() to persist reference-line state in Journal payloads.
    - Plot style edits are runtime visual edits; persist them in host payloads only if a module needs that later.
    """

    def __init__(
        self,
        host: QWidget,
        figure_getter: Callable[[], Any],
        canvas_getter: Callable[[], Any],
        decimals_getter: Optional[Callable[[], int]] = None,
        on_changed: Optional[Callable[[], None]] = None,
        on_status: Optional[Callable[[str], None]] = None,
    ):
        self.host = host
        self._figure_getter = figure_getter
        self._canvas_getter = canvas_getter
        self._decimals_getter = decimals_getter or (lambda: 2)
        self._on_changed = on_changed
        self._on_status = on_status

        self.active_axis = None
        self.reference_lines: list[ReferenceLineSpec] = []
        self._dragging_spec: Optional[ReferenceLineSpec] = None
        self._dragging_axis = None
        self._drag_started = False
        self._connected = False

    @property
    def figure(self):
        return self._figure_getter()

    @property
    def canvas(self):
        return self._canvas_getter()

    @property
    def decimals(self) -> int:
        try:
            return int(self._decimals_getter())
        except Exception:
            return 2

    def connect_canvas(self):
        if self._connected:
            return
        self.canvas.setContextMenuPolicy(Qt.CustomContextMenu)
        self.canvas.customContextMenuRequested.connect(self.show_context_menu)
        self.canvas.mpl_connect("button_press_event", self.on_button_press)
        self.canvas.mpl_connect("motion_notify_event", self.on_mouse_motion)
        self.canvas.mpl_connect("button_release_event", self.on_button_release)
        self._connected = True

    def register_axes(self):
        for idx, ax in enumerate(self.figure.axes):
            try:
                ax._sps_axis_index = idx
            except Exception:
                pass
        if self.figure.axes and self.active_axis not in self.figure.axes:
            self.active_axis = self.figure.axes[0]

    def selected_axis(self):
        if self.active_axis is not None and self.active_axis in self.figure.axes:
            return self.active_axis
        if self.figure.axes:
            self.active_axis = self.figure.axes[0]
            return self.active_axis
        return None

    # ─────────────────────────────────────────────
    # Event handling
    # ─────────────────────────────────────────────

    def on_button_press(self, event):
        if event.inaxes is not None:
            self.active_axis = event.inaxes

        if getattr(event, "dblclick", False):
            if self._try_edit_reference_line_from_event(event):
                return

            # Direct style editing: double-click a bar, box, point, line or filled element.
            # This avoids forcing users to guess from generic Element 1 / Element 2 lists.
            if self.edit_plot_element_style_from_event(event):
                return

            self.handle_double_click(event)
            return

        if getattr(event, "button", None) == 1:
            spec = self.reference_line_from_event(event)
            if spec is not None:
                self._dragging_spec = spec
                self._dragging_axis = spec.artist.axes if spec.artist is not None else event.inaxes
                self._drag_started = False
                self.active_axis = self._dragging_axis

    def on_mouse_motion(self, event):
        if self._dragging_spec is None:
            return
        spec = self._dragging_spec
        if event.inaxes is not self._dragging_axis:
            return
        if spec.axis == "x":
            if event.xdata is None:
                return
            spec.value = float(event.xdata)
        else:
            if event.ydata is None:
                return
            spec.value = float(event.ydata)
        if spec.show_value and not spec.label:
            spec.label = self._default_reference_label(spec)
        self._move_reference_artist(spec)
        self._drag_started = True
        self.canvas.draw_idle()

    def on_button_release(self, event):
        if self._dragging_spec is None:
            return
        spec = self._dragging_spec
        changed = self._drag_started
        self._dragging_spec = None
        self._dragging_axis = None
        self._drag_started = False
        if changed:
            self._sync_host_reference_lines()
            self.apply_reference_lines()
            self._redraw_changed(f"Reference line moved to {self._format_reference_value(spec.axis, spec.value)}.")

    # ─────────────────────────────────────────────
    # Context menu
    # ─────────────────────────────────────────────

    def show_context_menu(self, pos):
        if not self.figure.axes:
            return

        selected_spec = self.reference_line_at_widget_pos(pos)
        if selected_spec is not None:
            self._show_reference_line_menu(pos, selected_spec)
            return

        clicked_target = self._plot_target_from_widget_pos(pos)
        if clicked_target is not None:
            self._show_plot_element_menu(pos, clicked_target)
            return

        menu = QMenu(self.host)
        menu.addAction("Edit Panel Title", self.edit_selected_panel_title)
        menu.addAction("Edit X Axis Label", lambda: self.edit_axis_label("x"))
        menu.addAction("Edit Y Axis Label", lambda: self.edit_axis_label("y"))
        menu.addSeparator()
        menu.addAction("Edit X Axis Scale", lambda: self.edit_axis_scale("x"))
        menu.addAction("Edit Y Axis Scale", lambda: self.edit_axis_scale("y"))
        menu.addAction("Reset Axis Scales", self.reset_axis_scales)
        menu.addSeparator()
        menu.addAction("Edit Plot Element Style", lambda: self.edit_plot_element_style_at_pos(pos))
        menu.addAction("Edit All Plot Colors", self.edit_all_plot_colors)
        menu.addSeparator()
        menu.addAction("Add X Reference Line(s) - Vertical", lambda: self.add_reference_lines_by_text("x"))
        menu.addAction("Add Y Reference Line(s) - Horizontal", lambda: self.add_reference_lines_by_text("y"))
        menu.addSeparator()
        menu.addAction("Remove X Reference Line(s) by Value", lambda: self.remove_reference_lines_by_value("x"))
        menu.addAction("Remove Y Reference Line(s) by Value", lambda: self.remove_reference_lines_by_value("y"))
        menu.addSeparator()
        menu.addAction("Clear X Reference Lines", lambda: self.clear_reference_lines("x"))
        menu.addAction("Clear Y Reference Lines", lambda: self.clear_reference_lines("y"))
        menu.addAction("Clear All Reference Lines", lambda: self.clear_reference_lines("all"))
        menu.exec(self.canvas.mapToGlobal(pos))

    def _show_reference_line_menu(self, pos, spec: ReferenceLineSpec):
        current_value = self._format_reference_value(spec.axis, spec.value)
        menu = QMenu(self.host)
        menu.addAction(f"Edit Value ({spec.axis.upper()} = {current_value})", lambda: self.edit_reference_line_value(spec))
        menu.addAction("Edit Reference Line Details", lambda: self.edit_reference_line(spec))
        menu.addAction("Change Color", lambda: self.edit_reference_line_color(spec))
        menu.addAction("Delete Reference Line", lambda: self.delete_reference_line(spec))
        menu.exec(self.canvas.mapToGlobal(pos))

    def _show_plot_element_menu(self, pos, target: dict):
        """Context menu shown when the user right-clicks directly on a plotted element."""
        menu = QMenu(self.host)
        label = target.get("label", "Selected Plot Element")
        menu.addAction(f"Edit Selected Element Style", lambda: self._edit_plot_target_style(target))

        series_target = self._series_target_from_point_target(target)
        if series_target is None:
            series_target = self._series_target_from_artist_target(target)
        if series_target is not None:
            menu.addAction("Edit Entire Series Style", lambda: self._edit_plot_target_style(series_target))

        menu.addSeparator()
        menu.addAction("Edit All Plot Colors", self.edit_all_plot_colors)
        menu.addSeparator()
        menu.addAction("Add X Reference Line(s) - Vertical", lambda: self.add_reference_lines_by_text("x"))
        menu.addAction("Add Y Reference Line(s) - Horizontal", lambda: self.add_reference_lines_by_text("y"))
        menu.setTitle(label)
        menu.exec(self.canvas.mapToGlobal(pos))

    # ─────────────────────────────────────────────
    # Title / axis labels / axis scale
    # ─────────────────────────────────────────────

    def edit_selected_panel_title(self):
        ax = self.selected_axis()
        if ax is None:
            return
        current = ax.get_title()
        text, ok = QInputDialog.getText(self.host, "Edit Panel Title", "Title:", text=current)
        if not ok:
            return
        new_text = text.strip()
        ax.set_title(new_text, fontweight="bold", pad=8)
        if hasattr(self.host, "cmb_title"):
            try:
                self.host.cmb_title.setEditText(new_text)
                self.host._last_auto_title = new_text
            except Exception:
                pass
        self._redraw_changed("Panel title updated.")

    def edit_axis_label(self, axis):
        ax = self.selected_axis()
        if ax is None:
            return
        if axis == "x":
            current = ax.get_xlabel()
            title = "Edit X Axis Label"
            prompt = "X axis label:"
        else:
            current = ax.get_ylabel()
            title = "Edit Y Axis Label"
            prompt = "Y axis label:"
        text, ok = QInputDialog.getText(self.host, title, prompt, text=current)
        if not ok:
            return
        new_text = text.strip()
        if axis == "x":
            ax.set_xlabel(new_text)
        else:
            ax.set_ylabel(new_text)
        if hasattr(self.host, "_axis_label_overrides"):
            self.host._axis_label_overrides[axis] = new_text
        self._redraw_changed("Axis label updated.")

    def apply_axis_labels(self, ax, label_overrides: Optional[dict] = None):
        label_overrides = label_overrides or {}
        if label_overrides.get("x"):
            ax.set_xlabel(label_overrides["x"])
        if label_overrides.get("y"):
            ax.set_ylabel(label_overrides["y"])

    def parse_axis_limit_value(self, axis, text):
        text = str(text).strip()
        if not text or text.lower() in {"auto", "none", "na", "n/a"}:
            return None
        try:
            return float(text)
        except ValueError:
            pass
        if axis == "x" and self._host_graph_type() == "Time Series Plot" and pd is not None and mdates is not None:
            try:
                dt = pd.to_datetime(text, errors="raise")
                return mdates.date2num(dt.to_pydatetime())
            except Exception:
                pass
        raise ValueError(f"Invalid limit value: {text}")

    def edit_axis_scale(self, axis):
        ax = self.selected_axis()
        if ax is None:
            return
        axis_name = "X" if axis == "x" else "Y"
        current_min, current_max = ax.get_xlim() if axis == "x" else ax.get_ylim()
        saved = getattr(self.host, "_axis_limits", {}).get(axis, [None, None]) if hasattr(self.host, "_axis_limits") else [None, None]
        auto_min = saved[0] is None
        auto_max = saved[1] is None

        dialog = AxisScaleDialog(
            self.host,
            axis_name,
            current_min if auto_min else saved[0],
            current_max if auto_max else saved[1],
            current_auto_min=auto_min,
            current_auto_max=auto_max,
        )
        if dialog.exec() != QDialog.Accepted:
            return
        values = dialog.values()
        try:
            lower = None if values["auto_min"] else self.parse_axis_limit_value(axis, values["min_text"])
            upper = None if values["auto_max"] else self.parse_axis_limit_value(axis, values["max_text"])
        except ValueError as exc:
            QMessageBox.warning(self.host, "Invalid scale", str(exc))
            return
        if lower is not None and upper is not None and lower >= upper:
            QMessageBox.warning(self.host, "Invalid scale", "Minimum must be lower than maximum.")
            return
        if hasattr(self.host, "_axis_limits"):
            self.host._axis_limits[axis] = [lower, upper]
        self.apply_axis_limits(ax, getattr(self.host, "_axis_limits", {axis: [lower, upper]}))
        self.apply_reference_lines()
        self._redraw_changed("Axis scale updated.")

    def apply_axis_limits(self, ax, axis_limits: Optional[dict] = None):
        axis_limits = axis_limits or getattr(self.host, "_axis_limits", {}) or {}
        for axis in ["x", "y"]:
            lower, upper = axis_limits.get(axis, [None, None])
            if lower is None and upper is None:
                continue
            current = ax.get_xlim() if axis == "x" else ax.get_ylim()
            lo = current[0] if lower is None else lower
            hi = current[1] if upper is None else upper
            if lo < hi:
                if axis == "x":
                    ax.set_xlim(lo, hi)
                else:
                    ax.set_ylim(lo, hi)

    def reset_axis_scales(self):
        if hasattr(self.host, "_axis_limits"):
            self.host._axis_limits = {"x": [None, None], "y": [None, None]}
        for ax in self.figure.axes:
            try:
                ax.autoscale(enable=True, axis="both")
            except Exception:
                pass
        self.apply_reference_lines()
        self._redraw_changed("Axis scales reset.")

    def handle_double_click(self, event):
        if not self.figure.axes:
            return
        try:
            renderer = self.canvas.get_renderer()
        except Exception:
            self.canvas.draw()
            renderer = self.canvas.get_renderer()
        px, py = event.x, event.y
        for ax in self.figure.axes:
            try:
                if ax.title.get_window_extent(renderer=renderer).expanded(1.2, 1.8).contains(px, py):
                    self.active_axis = ax
                    self.edit_selected_panel_title()
                    return
            except Exception:
                pass
            try:
                if ax.xaxis.label.get_window_extent(renderer=renderer).expanded(1.4, 2.2).contains(px, py):
                    self.active_axis = ax
                    self.edit_axis_label("x")
                    return
            except Exception:
                pass
            try:
                if ax.yaxis.label.get_window_extent(renderer=renderer).expanded(1.8, 1.4).contains(px, py):
                    self.active_axis = ax
                    self.edit_axis_label("y")
                    return
            except Exception:
                pass
            for tick in ax.get_xticklabels():
                try:
                    if not tick.get_visible() or not str(tick.get_text()).strip():
                        continue
                    if tick.get_window_extent(renderer=renderer).expanded(1.8, 2.6).contains(px, py):
                        self.active_axis = ax
                        self.edit_axis_scale("x")
                        return
                except Exception:
                    pass
            for tick in ax.get_yticklabels():
                try:
                    if not tick.get_visible() or not str(tick.get_text()).strip():
                        continue
                    if tick.get_window_extent(renderer=renderer).expanded(2.6, 1.8).contains(px, py):
                        self.active_axis = ax
                        self.edit_axis_scale("y")
                        return
                except Exception:
                    pass

    # ─────────────────────────────────────────────
    # Reference lines - Graph Builder visual style
    # ─────────────────────────────────────────────

    def add_reference_line_dialog(self, axis=None):
        ax = self.selected_axis()
        if ax is None:
            return
        axis_index = getattr(ax, "_sps_axis_index", self.figure.axes.index(ax))
        if axis is None:
            axis, ok = QInputDialog.getItem(self.host, "Reference Line Axis", "Axis:", ["x", "y"], 1, False)
            if not ok:
                return
        default_value = self._default_reference_value(ax, axis)
        spec = ReferenceLineSpec(axis_index=axis_index, axis=axis, value=default_value)
        dialog = ReferenceLineDialog(self.host, spec, decimals=self.decimals, title="Add Reference Line")
        if dialog.exec() != QDialog.Accepted:
            return
        new_spec = dialog.to_spec(axis_index)
        if not new_spec.label and new_spec.show_value:
            new_spec.label = self._default_reference_label(new_spec)
        self.reference_lines.append(new_spec)
        self._sync_host_reference_lines()
        self.apply_reference_lines()
        self._redraw_changed("Reference line added.")

    def add_reference_lines_by_text(self, axis):
        ax = self.selected_axis()
        if ax is None:
            return
        axis_index = getattr(ax, "_sps_axis_index", self.figure.axes.index(ax))
        text, ok = QInputDialog.getText(
            self.host,
            f"Add {axis.upper()} Reference Lines",
            "Enter one or more values separated by spaces:",
        )
        if not ok or not text.strip():
            return
        values, invalid = self._parse_reference_values(axis, text)
        if invalid:
            QMessageBox.warning(self.host, "Invalid values", f"Ignored: {', '.join(invalid)}")
        for value in values:
            self.reference_lines.append(ReferenceLineSpec(axis_index=axis_index, axis=axis, value=float(value)))
        self._sync_host_reference_lines()
        self.apply_reference_lines()
        self._redraw_changed("Reference line added.")

    def edit_reference_line(self, spec: ReferenceLineSpec):
        dialog = ReferenceLineDialog(self.host, spec, decimals=self.decimals, title="Edit Reference Line")
        if dialog.exec() != QDialog.Accepted:
            return
        updated = dialog.to_spec(spec.axis_index)
        if not updated.label and updated.show_value:
            updated.label = self._default_reference_label(updated)
        spec.axis = updated.axis
        spec.value = updated.value
        spec.label = updated.label
        spec.color = updated.color
        spec.linestyle = updated.linestyle
        spec.linewidth = updated.linewidth
        spec.show_value = updated.show_value
        self._sync_host_reference_lines()
        self.apply_reference_lines()
        self._redraw_changed("Reference line updated.")

    def edit_reference_line_value(self, spec: ReferenceLineSpec):
        current = self._format_reference_value(spec.axis, spec.value)
        text, ok = QInputDialog.getText(
            self.host,
            "Edit Reference Line Value",
            f"New {spec.axis.upper()} value:",
            text=current,
        )
        if not ok or not text.strip():
            return
        try:
            values, invalid = self._parse_reference_values(spec.axis, text)
            if invalid or not values:
                raise ValueError(text)
            spec.value = float(values[0])
            if spec.show_value and not spec.label:
                spec.label = self._default_reference_label(spec)
        except Exception:
            QMessageBox.warning(self.host, "Invalid value", "Enter a valid numeric value or date value for time series X axis.")
            return
        self._sync_host_reference_lines()
        self.apply_reference_lines()
        self._redraw_changed("Reference line updated.")

    def edit_reference_line_color(self, spec: ReferenceLineSpec):
        color = QColorDialog.getColor(parent=self.host, title="Select Reference Line Color")
        if not color.isValid():
            return
        spec.color = color.name()
        self._sync_host_reference_lines()
        self.apply_reference_lines()
        self._redraw_changed("Reference line color updated.")

    def delete_reference_line(self, spec: ReferenceLineSpec):
        try:
            self.reference_lines.remove(spec)
        except ValueError:
            return
        self._sync_host_reference_lines()
        self.apply_reference_lines()
        self._redraw_changed("Reference line deleted.")

    def remove_reference_lines_by_value(self, axis):
        current = " ".join(self._format_reference_value(axis, spec.value) for spec in self.reference_lines if spec.axis == axis)
        if not current:
            return
        text, ok = QInputDialog.getText(
            self.host,
            f"Remove {axis.upper()} Reference Lines",
            f"Current values:\n{current}\n\nEnter values to remove:",
        )
        if not ok or not text.strip():
            return
        values, invalid = self._parse_reference_values(axis, text)
        if invalid:
            QMessageBox.warning(self.host, "Invalid values", f"Ignored: {', '.join(invalid)}")
        self.reference_lines = [
            spec for spec in self.reference_lines
            if not (spec.axis == axis and any(self._reference_values_match(spec.value, value, axis) for value in values))
        ]
        self._sync_host_reference_lines()
        self.apply_reference_lines()
        self._redraw_changed("Reference line removed.")

    def clear_reference_lines(self, axis="all"):
        if axis == "all":
            self.reference_lines = []
        else:
            self.reference_lines = [spec for spec in self.reference_lines if spec.axis != axis]
        self._sync_host_reference_lines()
        self.apply_reference_lines()
        self._redraw_changed("Reference lines cleared.")

    def apply_reference_lines(self):
        self.register_axes()
        self._remove_reference_artists()
        for spec in self.reference_lines:
            idx = int(spec.axis_index or 0)
            if idx < 0 or idx >= len(self.figure.axes):
                continue
            ax = self.figure.axes[idx]
            if spec.axis == "x":
                line = ax.axvline(
                    spec.value,
                    color=spec.color,
                    linestyle=spec.linestyle,
                    linewidth=spec.linewidth,
                    alpha=0.95,
                    picker=8,
                    zorder=10,
                )
                text_artist = None
                if spec.show_value:
                    text_artist = ax.text(
                        spec.value,
                        0.98,
                        spec.label or self._format_reference_value("x", spec.value),
                        transform=ax.get_xaxis_transform(),
                        ha="center",
                        va="top",
                        fontsize=8,
                        color=spec.color,
                        bbox={"facecolor": "white", "edgecolor": spec.color, "alpha": 0.78, "boxstyle": "round,pad=0.20"},
                        zorder=11,
                    )
            else:
                line = ax.axhline(
                    spec.value,
                    color=spec.color,
                    linestyle=spec.linestyle,
                    linewidth=spec.linewidth,
                    alpha=0.95,
                    picker=8,
                    zorder=10,
                )
                text_artist = None
                if spec.show_value:
                    text_artist = ax.text(
                        0.985,
                        spec.value,
                        spec.label or self._format_reference_value("y", spec.value),
                        transform=ax.get_yaxis_transform(),
                        ha="right",
                        va="center",
                        fontsize=8,
                        color=spec.color,
                        bbox={"facecolor": "white", "edgecolor": spec.color, "alpha": 0.78, "boxstyle": "round,pad=0.20"},
                        zorder=11,
                    )
            line._sps_reference_line = True
            if text_artist is not None:
                text_artist._sps_reference_line = True
            spec.artist = line
            spec.text_artist = text_artist
        self.canvas.draw_idle()

    def _remove_reference_artists(self):
        for spec in self.reference_lines:
            for artist in [spec.artist, spec.text_artist]:
                if artist is not None:
                    try:
                        artist.remove()
                    except Exception:
                        pass
            spec.artist = None
            spec.text_artist = None
        for ax in self.figure.axes:
            for artist in list(ax.lines) + list(ax.texts):
                if getattr(artist, "_sps_reference_line", False):
                    try:
                        artist.remove()
                    except Exception:
                        pass

    def reference_line_from_event(self, event) -> Optional[ReferenceLineSpec]:
        for spec in reversed(self.reference_lines):
            artist = spec.artist
            if artist is None:
                continue
            try:
                contains, _ = artist.contains(event)
            except Exception:
                contains = False
            if contains:
                return spec
        return None

    def reference_line_at_widget_pos(self, pos) -> Optional[ReferenceLineSpec]:
        if not self.figure.axes:
            return None
        height = self.canvas.height()
        mouse_x = float(pos.x())
        mouse_y = float(height - pos.y())
        tolerance_px = 10
        for spec in reversed(self.reference_lines):
            if spec.axis_index < 0 or spec.axis_index >= len(self.figure.axes):
                continue
            ax = self.figure.axes[spec.axis_index]
            try:
                if spec.axis == "x":
                    px = ax.transData.transform((float(spec.value), ax.get_ylim()[0]))[0]
                    if abs(mouse_x - px) <= tolerance_px:
                        return spec
                else:
                    py = ax.transData.transform((ax.get_xlim()[0], float(spec.value)))[1]
                    if abs(mouse_y - py) <= tolerance_px:
                        return spec
            except Exception:
                pass
        try:
            event = MouseEvent("button_press_event", self.canvas, mouse_x, mouse_y)
            return self.reference_line_from_event(event)
        except Exception:
            return None

    def _move_reference_artist(self, spec: ReferenceLineSpec):
        artist = spec.artist
        if artist is None:
            return
        ax = artist.axes
        if spec.axis == "x":
            try:
                artist.set_xdata([spec.value, spec.value])
            except Exception:
                pass
            if spec.text_artist is not None:
                try:
                    spec.text_artist.set_position((spec.value, 0.98))
                    spec.text_artist.set_text(spec.label or self._format_reference_value("x", spec.value))
                except Exception:
                    pass
        else:
            try:
                artist.set_ydata([spec.value, spec.value])
            except Exception:
                pass
            if spec.text_artist is not None:
                try:
                    spec.text_artist.set_position((0.985, spec.value))
                    spec.text_artist.set_text(spec.label or self._format_reference_value("y", spec.value))
                except Exception:
                    pass

    def _try_edit_reference_line_from_event(self, event) -> bool:
        spec = self.reference_line_from_event(event)
        if spec is None:
            return False
        if spec.artist is not None:
            self.active_axis = spec.artist.axes
        self.edit_reference_line(spec)
        return True


    # ─────────────────────────────────────────────
    # Plot element style editing
    # ─────────────────────────────────────────────

    def edit_plot_element_style_from_event(self, event) -> bool:
        """Edit the exact plot element the user double-clicked.

        Supported direct targets:
        - Single bars / histogram bins / boxplot patches.
        - Single scatter points when the collection exposes point indices.
        - Full Line2D elements.
        - Filled/collection series when the collection is clicked.
        """
        target = self._plot_target_from_event(event)
        if target is None:
            return False
        self._edit_plot_target_style(target)
        return True

    def _plot_target_from_widget_pos(self, pos) -> Optional[dict]:
        if pos is None or not self.figure.axes:
            return None
        height = self.canvas.height()
        mouse_x = float(pos.x())
        mouse_y = float(height - pos.y())
        try:
            event = MouseEvent("button_press_event", self.canvas, mouse_x, mouse_y)
            return self._plot_target_from_event(event)
        except Exception:
            return None

    def _plot_target_from_event(self, event) -> Optional[dict]:
        ax = event.inaxes
        if ax is None:
            return None
        self.active_axis = ax
        axis_index = getattr(ax, "_sps_axis_index", 0)

        # Patches first: bars, histogram bins and boxplot boxes are usually patches.
        # Checking patches before collections makes bar/box editing feel precise.
        for idx, patch in enumerate(reversed(list(getattr(ax, "patches", []))), start=1):
            if getattr(patch, "_sps_reference_line", False):
                continue
            try:
                visible = patch.get_visible()
            except Exception:
                visible = True
            if not visible:
                continue
            try:
                contains, _ = patch.contains(event)
            except Exception:
                contains = False
            if contains:
                label = self._artist_legend_label(patch)
                return {
                    "label": f"Axis {axis_index + 1} - Selected Patch" + (f" ({label})" if label else ""),
                    "artists": [patch],
                    "kind": "patch",
                    "legend_label": label,
                    "axis": ax,
                }

        # Collections: scatter points, filled areas, contour collections.
        for idx, coll in enumerate(reversed(list(getattr(ax, "collections", []))), start=1):
            if getattr(coll, "_sps_reference_line", False):
                continue
            try:
                visible = coll.get_visible()
            except Exception:
                visible = True
            if not visible:
                continue
            try:
                contains, info = coll.contains(event)
            except Exception:
                contains, info = False, {}
            if contains:
                label = self._artist_legend_label(coll)
                inds = list(info.get("ind", [])) if isinstance(info, dict) else []
                kind = self._collection_kind_name(coll)
                if kind == "Scatter" and inds:
                    point_index = int(inds[0])
                    return {
                        "label": f"Axis {axis_index + 1} - Selected Scatter Point {point_index + 1}" + (f" ({label})" if label else ""),
                        "artists": [coll],
                        "kind": "collection_point",
                        "point_index": point_index,
                        "legend_label": label,
                        "axis": ax,
                    }
                return {
                    "label": f"Axis {axis_index + 1} - Selected {kind}" + (f" ({label})" if label else ""),
                    "artists": [coll],
                    "kind": "collection",
                    "legend_label": label,
                    "axis": ax,
                }

        # Lines last, excluding reference lines.
        for idx, line in enumerate(reversed(list(getattr(ax, "lines", []))), start=1):
            if getattr(line, "_sps_reference_line", False):
                continue
            try:
                visible = line.get_visible()
            except Exception:
                visible = True
            if not visible:
                continue
            try:
                contains, _ = line.contains(event)
            except Exception:
                contains = False
            if contains:
                label = self._artist_legend_label(line)
                return {
                    "label": f"Axis {axis_index + 1} - Selected Line" + (f" ({label})" if label else ""),
                    "artists": [line],
                    "kind": "line",
                    "legend_label": label,
                    "axis": ax,
                }

        return None

    def _series_target_from_point_target(self, target: dict) -> Optional[dict]:
        """For a selected scatter point, offer an optional whole-series edit."""
        if target.get("kind") != "collection_point":
            return None
        artists = target.get("artists") or []
        if not artists:
            return None
        label = target.get("legend_label", "")
        return {
            "label": "Entire Scatter Series" + (f" ({label})" if label else ""),
            "artists": artists,
            "kind": "collection",
            "legend_label": label,
            "axis": target.get("axis"),
        }

    def _series_target_from_artist_target(self, target: dict) -> Optional[dict]:
        """For bars/boxes/patches tagged with SPS metadata, edit all artists in the same series."""
        artists = target.get("artists") or []
        if not artists:
            return None
        first = artists[0]
        series_key = self._artist_series_key(first)
        label = self._clean_artist_label(target.get("legend_label", "")) or self._artist_legend_label(first)
        if not series_key and not label:
            return None
        ax = target.get("axis")
        if ax is None:
            try:
                ax = first.axes
            except Exception:
                ax = None
        if ax is None:
            return None
        related = []
        for candidate in list(getattr(ax, "lines", [])) + list(getattr(ax, "collections", [])) + list(getattr(ax, "patches", [])):
            if getattr(candidate, "_sps_reference_line", False):
                continue
            candidate_key = self._artist_series_key(candidate)
            candidate_label = self._artist_legend_label(candidate)
            if series_key and candidate_key == series_key:
                related.append(candidate)
            elif not series_key and label and candidate_label == label:
                related.append(candidate)
        if len(related) <= len(artists):
            return None
        return {
            "label": "Entire Series" + (f" ({label})" if label else ""),
            "artists": related,
            "kind": "series",
            "legend_label": label,
            "axis": ax,
        }

    def edit_plot_element_style_at_pos(self, pos=None):
        """Open a reusable style editor for one plotted element.

        Supports common Matplotlib objects generated by SPS modules:
        - Line2D: line plots, mean lines, trend lines.
        - PathCollection: scatter plots, individual points.
        - Patch groups: bars, histograms, boxplot patches.
        - Poly/Fill collections: area and contour-like filled objects.
        """
        targets = self._editable_plot_targets(pos)
        if not targets:
            QMessageBox.information(self.host, "No editable elements", "No editable plot elements were found in this graph.")
            return

        labels = [target["label"] for target in targets]
        selected_label, ok = QInputDialog.getItem(
            self.host,
            "Edit Plot Element Style",
            "Select plot element:",
            labels,
            0,
            False,
        )
        if not ok:
            return

        idx = labels.index(selected_label)
        target = targets[idx]
        self._edit_plot_target_style(target)

    def edit_all_plot_colors(self):
        """Change the color of all editable plot elements in the active axis.

        This is useful for quick formatting when the user wants one consistent color
        for a graph before copying/exporting it.
        """
        ax = self.selected_axis()
        if ax is None:
            return

        targets = self._editable_plot_targets_for_axis(ax)
        if not targets:
            QMessageBox.information(self.host, "No editable elements", "No editable plot elements were found in this graph.")
            return

        color = QColorDialog.getColor(parent=self.host, title="Select Plot Color")
        if not color.isValid():
            return

        color_hex = color.name()
        values = {
            "apply_color": True,
            "color": color_hex,
            "apply_alpha": False,
            "alpha": None,
            "apply_linewidth": False,
            "linewidth": None,
            "linestyle": "__keep__",
            "marker": "__keep__",
        }
        for target in targets:
            self._apply_plot_style_to_target(target, values)
            self._sync_legend_for_target(target, values)
        self._redraw_changed("Plot colors updated.")

    def _edit_plot_target_style(self, target: dict):
        artists = target.get("artists", [])
        if not artists:
            return

        first = artists[0]
        current_color = self._artist_current_color(first)
        current_alpha = self._artist_current_alpha(first)
        current_linewidth = self._artist_current_linewidth(first)

        dialog = PlotStyleDialog(
            self.host,
            title=f"Edit Style - {target.get('label', 'Plot Element')}",
            current_color=current_color,
            current_alpha=current_alpha,
            current_linewidth=current_linewidth,
        )
        if dialog.exec() != QDialog.Accepted:
            return

        values = dialog.values()
        self._apply_plot_style_to_target(target, values)
        self._sync_legend_for_target(target, values)
        self._redraw_changed("Plot element style updated.")

    def _editable_plot_targets(self, pos=None) -> list[dict]:
        self.register_axes()

        # Prefer the axis under the context menu. Fall back to active/first axis.
        ax = self._axis_from_widget_pos(pos) if pos is not None else None
        if ax is not None:
            self.active_axis = ax
            return self._editable_plot_targets_for_axis(ax)

        ax = self.selected_axis()
        if ax is not None:
            return self._editable_plot_targets_for_axis(ax)

        targets = []
        for axis in self.figure.axes:
            targets.extend(self._editable_plot_targets_for_axis(axis))
        return targets

    def _editable_plot_targets_for_axis(self, ax) -> list[dict]:
        targets = []
        axis_index = getattr(ax, "_sps_axis_index", 0)

        # Lines: one target per visible non-reference line.
        line_count = 0
        for line in list(getattr(ax, "lines", [])):
            if getattr(line, "_sps_reference_line", False):
                continue
            try:
                if not line.get_visible():
                    continue
            except Exception:
                pass
            line_count += 1
            label = self._artist_legend_label(line)
            targets.append({
                "label": f"Axis {axis_index + 1} - Line {line_count}" + (f" ({label})" if label else ""),
                "artists": [line],
                "kind": "line",
                "legend_label": label,
                "axis": ax,
            })

        # Collections: one target per collection. Scatter, fill_between, contours, etc.
        collection_count = 0
        for coll in list(getattr(ax, "collections", [])):
            if getattr(coll, "_sps_reference_line", False):
                continue
            try:
                if not coll.get_visible():
                    continue
            except Exception:
                pass
            collection_count += 1
            label = self._artist_legend_label(coll)
            kind = self._collection_kind_name(coll)
            targets.append({
                "label": f"Axis {axis_index + 1} - {kind} {collection_count}" + (f" ({label})" if label else ""),
                "artists": [coll],
                "kind": "collection",
                "legend_label": label,
                "axis": ax,
            })

        # Patches: group patches together to avoid showing dozens of histogram/bar rectangles.
        editable_patches = []
        for patch in list(getattr(ax, "patches", [])):
            if getattr(patch, "_sps_reference_line", False):
                continue
            try:
                if not patch.get_visible():
                    continue
            except Exception:
                pass
            # Skip near-empty decorative patches.
            try:
                if hasattr(patch, "get_width") and hasattr(patch, "get_height"):
                    if abs(float(patch.get_width())) <= 0 and abs(float(patch.get_height())) <= 0:
                        continue
            except Exception:
                pass
            editable_patches.append(patch)

        if editable_patches:
            grouped_by_series = {}
            untagged = []
            for patch in editable_patches:
                label = self._artist_legend_label(patch)
                key = self._artist_series_key(patch) or label
                if key:
                    grouped_by_series.setdefault((key, label), []).append(patch)
                else:
                    untagged.append(patch)
            for (key, label), patches in grouped_by_series.items():
                targets.append({
                    "label": f"Axis {axis_index + 1} - Patches ({label or key})",
                    "artists": patches,
                    "kind": "patches",
                    "legend_label": label,
                    "axis": ax,
                })
            if untagged:
                targets.append({
                    "label": f"Axis {axis_index + 1} - Bars / Patches ({len(untagged)})",
                    "artists": untagged,
                    "kind": "patches",
                    "legend_label": "",
                    "axis": ax,
                })

        return targets

    def _apply_plot_style_to_target(self, target: dict, values: dict):
        """Apply style values to a full target or a single point inside a collection."""
        if target.get("kind") == "collection_point":
            self._apply_plot_style_to_collection_point(target, values)
            return
        self._apply_plot_style_to_artists(target.get("artists", []), values)

    def _apply_plot_style_to_collection_point(self, target: dict, values: dict):
        artists = target.get("artists") or []
        if not artists:
            return
        coll = artists[0]
        index = int(target.get("point_index", 0))

        color = values.get("color")
        apply_color = bool(values.get("apply_color"))
        alpha = values.get("alpha")
        apply_alpha = bool(values.get("apply_alpha"))
        linewidth = values.get("linewidth")
        apply_linewidth = bool(values.get("apply_linewidth"))

        if apply_color and color:
            rgba = self._color_to_rgba(color, alpha if apply_alpha else None)
            self._set_collection_index_color(coll, index, rgba, face=True)
            self._set_collection_index_color(coll, index, rgba, face=False)

        elif apply_alpha and alpha is not None:
            # Preserve existing point color and update only alpha.
            current = self._collection_index_color(coll, index, face=True)
            if current is not None:
                rgba = list(current)
                rgba[3] = float(alpha)
                self._set_collection_index_color(coll, index, rgba, face=True)
            current_edge = self._collection_index_color(coll, index, face=False)
            if current_edge is not None:
                rgba = list(current_edge)
                rgba[3] = float(alpha)
                self._set_collection_index_color(coll, index, rgba, face=False)

        if apply_linewidth and linewidth is not None:
            try:
                widths = coll.get_linewidths()
                n = self._collection_point_count(coll)
                widths = self._expand_sequence(widths, n, default=1.0)
                if 0 <= index < len(widths):
                    widths[index] = float(linewidth)
                    coll.set_linewidths(widths)
            except Exception:
                pass

    def _sync_legend_for_target(self, target: dict, values: dict):
        """Synchronize legend color/style when a whole series/artist is recolored.

        Single scatter-point edits intentionally do not recolor the legend because the
        legend normally represents the full series, not one point. Use "Edit Entire
        Series Style" from the right-click menu when the legend should change.
        """
        if target.get("kind") == "collection_point":
            return
        if not values.get("apply_color") and values.get("linestyle") == "__keep__" and values.get("marker") == "__keep__":
            return

        ax = target.get("axis")
        if ax is None:
            artists = target.get("artists") or []
            if artists:
                try:
                    ax = artists[0].axes
                except Exception:
                    ax = None
        if ax is None:
            return

        label = self._clean_artist_label(target.get("legend_label", ""))
        if not label:
            artists = target.get("artists") or []
            if artists:
                label = self._artist_legend_label(artists[0])
        if not label:
            return

        legend = ax.get_legend()
        if legend is None:
            return

        try:
            legend_handles = legend.legend_handles
        except Exception:
            try:
                legend_handles = legend.legendHandles
            except Exception:
                legend_handles = []

        try:
            legend_texts = legend.get_texts()
        except Exception:
            legend_texts = []

        color = values.get("color") if values.get("apply_color") else None
        linewidth = values.get("linewidth") if values.get("apply_linewidth") else None
        linestyle = values.get("linestyle", "__keep__")
        marker = values.get("marker", "__keep__")

        for handle, text in zip(legend_handles, legend_texts):
            try:
                text_label = str(text.get_text()).strip()
            except Exception:
                text_label = ""
            if text_label != label:
                continue
            if color:
                self._set_artist_color(handle, color)
            if linewidth is not None:
                self._set_artist_linewidth(handle, linewidth)
            if linestyle and linestyle != "__keep__":
                self._set_artist_linestyle(handle, linestyle)
            if marker and marker != "__keep__":
                self._set_artist_marker(handle, marker)

    def _apply_plot_style_to_artists(self, artists: list, values: dict):
        color = values.get("color")
        apply_color = bool(values.get("apply_color"))
        alpha = values.get("alpha")
        apply_alpha = bool(values.get("apply_alpha"))
        linewidth = values.get("linewidth")
        apply_linewidth = bool(values.get("apply_linewidth"))
        linestyle = values.get("linestyle", "__keep__")
        marker = values.get("marker", "__keep__")

        for artist in artists:
            # Color handling by artist type/capabilities.
            if apply_color and color:
                self._set_artist_color(artist, color)

            if apply_alpha and alpha is not None:
                try:
                    artist.set_alpha(float(alpha))
                except Exception:
                    pass

            if apply_linewidth and linewidth is not None:
                self._set_artist_linewidth(artist, linewidth)

            if linestyle and linestyle != "__keep__":
                self._set_artist_linestyle(artist, linestyle)

            if marker and marker != "__keep__":
                self._set_artist_marker(artist, marker)

    def _set_artist_color(self, artist, color: str):
        # Line2D supports set_color directly.
        try:
            if hasattr(artist, "set_color"):
                artist.set_color(color)
        except Exception:
            pass

        # Collections and patches normally use face/edge colors.
        try:
            if hasattr(artist, "set_facecolor"):
                artist.set_facecolor(color)
        except Exception:
            pass
        try:
            if hasattr(artist, "set_facecolors"):
                artist.set_facecolors(color)
        except Exception:
            pass
        try:
            if hasattr(artist, "set_edgecolor"):
                artist.set_edgecolor(color)
        except Exception:
            pass
        try:
            if hasattr(artist, "set_edgecolors"):
                artist.set_edgecolors(color)
        except Exception:
            pass

    def _set_artist_linewidth(self, artist, linewidth):
        try:
            if hasattr(artist, "set_linewidth"):
                artist.set_linewidth(float(linewidth))
                return
        except Exception:
            pass
        try:
            if hasattr(artist, "set_linewidths"):
                artist.set_linewidths([float(linewidth)])
        except Exception:
            pass

    def _set_artist_linestyle(self, artist, linestyle):
        style = "" if linestyle == "None" else linestyle
        try:
            if hasattr(artist, "set_linestyle"):
                artist.set_linestyle(style)
                return
        except Exception:
            pass
        try:
            if hasattr(artist, "set_linestyles"):
                artist.set_linestyles(style)
        except Exception:
            pass

    def _set_artist_marker(self, artist, marker):
        marker_value = "" if marker == "None" else marker
        try:
            if hasattr(artist, "set_marker"):
                artist.set_marker(marker_value)
        except Exception:
            pass

    def _color_to_rgba(self, color: str, alpha: Optional[float] = None):
        try:
            if to_rgba is not None:
                rgba = list(to_rgba(color))
            else:
                rgba = list(color)
            if alpha is not None and len(rgba) >= 4:
                rgba[3] = float(alpha)
            return rgba
        except Exception:
            return color

    def _collection_point_count(self, coll) -> int:
        try:
            offsets = coll.get_offsets()
            return int(len(offsets))
        except Exception:
            pass
        try:
            facecolors = coll.get_facecolors()
            return int(len(facecolors))
        except Exception:
            pass
        return 1

    def _expand_color_array(self, colors, n: int, default_color=None):
        try:
            import numpy as np
            arr = np.array(colors, dtype=float)
            if arr.ndim == 1 and arr.size in {3, 4}:
                arr = arr.reshape(1, -1)
            if arr.size == 0:
                base = self._color_to_rgba(default_color or DEFAULT_PLOT_COLOR)
                arr = np.array([base], dtype=float)
            if arr.shape[0] == 1 and n > 1:
                arr = np.repeat(arr, n, axis=0)
            return arr
        except Exception:
            return colors

    def _expand_sequence(self, values, n: int, default=1.0):
        try:
            seq = list(values)
        except Exception:
            seq = [default]
        if not seq:
            seq = [default]
        if len(seq) == 1 and n > 1:
            seq = seq * n
        if len(seq) < n:
            seq.extend([seq[-1]] * (n - len(seq)))
        return seq

    def _collection_index_color(self, coll, index: int, face: bool = True):
        try:
            colors = coll.get_facecolors() if face else coll.get_edgecolors()
            n = self._collection_point_count(coll)
            colors = self._expand_color_array(colors, n)
            if 0 <= index < len(colors):
                return colors[index]
        except Exception:
            pass
        return None

    def _set_collection_index_color(self, coll, index: int, rgba, face: bool = True):
        try:
            n = self._collection_point_count(coll)
            getter = coll.get_facecolors if face else coll.get_edgecolors
            setter = coll.set_facecolors if face else coll.set_edgecolors
            colors = self._expand_color_array(getter(), n, default_color=DEFAULT_PLOT_COLOR)
            if 0 <= index < len(colors):
                colors[index] = rgba
                setter(colors)
        except Exception:
            pass

    def _axis_from_widget_pos(self, pos):
        if pos is None or not self.figure.axes:
            return None
        height = self.canvas.height()
        mouse_x = float(pos.x())
        mouse_y = float(height - pos.y())
        for ax in self.figure.axes:
            try:
                if ax.bbox.contains(mouse_x, mouse_y):
                    return ax
            except Exception:
                pass
        return None

    def _artist_current_color(self, artist) -> str:
        candidates = []
        for method_name in ["get_color", "get_facecolor", "get_facecolors", "get_edgecolor", "get_edgecolors"]:
            try:
                if hasattr(artist, method_name):
                    value = getattr(artist, method_name)()
                    candidates.append(value)
            except Exception:
                pass
        for value in candidates:
            color = self._coerce_color_to_hex(value)
            if color:
                return color
        return DEFAULT_PLOT_COLOR

    def _artist_current_alpha(self, artist) -> float:
        try:
            alpha = artist.get_alpha()
            if alpha is not None:
                return float(alpha)
        except Exception:
            pass
        return 1.0

    def _artist_current_linewidth(self, artist) -> float:
        for method_name in ["get_linewidth", "get_linewidths"]:
            try:
                if hasattr(artist, method_name):
                    value = getattr(artist, method_name)()
                    if isinstance(value, (list, tuple)) and value:
                        return float(value[0])
                    try:
                        # numpy arrays land here.
                        if hasattr(value, "__len__") and len(value) > 0:
                            return float(value[0])
                    except Exception:
                        pass
                    return float(value)
            except Exception:
                pass
        return 1.5

    def _coerce_color_to_hex(self, value) -> str:
        if value is None:
            return ""
        try:
            # Collections often return Nx4 arrays. Use first color.
            if hasattr(value, "shape") and len(getattr(value, "shape", [])) == 2 and value.shape[0] > 0:
                value = value[0]
            elif isinstance(value, (list, tuple)) and value and isinstance(value[0], (list, tuple)):
                value = value[0]
            if to_hex is not None:
                return to_hex(value, keep_alpha=False)
            if isinstance(value, str):
                return value
        except Exception:
            pass
        return ""

    def _collection_kind_name(self, coll) -> str:
        name = type(coll).__name__
        if name == "PathCollection":
            return "Scatter"
        if name in {"PolyCollection", "FillBetweenPolyCollection"}:
            return "Filled Area"
        if "Contour" in name:
            return "Contour"
        return "Collection"

    def _artist_legend_label(self, artist) -> str:
        """Return the SPS legend label assigned by a plotting module, falling back to Matplotlib label."""
        try:
            label = getattr(artist, "_sps_legend_label", "")
            label = self._clean_artist_label(label)
            if label:
                return label
        except Exception:
            pass
        try:
            return self._clean_artist_label(artist.get_label())
        except Exception:
            return ""

    def _artist_series_key(self, artist) -> str:
        """Return optional SPS series key used to group related artists across a legend series."""
        try:
            key = str(getattr(artist, "_sps_series_key", "") or "").strip()
            return key
        except Exception:
            return ""

    def _clean_artist_label(self, label) -> str:
        label = str(label or "").strip()
        if not label or label.startswith("_"):
            return ""
        return label

    # ─────────────────────────────────────────────
    # Serialization / compatibility helpers
    # ─────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "reference_lines": [self._spec_to_dict(spec) for spec in self.reference_lines],
            "reference_lines_full": self.to_graph_builder_reference_dict(),
        }

    def from_dict(self, payload: Optional[dict]):
        if not payload:
            self.reference_lines = []
            self._sync_host_reference_lines()
            return
        refs = []
        if isinstance(payload, dict) and isinstance(payload.get("reference_lines"), list):
            for item in payload.get("reference_lines", []):
                spec = self._dict_to_spec(item)
                if spec is not None:
                    refs.append(spec)
        elif isinstance(payload, dict) and ("x" in payload or "y" in payload):
            refs = self._graph_builder_dict_to_specs(payload)
        elif isinstance(payload, dict) and isinstance(payload.get("reference_lines_full"), dict):
            refs = self._graph_builder_dict_to_specs(payload.get("reference_lines_full"))
        self.reference_lines = refs
        self._sync_host_reference_lines()

    def to_graph_builder_reference_dict(self) -> dict:
        result = {"x": [], "y": []}
        for spec in self.reference_lines:
            result.setdefault(spec.axis, []).append({
                "value": spec.value,
                "color": spec.color,
                "linestyle": spec.linestyle,
                "linewidth": spec.linewidth,
                "label": spec.label,
                "show_value": spec.show_value,
                "axis_index": spec.axis_index,
            })
        return result

    def from_graph_builder_reference_dict(self, payload: Optional[dict]):
        self.reference_lines = self._graph_builder_dict_to_specs(payload or {})
        self._sync_host_reference_lines()

    def _spec_to_dict(self, spec: ReferenceLineSpec) -> dict:
        return {
            "axis_index": spec.axis_index,
            "axis": spec.axis,
            "value": spec.value,
            "label": spec.label,
            "color": spec.color,
            "linestyle": spec.linestyle,
            "linewidth": spec.linewidth,
            "show_value": spec.show_value,
        }

    def _dict_to_spec(self, item: dict) -> Optional[ReferenceLineSpec]:
        try:
            return ReferenceLineSpec(
                axis_index=int(item.get("axis_index", 0)),
                axis=item.get("axis", "y"),
                value=float(item.get("value")),
                label=item.get("label", "") or "",
                color=item.get("color", REFERENCE_LINE_COLOR) or REFERENCE_LINE_COLOR,
                linestyle=item.get("linestyle", "--") or "--",
                linewidth=float(item.get("linewidth", 1.4)),
                show_value=bool(item.get("show_value", True)),
            )
        except Exception:
            return None

    def _graph_builder_dict_to_specs(self, payload: dict) -> list[ReferenceLineSpec]:
        refs = []
        for axis in ["x", "y"]:
            for item in payload.get(axis, []) or []:
                try:
                    if isinstance(item, dict):
                        value = item.get("value")
                        refs.append(ReferenceLineSpec(
                            axis_index=int(item.get("axis_index", 0)),
                            axis=axis,
                            value=float(value),
                            label=item.get("label", "") or "",
                            color=item.get("color", REFERENCE_LINE_COLOR) or REFERENCE_LINE_COLOR,
                            linestyle=item.get("linestyle", "--") or "--",
                            linewidth=float(item.get("linewidth", 1.4)),
                            show_value=bool(item.get("show_value", True)),
                        ))
                    else:
                        refs.append(ReferenceLineSpec(axis_index=0, axis=axis, value=float(item)))
                except Exception:
                    pass
        return refs

    # ─────────────────────────────────────────────
    # Internal utilities
    # ─────────────────────────────────────────────

    def _sync_host_reference_lines(self):
        if hasattr(self.host, "reference_lines") and isinstance(getattr(self.host, "reference_lines"), dict):
            self.host.reference_lines = self.to_graph_builder_reference_dict()

    def _host_graph_type(self):
        try:
            return self.host.cmb_graph.currentText()
        except Exception:
            return ""

    def _default_reference_value(self, ax, axis) -> float:
        if axis == "x":
            x0, x1 = ax.get_xlim()
            return float((x0 + x1) / 2)
        y0, y1 = ax.get_ylim()
        return float((y0 + y1) / 2)

    def _default_reference_label(self, spec: ReferenceLineSpec) -> str:
        return self._format_reference_value(spec.axis, spec.value)

    def _parse_reference_values(self, axis, text):
        values, invalid = [], []
        allow_date_x = axis == "x" and self._host_graph_type() == "Time Series Plot" and pd is not None and mdates is not None
        for raw in str(text).replace(",", " ").split():
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

    def _format_reference_value(self, axis, value):
        if axis == "x" and self._host_graph_type() == "Time Series Plot" and mdates is not None:
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

    def _reference_values_match(self, a, b, axis=None):
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
                return abs(a - b) <= max(span * 0.003, 1e-6)
            return abs(a - b) <= max(1e-6, abs(a) * 1e-4)
        except Exception:
            return str(a) == str(b)

    def _redraw_changed(self, message: Optional[str] = None):
        self.canvas.draw_idle()
        if self._on_changed:
            self._on_changed()
        if message and self._on_status:
            self._on_status(message)


# ─────────────────────────────────────────────
# REUSABLE GRAPH INTERACTION BLOCK - CONVENIENCE API - START
# ─────────────────────────────────────────────

def enable_interaction(
    host: QWidget,
    figure_getter: Callable[[], Any],
    canvas_getter: Callable[[], Any],
    decimals_getter: Optional[Callable[[], int]] = None,
    on_changed: Optional[Callable[[], None]] = None,
    on_status: Optional[Callable[[str], None]] = None,
) -> PlotInteractionManager:
    """Create and connect a PlotInteractionManager for a host widget."""
    manager = PlotInteractionManager(
        host=host,
        figure_getter=figure_getter,
        canvas_getter=canvas_getter,
        decimals_getter=decimals_getter,
        on_changed=on_changed,
        on_status=on_status,
    )
    manager.connect_canvas()
    manager.register_axes()
    return manager

# ─────────────────────────────────────────────
# REUSABLE GRAPH INTERACTION BLOCK - CONVENIENCE API - END
# ─────────────────────────────────────────────
