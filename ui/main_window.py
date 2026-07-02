"""Main window orchestrator for SPS Studio.

This module keeps the application shell thin and delegates menu,
worksheet, and journal responsibilities to dedicated controllers.
"""

from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ui.controllers.journal_manager import JournalManager
from ui.controllers.menu_controller import MenuController
from ui.controllers.worksheet_manager import WorksheetManager
from core.app_config import (
    APP_NAME,
    APP_SUBTITLE,
    APP_VERSION,
    CONTACT,
    CREATOR,
    WINDOW_TITLE,
)

MAX_WORKSHEETS = 33
EXTENSION = ".COESPSFJ"


def resource_path(relative_path: str | Path) -> Path:
    """Return the absolute path for a bundled or development resource.

    Parameters
    ----------
    relative_path : str or pathlib.Path
        Resource path relative to the application root.

    Returns
    -------
    pathlib.Path
        Absolute path to the requested resource.
    """
    try:
        base_path = Path(sys._MEIPASS)  # type: ignore[attr-defined]
    except Exception:
        base_path = Path(__file__).resolve().parent.parent

    return base_path / relative_path


class SPSStudioMainWindow(QMainWindow):
    """Coordinate the SPS Studio application shell.

    The class owns shared application state and exposes a compatibility API
    used by analysis modules. UI responsibilities are delegated to controllers.

    Attributes
    ----------
    project_path : str or None
        Path to the currently opened project file.
    dirty : bool
        Whether the project contains unsaved changes.
    worksheets : dict[str, QWidget]
        Mapping of worksheet names to worksheet table widgets.
    current_ws_name : str or None
        Name of the active worksheet.
    journal_entries : list[dict]
        Stored analysis journal entries.
    """

    def __init__(self) -> None:
        """Initialize the main window and controller layer."""
        super().__init__()

        self.setWindowTitle(WINDOW_TITLE)
        self.resize(1500, 900)

        self.project_path = None
        self.dirty = False
        self.worksheets = {}
        self.current_ws_name = None
        self.journal_entries = []

        self.menu_controller = MenuController(self)
        self.worksheet_manager = WorksheetManager(self, max_worksheets=MAX_WORKSHEETS)
        self.journal_manager = JournalManager(self)

        self._build_menu()
        self._build_docks()
        self._build_central()

        self.add_worksheet("Worksheet 1", switch=True)
        self.show_home()

        self.statusBar().showMessage("Ready.")

    # ------------------------------------------------------------------
    # Application shell
    # ------------------------------------------------------------------

    def _build_menu(self) -> None:
        """Build the application menu bar."""
        self.menu_controller.build_menu()

    def _build_docks(self) -> None:
        """Build worksheet and journal dock widgets."""
        self.worksheet_manager.build_dock()
        self.journal_manager.build_dock()

    def _build_central(self) -> None:
        """Build the central tab container."""
        self.central_tabs = QTabWidget()
        self.setCentralWidget(self.central_tabs)

    def clear_central(self) -> None:
        """Remove every tab from the central widget."""
        self.central_tabs.clear()

    def show_home(self) -> None:
        """Display the SPS Studio home dashboard."""
        self.clear_central()

        home = QWidget()
        layout = QVBoxLayout(home)
        layout.setContentsMargins(28, 28, 28, 28)
        layout.setSpacing(14)

        logo_path = resource_path("img/Logo_SPS.webp")
        if logo_path.exists():
            pix = QPixmap(str(logo_path))
            if not pix.isNull():
                logo = QLabel()
                logo.setPixmap(pix.scaledToWidth(220, Qt.SmoothTransformation))
                logo.setAlignment(Qt.AlignCenter)
                layout.addWidget(logo)

        title = QLabel(APP_NAME)
        title.setObjectName("TitleLabel")
        title.setAlignment(Qt.AlignCenter)

        subtitle = QLabel(f"{APP_SUBTITLE}\nCreated by {CONTACT}")
        subtitle.setObjectName("SubtitleLabel")
        subtitle.setAlignment(Qt.AlignCenter)

        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addWidget(self._build_home_tool_dashboard())
        layout.addStretch()

        self.central_tabs.addTab(home, "Home")

    def _build_home_tool_dashboard(self) -> QGroupBox:
        """Create the tool availability dashboard.

        Returns
        -------
        PySide6.QtWidgets.QGroupBox
            Dashboard widget with release status by module.
        """
        box = QGroupBox("SPS Studio Tool Status")
        root = QVBoxLayout(box)
        root.setContentsMargins(16, 14, 16, 16)
        root.setSpacing(10)

        note = QLabel(
            "Available modules can be used in the current release. "
            "In-progress modules are visible in the menus but not yet released."
        )
        note.setWordWrap(True)
        note.setStyleSheet("color:#4F6680;")
        root.addWidget(note)

        grid = QGridLayout()
        grid.setHorizontalSpacing(14)
        grid.setVerticalSpacing(7)

        headers = ["Area", "Tool", "Status", "Access Path", "Notes"]
        for col, text in enumerate(headers):
            lbl = QLabel(text)
            lbl.setStyleSheet(
                "font-weight:bold; color:#17324D; padding:6px; "
                "border-bottom:1px solid #B8C7D6;"
            )
            grid.addWidget(lbl, 0, col)

        rows = [
            ("Core", "Worksheets", "Core", "Left panel / Worksheet menu", "Excel-like paste, headers in first row."),
            ("Core", "Project Save / Open", "Core", "File menu", f"Project extension: {EXTENSION}"),
            ("Graph", "Graph Builder", "Available", "Graph menu", "Histogram, scatterplot, boxplot, time series and more."),
            ("Quality", "Capability Study", "Available", "Stat > Quality Tools", "Capability summary, full diagnostic report and dashboard."),
            ("Quality", "Gage R&R Study", "Available", "Stat > Quality Tools > Gage Study", "Crossed Gage R&R with journal support."),
            ("Core", "Analysis Journal", "Available", "Right panel", "Review, edit, export and manage saved analyses."),
            ("Statistics", "Hypothesis Tests", "Available", "Stat > Basic Statistics", "1-Sample, 2-Sample, Paired t and Proportion Test."),
            ("Statistics", "Regression", "Available", "Stat > Regression", "Simple regression, Multiple regression."),
            ("Statistics", "ANOVA", "Available", "Stat > ANOVA", "Standard Deviations Test, One-Way ANOVA."),
            ("Statistics", "DOE", "Available", "Stat > DOE", "Factorial Design."),
            ("SPC", "Control Charts", "Available", "Stat > Control Charts", "Individuals chart with stages."),
            ("FTA", "FTA Builder", "In progress", "FTA menu", "PDT, FTA and Red X logic tree planned."),
        ]

        for row_idx, (area, tool, status, access, notes) in enumerate(rows, start=1):
            self._add_tool_status_row(grid, row_idx, area, tool, status, access, notes)

        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 2)
        grid.setColumnStretch(2, 1)
        grid.setColumnStretch(3, 2)
        grid.setColumnStretch(4, 3)

        root.addLayout(grid)
        return box

    def _add_tool_status_row(
        self,
        grid: QGridLayout,
        row: int,
        area: str,
        tool: str,
        status: str,
        access: str,
        notes: str,
    ) -> None:
        """Add one row to the home dashboard.

        Parameters
        ----------
        grid : PySide6.QtWidgets.QGridLayout
            Target grid layout.
        row : int
            Row index where the labels are inserted.
        area : str
            Functional area name.
        tool : str
            Tool name.
        status : str
            Tool release status.
        access : str
            Menu or panel path.
        notes : str
            Additional user-facing notes.
        """
        bg = "#FFFFFF" if row % 2 else "#F6F9FC"
        values = [area, tool, status, access, notes]

        for col, value in enumerate(values):
            lbl = QLabel(value)
            lbl.setWordWrap(True)
            lbl.setStyleSheet(f"background:{bg}; padding:6px; color:#1E2A36;")

            if col == 1:
                lbl.setStyleSheet(
                    f"background:{bg}; padding:6px; color:#17324D; font-weight:bold;"
                )

            if col == 2:
                lbl.setAlignment(Qt.AlignCenter)
                lbl.setStyleSheet(self._status_chip_style(status))

            grid.addWidget(lbl, row, col)

    def _status_chip_style(self, status: str) -> str:
        """Return CSS for a module status chip.

        Parameters
        ----------
        status : str
            Tool status label.

        Returns
        -------
        str
            Qt stylesheet fragment.
        """
        if status == "Available":
            return (
                "background:#DFF3E5; color:#1A7F37; font-weight:bold; "
                "border:1px solid #9CD3AA; border-radius:8px; padding:5px;"
            )
        if status == "Core":
            return (
                "background:#E5EEF7; color:#17324D; font-weight:bold; "
                "border:1px solid #AFC2D5; border-radius:8px; padding:5px;"
            )
        return (
            "background:#F1F3F5; color:#68717A; font-weight:bold; "
            "border:1px solid #CDD3DA; border-radius:8px; padding:5px;"
        )

    def make_dock_title(self, text: str) -> QWidget:
        """Create a consistent custom title bar for dock widgets.

        Parameters
        ----------
        text : str
            Title text.

        Returns
        -------
        PySide6.QtWidgets.QWidget
            Styled dock title widget.
        """
        w = QWidget()
        w.setStyleSheet("background:#17324D;")

        layout = QHBoxLayout(w)
        layout.setContentsMargins(8, 4, 8, 4)

        lbl = QLabel(text)
        lbl.setStyleSheet("color:white; font-weight:bold;")
        layout.addWidget(lbl)
        return w

    # ------------------------------------------------------------------
    # Project IO
    # ------------------------------------------------------------------

    def new_project(self) -> None:
        """Create a clean project after confirming unsaved changes."""
        if not self._confirm_unsaved():
            return

        for widget in self.worksheets.values():
            widget.deleteLater()

        self.worksheets.clear()
        self.ws_list.clear()
        self.journal_entries.clear()
        self._refresh_journal()
        self.project_path = None
        self.dirty = False

        self.add_worksheet("Worksheet 1", switch=True)
        self.show_home()

    def save_project(self) -> None:
        """Save the current project to its existing path."""
        if not self.project_path:
            self.save_project_as()
            return
        self._save_to_path(self.project_path)

    def save_project_as(self) -> None:
        """Ask the user for a project path and save the project there."""
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save SPS Studio Project",
            "",
            f"SPS Studio Project (*{EXTENSION});;All Files (*.*)",
        )
        if not path:
            return
        if not path.upper().endswith(EXTENSION):
            path += EXTENSION
        self._save_to_path(path)

    def _save_to_path(self, path: str) -> None:
        """Serialize the current project to disk.

        Parameters
        ----------
        path : str
            Destination project file path.
        """
        data = {
            "app": APP_NAME,
            "version": APP_VERSION,
            "worksheets": {
                name: widget.to_raw_rows()
                for name, widget in self.worksheets.items()
            },
            "active_worksheet": self.current_ws_name,
            "journal_entries": self.journal_entries,
        }

        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr(
                "project.json",
                json.dumps(data, ensure_ascii=False, indent=2, default=str),
            )

        self.project_path = path
        self.dirty = False
        self.statusBar().showMessage(f"Project saved: {path}")

    def open_project(self) -> None:
        """Open an SPS Studio project from disk."""
        if not self._confirm_unsaved():
            return

        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open SPS Studio Project",
            "",
            f"SPS Studio Project (*{EXTENSION});;All Files (*.*)",
        )
        if not path:
            return

        with zipfile.ZipFile(path, "r") as zf:
            data = json.loads(zf.read("project.json").decode("utf-8"))

        for widget in self.worksheets.values():
            widget.deleteLater()

        self.worksheets.clear()
        self.ws_list.clear()

        for name, raw_rows in data.get("worksheets", {}).items():
            self.add_worksheet(name, switch=False, raw_rows=raw_rows)

        if not self.worksheets:
            self.add_worksheet("Worksheet 1", switch=False)

        active = data.get("active_worksheet") or list(self.worksheets.keys())[0]
        self.current_ws_name = active if active in self.worksheets else list(self.worksheets.keys())[0]

        for idx in range(self.ws_list.count()):
            if self.ws_list.item(idx).text() == self.current_ws_name:
                self.ws_list.setCurrentRow(idx)
                break

        self.journal_entries = data.get("journal_entries", [])
        self._refresh_journal()

        self.project_path = path
        self.dirty = False
        self.show_home()
        self.statusBar().showMessage(f"Project opened: {path}")

    # ------------------------------------------------------------------
    # Shared utilities
    # ------------------------------------------------------------------

    def _confirm_unsaved(self) -> bool:
        """Return whether a destructive action can continue.

        Returns
        -------
        bool
            True when the project is clean or the user accepts discarding
            unsaved changes.
        """
        if not self.dirty:
            return True

        answer = QMessageBox.question(
            self,
            "Unsaved changes",
            "This project has unsaved changes. Continue without saving?",
        )
        return answer == QMessageBox.Yes

    def _mark_dirty(self) -> None:
        """Mark the current project as modified."""
        self.dirty = True

    def _now(self) -> str:
        """Return the current local timestamp.

        Returns
        -------
        str
            Timestamp formatted as ``YYYY-MM-DD HH:MM:SS``.
        """
        from datetime import datetime

        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def planned(self, module_name: str) -> None:
        """Show a placeholder dialog for an unreleased module.

        Parameters
        ----------
        module_name : str
            Name of the planned module.
        """
        QMessageBox.information(
            self,
            "Planned Module",
            f"{module_name} is planned for a future release.",
        )

    def about(self) -> None:
        """Display the application About dialog."""
        QMessageBox.information(
            self,
            "About SPS Studio",
            f"{APP_NAME} v{APP_VERSION}\n\n"
            f"{APP_SUBTITLE}\n\n"
            f"Created by:\n{CREATOR}\n\n"
            f"Contact:\n{CONTACT}\n\n"
            f"Project extension: {EXTENSION}",
        )

    # ------------------------------------------------------------------
    # Compatibility facade: worksheet manager
    # ------------------------------------------------------------------

    def add_worksheet_dialog(self) -> None:
        """Delegate worksheet creation dialog to the worksheet manager."""
        self.worksheet_manager.add_worksheet_dialog()

    def add_worksheet(self, name=None, switch=True, raw_rows=None):
        """Delegate worksheet creation to the worksheet manager."""
        return self.worksheet_manager.add_worksheet(name=name, switch=switch, raw_rows=raw_rows)

    def rename_current_worksheet(self) -> None:
        """Delegate active worksheet rename to the worksheet manager."""
        self.worksheet_manager.rename_current_worksheet()

    def delete_current_worksheet(self) -> None:
        """Delegate active worksheet deletion to the worksheet manager."""
        self.worksheet_manager.delete_current_worksheet()

    def show_worksheet(self, name=None) -> None:
        """Delegate worksheet display to the worksheet manager."""
        self.worksheet_manager.show_worksheet(name=name)

    def active_worksheet_name(self) -> str:
        """Return the active worksheet name."""
        return self.worksheet_manager.active_worksheet_name()

    def get_active_dataframe(self):
        """Return the active worksheet as a DataFrame."""
        return self.worksheet_manager.get_active_dataframe()

    # ------------------------------------------------------------------
    # Compatibility facade: journal manager
    # ------------------------------------------------------------------

    def add_journal_entry(self, entry: dict) -> None:
        """Delegate journal entry creation to the journal manager."""
        self.journal_manager.add_journal_entry(entry)

    def update_journal_entry(self, entry_id: str, updates: dict) -> bool:
        """Delegate journal entry update to the journal manager."""
        return self.journal_manager.update_journal_entry(entry_id, updates)

    def find_journal_entry(self, entry_id: str) -> dict | None:
        """Delegate journal entry lookup to the journal manager."""
        return self.journal_manager.find_journal_entry(entry_id)

    def _refresh_journal(self) -> None:
        """Refresh the journal tree when it has already been built."""
        if hasattr(self, "journal_manager"):
            self.journal_manager.refresh_journal()

    def selected_journal_index(self):
        """Return the selected journal index."""
        return self.journal_manager.selected_journal_index()

    def open_selected_journal(self) -> None:
        """Open the currently selected journal entry."""
        self.journal_manager.open_selected_journal()

    def open_journal_entry(self, entry: dict) -> None:
        """Open a specific journal entry."""
        self.journal_manager.open_journal_entry(entry)

    def rename_selected_journal(self) -> None:
        """Rename the selected journal entry."""
        self.journal_manager.rename_selected_journal()

    def delete_selected_journal(self) -> None:
        """Delete the selected journal entry."""
        self.journal_manager.delete_selected_journal()

    # ------------------------------------------------------------------
    # Compatibility facade: menu / module openers
    # ------------------------------------------------------------------

    def open_graph_builder(self, graph_type="Histogram") -> None:
        """Open Graph Builder."""
        self.menu_controller.open_graph_builder(graph_type)

    def open_gage_rr_crossed(self) -> None:
        """Open Gage R&R crossed study."""
        self.menu_controller.open_gage_rr_crossed()

    def open_capability_study(self) -> None:
        """Open Capability Study."""
        self.menu_controller.open_capability_study()

    def open_normality_test(self) -> None:
        """Open Normality Test."""
        self.menu_controller.open_normality_test()

    def open_hypothesis_test(self, study_type="two_sample") -> None:
        """Open Hypothesis Test Builder."""
        self.menu_controller.open_hypothesis_test(study_type)

    def open_simple_regression(self) -> None:
        """Open Simple Regression Builder."""
        self.menu_controller.open_simple_regression()

    def open_multiple_regression(self) -> None:
        """Open Multiple Regression Builder."""
        self.menu_controller.open_multiple_regression()

    def open_anova_analysis(self) -> None:
        """Open ANOVA Analysis Builder."""
        self.menu_controller.open_anova_analysis()

    def open_factorial_doe(self) -> None:
        """Open Factorial DOE Builder."""
        self.menu_controller.open_factorial_doe()

    def open_control_charts(self) -> None:
        """Open Control Charts Builder."""
        self.menu_controller.open_control_charts()

    def open_graph_builder_from_journal(self, entry_id) -> None:
        """Open Graph Builder from a journal entry."""
        self.menu_controller.open_graph_builder_from_journal(entry_id)

    def open_gage_study_from_journal(self, entry_id) -> None:
        """Open Gage Study from a journal entry."""
        self.menu_controller.open_gage_study_from_journal(entry_id)

    def open_capability_study_from_journal(self, entry_id) -> None:
        """Open Capability Study from a journal entry."""
        self.menu_controller.open_capability_study_from_journal(entry_id)

    def open_normality_test_from_journal(self, entry_id) -> None:
        """Open Normality Test from a journal entry."""
        self.menu_controller.open_normality_test_from_journal(entry_id)

    def open_hypothesis_test_from_journal(self, entry_id) -> None:
        """Open Hypothesis Test from a journal entry."""
        self.menu_controller.open_hypothesis_test_from_journal(entry_id)

    def open_simple_regression_from_journal(self, entry_id) -> None:
        """Open Simple Regression from a journal entry."""
        self.menu_controller.open_simple_regression_from_journal(entry_id)

    def open_multiple_regression_from_journal(self, entry_id) -> None:
        """Open Multiple Regression from a journal entry."""
        self.menu_controller.open_multiple_regression_from_journal(entry_id)

    def open_anova_analysis_from_journal(self, entry_id) -> None:
        """Open ANOVA Analysis from a journal entry."""
        self.menu_controller.open_anova_analysis_from_journal(entry_id)

    def open_factorial_doe_from_journal(self, entry_id) -> None:
        """Open Factorial DOE from a journal entry."""
        self.menu_controller.open_factorial_doe_from_journal(entry_id)

    def open_control_charts_from_journal(self, entry_id) -> None:
        """Open Control Charts from a journal entry."""
        self.menu_controller.open_control_charts_from_journal(entry_id)
