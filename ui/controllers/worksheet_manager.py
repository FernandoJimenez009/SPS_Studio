"""Worksheet dock and worksheet state manager for SPS Studio."""

from __future__ import annotations

import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDockWidget,
    QInputDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ui.worksheet_table import WorksheetTable


class WorksheetManager:
    """Manage worksheet UI and worksheet CRUD operations.

    Parameters
    ----------
    app_window : SPSStudioMainWindow
        Main application window that owns shared state.
    max_worksheets : int
        Maximum number of worksheets allowed per project.
    """

    def __init__(self, app_window, max_worksheets: int = 33) -> None:
        """Store dependencies and configuration."""
        self.app = app_window
        self.max_worksheets = max_worksheets

    def build_dock(self) -> None:
        """Create and attach the Worksheets dock."""
        dock = QDockWidget("Worksheets", self.app)
        dock.setTitleBarWidget(self.app.make_dock_title("Worksheets"))
        dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)

        container = QWidget()
        layout = QVBoxLayout(container)

        self.app.btn_home = QPushButton("Home")
        self.app.btn_home.clicked.connect(self.app.show_home)

        self.app.btn_add_ws = QPushButton("+ Add Worksheet")
        self.app.btn_add_ws.clicked.connect(self.add_worksheet_dialog)

        self.app.btn_rename_ws = QPushButton("Rename")
        self.app.btn_rename_ws.clicked.connect(self.rename_current_worksheet)

        self.app.btn_delete_ws = QPushButton("Delete")
        self.app.btn_delete_ws.setObjectName("DangerButton")
        self.app.btn_delete_ws.clicked.connect(self.delete_current_worksheet)

        layout.addWidget(self.app.btn_home)
        layout.addWidget(self.app.btn_add_ws)
        layout.addWidget(self.app.btn_rename_ws)
        layout.addWidget(self.app.btn_delete_ws)

        layout.addWidget(QLabel("Worksheet List"))
        self.app.ws_list = QListWidget()
        self.app.ws_list.currentItemChanged.connect(self._worksheet_selected)
        self.app.ws_list.itemClicked.connect(lambda item: self.show_worksheet(item.text()))
        layout.addWidget(self.app.ws_list)

        dock.setWidget(container)
        self.app.addDockWidget(Qt.LeftDockWidgetArea, dock)

    def add_worksheet_dialog(self) -> None:
        """Ask the user for a worksheet name and create it."""
        if len(self.app.worksheets) >= self.max_worksheets:
            QMessageBox.warning(
                self.app,
                "Worksheet limit",
                f"Maximum number of worksheets is {self.max_worksheets}.",
            )
            return

        name, ok = QInputDialog.getText(
            self.app,
            "Add Worksheet",
            "Worksheet name:",
            text=self._next_ws_name(),
        )
        if not ok or not name.strip():
            return

        self.add_worksheet(name.strip(), switch=True)

    def add_worksheet(self, name=None, switch: bool = True, raw_rows=None) -> None:
        """Create a worksheet and optionally switch to it.

        Parameters
        ----------
        name : str or None, optional
            Worksheet name. When None, the next available default name is used.
        switch : bool, default=True
            Whether to make the new worksheet active.
        raw_rows : list or None, optional
            Raw worksheet rows to load into the table.
        """
        if name is None:
            name = self._next_ws_name()

        if name in self.app.worksheets:
            QMessageBox.warning(
                self.app,
                "Duplicate Worksheet",
                "A worksheet with this name already exists.",
            )
            return

        table = WorksheetTable()
        if raw_rows is not None:
            table.load_raw_rows(raw_rows)
        table.data_changed.connect(self.app._mark_dirty)

        self.app.worksheets[name] = table

        item = QListWidgetItem(name)
        self.app.ws_list.addItem(item)

        if switch:
            self.app.ws_list.setCurrentItem(item)
            self.show_worksheet(name)

        self._update_ws_buttons()
        self.app._mark_dirty()

    def _next_ws_name(self) -> str:
        """Return the next available default worksheet name.

        Returns
        -------
        str
            Available worksheet name.
        """
        for idx in range(1, self.max_worksheets + 1):
            name = f"Worksheet {idx}"
            if name not in self.app.worksheets:
                return name
        return f"Worksheet {len(self.app.worksheets) + 1}"

    def _worksheet_selected(self, current, previous) -> None:
        """Show the selected worksheet from the list.

        Parameters
        ----------
        current : QListWidgetItem or None
            Current selected list item.
        previous : QListWidgetItem or None
            Previous selected list item.
        """
        if current is None:
            return
        self.show_worksheet(current.text())

    def show_worksheet(self, name=None) -> None:
        """Display a worksheet in the central tab area.

        Parameters
        ----------
        name : str or None, optional
            Worksheet name. When None, the current worksheet is used.
        """
        if name is None:
            name = self.app.current_ws_name

        if name not in self.app.worksheets:
            return

        self.app.clear_central()
        self.app.central_tabs.addTab(self.app.worksheets[name], name)
        self.app.current_ws_name = name
        self.app.statusBar().showMessage(f"Active worksheet: {name}")

    def rename_current_worksheet(self) -> None:
        """Rename the active worksheet."""
        if not self.app.current_ws_name:
            return

        old_name = self.app.current_ws_name
        new_name, ok = QInputDialog.getText(
            self.app,
            "Rename Worksheet",
            "New name:",
            text=old_name,
        )

        if not ok or not new_name.strip():
            return

        new_name = new_name.strip()
        if new_name in self.app.worksheets and new_name != old_name:
            QMessageBox.warning(
                self.app,
                "Duplicate Worksheet",
                "A worksheet with this name already exists.",
            )
            return

        self.app.worksheets[new_name] = self.app.worksheets.pop(old_name)
        self.app.current_ws_name = new_name

        for idx in range(self.app.ws_list.count()):
            item = self.app.ws_list.item(idx)
            if item.text() == old_name:
                item.setText(new_name)
                self.app.ws_list.setCurrentItem(item)
                break

        self.app._mark_dirty()
        self.show_worksheet(new_name)

    def delete_current_worksheet(self) -> None:
        """Delete the active worksheet after confirmation."""
        if len(self.app.worksheets) <= 1:
            QMessageBox.information(
                self.app,
                "Delete Worksheet",
                "At least one worksheet must remain.",
            )
            return

        name = self.app.current_ws_name
        if QMessageBox.question(self.app, "Delete Worksheet", f"Delete worksheet '{name}'?") != QMessageBox.Yes:
            return

        widget = self.app.worksheets.pop(name)
        widget.deleteLater()

        for idx in range(self.app.ws_list.count()):
            if self.app.ws_list.item(idx).text() == name:
                self.app.ws_list.takeItem(idx)
                break

        self.app.current_ws_name = self.app.ws_list.item(0).text()
        self.app.ws_list.setCurrentRow(0)
        self.show_worksheet(self.app.current_ws_name)

        self._update_ws_buttons()
        self.app._mark_dirty()

    def _update_ws_buttons(self) -> None:
        """Enable or disable worksheet buttons based on current state."""
        self.app.btn_delete_ws.setEnabled(len(self.app.worksheets) > 1)

    def active_worksheet_name(self) -> str:
        """Return the active worksheet name.

        Returns
        -------
        str
            Active worksheet name or an empty string.
        """
        return self.app.current_ws_name or ""

    def get_active_dataframe(self) -> pd.DataFrame:
        """Return the active worksheet data as a DataFrame.

        Returns
        -------
        pandas.DataFrame
            Active worksheet data, or an empty DataFrame when unavailable.
        """
        if not self.app.current_ws_name or self.app.current_ws_name not in self.app.worksheets:
            return pd.DataFrame()
        return self.app.worksheets[self.app.current_ws_name].to_dataframe()
