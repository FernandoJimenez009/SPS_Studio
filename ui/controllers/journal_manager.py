"""Analysis journal dock and journal entry manager for SPS Studio."""

from __future__ import annotations

import base64

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QDockWidget,
    QHBoxLayout,
    QLabel,
    QMenu,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTextEdit,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)


class JournalManager:
    """Manage the Analysis Journal dock and entry operations.

    Parameters
    ----------
    app_window : SPSStudioMainWindow
        Main application window that owns journal state.
    """

    EDITABLE_TYPES = {
        "Graph",
        "Gage Study",
        "Capability Study",
        "Hypothesis Test",
        "Normality Test",
        "Simple Regression",
        "Multiple Regression",
        "ANOVA Analysis",
        "Factorial DOE",
        "Control Chart",
    }

    def __init__(self, app_window) -> None:
        """Store the main window dependency."""
        self.app = app_window

    def build_dock(self) -> None:
        """Create and attach the Analysis Journal dock."""
        dock = QDockWidget("Analysis Journal", self.app)
        dock.setTitleBarWidget(self.app.make_dock_title("Analysis Journal"))
        dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)

        container = QWidget()
        layout = QVBoxLayout(container)

        toolbar = QHBoxLayout()
        self.app.btn_open_j = QPushButton("Open")
        self.app.btn_rename_j = QPushButton("Rename")
        self.app.btn_delete_j = QPushButton("Delete")
        self.app.btn_delete_j.setObjectName("DangerButton")

        self.app.btn_open_j.clicked.connect(self.open_selected_journal)
        self.app.btn_rename_j.clicked.connect(self.rename_selected_journal)
        self.app.btn_delete_j.clicked.connect(self.delete_selected_journal)

        toolbar.addWidget(self.app.btn_open_j)
        toolbar.addWidget(self.app.btn_rename_j)
        toolbar.addWidget(self.app.btn_delete_j)
        layout.addLayout(toolbar)

        self.app.journal = QTreeWidget()
        self.app.journal.setHeaderLabels(["Name", "Worksheet", "Type", "Status", "Modified"])
        self.app.journal.itemDoubleClicked.connect(lambda item, col: self.open_selected_journal())
        self.app.journal.setContextMenuPolicy(Qt.CustomContextMenu)
        self.app.journal.customContextMenuRequested.connect(self._journal_context_menu)
        layout.addWidget(self.app.journal)

        dock.setWidget(container)
        self.app.addDockWidget(Qt.RightDockWidgetArea, dock)

    def add_journal_entry(self, entry: dict) -> None:
        """Add an entry to the analysis journal.

        Parameters
        ----------
        entry : dict
            Journal entry payload.
        """
        self.app.journal_entries.append(entry)
        self.refresh_journal()
        self.app._mark_dirty()

        idx = self.app.journal.topLevelItemCount() - 1
        if idx >= 0:
            item = self.app.journal.topLevelItem(idx)
            self.app.journal.setCurrentItem(item)
            self.app.journal.scrollToItem(item)

    def update_journal_entry(self, entry_id: str, updates: dict) -> bool:
        """Update a journal entry by identifier.

        Parameters
        ----------
        entry_id : str
            Journal entry identifier.
        updates : dict
            Fields to merge into the entry.

        Returns
        -------
        bool
            True when an entry was updated; otherwise False.
        """
        if not entry_id:
            return False

        for entry in self.app.journal_entries:
            if entry.get("id") == entry_id:
                entry.update(updates)
                entry["modified_at"] = self.app._now()
                self.refresh_journal()
                self.app._mark_dirty()
                return True

        return False

    def find_journal_entry(self, entry_id: str) -> dict | None:
        """Find a journal entry by identifier.

        Parameters
        ----------
        entry_id : str
            Journal entry identifier.

        Returns
        -------
        dict or None
            Matching journal entry, or None when not found.
        """
        if not entry_id:
            return None
        for entry in self.app.journal_entries:
            if entry.get("id") == entry_id:
                return entry
        return None

    def refresh_journal(self) -> None:
        """Refresh the journal tree widget from journal state."""
        if not hasattr(self.app, "journal"):
            return

        self.app.journal.clear()
        for entry in self.app.journal_entries:
            item = QTreeWidgetItem(
                [
                    entry.get("name", "Unnamed"),
                    entry.get("worksheet", ""),
                    entry.get("type", ""),
                    entry.get("status", ""),
                    entry.get("modified_at", entry.get("created_at", "")),
                ]
            )
            self.app.journal.addTopLevelItem(item)

    def selected_journal_index(self):
        """Return the selected top-level journal index.

        Returns
        -------
        int or None
            Selected index, or None when no item is selected.
        """
        item = self.app.journal.currentItem()
        if item is None:
            return None
        return self.app.journal.indexOfTopLevelItem(item)

    def open_selected_journal(self) -> None:
        """Open the selected journal entry."""
        idx = self.selected_journal_index()
        if idx is None:
            return

        entry = self.app.journal_entries[idx]
        self.open_journal_entry(entry)

    def open_journal_entry(self, entry: dict) -> None:
        """Open a journal entry preview in the central area.

        Parameters
        ----------
        entry : dict
            Journal entry to display.
        """
        view = QWidget()
        layout = QVBoxLayout(view)
        layout.setContentsMargins(12, 12, 12, 12)

        title = QLabel(entry.get("name", "Analysis"))
        title.setObjectName("TitleLabel")
        layout.addWidget(title)

        meta = QLabel(
            f"Type: {entry.get('type', '')} | "
            f"Worksheet: {entry.get('worksheet', '')} | "
            f"Status: {entry.get('status', '')} | "
            f"Modified: {entry.get('modified_at', entry.get('created_at', ''))}"
        )
        meta.setObjectName("SubtitleLabel")
        layout.addWidget(meta)

        splitter = QSplitter(Qt.Vertical)
        image_area = self._build_image_area(entry)

        summary = QTextEdit()
        summary.setReadOnly(True)
        summary.setPlainText(entry.get("summary", "No summary available."))

        splitter.addWidget(image_area)
        splitter.addWidget(summary)
        splitter.setStretchFactor(0, 6)
        splitter.setStretchFactor(1, 4)
        layout.addWidget(splitter)

        layout.addLayout(self._build_entry_toolbar(entry, summary))

        self.app.clear_central()
        self.app.central_tabs.addTab(view, "Journal Entry")
        self.app.statusBar().showMessage(f"Opened journal entry: {entry.get('name', 'Analysis')}")

    def _build_image_area(self, entry: dict) -> QScrollArea:
        """Build the journal image preview area.

        Parameters
        ----------
        entry : dict
            Journal entry to preview.

        Returns
        -------
        PySide6.QtWidgets.QScrollArea
            Scroll area containing a graph image or fallback text.
        """
        image_area = QScrollArea()
        image_area.setWidgetResizable(True)

        image_container = QWidget()
        image_layout = QVBoxLayout(image_container)
        image_layout.setContentsMargins(8, 8, 8, 8)

        image_label = QLabel()
        image_label.setAlignment(Qt.AlignCenter)
        image_label.setText("No graph image available for this journal entry.")
        image_label.setMinimumHeight(360)

        pix = self._pixmap_from_base64(entry.get("figure_png_base64", ""))
        if pix and not pix.isNull():
            display_pix = self._scaled_journal_pixmap(pix)
            image_label.setPixmap(display_pix)
            image_label.setMinimumHeight(min(560, max(360, display_pix.height() + 24)))

        image_layout.addWidget(image_label)
        image_layout.addStretch()
        image_area.setWidget(image_container)
        return image_area

    def _build_entry_toolbar(self, entry: dict, summary: QTextEdit) -> QHBoxLayout:
        """Create journal entry action buttons.

        Parameters
        ----------
        entry : dict
            Active journal entry.
        summary : PySide6.QtWidgets.QTextEdit
            Summary text widget.

        Returns
        -------
        PySide6.QtWidgets.QHBoxLayout
            Toolbar layout.
        """
        toolbar = QHBoxLayout()

        btn_copy = QPushButton("Copy Summary")
        btn_copy.clicked.connect(lambda: self._copy_text_to_clipboard(summary.toPlainText()))

        btn_export = QPushButton("Export Image")
        btn_export.clicked.connect(lambda: self._export_journal_image(entry))

        btn_edit = QPushButton("Edit Analysis")
        btn_edit.setObjectName("PrimaryButton")
        btn_edit.setEnabled(entry.get("type") in self.EDITABLE_TYPES)
        btn_edit.clicked.connect(lambda: self._edit_journal_entry(entry))

        btn_close = QPushButton("Close View")
        btn_close.clicked.connect(self.app.show_home)

        toolbar.addWidget(btn_copy)
        toolbar.addWidget(btn_export)
        toolbar.addWidget(btn_edit)
        toolbar.addStretch()
        toolbar.addWidget(btn_close)
        return toolbar

    def _edit_journal_entry(self, entry: dict) -> None:
        """Route a journal entry to the correct builder.

        Parameters
        ----------
        entry : dict
            Journal entry to edit.
        """
        route = {
            "Graph": self.app.open_graph_builder_from_journal,
            "Gage Study": self.app.open_gage_study_from_journal,
            "Capability Study": self.app.open_capability_study_from_journal,
            "Hypothesis Test": self.app.open_hypothesis_test_from_journal,
            "Simple Regression": self.app.open_simple_regression_from_journal,
            "Normality Test": self.app.open_normality_test_from_journal,
            "ANOVA Analysis": self.app.open_anova_analysis_from_journal,
            "Multiple Regression": self.app.open_multiple_regression_from_journal,
            "Factorial DOE": self.app.open_factorial_doe_from_journal,
            "Control Chart": self.app.open_control_charts_from_journal,
        }
        opener = route.get(entry.get("type"))
        if opener:
            opener(entry.get("id"))

    def _scaled_journal_pixmap(self, pix: QPixmap) -> QPixmap:
        """Scale a journal image to a readable preview size.

        Parameters
        ----------
        pix : PySide6.QtGui.QPixmap
            Source image.

        Returns
        -------
        PySide6.QtGui.QPixmap
            Original or scaled pixmap.
        """
        if pix is None or pix.isNull():
            return pix

        max_w = 1100
        max_h = 520
        if pix.width() <= max_w and pix.height() <= max_h:
            return pix

        return pix.scaled(max_w, max_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)

    def _pixmap_from_base64(self, encoded: str):
        """Create a PNG pixmap from a base64 string.

        Parameters
        ----------
        encoded : str
            Base64-encoded PNG data.

        Returns
        -------
        PySide6.QtGui.QPixmap or None
            Decoded pixmap, or None when decoding fails.
        """
        if not encoded:
            return None

        try:
            raw = base64.b64decode(encoded)
            pix = QPixmap()
            if pix.loadFromData(raw, "PNG"):
                return pix
        except Exception:
            return None

        return None

    def _copy_text_to_clipboard(self, text: str) -> None:
        """Copy text to the system clipboard.

        Parameters
        ----------
        text : str
            Text to copy.
        """
        QApplication.clipboard().setText(text or "")
        self.app.statusBar().showMessage("Summary copied to clipboard.")

    def _export_journal_image(self, entry: dict) -> None:
        """Export the journal entry graph image as PNG.

        Parameters
        ----------
        entry : dict
            Journal entry containing ``figure_png_base64``.
        """
        encoded = entry.get("figure_png_base64", "")
        if not encoded:
            QMessageBox.information(
                self.app,
                "No image",
                "This journal entry does not contain a graph image.",
            )
            return

        path, _ = QFileDialog.getSaveFileName(
            self.app,
            "Export Journal Image",
            "",
            "PNG Image (*.png)",
        )
        if not path:
            return
        if not path.lower().endswith(".png"):
            path += ".png"

        try:
            raw = base64.b64decode(encoded)
            with open(path, "wb") as file_obj:
                file_obj.write(raw)
            self.app.statusBar().showMessage(f"Journal image exported: {path}")
        except Exception as exc:
            QMessageBox.critical(self.app, "Export failed", str(exc))

    def rename_selected_journal(self) -> None:
        """Rename the selected journal entry."""
        idx = self.selected_journal_index()
        if idx is None:
            return

        entry = self.app.journal_entries[idx]
        from PySide6.QtWidgets import QInputDialog

        name, ok = QInputDialog.getText(
            self.app,
            "Rename Analysis",
            "New name:",
            text=entry.get("name", ""),
        )
        if not ok or not name.strip():
            return

        entry["name"] = name.strip()
        entry["modified_at"] = self.app._now()
        self.refresh_journal()
        self.app._mark_dirty()

    def delete_selected_journal(self) -> None:
        """Delete the selected journal entry after confirmation."""
        idx = self.selected_journal_index()
        if idx is None:
            return

        entry = self.app.journal_entries[idx]
        if QMessageBox.question(
            self.app,
            "Delete Analysis",
            f"Delete '{entry.get('name')}' from journal?",
        ) != QMessageBox.Yes:
            return

        del self.app.journal_entries[idx]
        self.refresh_journal()
        self.app._mark_dirty()

    def _journal_context_menu(self, pos) -> None:
        """Display the context menu for the selected journal item.

        Parameters
        ----------
        pos : PySide6.QtCore.QPoint
            Local position where the context menu was requested.
        """
        menu = QMenu(self.app)
        menu.addAction("Open", self.open_selected_journal)

        idx = self.selected_journal_index()
        if idx is not None:
            entry = self.app.journal_entries[idx]
            edit_label = self._edit_label(entry.get("type"))
            if edit_label:
                menu.addAction(edit_label, lambda entry=entry: self._edit_journal_entry(entry))

        menu.addAction("Rename", self.rename_selected_journal)
        menu.addAction("Delete", self.delete_selected_journal)
        menu.exec(self.app.journal.mapToGlobal(pos))

    def _edit_label(self, entry_type: str | None) -> str | None:
        """Return the context-menu edit label for an entry type.

        Parameters
        ----------
        entry_type : str or None
            Journal entry type.

        Returns
        -------
        str or None
            Context-menu label, or None for unsupported types.
        """
        labels = {
            "Graph": "Edit Graph",
            "Gage Study": "Edit Gage Study",
            "Capability Study": "Edit Capability Study",
            "Hypothesis Test": "Edit Hypothesis Test",
            "Simple Regression": "Edit Simple Regression",
            "Normality Test": "Edit Normality Test",
            "ANOVA Analysis": "Edit ANOVA Analysis",
            "Multiple Regression": "Edit Multiple Regression",
            "Factorial DOE": "Edit Factorial DOE",
            "Control Chart": "Edit Control Chart",
        }
        return labels.get(entry_type)
