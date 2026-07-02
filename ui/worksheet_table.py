import re
from typing import List, Tuple

import pandas as pd
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QBrush, QColor, QFont, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QAbstractItemView,
    QHBoxLayout,
    QLabel,
    QMenu,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


class WorksheetTable(QWidget):
    """
    Excel-like worksheet table.

    Behavior:
    - First row is the variable/header row.
    - Header row text is automatically bold.
    - Row numbering starts at the first data row:
        row 0 = header row, vertical label = blank
        row 1 = data row 1, vertical label = 1
    - Ctrl+V pastes Excel/tab-delimited blocks.
    - Ctrl+C copies selected cells or complete selected columns.
    - Right-click supports copy, paste, clear, delete column and insert column.
    - Duplicate column names are prevented.
    - Visual column IDs show "-T" when the data below the header row is text/categorical.
    """

    data_changed = Signal()

    HEADER_ROW = 0
    DEFAULT_HEADER_PREFIX = "C"

    def __init__(self, rows=120, columns=12, parent=None):
        super().__init__(parent)
        self.default_rows = int(rows)
        self.default_columns = int(columns)
        self._internal_change = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        top = QHBoxLayout()
        self.info_label = QLabel(
            "First row = variable names. Data row numbering starts below the header row. "
            "Use Ctrl+V / Ctrl+C or right-click for worksheet actions."
        )
        top.addWidget(self.info_label)
        top.addStretch()

        self.btn_add_rows = QPushButton("+ Rows")
        self.btn_add_cols = QPushButton("+ Columns")
        self.btn_clear = QPushButton("Clear")
        top.addWidget(self.btn_add_rows)
        top.addWidget(self.btn_add_cols)
        top.addWidget(self.btn_clear)
        layout.addLayout(top)

        self.table = QTableWidget(rows, columns)
        self.table.setAlternatingRowColors(False)
        self.table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.table.setSelectionBehavior(QAbstractItemView.SelectItems)
        self.table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self._show_table_context_menu)

        self.table.horizontalHeader().setSectionsClickable(True)
        self.table.horizontalHeader().setContextMenuPolicy(Qt.CustomContextMenu)
        self.table.horizontalHeader().customContextMenuRequested.connect(self._show_header_context_menu)

        self.table.setHorizontalHeaderLabels([f"C{i + 1}" for i in range(columns)])
        self._refresh_vertical_headers()
        self._ensure_header_row_items()
        self._style_header_row()
        self._apply_all_alignments()

        self.table.itemChanged.connect(self._on_item_changed)
        layout.addWidget(self.table)

        self.btn_add_rows.clicked.connect(lambda: self.add_rows(20))
        self.btn_add_cols.clicked.connect(lambda: self.add_columns(5))
        self.btn_clear.clicked.connect(self.clear_table)

        self.copy_shortcut = QShortcut(QKeySequence.Copy, self.table)
        self.copy_shortcut.activated.connect(self.copy_selection_to_clipboard)

        self.paste_shortcut = QShortcut(QKeySequence.Paste, self.table)
        self.paste_shortcut.activated.connect(self.paste_from_clipboard)

        self.delete_shortcut = QShortcut(QKeySequence.Delete, self.table)
        self.delete_shortcut.activated.connect(self.clear_selection)

    # ─────────────────────────────────────────────
    # Styling / labels
    # ─────────────────────────────────────────────

    def _refresh_vertical_headers(self):
        labels = [""]
        labels.extend(str(i) for i in range(1, self.table.rowCount()))
        self.table.setVerticalHeaderLabels(labels[: self.table.rowCount()])

    def _ensure_header_row_items(self):
        self.table.blockSignals(True)
        for c in range(self.table.columnCount()):
            item = self.table.item(self.HEADER_ROW, c)
            if item is None:
                item = QTableWidgetItem("")
                self.table.setItem(self.HEADER_ROW, c, item)
        self.table.blockSignals(False)

    def _style_header_item(self, item: QTableWidgetItem):
        if item is None:
            return
        font = item.font()
        font.setBold(True)
        item.setFont(font)
        item.setBackground(QBrush(QColor("#EEF3F8")))
        item.setForeground(QBrush(QColor("#102033")))
        item.setTextAlignment(Qt.AlignCenter)

    def _style_header_row(self):
        self._ensure_header_row_items()
        self.table.blockSignals(True)
        for c in range(self.table.columnCount()):
            self._style_header_item(self.table.item(self.HEADER_ROW, c))
        self.table.blockSignals(False)

    def _is_numeric_text(self, text: str) -> bool:
        text = str(text).strip()
        if not text:
            return False
        normalized = text.replace(",", ".")
        try:
            float(normalized)
            return True
        except Exception:
            return False

    def _column_kind(self, col: int) -> str:
        """
        Return the visual/data kind for a worksheet column using data rows only.

        Returns:
        - "empty"   : no data values below the header row
        - "numeric" : all non-empty data values are numeric
        - "text"    : at least one non-empty data value is non-numeric

        The first row is intentionally ignored because it is the variable name row.
        """
        has_value = False
        for r in range(self.HEADER_ROW + 1, self.table.rowCount()):
            item = self.table.item(r, col)
            text = item.text().strip() if item else ""
            if not text:
                continue
            has_value = True
            if not self._is_numeric_text(text):
                return "text"
        return "numeric" if has_value else "empty"

    def _is_numeric_column(self, col: int) -> bool:
        return self._column_kind(col) == "numeric"

    def _visual_column_label(self, col: int) -> str:
        base = f"{self.DEFAULT_HEADER_PREFIX}{col + 1}"
        return f"{base}-T" if self._column_kind(col) == "text" else base

    def _set_horizontal_header_label(self, col: int):
        if col < 0 or col >= self.table.columnCount():
            return

        header_item = self.table.horizontalHeaderItem(col)
        if header_item is None:
            header_item = QTableWidgetItem()
            self.table.setHorizontalHeaderItem(col, header_item)

        header_item.setText(self._visual_column_label(col))
        header_item.setTextAlignment(Qt.AlignCenter)
        font = header_item.font()
        font.setBold(True)
        header_item.setFont(font)

    def _refresh_horizontal_headers(self):
        self.table.blockSignals(True)
        for c in range(self.table.columnCount()):
            self._set_horizontal_header_label(c)
        self.table.blockSignals(False)

    def _apply_column_alignment(self, col: int):
        if col < 0 or col >= self.table.columnCount():
            return

        numeric_column = self._is_numeric_column(col)
        data_alignment = (Qt.AlignRight | Qt.AlignVCenter) if numeric_column else (Qt.AlignLeft | Qt.AlignVCenter)

        self._internal_change = True
        self.table.blockSignals(True)
        self._set_horizontal_header_label(col)
        header_item = self.table.item(self.HEADER_ROW, col)
        if header_item is not None:
            self._style_header_item(header_item)

        for r in range(self.HEADER_ROW + 1, self.table.rowCount()):
            item = self.table.item(r, col)
            if item is None:
                continue
            item.setTextAlignment(data_alignment)
        self.table.blockSignals(False)
        self._internal_change = False

    def _apply_all_alignments(self):
        self._internal_change = True
        self.table.blockSignals(True)
        for c in range(self.table.columnCount()):
            self._set_horizontal_header_label(c)
            header_item = self.table.item(self.HEADER_ROW, c)
            if header_item is not None:
                self._style_header_item(header_item)
        self.table.blockSignals(False)
        self._internal_change = False

        for c in range(self.table.columnCount()):
            self._apply_column_alignment(c)

    # ─────────────────────────────────────────────
    # Events / validation
    # ─────────────────────────────────────────────

    def _on_item_changed(self, item: QTableWidgetItem):
        if self._internal_change:
            return

        if item.row() == self.HEADER_ROW:
            self._style_header_item(item)
            self._validate_header_item(item)
        else:
            self._apply_column_alignment(item.column())

        self.data_changed.emit()

    def _emit_change(self):
        self.data_changed.emit()

    def _header_text(self, col: int) -> str:
        item = self.table.item(self.HEADER_ROW, col)
        text = item.text().strip() if item else ""
        return text if text else f"C{col + 1}"

    def _all_header_texts(self) -> List[str]:
        return [self._header_text(c) for c in range(self.table.columnCount())]

    @staticmethod
    def _normalize_name(name: str) -> str:
        return str(name).strip().lower()

    @staticmethod
    def _split_base_counter(name: str) -> Tuple[str, int]:
        match = re.match(r"^(.*?)(\d+)$", str(name).strip())
        if match:
            base = match.group(1).strip()
            number = int(match.group(2))
            return base if base else str(name).strip(), number
        return str(name).strip(), 0

    def _make_unique_name(self, proposed: str, existing: List[str], current_col: int = -1) -> str:
        proposed = str(proposed).strip()
        if not proposed:
            proposed = f"C{current_col + 1}" if current_col >= 0 else "C"

        existing_norm = {self._normalize_name(x) for x in existing if str(x).strip()}
        if self._normalize_name(proposed) not in existing_norm:
            return proposed

        base, _ = self._split_base_counter(proposed)
        counter = 1
        while True:
            candidate = f"{base}{counter}"
            if self._normalize_name(candidate) not in existing_norm:
                return candidate
            counter += 1

    def _validate_header_item(self, item: QTableWidgetItem):
        col = item.column()
        proposed = item.text().strip()
        existing = [self._header_text(c) for c in range(self.table.columnCount()) if c != col]
        unique = self._make_unique_name(proposed, existing, current_col=col)

        if proposed != unique:
            QMessageBox.warning(
                self,
                "Duplicate Column Name",
                f"Column name '{proposed}' already exists. It was renamed to '{unique}'.",
            )
            self._internal_change = True
            item.setText(unique)
            self._style_header_item(item)
            self._internal_change = False

    def _deduplicate_headers(self, notify=False):
        existing: List[str] = []
        changed = []
        self._internal_change = True
        self.table.blockSignals(True)
        self._ensure_header_row_items()
        for c in range(self.table.columnCount()):
            item = self.table.item(self.HEADER_ROW, c)
            proposed = item.text().strip() if item else ""
            unique = self._make_unique_name(proposed, existing, current_col=c)
            if item is None:
                item = QTableWidgetItem(unique)
                self.table.setItem(self.HEADER_ROW, c, item)
            elif item.text().strip() != unique:
                changed.append((item.text().strip(), unique))
                item.setText(unique)
            self._style_header_item(item)
            existing.append(unique)
        self.table.blockSignals(False)
        self._internal_change = False

        if notify and changed:
            detail = "\n".join(f"'{old}' → '{new}'" for old, new in changed[:8])
            if len(changed) > 8:
                detail += "\n..."
            QMessageBox.information(
                self,
                "Column Names Adjusted",
                "Duplicate column names were automatically renamed:\n\n" + detail,
            )

    # ─────────────────────────────────────────────
    # Clipboard
    # ─────────────────────────────────────────────

    def _selected_rectangle(self):
        ranges = self.table.selectedRanges()
        if not ranges:
            r = self.table.currentRow()
            c = self.table.currentColumn()
            if r < 0 or c < 0:
                return None
            return r, c, r, c

        top = min(r.topRow() for r in ranges)
        left = min(r.leftColumn() for r in ranges)
        bottom = max(r.bottomRow() for r in ranges)
        right = max(r.rightColumn() for r in ranges)
        return top, left, bottom, right

    def copy_selection_to_clipboard(self):
        rect = self._selected_rectangle()
        if rect is None:
            return

        top, left, bottom, right = rect
        lines = []
        for r in range(top, bottom + 1):
            values = []
            for c in range(left, right + 1):
                item = self.table.item(r, c)
                values.append(item.text() if item else "")
            lines.append("\t".join(values))

        QApplication.clipboard().setText("\n".join(lines))

    def paste_from_clipboard(self):
        text = QApplication.clipboard().text()
        if not text or not text.strip():
            return

        start_row = self.table.currentRow()
        start_col = self.table.currentColumn()
        if start_row < 0:
            start_row = 0
        if start_col < 0:
            start_col = 0

        rows = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
        if rows and rows[-1] == "":
            rows = rows[:-1]

        parsed = [row.split("\t") for row in rows]
        if not parsed:
            return

        self.ensure_size(start_row + len(parsed), start_col + max(len(r) for r in parsed))

        self._internal_change = True
        self.table.blockSignals(True)
        for r, row_data in enumerate(parsed):
            target_row = start_row + r
            for c, value in enumerate(row_data):
                target_col = start_col + c
                item = self.table.item(target_row, target_col)
                if item is None:
                    item = QTableWidgetItem("")
                    self.table.setItem(target_row, target_col, item)
                item.setText(value.strip())
                if target_row == self.HEADER_ROW:
                    self._style_header_item(item)
        self.table.blockSignals(False)
        self._internal_change = False

        # If pasted data touched the header row, silently auto-rename duplicates.
        if start_row <= self.HEADER_ROW <= start_row + len(parsed) - 1:
            self._deduplicate_headers(notify=False)
        self._style_header_row()
        self._apply_all_alignments()
        self.data_changed.emit()

    # ─────────────────────────────────────────────
    # Context menus
    # ─────────────────────────────────────────────

    def _show_table_context_menu(self, pos):
        menu = QMenu(self)
        menu.addAction("Copy", self.copy_selection_to_clipboard)
        menu.addAction("Paste", self.paste_from_clipboard)
        menu.addAction("Clear Selection", self.clear_selection)
        menu.addSeparator()

        col = self.table.columnAt(pos.x())
        if col >= 0:
            menu.addAction("Copy Column", lambda c=col: self.copy_column(c))
            menu.addAction("Insert Column After", lambda c=col: self.insert_column_after(c))
            menu.addAction("Delete Column", lambda c=col: self.delete_column(c))

        menu.exec(self.table.viewport().mapToGlobal(pos))

    def _show_header_context_menu(self, pos):
        header = self.table.horizontalHeader()
        col = header.logicalIndexAt(pos)
        if col < 0:
            return

        menu = QMenu(self)
        menu.addAction("Copy Column", lambda c=col: self.copy_column(c))
        menu.addAction("Insert Column After", lambda c=col: self.insert_column_after(c))
        menu.addAction("Delete Column", lambda c=col: self.delete_column(c))
        menu.exec(header.mapToGlobal(pos))

    def copy_column(self, col: int):
        if col < 0 or col >= self.table.columnCount():
            return
        values = []
        for r in range(self.table.rowCount()):
            item = self.table.item(r, col)
            values.append(item.text() if item else "")
        # Trim trailing empty rows but keep header.
        while len(values) > 1 and not values[-1].strip():
            values.pop()
        QApplication.clipboard().setText("\n".join(values))

    def clear_selection(self):
        selected = self.table.selectedIndexes()
        if not selected:
            return

        self._internal_change = True
        self.table.blockSignals(True)
        for index in selected:
            item = self.table.item(index.row(), index.column())
            if item is None:
                item = QTableWidgetItem("")
                self.table.setItem(index.row(), index.column(), item)
            item.setText("")
            if index.row() == self.HEADER_ROW:
                self._style_header_item(item)
        self.table.blockSignals(False)
        self._internal_change = False
        self._deduplicate_headers(notify=False)
        self._apply_all_alignments()
        self.data_changed.emit()

    def insert_column_after(self, col: int):
        insert_at = min(max(col + 1, 0), self.table.columnCount())
        self.table.insertColumn(insert_at)
        self.table.setHorizontalHeaderItem(insert_at, QTableWidgetItem(f"C{insert_at + 1}"))
        self._ensure_header_row_items()
        item = self.table.item(self.HEADER_ROW, insert_at)
        item.setText(self._make_unique_name(f"C{insert_at + 1}", self._all_header_texts(), current_col=insert_at))
        self._style_header_row()
        self._apply_all_alignments()
        self._refresh_horizontal_headers()
        self.data_changed.emit()

    def delete_column(self, col: int):
        if self.table.columnCount() <= 1:
            QMessageBox.information(self, "Delete Column", "At least one column must remain.")
            return

        name = self._header_text(col)
        if QMessageBox.question(self, "Delete Column", f"Delete column '{name}'?") != QMessageBox.Yes:
            return

        self.table.removeColumn(col)
        self._refresh_horizontal_headers()
        self._deduplicate_headers(notify=False)
        self._apply_all_alignments()
        self.data_changed.emit()

    # ─────────────────────────────────────────────
    # Size / worksheet actions
    # ─────────────────────────────────────────────

    def ensure_size(self, rows, columns):
        if rows > self.table.rowCount():
            self.table.setRowCount(rows)
            self._refresh_vertical_headers()
        if columns > self.table.columnCount():
            old_cols = self.table.columnCount()
            self.table.setColumnCount(columns)
            self._refresh_horizontal_headers()
            self._ensure_header_row_items()
            for c in range(old_cols, columns):
                item = self.table.item(self.HEADER_ROW, c)
                if item is None:
                    item = QTableWidgetItem("")
                    self.table.setItem(self.HEADER_ROW, c, item)
                item.setText(self._make_unique_name(f"C{c + 1}", self._all_header_texts()[:c], current_col=c))
            self._style_header_row()
            self._apply_all_alignments()

    def add_rows(self, n=20):
        rows = self.table.rowCount() + int(n)
        self.table.setRowCount(rows)
        self._refresh_vertical_headers()
        self._ensure_header_row_items()
        self._style_header_row()
        self._apply_all_alignments()
        self.data_changed.emit()

    def add_columns(self, n=5):
        old_cols = self.table.columnCount()
        cols = old_cols + int(n)
        self.table.setColumnCount(cols)
        self._refresh_horizontal_headers()
        self._ensure_header_row_items()
        for c in range(old_cols, cols):
            existing = [self._header_text(i) for i in range(c)]
            self.table.item(self.HEADER_ROW, c).setText(self._make_unique_name(f"C{c + 1}", existing, current_col=c))
        self._style_header_row()
        self._apply_all_alignments()
        self.data_changed.emit()

    def clear_table(self):
        if QMessageBox.question(self, "Clear Worksheet", "Clear all worksheet data?") != QMessageBox.Yes:
            return
        self._internal_change = True
        self.table.blockSignals(True)
        self.table.clearContents()
        self._ensure_header_row_items()
        self.table.blockSignals(False)
        self._internal_change = False
        self._style_header_row()
        self._apply_all_alignments()
        self.data_changed.emit()

    # ─────────────────────────────────────────────
    # Serialization / dataframe
    # ─────────────────────────────────────────────

    def to_raw_rows(self):
        rows = []
        max_row_used = -1
        max_col_used = -1
        for r in range(self.table.rowCount()):
            row = []
            row_has_data = False
            for c in range(self.table.columnCount()):
                item = self.table.item(r, c)
                val = item.text() if item else ""
                if val.strip():
                    row_has_data = True
                    max_col_used = max(max_col_used, c)
                row.append(val)
            if row_has_data:
                max_row_used = r
            rows.append(row)
        if max_row_used < 0:
            return [[""]]
        return [row[: max_col_used + 1] for row in rows[: max_row_used + 1]]

    def load_raw_rows(self, rows):
        if not rows:
            rows = [[""]]
        n_rows = max(self.default_rows, len(rows))
        n_cols = max(self.default_columns, max(len(r) for r in rows))

        self._internal_change = True
        self.table.blockSignals(True)
        self.table.setRowCount(n_rows)
        self.table.setColumnCount(n_cols)
        self._refresh_horizontal_headers()
        self._refresh_vertical_headers()
        self.table.clearContents()
        self._ensure_header_row_items()
        for r, row in enumerate(rows):
            for c, value in enumerate(row):
                if value != "":
                    item = QTableWidgetItem(str(value))
                    self.table.setItem(r, c, item)
        self.table.blockSignals(False)
        self._internal_change = False
        self._deduplicate_headers(notify=False)
        self._style_header_row()
        self._apply_all_alignments()
        self.data_changed.emit()

    def to_dataframe(self):
        raw = self.to_raw_rows()
        if not raw or len(raw) < 2:
            return pd.DataFrame()

        headers = []
        existing = []
        for i, h in enumerate(raw[0]):
            name = str(h).strip() if str(h).strip() else f"C{i + 1}"
            unique = self._make_unique_name(name, existing, current_col=i)
            headers.append(unique)
            existing.append(unique)

        df = pd.DataFrame(raw[1:], columns=headers)
        for col in df.columns:
            numeric = pd.to_numeric(df[col].astype(str).str.replace(",", ".", regex=False), errors="coerce")
            if numeric.notna().sum() > 0:
                df[col] = numeric
            else:
                df[col] = df[col].astype(str)
        return df.dropna(how="all")

    def get_numeric_columns(self):
        df = self.to_dataframe()
        if df.empty:
            return []
        return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

    def get_all_columns(self):
        df = self.to_dataframe()
        return list(df.columns) if not df.empty else []
