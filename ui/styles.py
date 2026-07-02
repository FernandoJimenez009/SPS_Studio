from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QPalette, QColor


def apply_app_style(app: QApplication):
    app.setStyle("Fusion")

    palette = QPalette()
    palette.setColor(QPalette.Window, QColor("#EEF1F4"))
    palette.setColor(QPalette.WindowText, QColor("#17202A"))
    palette.setColor(QPalette.Base, QColor("#FFFFFF"))
    palette.setColor(QPalette.AlternateBase, QColor("#FFFFFF"))
    palette.setColor(QPalette.Text, QColor("#17202A"))
    palette.setColor(QPalette.Button, QColor("#F8FAFC"))
    palette.setColor(QPalette.ButtonText, QColor("#17202A"))
    palette.setColor(QPalette.Highlight, QColor("#D6E8FA"))
    palette.setColor(QPalette.HighlightedText, QColor("#17202A"))
    app.setPalette(palette)

    app.setStyleSheet("""
    QMainWindow {
        background: #EEF1F4;
        color: #17202A;
    }

    QWidget {
        background: #EEF1F4;
        color: #17202A;
        font-family: Segoe UI;
        font-size: 9pt;
    }

    QMenuBar {
        background: #17324D;
        color: #FFFFFF;
        padding: 4px;
        font-size: 9pt;
        font-weight: 600;
    }

    QMenuBar::item {
        padding: 6px 12px;
        background: transparent;
        color: #FFFFFF;
    }

    QMenuBar::item:selected,
    QMenuBar::item:pressed {
        background: #2F5D87;
        color: #FFFFFF;
    }

    QMenu {
        background: #FFFFFF;
        color: #17202A;
        border: 1px solid #C8D0DA;
    }

    QMenu::item {
        padding: 6px 22px;
        color: #17202A;
        background: #FFFFFF;
    }

    QMenu::item:selected {
        background: #D6E8FA;
        color: #17202A;
    }

    QDockWidget {
        background: #EEF1F4;
        color: #17202A;
        font-weight: bold;
    }

    QDockWidget::title {
        background: #17324D;
        color: #FFFFFF;
        padding: 8px;
        text-align: left;
        font-weight: bold;
    }

    QLabel {
        background: transparent;
        color: #17202A;
    }

    QLabel#TitleLabel {
        font-size: 24pt;
        font-weight: bold;
        color: #17324D;
        background: transparent;
    }

    QLabel#SubtitleLabel {
        font-size: 11pt;
        color: #627080;
        background: transparent;
    }

    QLabel#SectionTitle {
        font-size: 12pt;
        font-weight: bold;
        color: #17324D;
        background: transparent;
    }

    QGroupBox {
        background: #FFFFFF;
        color: #17324D;
        border: 1px solid #D4DAE2;
        border-radius: 10px;
        margin-top: 12px;
        padding: 10px;
        font-weight: bold;
    }

    QGroupBox::title {
        subcontrol-origin: margin;
        left: 12px;
        padding: 0px 4px;
        background: #FFFFFF;
        color: #17324D;
        font-weight: bold;
    }

    QPushButton {
        background: #F8FAFC;
        color: #17202A;
        border: 1px solid #C8D0DA;
        border-radius: 7px;
        padding: 7px 10px;
    }

    QPushButton:hover {
        background: #E8EEF5;
        color: #17202A;
    }

    QPushButton:pressed {
        background: #D6E8FA;
        color: #17202A;
    }

    QPushButton:disabled {
        background: #D7DCE2;
        color: #777777;
        border: 1px solid #C8D0DA;
    }

    QPushButton#PrimaryButton {
        background: #2F5D87;
        color: #FFFFFF;
        border: 1px solid #17324D;
        font-weight: bold;
        padding: 10px 14px;
    }

    QPushButton#PrimaryButton:hover {
        background: #17324D;
        color: #FFFFFF;
    }

    QPushButton#DangerButton {
        background: #B42318;
        color: #FFFFFF;
        border: 1px solid #8C1D14;
        font-weight: bold;
    }

    QPushButton#DangerButton:hover {
        background: #8C1D14;
        color: #FFFFFF;
    }

    QPushButton#DangerButton:disabled {
        background: #D7DCE2;
        color: #777777;
        border: 1px solid #C8D0DA;
    }

    QLineEdit,
    QComboBox,
    QTextEdit,
    QPlainTextEdit {
        background: #FFFFFF;
        color: #17202A;
        border: 1px solid #C8D0DA;
        border-radius: 5px;
        padding: 4px 8px;
        selection-background-color: #D6E8FA;
        selection-color: #17202A;
    }

    QComboBox::drop-down {
        border-left: 1px solid #C8D0DA;
        background: #F8FAFC;
        width: 24px;
    }

    QComboBox QAbstractItemView {
        background: #FFFFFF;
        color: #17202A;
        border: 1px solid #C8D0DA;
        selection-background-color: #D6E8FA;
        selection-color: #17202A;
    }

    QCheckBox {
        background: transparent;
        color: #17202A;
        spacing: 6px;
    }

    QCheckBox::indicator {
        width: 14px;
        height: 14px;
        border: 1px solid #8A97A8;
        background: #FFFFFF;
    }

    QCheckBox::indicator:checked {
        background: #2F5D87;
        border: 1px solid #17324D;
    }

    QListWidget,
    QTreeWidget {
        background: #FFFFFF;
        color: #17202A;
        border: 1px solid #C8D0DA;
        selection-background-color: #D6E8FA;
        selection-color: #17202A;
        alternate-background-color: #FFFFFF;
    }

    QListWidget::item,
    QTreeWidget::item {
        background: #FFFFFF;
        color: #17202A;
        padding: 3px;
    }

    QListWidget::item:selected,
    QTreeWidget::item:selected {
        background: #D6E8FA;
        color: #17202A;
    }

    QTableWidget {
        background: #FFFFFF;
        color: #17202A;
        alternate-background-color: #FFFFFF;
        gridline-color: #D4DAE2;
        border: 1px solid #C8D0DA;
        selection-background-color: #D6E8FA;
        selection-color: #17202A;
    }

    QTableWidget::item {
        background: #FFFFFF;
        color: #17202A;
    }

    QTableWidget::item:selected {
        background: #D6E8FA;
        color: #17202A;
    }

    QHeaderView::section {
        background: #E7ECF2;
        color: #17202A;
        padding: 5px;
        border: 1px solid #C8D0DA;
        font-weight: bold;
    }

    QTabWidget::pane {
        border: 1px solid #C8D0DA;
        background: #FFFFFF;
    }

    QTabBar::tab {
        background: #F8FAFC;
        color: #17202A;
        padding: 8px 14px;
        border: 1px solid #C8D0DA;
        border-bottom: none;
    }

    QTabBar::tab:selected {
        background: #FFFFFF;
        color: #17324D;
        font-weight: bold;
    }

    QTabBar::tab:hover {
        background: #E8EEF5;
        color: #17202A;
    }

    QScrollBar:vertical,
    QScrollBar:horizontal {
        background: #F4F6F8;
        border: 1px solid #D4DAE2;
    }

    QScrollBar::handle:vertical,
    QScrollBar::handle:horizontal {
        background: #BFC7D1;
        border-radius: 4px;
    }

    QScrollBar::handle:vertical:hover,
    QScrollBar::handle:horizontal:hover {
        background: #8A97A8;
    }

    QMessageBox,
    QDialog {
        background: #FFFFFF;
        color: #17202A;
    }

    QMessageBox QLabel,
    QDialog QLabel {
        background: transparent;
        color: #17202A;
    }
    """)