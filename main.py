import sys
import os
import ctypes
import webbrowser

from PySide6.QtWidgets import (
    QApplication, QDialog, QVBoxLayout, QLabel,
    QPushButton, QHBoxLayout
)
from PySide6.QtGui import QIcon
from PySide6.QtCore import Qt

from ui.main_window import SPSStudioMainWindow
from ui.styles import apply_app_style
from core.startup_guard import validate_startup
from core.app_config import APP_ID, STARTUP_ERROR_TITLE


def resource_path(relative_path):
    """
    Return the absolute path to a resource.

    This function supports both development environments and bundled
    executables (e.g., created with PyInstaller).

    Parameters
    ----------
    relative_path : str
        Relative path to the resource within the project.

    Returns
    -------
    str
        Absolute path to the requested resource.
    """
    if hasattr(sys, "_MEIPASS"):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)


class StartupErrorDialog(QDialog):
    """
    Dialog displayed when startup validation fails.

    This dialog shows an error message to the user and optionally
    provides a link to download the latest version of the application.
    """

    def __init__(self, message, download_url=None):
        """
        Initialize the startup error dialog.

        Parameters
        ----------
        message : str
            Message displayed to the user. HTML formatting is supported.
        download_url : str, optional
            URL to download the latest version of the application.

        Notes
        -----
        If `download_url` is provided, a button is displayed that opens
        the link in the system's default web browser.
        """
        super().__init__()

        self.setWindowTitle(STARTUP_ERROR_TITLE)
        self.setMinimumWidth(420)

        layout = QVBoxLayout()

        label = QLabel()
        label.setText(message)
        label.setTextFormat(Qt.RichText)
        label.setWordWrap(True)
        layout.addWidget(label)

        button_layout = QHBoxLayout()

        if download_url:
            download_btn = QPushButton("Download Latest Version")
            download_btn.clicked.connect(
                lambda: webbrowser.open(download_url)
            )
            button_layout.addWidget(download_btn)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        button_layout.addWidget(close_btn)

        layout.addLayout(button_layout)
        self.setLayout(layout)


def main():
    """
    Application entry point for SPS Studio.

    This function initializes the GUI environment, performs startup
    validation, and launches the main application window.

    Notes
    -----
    Execution flow:

    1. Configure AppUserModelID (Windows only).
    2. Initialize QApplication.
    3. Execute startup validation.
    4. If validation fails:
       - Display error dialog.
       - Terminate the application.
    5. If validation succeeds:
       - Apply global styles.
       - Create and show the main window.
       - Start the Qt event loop.

    Raises
    ------
    SystemExit
        Raised when the application exits or if startup validation fails.
    """
    # Windows-specific configuration
    if sys.platform == "win32":
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(APP_ID)

    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon(resource_path("img/SPS_icon.ico")))

    is_valid, message, download_url = validate_startup()

    if not is_valid:
        dialog = StartupErrorDialog(message, download_url)
        dialog.exec()
        sys.exit(1)

    apply_app_style(app)

    window = SPSStudioMainWindow()
    window.setWindowIcon(QIcon(resource_path("img/Logo_SPS.webp")))
    window.showMaximized()

    sys.exit(app.exec())


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # Placeholder for future logging integration
        print(f"Fatal error during execution: {e}")
        sys.exit(1)