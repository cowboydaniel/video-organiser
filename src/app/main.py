"""Entry point for the PySide6 GUI application."""

from __future__ import annotations

import sys

from PySide6 import QtCore, QtWidgets


class MainWindow(QtWidgets.QMainWindow):
    """Simple main window placeholder for the video organiser."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Video Organiser")
        self.resize(800, 600)
        self._setup_ui()

    def _setup_ui(self) -> None:
        central_widget = QtWidgets.QWidget(self)
        layout = QtWidgets.QVBoxLayout(central_widget)

        label = QtWidgets.QLabel(
            "Welcome to Video Organiser! Replace this view with the real UI.",
            alignment=QtCore.Qt.AlignmentFlag.AlignCenter,
        )
        layout.addWidget(label)

        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)


def main() -> None:
    """Launch the GUI application."""

    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
