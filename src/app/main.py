"""Entry point for the PySide6 GUI application."""
from __future__ import annotations

import sys

from PySide6 import QtWidgets

from .ui import MainWindow


def main() -> None:
    """Launch the GUI application."""

    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
