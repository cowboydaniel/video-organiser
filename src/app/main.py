"""Entry point for the PySide6 GUI application."""
from __future__ import annotations

import sys
from pathlib import Path

# Add the src directory to sys.path to enable absolute imports
src_dir = Path(__file__).resolve().parent.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from PySide6 import QtWidgets

from app.ui import MainWindow


def main() -> None:
    """Launch the GUI application."""

    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
