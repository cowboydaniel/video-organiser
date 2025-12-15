"""Entry point for the PySide6 GUI application."""
from __future__ import annotations

import logging
import sys
from pathlib import Path

# Add the src directory to sys.path to enable absolute imports
src_dir = Path(__file__).resolve().parent.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from PySide6 import QtWidgets

from app.ui import MainWindow


def setup_logging() -> None:
    """Configure logging to output to console and file."""
    # Create logs directory
    logs_dir = Path.home() / ".config" / "video-organiser" / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_file = logs_dir / "application.log"

    # Configure root logger
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),  # Console output
            logging.FileHandler(log_file, mode='a', encoding='utf-8')  # File output
        ]
    )

    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("Video Organiser started")
    logger.info("=" * 60)


def main() -> None:
    """Launch the GUI application."""
    setup_logging()

    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
