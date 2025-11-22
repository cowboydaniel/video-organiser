"""Shared filesystem locations for application data."""
from __future__ import annotations

from pathlib import Path

CONFIG_DIR = Path.home() / ".config" / "video-organiser"


def ensure_config_dir() -> Path:
    """Ensure the configuration directory exists and return it."""

    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    return CONFIG_DIR
