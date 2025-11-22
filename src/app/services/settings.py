"""Persistent storage for user-facing configuration."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .paths import CONFIG_DIR, ensure_config_dir


@dataclass(slots=True)
class RulePresets:
    """Reusable rule configuration preferences."""

    destination_root: str = str(Path.home() / "Videos" / "Organized")
    folder_template: str = "{date:%Y/%m}"
    filename_template: str = "{name}_{resolution}"


@dataclass(slots=True)
class UserSettings:
    """Top-level settings stored on disk."""

    last_directory: str | None = None
    rule_presets: RulePresets = field(default_factory=RulePresets)
    theme: str = "light"


class SettingsManager:
    """Load and persist user settings to the config directory."""

    def __init__(self, path: Path | None = None) -> None:
        ensure_config_dir()
        self.path = Path(path or CONFIG_DIR / "config.json").expanduser()
        self._settings = self._load()

    @property
    def settings(self) -> UserSettings:
        """Current settings in memory."""

        return self._settings

    def _load(self) -> UserSettings:
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text())
                return self._from_dict(data)
            except (OSError, json.JSONDecodeError):
                pass
        return UserSettings()

    def save(self) -> None:
        """Write current settings to disk."""

        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(asdict(self._settings), indent=2)
        self.path.write_text(payload)

    def update_last_directory(self, directory: Path | None) -> None:
        """Persist the most recently used directory."""

        self._settings.last_directory = str(directory) if directory else None
        self.save()

    def update_rule_presets(
        self,
        *,
        destination_root: str | None = None,
        folder_template: str | None = None,
        filename_template: str | None = None,
    ) -> None:
        """Persist preferred rule values."""

        presets = self._settings.rule_presets
        if destination_root is not None:
            presets.destination_root = destination_root
        if folder_template is not None:
            presets.folder_template = folder_template
        if filename_template is not None:
            presets.filename_template = filename_template
        self.save()

    def set_theme(self, theme: str) -> None:
        """Persist the user's preferred theme."""

        self._settings.theme = theme
        self.save()

    def _from_dict(self, data: dict) -> UserSettings:
        presets_data = data.get("rule_presets", {}) if isinstance(data, dict) else {}
        default_presets = RulePresets()
        presets = RulePresets(
            destination_root=str(presets_data.get("destination_root") or default_presets.destination_root),
            folder_template=str(presets_data.get("folder_template") or default_presets.folder_template),
            filename_template=str(
                presets_data.get("filename_template") or default_presets.filename_template
            ),
        )
        last_directory = data.get("last_directory") if isinstance(data, dict) else None
        theme_value = data.get("theme") if isinstance(data, dict) else None
        theme = str(theme_value) if theme_value else "light"
        return UserSettings(last_directory=last_directory, rule_presets=presets, theme=theme)
