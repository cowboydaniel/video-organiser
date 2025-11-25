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
class TranscriptionSettings:
    """Preferences for speech-to-text processing."""

    model_size: str = "base"
    device: str = "cpu"
    sample_rate: int = 16000
    chunk_duration: float = 300.0
    confidence_threshold: float = 0.0
    diarization: bool = False
    language_detection: bool = True


@dataclass(slots=True)
class ProcessingSettings:
    """Preferences for scene sampling and analysis."""

    scene_interval: float = 5.0


@dataclass(slots=True)
class UserSettings:
    """Top-level settings stored on disk."""

    last_directory: str | None = None
    rule_presets: RulePresets = field(default_factory=RulePresets)
    theme: str = "light"
    transcription: TranscriptionSettings = field(default_factory=TranscriptionSettings)
    processing: ProcessingSettings = field(default_factory=ProcessingSettings)


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

    def update_transcription_settings(
        self,
        *,
        model_size: str | None = None,
        device: str | None = None,
        sample_rate: int | None = None,
        chunk_duration: float | None = None,
        confidence_threshold: float | None = None,
        diarization: bool | None = None,
        language_detection: bool | None = None,
    ) -> None:
        """Persist preferred transcription configuration."""

        preferences = self._settings.transcription
        if model_size is not None:
            preferences.model_size = model_size
        if device is not None:
            preferences.device = device
        if sample_rate is not None:
            preferences.sample_rate = sample_rate
        if chunk_duration is not None:
            preferences.chunk_duration = chunk_duration
        if confidence_threshold is not None:
            preferences.confidence_threshold = confidence_threshold
        if diarization is not None:
            preferences.diarization = diarization
        if language_detection is not None:
            preferences.language_detection = language_detection
        self.save()

    def update_processing_settings(self, *, scene_interval: float | None = None) -> None:
        """Persist preferred processing configuration."""

        settings = self._settings.processing
        if scene_interval is not None:
            settings.scene_interval = scene_interval
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
        transcription_data = data.get("transcription") if isinstance(data, dict) else None
        default_transcription = TranscriptionSettings()
        transcription = TranscriptionSettings(
            model_size=str(
                (transcription_data or {}).get("model_size") or default_transcription.model_size
            ),
            device=str((transcription_data or {}).get("device") or default_transcription.device),
            sample_rate=int(
                (transcription_data or {}).get("sample_rate")
                if (transcription_data or {}).get("sample_rate") is not None
                else default_transcription.sample_rate
            ),
            chunk_duration=float(
                (transcription_data or {}).get("chunk_duration")
                if (transcription_data or {}).get("chunk_duration") is not None
                else default_transcription.chunk_duration
            ),
            confidence_threshold=float(
                (transcription_data or {}).get("confidence_threshold")
                if (transcription_data or {}).get("confidence_threshold") is not None
                else default_transcription.confidence_threshold
            ),
            diarization=bool((transcription_data or {}).get("diarization", default_transcription.diarization)),
            language_detection=bool(
                (transcription_data or {}).get("language_detection", default_transcription.language_detection)
            ),
        )
        processing_data = data.get("processing") if isinstance(data, dict) else None
        default_processing = ProcessingSettings()
        processing = ProcessingSettings(
            scene_interval=float(
                (processing_data or {}).get("scene_interval")
                if (processing_data or {}).get("scene_interval") is not None
                else default_processing.scene_interval
            )
        )
        return UserSettings(
            last_directory=last_directory,
            rule_presets=presets,
            theme=theme,
            transcription=transcription,
            processing=processing,
        )
