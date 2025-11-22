"""Rule engine for mapping media metadata to destination paths."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from string import Formatter
from typing import Dict, Mapping

from .metadata import MediaMetadata


def _coerce_datetime(path: Path, metadata: MediaMetadata) -> datetime:
    """Choose a datetime for templating.

    Preference order:
    1. ``metadata.tags['creation_time']`` if present and parseable as ISO 8601.
    2. File modification time from :func:`Path.stat`.
    """

    tag_value = metadata.tags.get("creation_time") if metadata.tags else None
    if isinstance(tag_value, str):
        try:
            return datetime.fromisoformat(tag_value.replace("Z", "+00:00"))
        except ValueError:
            pass

    stat = path.stat()
    return datetime.fromtimestamp(stat.st_mtime)


def _safe_component(value: str) -> str:
    """Return a filesystem-safe component."""

    return value.replace(Path.sep, "_")


def _format_resolution(metadata: MediaMetadata) -> str:
    if metadata.resolution:
        return f"{metadata.resolution[0]}x{metadata.resolution[1]}"
    return "unknown"


class _TemplateFormatter(Formatter):
    """Formatter that understands our templating keys."""

    def __init__(self, context: Mapping[str, object]):
        super().__init__()
        self.context = context

    def get_value(self, key: str, args, kwargs):  # type: ignore[override]
        if key in kwargs:
            return kwargs[key]
        if isinstance(key, str) and key in self.context:
            return self.context[key]
        return super().get_value(key, args, kwargs)

    def format_field(self, value, format_spec: str):  # type: ignore[override]
        if isinstance(value, datetime) and format_spec:
            return value.strftime(format_spec)
        return super().format_field(value, format_spec)


@dataclass(slots=True)
class RuleContext:
    """Values available while rendering templates."""

    source: Path
    metadata: MediaMetadata
    custom_tags: Dict[str, str]
    date: datetime

    def mapping(self) -> Dict[str, object]:
        mapping: Dict[str, object] = {
            "name": self.source.stem,
            "ext": self.source.suffix.lstrip("."),
            "resolution": _format_resolution(self.metadata),
            "codec": self.metadata.codec or "unknown",
            "date": self.date,
        }

        for key, value in self.custom_tags.items():
            mapping[f"tag_{key}"] = value
        for key, value in self.metadata.tags.items():
            mapping[f"tag_{key}"] = value

        return mapping


@dataclass(slots=True)
class DestinationPlan:
    """Represents a resolved destination for a media file."""

    source: Path
    destination: Path
    metadata: MediaMetadata


class RuleEngine:
    """Compute destination paths for media files using templates."""

    def __init__(
        self,
        destination_root: Path,
        folder_template: str = "{date:%Y/%m}",
        filename_template: str = "{name}_{resolution}",
    ) -> None:
        self.destination_root = destination_root
        self.folder_template = folder_template
        self.filename_template = filename_template

    def build_context(
        self, source: Path, metadata: MediaMetadata, custom_tags: Mapping[str, str] | None = None
    ) -> RuleContext:
        return RuleContext(
            source=source,
            metadata=metadata,
            custom_tags=dict(custom_tags or {}),
            date=_coerce_datetime(source, metadata),
        )

    def render(self, template: str, context: RuleContext) -> str:
        formatter = _TemplateFormatter(context.mapping())
        rendered = formatter.format(template)
        return _safe_component(rendered)

    def resolve(
        self, source: Path, metadata: MediaMetadata, custom_tags: Mapping[str, str] | None = None
    ) -> DestinationPlan:
        context = self.build_context(source, metadata, custom_tags)
        folder = self.render(self.folder_template, context)
        filename = self.render(self.filename_template, context)
        filename_with_ext = f"{filename}.{context.source.suffix.lstrip('.') or 'mp4'}"
        destination = self.destination_root / folder / filename_with_ext
        return DestinationPlan(source=source, destination=destination, metadata=metadata)
