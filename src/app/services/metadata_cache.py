"""Lightweight persistence for media metadata."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Tuple

from .paths import CONFIG_DIR, ensure_config_dir

if TYPE_CHECKING:  # pragma: no cover - for type checkers only
    from .metadata import MediaMetadata

Signature = Tuple[int, int]


@dataclass(slots=True)
class CachedMetadata:
    """Serialized metadata tied to a specific file signature."""

    mtime_ns: int
    size: int
    metadata: "MediaMetadata"


class MetadataCache:
    """Store and retrieve metadata for media files."""

    def __init__(self, path: Path | None = None) -> None:
        ensure_config_dir()
        self.path = Path(path or CONFIG_DIR / "metadata_cache.json").expanduser()
        self._cache: Dict[str, CachedMetadata] = {}
        self._load()

    def get(self, path: Path, signature: Signature) -> "MediaMetadata" | None:
        key = str(path)
        entry = self._cache.get(key)
        if not entry:
            return None
        if entry.mtime_ns != signature[0] or entry.size != signature[1]:
            return None
        return entry.metadata

    def set(self, path: Path, signature: Signature, metadata: "MediaMetadata") -> None:
        key = str(path)
        self._cache[key] = CachedMetadata(mtime_ns=signature[0], size=signature[1], metadata=metadata)
        self.save()

    def save(self) -> None:
        payload = {key: self._serialize(entry) for key, entry in self._cache.items()}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(payload, indent=2))

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text())
        except (OSError, json.JSONDecodeError):
            return
        for key, value in data.items():
            metadata = self._deserialize_metadata(value.get("metadata", {}))
            if metadata is None:
                continue
            entry = CachedMetadata(
                mtime_ns=int(value.get("mtime_ns", 0)),
                size=int(value.get("size", 0)),
                metadata=metadata,
            )
            self._cache[key] = entry

    @staticmethod
    def _serialize(entry: CachedMetadata) -> dict:
        metadata = entry.metadata
        resolution = metadata.resolution if metadata.resolution is None else list(metadata.resolution)
        return {
            "mtime_ns": entry.mtime_ns,
            "size": entry.size,
            "metadata": {
                "duration": metadata.duration,
                "resolution": resolution,
                "codec": metadata.codec,
                "tags": metadata.tags,
            },
        }

    @staticmethod
    def _deserialize_metadata(data: dict) -> "MediaMetadata" | None:
        if not isinstance(data, dict):
            return None
        from .metadata import MediaMetadata

        resolution = data.get("resolution")
        resolution_tuple: tuple[int, int] | None = None
        if isinstance(resolution, list) and len(resolution) == 2:
            try:
                resolution_tuple = (int(resolution[0]), int(resolution[1]))
            except (TypeError, ValueError):
                resolution_tuple = None
        return MediaMetadata(
            duration=data.get("duration"),
            resolution=resolution_tuple,
            codec=data.get("codec"),
            tags=data.get("tags") or {},
        )
