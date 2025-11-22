"""Metadata extraction helpers powered by ffprobe or mediainfo."""
from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

from .metadata_cache import MetadataCache, Signature


@dataclass(slots=True)
class MediaMetadata:
    """Simplified metadata for a media file."""

    duration: float | None
    resolution: Tuple[int, int] | None
    codec: str | None
    tags: Dict[str, Any]

    @property
    def duration_seconds(self) -> float | None:
        """Return the duration expressed in seconds."""

        if self.duration is None:
            return None
        return float(self.duration)


class MetadataReader:
    """Read metadata using ffprobe or mediainfo.

    The implementation first attempts ``ffprobe`` for detailed JSON output and falls
    back to ``mediainfo`` when FFmpeg tools are unavailable.
    """

    def __init__(self, cache: MetadataCache | None = None) -> None:
        self._ffprobe_path = shutil.which("ffprobe")
        self._mediainfo_path = shutil.which("mediainfo")
        self.cache = cache

    def read(self, path: Path) -> MediaMetadata:
        """Return a :class:`MediaMetadata` instance for ``path``.

        Errors from external tools are caught and translated into a metadata object with
        ``None`` fields where values could not be determined.
        """

        if not path.exists() or not path.is_file():
            raise FileNotFoundError(path)

        signature = self._signature(path)

        if self.cache:
            cached = self.cache.get(path, signature)
            if cached:
                return cached

        if self._ffprobe_path:
            metadata = self._read_with_ffprobe(path)
            if metadata:
                self._persist(path, signature, metadata)
                return metadata

        if self._mediainfo_path:
            metadata = self._read_with_mediainfo(path)
            if metadata:
                self._persist(path, signature, metadata)
                return metadata

        metadata = MediaMetadata(duration=None, resolution=None, codec=None, tags={})
        self._persist(path, signature, metadata)
        return metadata

    def _read_with_ffprobe(self, path: Path) -> MediaMetadata | None:
        try:
            output = subprocess.check_output(
                [
                    self._ffprobe_path or "ffprobe",
                    "-v",
                    "error",
                    "-print_format",
                    "json",
                    "-show_streams",
                    "-show_format",
                    str(path),
                ],
                stderr=subprocess.STDOUT,
            )
        except (OSError, subprocess.CalledProcessError):
            return None

        data = json.loads(output)
        streams = data.get("streams", [])
        fmt = data.get("format", {})

        video_stream = next((s for s in streams if s.get("codec_type") == "video"), None)
        duration = self._coerce_duration(fmt.get("duration") or video_stream and video_stream.get("duration"))
        resolution = None
        codec = None
        if video_stream:
            width = self._coerce_int(video_stream.get("width"))
            height = self._coerce_int(video_stream.get("height"))
            if width and height:
                resolution = (width, height)
            codec = video_stream.get("codec_name")

        tags = fmt.get("tags", {}) if isinstance(fmt.get("tags", {}), dict) else {}
        return MediaMetadata(duration=duration, resolution=resolution, codec=codec, tags=tags)

    def _read_with_mediainfo(self, path: Path) -> MediaMetadata | None:
        try:
            output = subprocess.check_output(
                [self._mediainfo_path or "mediainfo", "--Output=JSON", str(path)],
                stderr=subprocess.STDOUT,
            )
        except (OSError, subprocess.CalledProcessError):
            return None

        data = json.loads(output)
        media = data.get("media", {})
        tracks = media.get("track", [])
        video_track = next((t for t in tracks if t.get("@type") == "Video"), None)
        general_track = next((t for t in tracks if t.get("@type") == "General"), {})

        duration = self._coerce_duration(general_track.get("Duration"))
        resolution = None
        codec = None
        if video_track:
            width = self._coerce_int(video_track.get("Width"))
            height = self._coerce_int(video_track.get("Height"))
            if width and height:
                resolution = (width, height)
            codec = video_track.get("CodecID") or video_track.get("Format")

        tags = {k: v for k, v in general_track.items() if k not in {"Duration"}}
        return MediaMetadata(duration=duration, resolution=resolution, codec=codec, tags=tags)

    @staticmethod
    def _coerce_duration(value: Any) -> float | None:
        try:
            if value is None:
                return None
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _coerce_int(value: Any) -> int | None:
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _signature(path: Path) -> Signature:
        stat = path.stat()
        return (stat.st_mtime_ns, stat.st_size)

    def _persist(self, path: Path, signature: Signature, metadata: MediaMetadata) -> None:
        if self.cache:
            self.cache.set(path, signature, metadata)
