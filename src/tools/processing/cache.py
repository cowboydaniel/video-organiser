"""Cache processing artefacts like transcripts, tags, and summaries."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, Sequence

from app.services.paths import CONFIG_DIR, ensure_config_dir

from .summarizer import SceneEvidence, SummaryResult

if TYPE_CHECKING:  # pragma: no cover - typing aid only
    from .pipeline import TranscriptSegment, VisualTag

logger = logging.getLogger(__name__)


DEFAULT_CACHE_PATH = CONFIG_DIR / "processing_cache.json"


@dataclass(slots=True)
class ProcessingCacheEntry:
    """Cached payload keyed by a media checksum."""

    checksum: str
    transcript: list[TranscriptSegment] | None = None
    tags: list[VisualTag] | None = None
    summary: SummaryResult | None = None


class ProcessingCache:
    """Persisted cache for transcripts, tags, and summaries."""

    def __init__(self, path: Path | None = None) -> None:
        ensure_config_dir()
        self.path = Path(path or DEFAULT_CACHE_PATH).expanduser()
        self._payload: Dict[str, dict[str, Any]] = {}
        self._load()

    def get(self, checksum: str) -> ProcessingCacheEntry | None:
        entry = self._payload.get(checksum)
        if not entry:
            return None
        return ProcessingCacheEntry(
            checksum=checksum,
            transcript=self._deserialize_transcript(entry.get("transcript", [])),
            tags=self._deserialize_tags(entry.get("tags", [])),
            summary=self._deserialize_summary(entry.get("summary")),
        )

    def store_transcript(self, checksum: str, transcript: Iterable[TranscriptSegment]) -> None:
        entry = self._payload.get(checksum, {})
        entry["transcript"] = [self._serialize_segment(segment) for segment in transcript]
        self._payload[checksum] = entry
        self._save()

    def store_tags(self, checksum: str, tags: Sequence[VisualTag]) -> None:
        entry = self._payload.get(checksum, {})
        entry["tags"] = [self._serialize_tag(tag) for tag in tags]
        self._payload[checksum] = entry
        self._save()

    def store_summary(self, checksum: str, summary: SummaryResult) -> None:
        entry = self._payload.get(checksum, {})
        entry["summary"] = self._serialize_summary(summary)
        self._payload[checksum] = entry
        self._save()

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.path.write_text(json.dumps(self._payload, indent=2))
        except OSError as exc:  # pragma: no cover - filesystem issues
            logger.warning("Failed to write processing cache: %s", exc)

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text())
        except (OSError, json.JSONDecodeError) as exc:  # pragma: no cover - defensive
            logger.warning("Could not load processing cache: %s", exc)
            return
        if isinstance(data, dict):
            self._payload = data

    @staticmethod
    def _serialize_segment(segment: TranscriptSegment) -> dict[str, Any]:
        return {
            "start": segment.start,
            "end": segment.end,
            "text": segment.text,
            "speaker": segment.speaker,
            "language": segment.language,
            "confidence": segment.confidence,
        }

    @staticmethod
    def _deserialize_transcript(raw: Iterable[dict[str, Any]]) -> list[TranscriptSegment]:
        from .pipeline import TranscriptSegment

        segments: list[TranscriptSegment] = []
        for item in raw:
            try:
                segments.append(
                    TranscriptSegment(
                        start=float(item.get("start", 0.0)),
                        end=float(item.get("end", 0.0)),
                        text=str(item.get("text", "")),
                        speaker=item.get("speaker"),
                        language=item.get("language"),
                        confidence=item.get("confidence"),
                    )
                )
            except Exception:
                continue
        return segments

    @staticmethod
    def _serialize_tag(tag: VisualTag) -> dict[str, Any]:
        return {
            "timestamp": tag.timestamp,
            "label": tag.label,
            "confidence": tag.confidence,
            "source": tag.source,
        }

    @staticmethod
    def _deserialize_tags(raw: Iterable[dict[str, Any]]) -> list[VisualTag]:
        from .pipeline import VisualTag

        tags: list[VisualTag] = []
        for item in raw:
            try:
                tags.append(
                    VisualTag(
                        timestamp=float(item.get("timestamp", 0.0)),
                        label=str(item.get("label", "")),
                        confidence=item.get("confidence"),
                        source=item.get("source"),
                    )
                )
            except Exception:
                continue
        return tags

    @staticmethod
    def _serialize_summary(summary: SummaryResult) -> dict[str, Any]:
        return {
            "title": summary.title,
            "primary_subjects": summary.primary_subjects,
            "event_or_topic": summary.event_or_topic,
            "date_inference": summary.date_inference,
            "confidences": summary.confidences,
            "folder_suggestion": summary.folder_suggestion,
            "raw_prompt": summary.raw_prompt,
            "raw_response": summary.raw_response,
            "scene_evidence": [
                {
                    "index": evidence.index,
                    "start": evidence.start,
                    "end": evidence.end,
                    "transcript_snippet": evidence.transcript_snippet,
                    "visual_tags": evidence.visual_tags,
                }
                for evidence in summary.scene_evidence
            ],
        }

    @staticmethod
    def _deserialize_summary(raw: dict[str, Any] | None) -> SummaryResult | None:
        if not isinstance(raw, dict):
            return None
        try:
            scene_evidence = [
                SceneEvidence(
                    index=int(item.get("index", 0)),
                    start=float(item.get("start", 0.0)),
                    end=float(item.get("end", 0.0)),
                    transcript_snippet=str(item.get("transcript_snippet", "")),
                    visual_tags=list(item.get("visual_tags", [])),
                )
                for item in raw.get("scene_evidence", [])
            ]
            return SummaryResult(
                title=str(raw.get("title", "")),
                primary_subjects=list(raw.get("primary_subjects", [])),
                event_or_topic=str(raw.get("event_or_topic", "")),
                date_inference=str(raw.get("date_inference", "")),
                confidences=dict(raw.get("confidences", {})),
                folder_suggestion=str(raw.get("folder_suggestion", "uncategorized")),
                raw_prompt=raw.get("raw_prompt"),
                raw_response=raw.get("raw_response"),
                scene_evidence=scene_evidence,
            )
        except Exception:
            return None


__all__ = ["ProcessingCache", "ProcessingCacheEntry"]

