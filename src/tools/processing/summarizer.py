"""LLM-backed summarization that fuses transcript and vision evidence."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Callable, Sequence


@dataclass(slots=True)
class SceneEvidence:
    """Summarized evidence for a single scene."""

    index: int
    start: float
    end: float
    transcript_snippet: str
    visual_tags: list[str]


@dataclass(slots=True)
class SummaryResult:
    """Structured summary produced by the ``Summarizer``."""

    title: str
    primary_subjects: list[str]
    event_or_topic: str
    date_inference: str
    confidences: dict[str, float] = field(default_factory=dict)
    folder_suggestion: str = "uncategorized"
    raw_prompt: str | None = None
    raw_response: str | None = None
    scene_evidence: list[SceneEvidence] = field(default_factory=list)

    def to_text(self) -> str:
        """Render a readable multi-line summary."""

        subjects = ", ".join(self.primary_subjects) if self.primary_subjects else "Unknown"
        parts = [
            f"Title: {self.title or 'Untitled'}",
            f"Primary subjects: {subjects}",
            f"Event/Topic: {self.event_or_topic or 'Unspecified'}",
            f"Date inference: {self.date_inference or 'Unknown'}",
            f"Suggested folder: {self.folder_suggestion}",
        ]
        return "\n".join(parts)


@dataclass(slots=True)
class SummarizerConfig:
    """Configuration for the ``Summarizer``."""

    max_tags_per_scene: int = 3
    max_transcript_chars: int = 320
    default_confidence: float = 0.4
    confidence_floor: float = 0.5


class Summarizer:
    """Fuse transcript snippets and visual tags into a structured summary."""

    def __init__(
        self,
        llm_client: Callable[[str], str] | None = None,
        config: SummarizerConfig | None = None,
    ) -> None:
        self.llm_client = llm_client
        self.config = config or SummarizerConfig()

    def summarize(
        self,
        transcript: Sequence["TranscriptSegment"],
        scenes: Sequence["SceneSegment"],
        visual_tags: Sequence["VisualTag"],
    ) -> SummaryResult:
        """Generate a structured summary using available modalities."""

        evidence = self._build_scene_evidence(transcript, scenes, visual_tags)
        prompt = self._build_prompt(evidence)
        raw_response = self._invoke_llm(prompt, evidence)
        parsed = self._parse_response(raw_response, transcript)
        sanitized = self._apply_fallbacks(parsed, transcript)
        folder_suggestion = self._build_folder_suggestion(sanitized)

        return SummaryResult(
            title=sanitized["title"],
            primary_subjects=sanitized["primary_subjects"],
            event_or_topic=sanitized["event_or_topic"],
            date_inference=sanitized["date_inference"],
            confidences=sanitized["confidences"],
            folder_suggestion=folder_suggestion,
            raw_prompt=prompt,
            raw_response=raw_response,
            scene_evidence=evidence,
        )

    def _build_scene_evidence(
        self,
        transcript: Sequence["TranscriptSegment"],
        scenes: Sequence["SceneSegment"],
        visual_tags: Sequence["VisualTag"],
    ) -> list[SceneEvidence]:
        evidence: list[SceneEvidence] = []
        for index, scene in enumerate(scenes):
            snippet = self._transcript_for_scene(transcript, scene.start, scene.end)
            tags = self._tags_for_scene(visual_tags, scene.start, scene.end)
            evidence.append(
                SceneEvidence(
                    index=index,
                    start=scene.start,
                    end=scene.end,
                    transcript_snippet=snippet,
                    visual_tags=tags,
                )
            )
        return evidence

    def _transcript_for_scene(
        self, transcript: Sequence["TranscriptSegment"], start: float, end: float
    ) -> str:
        pieces: list[str] = []
        for segment in transcript:
            if segment.end < start or segment.start > end:
                continue
            pieces.append(segment.text)
        combined = " ".join(piece.strip() for piece in pieces if piece.strip())
        if len(combined) > self.config.max_transcript_chars:
            combined = combined[: self.config.max_transcript_chars].rstrip() + "…"
        return combined

    def _tags_for_scene(
        self, visual_tags: Sequence["VisualTag"], start: float, end: float
    ) -> list[str]:
        relevant = [
            tag for tag in visual_tags if start <= tag.timestamp <= end or not visual_tags
        ]
        sorted_tags = sorted(
            relevant,
            key=lambda t: (t.confidence if t.confidence is not None else 0.0),
            reverse=True,
        )
        trimmed = sorted_tags[: self.config.max_tags_per_scene]
        return [tag.label for tag in trimmed]

    def _build_prompt(self, evidence: Sequence[SceneEvidence]) -> str:
        lines = [
            "You are a media organiser. Given scene-level transcript snippets and top visual tags,",
            "produce a concise JSON object with keys: title (string), primary_subjects (list of strings),",
            "event_or_topic (string), date_inference (string such as YYYY or YYYY-MM-DD),",
            "and confidences (object with numeric scores for title, primary_subjects, event_or_topic, date_inference).",
            "The JSON should be parseable and not wrapped in markdown fences.",
            "Scenes:",
        ]
        for scene in evidence:
            lines.append(
                f"- Scene {scene.index} [{scene.start:.2f}-{scene.end:.2f}s]:"
                f" transcript='{scene.transcript_snippet}' tags={scene.visual_tags}"
            )
        lines.append(
            "Focus on concise phrasing and grounded in the provided evidence; do not hallucinate details."
        )
        return "\n".join(lines)

    def _invoke_llm(self, prompt: str, evidence: Sequence[SceneEvidence]) -> str:
        if self.llm_client is None:
            return json.dumps(self._heuristic_summary(evidence))
        try:
            response = self.llm_client(prompt)
        except Exception:
            return json.dumps(self._heuristic_summary(evidence))
        return response or json.dumps(self._heuristic_summary(evidence))

    def _heuristic_summary(self, evidence: Sequence[SceneEvidence]) -> dict:
        transcript_text = " ".join(scene.transcript_snippet for scene in evidence if scene.transcript_snippet)
        title = (transcript_text[:80] + "…") if transcript_text else "Untitled clip"
        subjects = []
        if transcript_text:
            first_words = transcript_text.split()
            subjects = [" ".join(first_words[:3])] if first_words else []
        tags = [tag for scene in evidence for tag in scene.visual_tags]
        event_topic = tags[0] if tags else (subjects[0] if subjects else "general")
        return {
            "title": title.strip(),
            "primary_subjects": subjects or ["Unknown"],
            "event_or_topic": event_topic,
            "date_inference": "unknown",
            "confidences": {
                "title": self.config.default_confidence,
                "primary_subjects": self.config.default_confidence,
                "event_or_topic": self.config.default_confidence,
                "date_inference": self.config.default_confidence,
            },
        }

    def _parse_response(self, raw_response: str, transcript: Sequence["TranscriptSegment"]) -> dict:
        candidate = raw_response.strip()
        if candidate.startswith("```"):
            candidate = candidate.strip("` ")
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            return self._heuristic_summary([])

        return {
            "title": str(payload.get("title", "")).strip(),
            "primary_subjects": [str(item).strip() for item in payload.get("primary_subjects", []) if str(item).strip()],
            "event_or_topic": str(payload.get("event_or_topic", "")).strip(),
            "date_inference": str(payload.get("date_inference", "")).strip(),
            "confidences": self._normalize_confidences(payload.get("confidences", {})),
        }

    def _normalize_confidences(self, raw: dict) -> dict[str, float]:
        normalized: dict[str, float] = {}
        for key in ["title", "primary_subjects", "event_or_topic", "date_inference"]:
            value = raw.get(key, self.config.default_confidence)
            try:
                normalized[key] = float(value)
            except (TypeError, ValueError):
                normalized[key] = self.config.default_confidence
        return normalized

    def _apply_fallbacks(self, parsed: dict, transcript: Sequence["TranscriptSegment"]) -> dict:
        confidences = parsed.get("confidences", {})
        title = parsed.get("title") or self._fallback_title(transcript)
        subjects = parsed.get("primary_subjects") or ["Unknown"]
        event_topic = parsed.get("event_or_topic") or "general"
        date = parsed.get("date_inference") or "unknown"

        return {
            "title": title,
            "primary_subjects": subjects,
            "event_or_topic": event_topic,
            "date_inference": date,
            "confidences": confidences,
        }

    def _fallback_title(self, transcript: Sequence["TranscriptSegment"]) -> str:
        if not transcript:
            return "Untitled clip"
        combined = " ".join(segment.text for segment in transcript if segment.text)
        return (combined[:80] + "…") if combined else "Untitled clip"

    def _build_folder_suggestion(self, parsed: dict) -> str:
        confidences = parsed.get("confidences", {})
        subject_component = (
            self._sanitize(parsed["primary_subjects"][0])
            if parsed["primary_subjects"]
            and confidences.get("primary_subjects", 0.0) >= self.config.confidence_floor
            else "uncategorized"
        )
        event_component = (
            self._sanitize(parsed["event_or_topic"])
            if confidences.get("event_or_topic", 0.0) >= self.config.confidence_floor
            else "unspecified"
        )
        date_component = self._sanitize(parsed.get("date_inference") or "unknown")
        return "/".join([subject_component, event_component, date_component])

    def _sanitize(self, component: str) -> str:
        safe = re.sub(r"[^\w\-]+", "_", component.strip())
        return safe.strip("_") or "unknown"


__all__ = ["Summarizer", "SummarizerConfig", "SummaryResult", "SceneEvidence"]

