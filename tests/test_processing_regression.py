"""Regression tests for the media processing pipeline using curated samples."""
from __future__ import annotations

import json
import wave
from pathlib import Path
from typing import Iterable

from tools.processing import ProcessingPipeline
from tools.processing.metrics import (
    tag_precision_recall,
    title_relevance_score,
    word_error_rate,
)
from tools.processing.pipeline import AnalysisResult, SceneSegment, TranscriptSegment, VisualTag
from tools.processing.summarizer import SummaryResult
from tools.processing.transcription import TranscriptionConfig, TranscriptionResult


class StubTranscriber:
    def __init__(self, transcript_text: str) -> None:
        self.config = TranscriptionConfig()
        self.transcript_text = transcript_text

    def transcribe(
        self, audio_path: Path, *, progress_callback=None
    ) -> TranscriptionResult:  # pragma: no cover - simple stub
        if progress_callback:
            progress_callback("Transcribing stub chunk", 0.5)
        segment = TranscriptSegment(
            start=0.0,
            end=5.0,
            text=self.transcript_text,
            speaker="Narrator",
            language="en",
            confidence=0.99,
        )
        return TranscriptionResult(segments=[segment], detected_language="en")


class StubVision:
    def __init__(self, tags: Iterable[str]) -> None:
        self.tags = [VisualTag(timestamp=0.0, label=tag, confidence=0.9, source="stub.jpg") for tag in tags]

    def analyze(self, video_path: Path, artifacts_dir: Path) -> tuple[list[SceneSegment], list[VisualTag]]:
        frame_dir = artifacts_dir / "frames"
        frame_dir.mkdir(parents=True, exist_ok=True)
        keyframe = frame_dir / "scene_0000.jpg"
        keyframe.touch()
        scene = SceneSegment(start=0.0, end=5.0, keyframe=keyframe)
        return [scene], self.tags


class StubSummarizer:
    def __init__(self, expected_sample: dict) -> None:
        self.expected = expected_sample

    def summarize(
        self, transcript: list[TranscriptSegment], scenes: list[SceneSegment], visual_tags: list[VisualTag]
    ) -> SummaryResult:  # pragma: no cover - deterministic stub
        return SummaryResult(
            title=self.expected["title"],
            primary_subjects=self.expected.get("primary_subjects", []),
            event_or_topic=self.expected.get("category", ""),
            date_inference="2024",
            confidences={
                "title": 0.95,
                "primary_subjects": 0.9,
                "event_or_topic": 0.85,
                "date_inference": 0.6,
            },
            folder_suggestion=self.expected.get("expected_folder", "uncategorized"),
            raw_prompt=None,
            raw_response=json.dumps(self.expected),
            scene_evidence=[],
        )


def _read_samples() -> list[dict]:
    fixture_path = Path(__file__).parent / "fixtures" / "sample_videos.json"
    return json.loads(fixture_path.read_text())


def _write_silent_wav(path: Path, sample_rate: int = 16000, duration_seconds: float = 1.0) -> None:
    frames = int(sample_rate * duration_seconds)
    with wave.open(str(path), "w") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(b"\x00\x00" * frames)


def _stub_audio_extraction(video_path: Path, artifacts_dir: Path, sample_rate: int = 16000) -> Path:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    audio_path = artifacts_dir / f"{video_path.stem}_audio.wav"
    if not audio_path.exists():
        _write_silent_wav(audio_path, sample_rate=sample_rate, duration_seconds=1.0)
    return audio_path


def test_processing_pipeline_samples(tmp_path, monkeypatch):
    """Run the pipeline on curated samples and track regression metrics."""

    samples = _read_samples()
    # Avoid external media dependencies during tests.
    monkeypatch.setattr("tools.processing.pipeline.extract_audio_track", _stub_audio_extraction)

    for sample in samples:
        video_path = tmp_path / f"{sample['id']}.mp4"
        video_path.write_text(sample["description"])

        pipeline = ProcessingPipeline(cache_root=tmp_path / "artifacts")
        pipeline.transcriber = StubTranscriber(sample["transcript"])
        pipeline.vision = StubVision(sample["tags"])
        pipeline.summarizer = StubSummarizer(sample)

        result: AnalysisResult = pipeline.process(video_path)

        transcript_text = " ".join(segment.text.lower() for segment in result.transcript)
        for phrase in sample["key_phrases"]:
            assert phrase in transcript_text

        observed_tags = {tag.label for tag in result.visual_tags}
        for expected_tag in sample["tags"]:
            assert expected_tag in observed_tags

        assert sample["title"].lower() in result.summary.title.lower()

        wer = word_error_rate(sample["transcript"], transcript_text)
        precision, recall = tag_precision_recall(sample["tags"], observed_tags)
        relevance = title_relevance_score(sample["title"], result.summary.title)

        assert wer <= 0.1
        assert precision >= 0.8 and recall >= 0.8
        assert relevance >= 0.6
        assert result.summary.folder_suggestion == sample.get("expected_folder")
