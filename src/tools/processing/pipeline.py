"""Media processing pipeline for extracting audio, frames, and semantic signals."""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from app.services.paths import ensure_config_dir

from .cache import ProcessingCache
from .summarizer import Summarizer, SummaryResult
from .utils import retry_with_backoff

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class TranscriptSegment:
    """Represents a span of text derived from speech."""

    start: float
    end: float
    text: str
    speaker: str | None = None
    language: str | None = None
    confidence: float | None = None


@dataclass(slots=True)
class VisualTag:
    """Tagging result for a given visual slice or keyframe."""

    timestamp: float
    label: str
    confidence: float | None = None
    source: str | None = None


@dataclass(slots=True)
class SceneSegment:
    """Lightweight container describing a sampled scene."""

    start: float
    end: float
    keyframe: Path


@dataclass(slots=True)
class AnalysisResult:
    """Aggregated output of the processing pipeline."""

    video_path: Path
    audio_path: Path
    transcript: list[TranscriptSegment]
    scenes: list[SceneSegment]
    visual_tags: list[VisualTag]
    fused_summary: str
    summary: SummaryResult
    artifacts_dir: Path


class ProcessingPipeline:
    """Orchestrate media processing for multi-modal analysis."""

    def __init__(
        self,
        cache_root: Path | None = None,
        scene_interval: float = 5.0,
        transcription_config: "TranscriptionConfig | None" = None,
        metadata_cache: ProcessingCache | None = None,
    ) -> None:
        from .transcription import TranscriptionConfig, TranscriptionService
        from .vision import VisionAnalyzer

        default_cache_root = ensure_config_dir() / "processing_artifacts"
        self.cache_root = cache_root or default_cache_root
        self.scene_interval = scene_interval
        self.transcriber = TranscriptionService(transcription_config)
        self.vision = VisionAnalyzer(scene_interval=scene_interval, cache_root=self.cache_root)
        self.summarizer = Summarizer()
        self.metadata_cache = metadata_cache or ProcessingCache()

    def process(
        self,
        video_path: Path,
        progress_callback: Callable[[str, float | None], None] | None = None,
    ) -> AnalysisResult:
        """Run the pipeline over ``video_path`` and return structured outputs."""
        from .vision import media_signature

        def report(message: str, progress: float | None = None) -> None:
            if progress_callback:
                progress_callback(message, progress)

        steps = [
            "extract_audio",
            "transcribe",
            "vision",
            "summarize",
        ]
        step_size = 1.0 / max(len(steps), 1)

        normalized_path = video_path.expanduser().resolve()
        checksum = media_signature(normalized_path)
        cached_entry = self.metadata_cache.get(checksum)
        artifacts_dir = self._prepare_artifacts_dir(normalized_path)

        report("Extracting audio track", step_size)
        audio_path = extract_audio_track(
            normalized_path, artifacts_dir, sample_rate=self.transcriber.config.sample_rate
        )
        transcript = cached_entry.transcript if cached_entry and cached_entry.transcript else None
        if transcript is None:
            report("Transcribing audio", step_size * 2)
            transcript = self._transcribe_audio(audio_path)
            self.metadata_cache.store_transcript(checksum, transcript)
        else:
            logger.info("Using cached transcript for %s", normalized_path)

        try:
            report("Running visual analysis", step_size * 3)
            scenes, visual_tags = self.vision.analyze(normalized_path, artifacts_dir)
        except Exception as exc:  # pragma: no cover - defensive fallback
            logger.exception("Vision analysis failed for %s: %s", normalized_path, exc)
            scenes, visual_tags = [], []

        if not visual_tags and cached_entry and cached_entry.tags:
            logger.info("Using cached visual tags for %s", normalized_path)
            visual_tags = cached_entry.tags
        elif visual_tags:
            self.metadata_cache.store_tags(checksum, visual_tags)

        summary = cached_entry.summary if cached_entry and cached_entry.summary else None
        if summary is None:
            report("Fusing transcript and visual tags", step_size * 4)
            summary = self._fuse_modalities(transcript, scenes, visual_tags)
            self.metadata_cache.store_summary(checksum, summary)
        else:
            logger.info("Using cached summary for %s", normalized_path)

        return AnalysisResult(
            video_path=normalized_path,
            audio_path=audio_path,
            transcript=transcript,
            scenes=scenes,
            visual_tags=visual_tags,
            fused_summary=summary.to_text(),
            summary=summary,
            artifacts_dir=artifacts_dir,
        )

    def _prepare_artifacts_dir(self, video_path: Path) -> Path:
        stem = video_path.stem.replace(" ", "_")
        target = self.cache_root / f"processing_{stem}"
        target.mkdir(parents=True, exist_ok=True)
        return target

    def _transcribe_audio(self, audio_path: Path) -> list[TranscriptSegment]:
        """Transcribe audio into segments using the configured backend."""
        try:
            result = retry_with_backoff(
                lambda: self.transcriber.transcribe(audio_path),
                attempts=3,
                base_delay=1.0,
                logger=logger,
                description="transcription",
            )
        except Exception as exc:
            logger.error("Transcription failed for %s: %s", audio_path, exc)
            raise
        if not result.segments:
            return [
                TranscriptSegment(
                    start=0.0,
                    end=0.0,
                    text="No transcription available.",
                    speaker=None,
                    language=result.detected_language,
                )
            ]
        return result.segments

    def _tag_scenes(self, scenes: Sequence[SceneSegment]) -> list[VisualTag]:
        """Placeholder visual tagging over keyframes."""

        tags: list[VisualTag] = []
        for index, scene in enumerate(scenes):
            label = f"scene_{index:02d}"
            tags.append(
                VisualTag(
                    timestamp=scene.start,
                    label=label,
                    confidence=None,
                    source=scene.keyframe.name,
                )
            )
        return tags

    def _fuse_modalities(
        self,
        transcript: Sequence[TranscriptSegment],
        scenes: Sequence[SceneSegment],
        visual_tags: Sequence[VisualTag],
    ) -> SummaryResult:
        """Fuse text and visual tags into a structured summary."""

        return self.summarizer.summarize(transcript, scenes, visual_tags)


def extract_audio_track(video_path: Path, artifacts_dir: Path, *, sample_rate: int = 16000) -> Path:
    """Extract the audio track using ffmpeg or moviepy fallbacks."""

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    audio_path = artifacts_dir / f"{video_path.stem}_audio.wav"

    if audio_path.exists():
        return audio_path

    ffmpeg_binary = shutil.which("ffmpeg")
    if ffmpeg_binary:
        command = [
            ffmpeg_binary,
            "-y",
            "-i",
            str(video_path),
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ar",
            str(sample_rate),
            "-ac",
            "1",
            str(audio_path),
        ]
        try:
            subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return audio_path
        except subprocess.CalledProcessError:
            audio_path.unlink(missing_ok=True)

    try:
        from moviepy.editor import AudioFileClip, VideoFileClip
    except Exception:  # pragma: no cover - optional dependency
        raise RuntimeError("Neither ffmpeg nor moviepy are available to extract audio.")

    with VideoFileClip(str(video_path)) as clip:
        audio_clip = clip.audio
        if audio_clip is None:
            raise RuntimeError("Input video does not contain an audio track.")
        audio: AudioFileClip = audio_clip
        audio.write_audiofile(str(audio_path), fps=sample_rate, nbytes=2, codec="pcm_s16le")
    return audio_path


def sample_scene_keyframes(
    video_path: Path, artifacts_dir: Path, interval_seconds: float = 5.0
) -> list[SceneSegment]:
    """Sample keyframes at a fixed interval using ffmpeg with a moviepy fallback."""

    frames_dir = artifacts_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    duration = _probe_duration(video_path)
    frame_pattern = frames_dir / "frame_%04d.jpg"

    ffmpeg_binary = shutil.which("ffmpeg")
    if ffmpeg_binary:
        command = [
            ffmpeg_binary,
            "-y",
            "-i",
            str(video_path),
            "-vf",
            f"fps=1/{interval_seconds}",
            str(frame_pattern),
        ]
        try:
            subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError:
            pass

    frames = sorted(frames_dir.glob("frame_*.jpg"))
    if not frames:
        frames = _sample_with_moviepy(video_path, frames_dir, interval_seconds)

    scenes: list[SceneSegment] = []
    for index, frame in enumerate(frames):
        start = index * interval_seconds
        end = min(start + interval_seconds, duration or start + interval_seconds)
        scenes.append(SceneSegment(start=start, end=end, keyframe=frame))

    return scenes


def _sample_with_moviepy(video_path: Path, frames_dir: Path, interval_seconds: float) -> list[Path]:
    try:
        from moviepy.editor import VideoFileClip
    except Exception:  # pragma: no cover - optional dependency
        raise RuntimeError("Neither ffmpeg nor moviepy are available to sample frames.")

    frames: list[Path] = []
    with VideoFileClip(str(video_path)) as clip:
        t = 0.0
        index = 0
        while t < clip.duration:
            frame_path = frames_dir / f"frame_{index:04d}.jpg"
            clip.save_frame(str(frame_path), t)
            frames.append(frame_path)
            t += interval_seconds
            index += 1
    return frames


def _probe_duration(video_path: Path) -> float | None:
    ffprobe_binary = shutil.which("ffprobe")
    if not ffprobe_binary:
        return None

    command = [
        ffprobe_binary,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    try:
        output = subprocess.check_output(command, stderr=subprocess.DEVNULL)
        return float(output.decode().strip())
    except (subprocess.CalledProcessError, ValueError):
        return None
