"""Media processing pipeline for extracting audio, frames, and semantic signals."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


@dataclass(slots=True)
class TranscriptSegment:
    """Represents a span of text derived from speech."""

    start: float
    end: float
    text: str
    speaker: str | None = None


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
    artifacts_dir: Path


class ProcessingPipeline:
    """Orchestrate media processing for multi-modal analysis."""

    def __init__(self, cache_root: Path | None = None, scene_interval: float = 5.0) -> None:
        self.cache_root = cache_root or Path(tempfile.gettempdir())
        self.scene_interval = scene_interval

    def process(self, video_path: Path) -> AnalysisResult:
        """Run the pipeline over ``video_path`` and return structured outputs."""

        normalized_path = video_path.expanduser().resolve()
        artifacts_dir = self._prepare_artifacts_dir(normalized_path)

        audio_path = extract_audio_track(normalized_path, artifacts_dir)
        transcript = self._transcribe_audio(audio_path)
        scenes = sample_scene_keyframes(normalized_path, artifacts_dir, interval_seconds=self.scene_interval)
        visual_tags = self._tag_scenes(scenes)
        fused_summary = self._fuse_modalities(transcript, visual_tags)

        return AnalysisResult(
            video_path=normalized_path,
            audio_path=audio_path,
            transcript=transcript,
            scenes=scenes,
            visual_tags=visual_tags,
            fused_summary=fused_summary,
            artifacts_dir=artifacts_dir,
        )

    def _prepare_artifacts_dir(self, video_path: Path) -> Path:
        stem = video_path.stem.replace(" ", "_")
        target = self.cache_root / f"processing_{stem}"
        target.mkdir(parents=True, exist_ok=True)
        return target

    def _transcribe_audio(self, audio_path: Path) -> list[TranscriptSegment]:
        """Placeholder transcription step producing time-aligned segments."""

        # In a full implementation this would call a speech-to-text engine.
        return [
            TranscriptSegment(start=0.0, end=5.0, text="Transcription not implemented.", speaker=None),
            TranscriptSegment(start=5.0, end=10.0, text="Replace with real ASR output when available.", speaker=None),
        ]

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
        self, transcript: Sequence[TranscriptSegment], visual_tags: Sequence[VisualTag]
    ) -> str:
        """Fuse text and visual tags into a human-readable summary."""

        transcript_text = " ".join(segment.text for segment in transcript)
        tag_labels = ", ".join(tag.label for tag in visual_tags)
        if not tag_labels:
            return transcript_text
        return f"{transcript_text}\nDetected visuals: {tag_labels}."


def extract_audio_track(video_path: Path, artifacts_dir: Path) -> Path:
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
            "16000",
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
        audio.write_audiofile(str(audio_path), fps=16000, nbytes=2, codec="pcm_s16le")
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
