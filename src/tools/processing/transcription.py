"""Transcription helpers wrapping Whisper models with chunking support."""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence, Tuple

from .pipeline import TranscriptSegment
from .utils import resolve_device

logger = logging.getLogger(__name__)

@dataclass(slots=True)
class TranscriptionConfig:
    """Configuration for the transcription backend."""

    model_size: str = "base"
    device: str = "cpu"
    sample_rate: int = 16000
    chunk_duration: float = 300.0
    diarization: bool = False
    auto_detect_language: bool = True
    confidence_threshold: float = 0.0


@dataclass(slots=True)
class TranscriptionResult:
    """Container describing the output of a transcription run."""

    segments: list[TranscriptSegment]
    detected_language: str | None


class TranscriptionService:
    """Wrap Whisper (or whisper.cpp) models with chunking for long files."""

    def __init__(self, config: TranscriptionConfig | None = None) -> None:
        self.config = config or TranscriptionConfig()
        self._model = None

    def transcribe(
        self,
        audio_path: Path,
        *,
        progress_callback: Callable[[str, float], None] | None = None,
    ) -> TranscriptionResult:
        """Transcribe ``audio_path`` into time-aligned segments.

        ``progress_callback`` accepts ``(message: str, fraction: float)`` to report
        progress within the transcription stage.
        """

        model = self._load_model()
        duration = self._probe_duration(audio_path)
        chunks = self._chunk_audio(audio_path, duration)

        all_segments: list[TranscriptSegment] = []
        detected_language: str | None = None

        total_chunks = len(chunks)
        for index, (chunk_path, offset) in enumerate(chunks):
            if progress_callback:
                progress_callback(
                    f"Transcribing chunk {index + 1}/{total_chunks} (offset {int(offset)}s)",
                    index / max(total_chunks, 1),
                )
            try:
                result = model.transcribe(
                    str(chunk_path),
                    word_timestamps=True,
                    language=None,
                    fp16=self.config.device != "cpu",
                )
            except Exception as exc:
                logger.error("Model transcription failed for chunk %s: %s", chunk_path, exc)
                continue
            if self.config.auto_detect_language:
                detected_language = detected_language or result.get("language")
            language_tag = detected_language if self.config.auto_detect_language else None
            all_segments.extend(self._parse_segments(result.get("segments", []), offset, language_tag))

            if progress_callback:
                progress_callback(
                    f"Finished chunk {index + 1}/{total_chunks}",
                    (index + 1) / max(total_chunks, 1),
                )

        return TranscriptionResult(segments=all_segments, detected_language=detected_language)

    def _load_model(self):  # type: ignore[override]
        if self._model is not None:
            return self._model

        import importlib

        whisper_spec = importlib.util.find_spec("whisper")
        if whisper_spec is None:
            raise RuntimeError(
                "Whisper is not installed. Install `openai-whisper` or configure whisper.cpp for transcription."
            )

        logger.info("Loading Whisper model '%s' on device '%s'...", self.config.model_size, self.config.device)
        logger.info("(First-time model download may take several minutes)")
        whisper = importlib.import_module("whisper")
        device = resolve_device(self.config.device)
        self._model = whisper.load_model(self.config.model_size, device=device)
        logger.info("Whisper model loaded successfully")
        return self._model

    def _chunk_audio(self, audio_path: Path, duration: float | None) -> list[Tuple[Path, float]]:
        if duration is None or duration <= self.config.chunk_duration:
            return [(audio_path, 0.0)]

        if not shutil.which("ffmpeg"):
            return [(audio_path, 0.0)]

        chunks: list[Tuple[Path, float]] = []
        temp_dir = Path(tempfile.mkdtemp(prefix="transcription_chunks_"))
        total_chunks = int(math.ceil(duration / self.config.chunk_duration))

        for index in range(total_chunks):
            start = index * self.config.chunk_duration
            chunk_path = temp_dir / f"chunk_{index:04d}.wav"
            command = [
                "ffmpeg",
                "-y",
                "-i",
                str(audio_path),
                "-ss",
                str(start),
                "-t",
                str(self.config.chunk_duration),
                str(chunk_path),
            ]
            try:
                subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except subprocess.CalledProcessError:
                logger.warning("Failed to extract audio chunk at %ss", start)
                continue
            chunks.append((chunk_path, start))

        return chunks or [(audio_path, 0.0)]

    def _parse_segments(
        self, segments: Sequence[dict], offset: float, detected_language: str | None
    ) -> Iterable[TranscriptSegment]:
        for raw in segments:
            confidence = self._segment_confidence(raw)
            if confidence is not None and confidence < self.config.confidence_threshold:
                continue

            speaker_label = None
            if self.config.diarization:
                speaker_index = raw.get("speaker") or 0
                speaker_label = f"Speaker {int(speaker_index) + 1}"

            yield TranscriptSegment(
                start=float(raw.get("start", 0.0)) + offset,
                end=float(raw.get("end", 0.0)) + offset,
                text=str(raw.get("text", "")).strip(),
                speaker=speaker_label,
                language=detected_language,
                confidence=confidence,
            )

    def _segment_confidence(self, segment: dict) -> float | None:
        words = segment.get("words")
        if not words:
            return None

        probabilities = [word.get("probability") for word in words if word.get("probability") is not None]
        if not probabilities:
            return None

        return float(sum(probabilities) / len(probabilities))

    def _probe_duration(self, audio_path: Path) -> float | None:
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
            str(audio_path),
        ]
        try:
            output = subprocess.check_output(command, stderr=subprocess.DEVNULL)
            return float(output.decode().strip())
        except (subprocess.CalledProcessError, ValueError):
            return None
