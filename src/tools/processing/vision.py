"""Vision utilities for sampling frames, scene detection, and tagging."""
from __future__ import annotations

import hashlib
import json
import logging
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from .pipeline import SceneSegment, VisualTag, _probe_duration
from .utils import retry_with_backoff

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class VisionCache:
    """Cached representation of a prior vision analysis."""

    signature: str
    scenes: list[SceneSegment]
    tags: list[VisualTag]


class VisionAnalyzer:
    """Detect scene boundaries and tag visual content with caching support."""

    def __init__(self, scene_interval: float = 5.0, cache_root: Path | None = None) -> None:
        self.scene_interval = scene_interval
        self.cache_root = cache_root or Path(tempfile.gettempdir())

    def analyze(self, video_path: Path, artifacts_dir: Path) -> tuple[list[SceneSegment], list[VisualTag]]:
        """Return scenes and visual tags for ``video_path`` using cached results when available."""

        signature = media_signature(video_path)
        cached = load_cache(artifacts_dir, signature)
        if cached:
            return cached.scenes, cached.tags

        frames_dir = artifacts_dir / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)

        scenes = self._detect_and_sample_scenes(video_path, frames_dir)
        tags = self._tag_scenes(scenes)

        save_cache(artifacts_dir, signature, scenes, tags)
        return scenes, tags

    def _detect_and_sample_scenes(self, video_path: Path, frames_dir: Path) -> list[SceneSegment]:
        boundaries = list(self._detect_scene_boundaries(video_path))
        if not boundaries:
            boundaries = list(self._fallback_intervals(video_path))

        scenes: list[SceneSegment] = []
        for index, (start, end) in enumerate(boundaries):
            midpoint = start + (end - start) / 2 if end > start else start
            keyframe = self._extract_frame(video_path, midpoint, frames_dir, index)
            if keyframe is None:
                continue
            scenes.append(SceneSegment(start=start, end=end, keyframe=keyframe))

        if scenes:
            return scenes

        fallback_boundaries = list(self._fallback_intervals(video_path))
        for index, (start, end) in enumerate(fallback_boundaries):
            midpoint = start + (end - start) / 2 if end > start else start
            keyframe = self._extract_frame(video_path, midpoint, frames_dir, index)
            if keyframe is None:
                continue
            scenes.append(SceneSegment(start=start, end=end, keyframe=keyframe))

        return scenes

    def _detect_scene_boundaries(self, video_path: Path) -> Iterable[tuple[float, float]]:
        try:
            from scenedetect import ContentDetector, SceneManager, VideoManager
        except Exception:
            logger.debug("PySceneDetect not available; falling back to interval sampling")
            return []

        video_manager = VideoManager([str(video_path)])
        scene_manager = SceneManager()
        scene_manager.add_detector(ContentDetector())

        try:
            video_manager.start()
            scene_manager.detect_scenes(video_manager)
            scene_list = scene_manager.get_scene_list()
        finally:
            video_manager.release()

        if not scene_list:
            return []

        boundaries: list[tuple[float, float]] = []
        for start_time, end_time in scene_list:
            boundaries.append((start_time.get_seconds(), end_time.get_seconds()))

        return boundaries

    def _fallback_intervals(self, video_path: Path) -> Iterable[tuple[float, float]]:
        duration = _probe_duration(video_path) or 0.0
        index = 0
        current_start = 0.0
        while current_start < max(duration, self.scene_interval):
            end = min(current_start + self.scene_interval, duration or current_start + self.scene_interval)
            yield (current_start, end)
            index += 1
            current_start = index * self.scene_interval

    def _extract_frame(
        self, video_path: Path, timestamp: float, frames_dir: Path, index: int
    ) -> Path | None:
        frame_path = frames_dir / f"scene_{index:04d}.jpg"

        ffmpeg_binary = shutil.which("ffmpeg")
        if ffmpeg_binary:
            command = [
                ffmpeg_binary,
                "-y",
                "-ss",
                str(timestamp),
                "-i",
                str(video_path),
                "-frames:v",
                "1",
                str(frame_path),
            ]
            try:
                subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return frame_path
            except subprocess.CalledProcessError:
                frame_path.unlink(missing_ok=True)

        try:
            from moviepy.editor import VideoFileClip
        except Exception:
            logger.warning("Could not extract frame; moviepy unavailable and ffmpeg failed.")
            return None

        with VideoFileClip(str(video_path)) as clip:
            if timestamp > clip.duration:
                timestamp = max(0.0, clip.duration - 0.01)
            clip.save_frame(str(frame_path), timestamp)
        return frame_path

    def _tag_scenes(self, scenes: Sequence[SceneSegment]) -> list[VisualTag]:
        classifier = self._load_classifier()
        tags: list[VisualTag] = []
        for scene in scenes:
            predictions = self._classify_frame(scene.keyframe, classifier)
            for prediction in predictions:
                tags.append(
                    VisualTag(
                        timestamp=scene.start,
                        label=prediction["label"],
                        confidence=prediction.get("score"),
                        source=scene.keyframe.name,
                    )
                )
        return tags

    def _load_classifier(self):
        try:
            from transformers import pipeline

            return pipeline(
                "image-classification",
                model="google/vit-base-patch16-224",
                top_k=3,
            )
        except Exception:
            logger.debug("Transformers pipeline unavailable; using heuristic fallback for tags")
            return None

    def _classify_frame(self, frame_path: Path, classifier) -> list[dict]:
        if classifier is None:
            return self._heuristic_tags(frame_path)

        try:
            predictions = retry_with_backoff(
                lambda: classifier(str(frame_path)),
                attempts=3,
                base_delay=0.5,
                logger=logger,
                description="vision model inference",
            )
        except Exception:
            logger.exception("Model inference failed after retries; using heuristic tags instead")
            return self._heuristic_tags(frame_path)

        if isinstance(predictions, dict):
            predictions = [predictions]
        formatted: list[dict] = []
        for item in predictions[:3]:
            formatted.append({"label": str(item.get("label", "unknown")), "score": float(item.get("score", 0.0))})
        return formatted

    def _heuristic_tags(self, frame_path: Path) -> list[dict]:
        try:
            from PIL import Image, ImageStat

            with Image.open(frame_path) as image:
                grayscale = image.convert("L")
                stat = ImageStat.Stat(grayscale)
                mean_brightness = stat.mean[0]
        except Exception:
            return [
                {"label": "unknown_scene", "score": None},
            ]

        label = "bright_scene" if mean_brightness > 127 else "dim_scene"
        return [
            {"label": label, "score": None},
        ]


def media_signature(video_path: Path) -> str:
    """Create a cache signature derived from file hash and modification time."""

    stat_result = video_path.stat()
    mtime = int(stat_result.st_mtime)
    file_hash = _hash_file(video_path)
    return f"{mtime}:{file_hash}"


def _hash_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def load_cache(artifacts_dir: Path, signature: str) -> VisionCache | None:
    cache_path = artifacts_dir / "vision_analysis.json"
    if not cache_path.exists():
        return None

    try:
        payload = json.loads(cache_path.read_text())
    except json.JSONDecodeError:
        return None

    if payload.get("signature") != signature:
        return None

    scenes: list[SceneSegment] = []
    for raw in payload.get("scenes", []):
        keyframe_ref = raw.get("keyframe")
        if not keyframe_ref:
            return None
        keyframe = artifacts_dir / keyframe_ref
        if not keyframe.exists():
            return None
        scenes.append(
            SceneSegment(
                start=float(raw.get("start", 0.0)),
                end=float(raw.get("end", 0.0)),
                keyframe=keyframe,
            )
        )

    tags: list[VisualTag] = []
    for raw in payload.get("tags", []):
        tags.append(
            VisualTag(
                timestamp=float(raw.get("timestamp", 0.0)),
                label=str(raw.get("label", "")),
                confidence=raw.get("confidence"),
                source=str(raw.get("source")),
            )
        )

    return VisionCache(signature=signature, scenes=scenes, tags=tags)


def save_cache(
    artifacts_dir: Path, signature: str, scenes: Sequence[SceneSegment], tags: Sequence[VisualTag]
) -> None:
    cache_path = artifacts_dir / "vision_analysis.json"

    serialized_scenes = []
    for scene in scenes:
        try:
            relative_keyframe = scene.keyframe.relative_to(artifacts_dir)
        except ValueError:
            relative_keyframe = scene.keyframe.name
        serialized_scenes.append(
            {
                "start": scene.start,
                "end": scene.end,
                "keyframe": str(relative_keyframe),
            }
        )
    serialized_tags = [
        {
            "timestamp": tag.timestamp,
            "label": tag.label,
            "confidence": tag.confidence,
            "source": tag.source,
        }
        for tag in tags
    ]

    payload = {"signature": signature, "scenes": serialized_scenes, "tags": serialized_tags}
    cache_path.write_text(json.dumps(payload, indent=2))
