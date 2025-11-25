"""Processing pipeline helpers for multi-modal analysis."""

from .pipeline import (
    AnalysisResult,
    ProcessingPipeline,
    SceneSegment,
    TranscriptSegment,
    VisualTag,
)
from .metrics import tag_precision_recall, title_relevance_score, word_error_rate
from .summarizer import SceneEvidence, Summarizer, SummarizerConfig, SummaryResult
from .transcription import TranscriptionConfig, TranscriptionResult, TranscriptionService
from .vision import VisionAnalyzer, media_signature

__all__ = [
    "AnalysisResult",
    "ProcessingPipeline",
    "SceneSegment",
    "TranscriptSegment",
    "VisualTag",
    "tag_precision_recall",
    "title_relevance_score",
    "word_error_rate",
    "SceneEvidence",
    "Summarizer",
    "SummarizerConfig",
    "SummaryResult",
    "TranscriptionConfig",
    "TranscriptionResult",
    "TranscriptionService",
    "VisionAnalyzer",
    "media_signature",
]
