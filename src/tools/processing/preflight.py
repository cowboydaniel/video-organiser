"""Preflight checks for verifying model and binary availability."""
from __future__ import annotations

import importlib
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

from .utils import resolve_device


@dataclass(slots=True)
class CheckResult:
    """Represents the outcome of a preflight step."""

    name: str
    status: str
    detail: str


def check_required_binaries() -> list[CheckResult]:
    """Ensure ffmpeg/ffprobe are available for audio and frame extraction."""

    results: list[CheckResult] = []
    for binary in ("ffmpeg", "ffprobe"):
        path = shutil.which(binary)
        if path:
            results.append(CheckResult(name=f"{binary} binary", status="ok", detail=f"Found at {path}"))
        else:
            results.append(
                CheckResult(name=f"{binary} binary", status="error", detail="Missing; install via your package manager")
            )
    return results


def check_whisper(model_size: str, device: str) -> list[CheckResult]:
    """Verify Whisper can be imported and the requested device is usable."""

    results: list[CheckResult] = []
    whisper_spec = importlib.util.find_spec("whisper")
    if whisper_spec is None:
        results.append(
            CheckResult(
                name="Whisper package",
                status="error",
                detail="Install `openai-whisper` to enable transcription",
            )
        )
        return results

    whisper = importlib.import_module("whisper")
    resolved_device = resolve_device(device)

    try:
        whisper.load_model(model_size, device=resolved_device)
        results.append(
            CheckResult(
                name="Whisper model",
                status="ok",
                detail=f"{model_size} model load succeeded on {resolved_device}",
            )
        )
    except Exception as exc:  # pragma: no cover - runtime specific
        results.append(
            CheckResult(
                name="Whisper model",
                status="error",
                detail=f"Could not load {model_size} on {resolved_device}: {exc}",
            )
        )

    return results


def check_vision_pipeline(device: str) -> list[CheckResult]:
    """Verify transformers image pipeline can be constructed."""

    results: list[CheckResult] = []
    transformers_spec = importlib.util.find_spec("transformers")
    if transformers_spec is None:
        results.append(
            CheckResult(
                name="Transformers package",
                status="error",
                detail="Install `transformers` to enable vision tagging",
            )
        )
        return results

    transformers = importlib.import_module("transformers")
    resolved_device = resolve_device(device)
    device_argument: int | str = -1 if resolved_device == "cpu" else resolved_device

    try:
        torch_spec = importlib.util.find_spec("torch")
        if torch_spec is not None and resolved_device != "cpu":
            torch = importlib.import_module("torch")
            device_argument = torch.device(resolved_device)
    except Exception:  # pragma: no cover - defensive
        device_argument = -1

    try:
        transformers.pipeline(
            "image-classification",
            model="google/vit-base-patch16-224",
            top_k=1,
            device=device_argument,
        )
        results.append(
            CheckResult(
                name="Vision model",
                status="ok",
                detail=f"ViT classifier ready on {resolved_device}",
            )
        )
    except Exception as exc:  # pragma: no cover - runtime specific
        results.append(
            CheckResult(
                name="Vision model",
                status="error",
                detail=f"Could not initialise vision model on {resolved_device}: {exc}",
            )
        )

    return results


def run_preflight_checks(model_size: str, transcription_device: str, vision_device: str) -> list[CheckResult]:
    """Run preflight checks concurrently and flatten the results."""

    checks: list[CheckResult] = []
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [
            executor.submit(check_required_binaries),
            executor.submit(check_whisper, model_size, transcription_device),
            executor.submit(check_vision_pipeline, vision_device),
        ]
        for future in as_completed(futures):
            checks.extend(future.result())

    return checks
