"""Utility helpers for processing tasks."""

from __future__ import annotations

import logging
import time
import importlib
from typing import Callable, Iterable, TypeVar


T = TypeVar("T")


def retry_with_backoff(
    operation: Callable[[], T],
    *,
    attempts: int = 3,
    base_delay: float = 0.5,
    exceptions: Iterable[type[Exception]] | tuple[type[Exception], ...] = (Exception,),
    logger: logging.Logger | None = None,
    description: str = "operation",
) -> T:
    """Execute ``operation`` with simple exponential backoff.

    Parameters
    ----------
    operation:
        Callable that performs the work and may raise exceptions.
    attempts:
        Number of attempts to make before surfacing the exception.
    base_delay:
        Initial delay in seconds before retrying; doubled on each attempt.
    exceptions:
        Exception types that should trigger a retry.
    logger:
        Optional logger for emitting diagnostic messages.
    description:
        Human readable label for the operation being attempted.
    """

    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return operation()
        except tuple(exceptions) as exc:  # type: ignore[arg-type]
            last_error = exc
            if logger:
                logger.warning(
                    "%s failed on attempt %s/%s: %s", description, attempt, attempts, exc
                )
            if attempt == attempts:
                break
            time.sleep(base_delay * (2 ** (attempt - 1)))

    assert last_error is not None  # for type checkers
    raise last_error


def resolve_device(preference: str) -> str:
    """Resolve a user-provided device string to a concrete runtime target.

    "auto" prefers CUDA when available, otherwise falls back to CPU. Any other
    value is returned unchanged so callers can opt into explicit devices such as
    ``cuda:0`` or ``mps`` when supported by the local torch build.
    """

    if preference != "auto":
        return preference

    torch_spec = importlib.util.find_spec("torch")
    if torch_spec is None:
        return "cpu"

    torch = importlib.import_module("torch")
    if torch.cuda.is_available():
        return "cuda"

    mps_backend = getattr(getattr(torch, "backends", None), "mps", None)
    if mps_backend and mps_backend.is_available():
        return "mps"

    return "cpu"

