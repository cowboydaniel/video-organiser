"""Utility helpers for processing tasks."""

from __future__ import annotations

import logging
import time
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

