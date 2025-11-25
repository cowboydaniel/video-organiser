"""Async task runner with progress event support."""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Optional

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class TaskProgress:
    """Represents the lifecycle of a queued task."""

    task_id: str
    status: str
    message: str | None = None
    progress: float | None = None


@dataclass(slots=True)
class QueuedTask:
    """Internal representation of a pending task."""

    description: str
    operation: Callable[[Callable[[str, float | None], None]], Awaitable[object]]
    task_id: str = field(default_factory=lambda: uuid.uuid4().hex)


class TaskRunner:
    """Run long-lived tasks sequentially with progress callbacks."""

    def __init__(self, *, max_concurrency: int = 1, loop: Optional[asyncio.AbstractEventLoop] = None):
        self.loop = loop or asyncio.get_event_loop()
        self.max_concurrency = max_concurrency
        self.queue: asyncio.Queue[QueuedTask] = asyncio.Queue()
        self._workers: list[asyncio.Task] = []
        self._subscribers: list[Callable[[TaskProgress], None]] = []
        self._stopped = False

    def subscribe(self, callback: Callable[[TaskProgress], None]) -> None:
        """Register a subscriber that will receive progress updates."""

        self._subscribers.append(callback)

    async def start(self) -> None:
        """Start background workers if they are not already running."""

        if self._workers:
            return
        for _ in range(self.max_concurrency):
            self._workers.append(self.loop.create_task(self._worker()))

    async def stop(self) -> None:
        """Signal workers to exit once the queue drains."""

        self._stopped = True
        await self.queue.join()
        for worker in self._workers:
            worker.cancel()
        self._workers.clear()

    async def enqueue(
        self,
        operation: Callable[[Callable[[str, float | None], None]], Awaitable[object]],
        *,
        description: str = "",
    ) -> str:
        """Enqueue an async operation for execution and return its task id."""

        task = QueuedTask(description=description, operation=operation)
        await self.queue.put(task)
        self._emit(TaskProgress(task_id=task.task_id, status="queued", message=description))
        return task.task_id

    async def _worker(self) -> None:
        while not self._stopped:
            task = await self.queue.get()
            reporter = self._build_reporter(task)
            try:
                self._emit(TaskProgress(task_id=task.task_id, status="started", message=task.description))
                await task.operation(reporter)
                self._emit(TaskProgress(task_id=task.task_id, status="completed", message=task.description))
            except asyncio.CancelledError:
                self._emit(TaskProgress(task_id=task.task_id, status="cancelled", message=task.description))
                raise
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.exception("Task %s failed: %s", task.task_id, exc)
                self._emit(
                    TaskProgress(
                        task_id=task.task_id,
                        status="failed",
                        message=str(exc),
                    )
                )
            finally:
                self.queue.task_done()

    def _build_reporter(self, task: QueuedTask) -> Callable[[str, float | None], None]:
        def reporter(message: str, progress: float | None = None) -> None:
            self._emit(
                TaskProgress(
                    task_id=task.task_id,
                    status="running",
                    message=message,
                    progress=progress,
                )
            )

        return reporter

    def _emit(self, progress: TaskProgress) -> None:
        for subscriber in list(self._subscribers):
            try:
                subscriber(progress)
            except Exception:  # pragma: no cover - user provided callbacks
                logger.exception("Progress subscriber failed for task %s", progress.task_id)


__all__ = ["TaskRunner", "TaskProgress"]

