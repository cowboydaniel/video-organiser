"""Threading helpers for running IO outside the GUI thread."""
from __future__ import annotations

from typing import Any, Callable

from PySide6 import QtConcurrent, QtCore


class WorkerSignals(QtCore.QObject):
    """Signals emitted by background workers."""

    finished = QtCore.Signal(object)
    error = QtCore.Signal(object)
    result = QtCore.Signal(object)


class _Worker(QtCore.QObject):
    def __init__(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

    @QtCore.Slot()
    def run(self) -> None:  # pragma: no cover - Qt callback
        try:
            result = self.func(*self.args, **self.kwargs)
        except Exception as exc:  # pragma: no cover - forwards to UI layer
            self.signals.error.emit(exc)
        else:
            self.signals.result.emit(result)
        finally:
            self.signals.finished.emit(None)


def run_in_qthread(func: Callable[..., Any], *args: Any, **kwargs: Any) -> tuple[QtCore.QThread, WorkerSignals]:
    """Execute ``func`` in a ``QThread`` and return the thread and signals."""

    thread = QtCore.QThread()
    worker = _Worker(func, *args, **kwargs)
    worker.moveToThread(thread)

    thread.started.connect(worker.run)
    worker.signals.finished.connect(thread.quit)
    worker.signals.finished.connect(worker.deleteLater)
    thread.finished.connect(thread.deleteLater)

    thread.start()
    return thread, worker.signals


def run_in_qtconcurrent(func: Callable[..., Any], *args: Any, **kwargs: Any) -> QtConcurrent.QFuture:
    """Run ``func`` using :mod:`QtConcurrent` and return the future handle."""

    return QtConcurrent.run(func, *args, **kwargs)
