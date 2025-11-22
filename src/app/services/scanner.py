"""Directory scanning utilities for locating media files."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from queue import Queue
from typing import Callable, Iterator, List, Sequence

from PySide6 import QtCore

VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".flv", ".wmv"}


@dataclass(slots=True)
class ScanFilters:
    """Filters used during media scanning."""

    include_extensions: Sequence[str] = tuple(sorted(VIDEO_EXTENSIONS))
    exclude_directories: Sequence[str] = (".git", "__pycache__")
    skip_hidden: bool = True

    def normalized_exts(self) -> set[str]:
        return {ext.lower() for ext in self.include_extensions}


@dataclass(slots=True)
class ScanResult:
    """Represents a single scan discovery."""

    path: Path
    is_valid: bool


class MediaScanner(QtCore.QObject):
    """Walk directories and emit progress updates for media files."""

    discovered = QtCore.Signal(object)  # emits ScanResult
    progressed = QtCore.Signal(int, int)  # processed, total
    finished = QtCore.Signal(list)

    def __init__(
        self,
        root: Path,
        filters: ScanFilters | None = None,
        queue: Queue[ScanResult] | None = None,
        parent: QtCore.QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self.root = root
        self.filters = filters or ScanFilters()
        self.queue = queue
        self._stopped = False

    def stop(self) -> None:
        """Request the scan loop to stop early."""

        self._stopped = True

    def scan(self) -> List[ScanResult]:
        """Perform the directory scan synchronously.

        Discovered results are emitted through Qt signals and optionally placed onto a
        thread-safe :class:`queue.Queue` for consumption by non-Qt components.
        """

        results: List[ScanResult] = []
        files = list(self._iter_files())
        total = len(files)
        for index, path in enumerate(files, start=1):
            if self._stopped:
                break
            result = ScanResult(path=path, is_valid=True)
            results.append(result)
            self.discovered.emit(result)
            if self.queue is not None:
                self.queue.put(result)
            self.progressed.emit(index, total)
        self.finished.emit(results)
        return results

    def _iter_files(self) -> Iterator[Path]:
        exts = self.filters.normalized_exts()
        for root, dirs, files in os.walk(self.root):
            if self._stopped:
                break
            dirs[:] = [d for d in dirs if not self._should_skip_dir(d)]
            for name in files:
                if self._stopped:
                    break
                if self.filters.skip_hidden and name.startswith('.'):
                    continue
                path = Path(root, name)
                if path.suffix.lower() in exts:
                    yield path

    def _should_skip_dir(self, directory: str) -> bool:
        if self.filters.skip_hidden and directory.startswith('.'):
            return True
        return directory in self.filters.exclude_directories


def scan_media(
    root: Path,
    filters: ScanFilters | None = None,
    on_discovered: Callable[[ScanResult], None] | None = None,
) -> List[ScanResult]:
    """Convenience function to scan media without Qt consumers."""

    scanner = MediaScanner(root=root, filters=filters)

    if on_discovered is not None:
        scanner.discovered.connect(on_discovered)

    return scanner.scan()
