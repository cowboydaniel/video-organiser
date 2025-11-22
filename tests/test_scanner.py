from pathlib import Path

from app.services.scanner import ScanFilters, scan_media


def test_scan_media_respects_filters(tmp_path: Path) -> None:
    visible = tmp_path / "video.mp4"
    visible.touch()
    hidden = tmp_path / ".hidden.mkv"
    hidden.touch()
    excluded_dir = tmp_path / "__pycache__"
    excluded_dir.mkdir()
    (excluded_dir / "skip.mp4").touch()

    results = scan_media(tmp_path, filters=ScanFilters(skip_hidden=True))
    discovered_paths = {result.path for result in results}

    assert visible in discovered_paths
    assert hidden not in discovered_paths
    assert all(path.parent != excluded_dir for path in discovered_paths)
