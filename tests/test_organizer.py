from pathlib import Path

from pathlib import Path

from app.services.metadata import MediaMetadata
from app.services.organizer import Organizer
from app.services.rules import RuleEngine


class StaticMetadataReader:
    def read(self, path: Path) -> MediaMetadata:
        return MediaMetadata(duration=None, resolution=None, codec=None, tags={})


def test_commit_copies_files(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    dest_dir = tmp_path / "dest"
    source_dir.mkdir()
    dest_dir.mkdir()

    first = source_dir / "first.mp4"
    second = source_dir / "second.mp4"
    first.write_text("one")
    second.write_text("two")

    engine = RuleEngine(destination_root=dest_dir, folder_template="sorted", filename_template="{name}")
    organizer = Organizer(rule_engine=engine, metadata_reader=StaticMetadataReader())

    plans = organizer.preview([first, second])
    reports = organizer.commit(plans, copy_files=True)

    assert all(report.success for report in reports)
    for source in [first, second]:
        destination = dest_dir / "sorted" / f"{source.stem}.mp4"
        assert destination.exists()
        assert source.exists()


def test_commit_rolls_back_on_failure(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    dest_dir = tmp_path / "dest"
    source_dir.mkdir()
    dest_dir.mkdir()

    first = source_dir / "first.mp4"
    second = source_dir / "second.mp4"
    first.write_text("one")
    second.write_text("two")

    engine = RuleEngine(destination_root=dest_dir, folder_template="sorted", filename_template="{name}")
    organizer = Organizer(rule_engine=engine, metadata_reader=StaticMetadataReader())

    plans = organizer.preview([first, second])
    conflict = dest_dir / "sorted" / "second.mp4"
    conflict.parent.mkdir(parents=True, exist_ok=True)
    conflict.write_text("existing")

    reports = organizer.commit(plans, copy_files=False)

    assert any(not report.success for report in reports)
    assert first.exists()
    assert second.exists()
    assert not (dest_dir / "sorted" / "first.mp4").exists()
