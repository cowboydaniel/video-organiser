import os
from types import MethodType

from app.services.metadata import MediaMetadata, MetadataReader
from app.services.metadata_cache import MetadataCache


def test_metadata_reader_hits_cache(tmp_path) -> None:
    media_file = tmp_path / "sample.mp4"
    media_file.write_text("content")

    cache = MetadataCache(path=tmp_path / "cache.json")
    reader = MetadataReader(cache=cache)
    reader._ffprobe_path = "ffprobe"

    calls: list[str] = []

    def fake_ffprobe(self, path):
        calls.append(str(path))
        return MediaMetadata(duration=10.0, resolution=(1920, 1080), codec="h264", tags={})

    reader._read_with_ffprobe = MethodType(fake_ffprobe, reader)

    first = reader.read(media_file)
    assert first.resolution == (1920, 1080)
    assert len(calls) == 1

    reader._read_with_ffprobe = MethodType(
        lambda self, path: (_ for _ in ()).throw(RuntimeError("should use cache")), reader
    )
    second = reader.read(media_file)
    assert second.duration == 10.0
    assert len(calls) == 1

    reader._read_with_ffprobe = MethodType(fake_ffprobe, reader)
    media_file.write_text("updated content")
    os.utime(media_file, None)
    third = reader.read(media_file)

    assert third.codec == "h264"
    assert len(calls) == 2
