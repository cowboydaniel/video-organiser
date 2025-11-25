# Video Organiser

Video Organiser is a PySide6 desktop tool for scanning media folders, extracting metadata, and applying rule-based organisation to your library. A lightweight CLI is also provided for headless workflows.

## Setup

1. Install Python 3.11 or later.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. (Optional) For development, install linting and test tooling:
   ```bash
   pip install -r requirements-dev.txt
   ```

## Running the application

### GUI

Launch the Qt application:

```bash
python -m app.main
```

Select a source directory, adjust rule presets (output folder, folder template, filename template, and custom tags), then preview or apply the organiser. The app persists your last-used directory, rule presets, and theme to `~/.config/video-organiser/config.json` via the settings manager.

### CLI

Use the Typer-powered CLI to organise videos without the UI:

```bash
video-organiser-cli organize /path/to/videos --destination ~/Videos/Organized \
  --folder-template "{date:%Y/%m}" --filename-template "{name}_{resolution}" --copy
```

Run `video-organiser-cli --help` for the full command list.

## Metadata caching and settings

* Metadata reads are cached to `~/.config/video-organiser/metadata_cache.json` to speed up repeat scans. Cache entries are keyed by the file path, size, and modification time so updates invalidate stale values.
* User preferences (last directory, rule presets, theme) are stored in `~/.config/video-organiser/config.json`. You can delete this file to reset to defaults.

## Tests and linting

Run the test suite with pytest:

```bash
pytest
```

Regression coverage includes a curated metadata-only sample set in `tests/fixtures/sample_videos.json` with expected titles,
categories, and tags. The processing pipeline tests stub external dependencies, generate placeholder audio, and track metrics
such as transcript word error rate, tag precision/recall, and title relevance against the descriptors.

Run linting with Ruff:

```bash
ruff check .
```

## Continuous integration

GitHub Actions run linting and tests on pushes and pull requests to ensure service correctness and prevent regressions.

## Screenshots

Screenshots of the GUI will be added once the interface stabilises.
