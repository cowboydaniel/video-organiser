# Video Organiser

Video Organiser is a PySide6 desktop tool for scanning media folders, extracting metadata, and applying rule-based organisation to your library. A lightweight CLI is also provided for headless workflows.

## Setup

1. Install Python 3.11 or later.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   * Install `ffmpeg`/`ffprobe` via your OS package manager so audio extraction and
     frame sampling work reliably.
   * For GPU acceleration, install a CUDA-enabled build of PyTorch before
     installing the requirements file (e.g. `pip install torch --index-url https://download.pytorch.org/whl/cu121`).
   * Whisper and the vision classifier rely on `openai-whisper` and
     `transformers` respectively; both are pulled in via `requirements.txt`.
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

Run a background preflight to verify required binaries and model availability:

```bash
video-organiser-cli preflight --model-size base --device auto --vision-device auto
```

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

## Performance and quality tuning

* **Frame sampling interval** (`--scene-interval`): lower values (e.g. 2–3s) produce
  more keyframes and potentially better visual tags at the cost of extra ffmpeg
  work. Laptops should start with 5s; desktops/GPUs can drop to 3s.
* **Whisper model size** (`--model-size`): `small` runs comfortably on laptops
  (CPU or integrated GPU) while `base` is a good default. Desktops with a recent
  GPU can use `medium` or `large` for higher accuracy.
* **Devices** (`--device`, `--vision-device`): set to `auto` to pick the best
  available accelerator. Force `cpu` to reduce heat on ultraportables; set
  `cuda`/`mps` to guarantee GPU usage when available.
* **Audio sample rate** (`--sample-rate`): 16 kHz is the sweet spot for Whisper;
  only raise this if you have ample CPU headroom.

Recommended presets:

* **Laptops/ultrabooks**: `--model-size base --device cpu --vision-device cpu --scene-interval 5.0`
* **Desktops with GPU**: `--model-size small --device cuda --vision-device cuda --scene-interval 3.0`

## Continuous integration

GitHub Actions run linting and tests on pushes and pull requests to ensure service correctness and prevent regressions.

## Screenshots

Screenshots of the GUI will be added once the interface stabilises.
