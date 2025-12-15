"""CLI helpers powered by Typer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
from pydantic import BaseModel, Field

from app.services.metadata import MetadataReader
from app.services.metadata_cache import MetadataCache
from app.services.organizer import Organizer
from app.services.rules import RuleEngine
from app.services.scanner import VIDEO_EXTENSIONS
from tools.processing import ProcessingPipeline, TranscriptionConfig
from tools.processing.preflight import run_preflight_checks

app = typer.Typer(help="Utility commands for the Video Organiser.")


class AppConfig(BaseModel):
    """Example configuration for CLI helpers."""

    library_path: Path = Field(Path.home() / "Videos", description="Folder for library scanning.")
    dry_run: bool = Field(False, description="Whether to skip writing changes.")


@app.command()
def init_config(output: Optional[Path] = typer.Option(None, help="Where to write the sample config.")) -> None:
    """Write a sample configuration file in JSON format."""

    config = AppConfig()
    target = output or Path.cwd() / "video_organiser.config.json"
    target.write_text(config.model_dump_json(indent=2))
    typer.echo(f"Wrote config to {target}")


@app.command()
def greet(name: str = typer.Argument("friend", help="Name to greet.")) -> None:
    """Greet a user from the CLI."""

    typer.echo(f"Hello, {name}!")


@app.command()
def transcribe(
    video: Path = typer.Argument(..., exists=True, help="Video file to transcribe."),
    model_size: str = typer.Option("base", "--model-size", "-m", help="Whisper model size to load."),
    device: str = typer.Option(
        "auto", "--device", "-d", help="Compute device to run the Whisper model on."
    ),
    sample_rate: int = typer.Option(16000, help="Target audio sample rate for extraction."),
    confidence_threshold: float = typer.Option(
        0.0,
        "--confidence-threshold",
        min=0.0,
        max=1.0,
        help="Discard segments with average confidence below this value.",
    ),
    diarization: bool = typer.Option(False, help="Attach lightweight speaker labels to each segment."),
    language_detection: bool = typer.Option(True, help="Detect spoken language automatically."),
    chunk_duration: float = typer.Option(300.0, help="Chunk duration (seconds) for long recordings."),
) -> None:
    """Extract audio and run Whisper transcription with chunking support."""

    config = TranscriptionConfig(
        model_size=model_size,
        device=device,
        sample_rate=sample_rate,
        confidence_threshold=confidence_threshold,
        diarization=diarization,
        auto_detect_language=language_detection,
        chunk_duration=chunk_duration,
    )
    pipeline = ProcessingPipeline(transcription_config=config)
    result = pipeline.process(video)

    if not result.transcript:
        typer.echo("No transcript generated.")
        return

    typer.echo(f"Detected language: {result.transcript[0].language or 'unknown'}")
    for segment in result.transcript:
        speaker = f" [{segment.speaker}]" if segment.speaker else ""
        lang = f" ({segment.language})" if segment.language else ""
        conf = f" ({segment.confidence:.2f})" if segment.confidence is not None else ""
        typer.echo(f"[{segment.start:6.2f} - {segment.end:6.2f}]{speaker}{lang}{conf} {segment.text}")


@app.command()
def analyze(
    video: Path = typer.Argument(..., exists=True, help="Video file to analyze."),
    model_size: str = typer.Option("base", "--model-size", "-m", help="Whisper model size to load."),
    device: str = typer.Option(
        "auto", "--device", "-d", help="Compute device to run the Whisper model on."
    ),
    sample_rate: int = typer.Option(16000, help="Target audio sample rate for extraction."),
    confidence_threshold: float = typer.Option(
        0.0,
        "--confidence-threshold",
        min=0.0,
        max=1.0,
        help="Discard segments with average confidence below this value.",
    ),
    diarization: bool = typer.Option(False, help="Attach lightweight speaker labels to each segment."),
    language_detection: bool = typer.Option(True, help="Detect spoken language automatically."),
    chunk_duration: float = typer.Option(300.0, help="Chunk duration (seconds) for long recordings."),
    scene_interval: float = typer.Option(5.0, help="Keyframe sampling interval in seconds."),
    vision_device: str = typer.Option(
        "auto",
        "--vision-device",
        help="Compute device for the vision model (image classification).",
    ),
) -> None:
    """Run the full analysis pipeline and print structured JSON suggestions."""

    config = TranscriptionConfig(
        model_size=model_size,
        device=device,
        sample_rate=sample_rate,
        confidence_threshold=confidence_threshold,
        diarization=diarization,
        auto_detect_language=language_detection,
        chunk_duration=chunk_duration,
    )
    pipeline = ProcessingPipeline(
        transcription_config=config, scene_interval=scene_interval, vision_device=vision_device
    )

    with typer.progressbar(length=4, label="Analyzing") as progress:

        def _report(message: str, _progress: float | None = None) -> None:
            typer.echo(message)
            progress.update(1)

        result = pipeline.process(video, progress_callback=_report)

    payload = {
        "video": str(result.video_path),
        "title": result.summary.title,
        "event_or_topic": result.summary.event_or_topic,
        "folder_suggestion": result.summary.folder_suggestion,
        "visual_tags": [tag.label for tag in result.visual_tags],
        "transcript_snippets": [seg.text for seg in result.transcript[:5]],
    }
    typer.echo(json.dumps(payload, indent=2))


@app.command()
def preflight(
    model_size: str = typer.Option("base", "--model-size", "-m", help="Whisper model size to check."),
    transcription_device: str = typer.Option(
        "auto", "--device", "-d", help="Compute device to validate for Whisper."
    ),
    vision_device: str = typer.Option(
        "auto", "--vision-device", help="Compute device to validate for vision tagging."
    ),
) -> None:
    """Run dependency and model checks without blocking the main workflow."""

    typer.echo("Starting background preflight checks...")
    results = run_preflight_checks(model_size, transcription_device, vision_device)

    status_counts = {"ok": 0, "warning": 0, "error": 0}
    for result in results:
        status_counts[result.status] = status_counts.get(result.status, 0) + 1
        status_label = result.status.upper()
        typer.echo(f"[{status_label}] {result.name}: {result.detail}")

    if status_counts.get("error"):
        raise typer.Exit(code=1)


@app.command()
def organize(
    source: Path = typer.Argument(..., exists=True, file_okay=False, help="Directory containing media files."),
    destination: Path = typer.Option(Path.home() / "Videos" / "Organized", "--destination", "-d", help="Output root folder."),
    folder_template: str = typer.Option("{tag_event_or_topic}", help="Folder template including subject/topic placeholders."),
    filename_template: str = typer.Option("{name}_{resolution}", help="Filename template including metadata placeholders."),
    copy: bool = typer.Option(False, help="Copy files instead of moving them."),
    dry_run: bool = typer.Option(False, help="Only preview planned moves/copies."),
    tag: list[str] = typer.Option([], help="Custom tag in key=value form. Can be provided multiple times."),
    analyze: bool = typer.Option(False, help="Run the analysis pipeline before organizing."),
    scene_interval: float = typer.Option(5.0, help="Keyframe sampling interval in seconds."),
    vision_device: str = typer.Option(
        "auto",
        "--vision-device",
        help="Compute device for the vision model when running analysis.",
    ),
) -> None:
    """Apply rule-based organization in headless mode."""

    custom_tags: dict[str, str] = {}
    for entry in tag:
        if "=" in entry:
            key, value = entry.split("=", 1)
            custom_tags[key.strip()] = value.strip()

    rule_engine = RuleEngine(
        destination_root=destination.expanduser(),
        folder_template=folder_template,
        filename_template=filename_template,
    )
    organizer = Organizer(rule_engine=rule_engine, metadata_reader=MetadataReader(cache=MetadataCache()))
    pipeline: ProcessingPipeline | None = None

    media_paths = [
        path
        for path in source.expanduser().iterdir()
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    ]
    if not media_paths:
        typer.echo("No media files found to process.")
        raise typer.Exit(code=0)

    typer.echo(f"Discovered {len(media_paths)} file(s). Building plans...")
    plans = []
    for index, path in enumerate(media_paths, start=1):
        analysis_tags: dict[str, str] = {}
        if analyze:
            typer.echo(f"[{index}/{len(media_paths)}] Analyzing {path.name}")
            if pipeline is None:
                pipeline = ProcessingPipeline(scene_interval=scene_interval, vision_device=vision_device)
            with typer.progressbar(length=4, label=f"Analyze {path.name}") as progress:

                def _report(message: str, _progress: float | None = None) -> None:
                    typer.echo(message)
                    progress.update(1)

                result = pipeline.process(path, progress_callback=_report)
            analysis_tags = {
                "generated_title": result.summary.title,
                "generated_folder": result.summary.folder_suggestion,
                "event_or_topic": result.summary.event_or_topic,
            }
            typer.echo(
                json.dumps(
                    {
                        "file": path.name,
                        "title": result.summary.title,
                        "folder_suggestion": result.summary.folder_suggestion,
                        "event_or_topic": result.summary.event_or_topic,
                    },
                    indent=2,
                )
            )
        merged_tags = {**custom_tags, **analysis_tags}
        plan = organizer.rule_engine.resolve(
            path, organizer.metadata_reader.read(path), merged_tags
        )
        plans.append(plan)
        typer.echo(f"{plan.source.name} -> {plan.destination}")

    if dry_run:
        typer.echo("Dry-run enabled. No file operations performed.")
        return

    typer.echo("Applying plans...")

    def _progress(processed: int, total: int) -> None:
        typer.echo(f"Processed {processed}/{total}")

    reports = organizer.commit(plans, copy_files=copy, progress_callback=_progress)
    failures = [report for report in reports if not report.success]
    if failures:
        typer.echo("Some operations failed:")
        for report in failures:
            typer.echo(f"- {report.plan.source}: {report.error}")
        raise typer.Exit(code=1)

    typer.echo("Completed successfully.")


def main() -> None:
    """Entrypoint used by `python -m` invocations."""

    app()


if __name__ == "__main__":
    main()
