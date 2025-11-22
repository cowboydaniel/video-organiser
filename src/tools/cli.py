"""CLI helpers powered by Typer."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from pydantic import BaseModel, Field

from app.services.metadata import MetadataReader
from app.services.organizer import Organizer
from app.services.rules import RuleEngine
from app.services.scanner import VIDEO_EXTENSIONS

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
def organize(
    source: Path = typer.Argument(..., exists=True, file_okay=False, help="Directory containing media files."),
    destination: Path = typer.Option(Path.home() / "Videos" / "Organized", "--destination", "-d", help="Output root folder."),
    folder_template: str = typer.Option("{date:%Y/%m}", help="Folder template including date placeholders."),
    filename_template: str = typer.Option("{name}_{resolution}", help="Filename template including metadata placeholders."),
    copy: bool = typer.Option(False, help="Copy files instead of moving them."),
    dry_run: bool = typer.Option(False, help="Only preview planned moves/copies."),
    tag: list[str] = typer.Option([], help="Custom tag in key=value form. Can be provided multiple times."),
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
    organizer = Organizer(rule_engine=rule_engine, metadata_reader=MetadataReader())

    media_paths = [
        path
        for path in source.expanduser().iterdir()
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    ]
    if not media_paths:
        typer.echo("No media files found to process.")
        raise typer.Exit(code=0)

    typer.echo(f"Discovered {len(media_paths)} file(s). Building plans...")
    plans = organizer.preview(media_paths, custom_tags)
    for plan in plans:
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
