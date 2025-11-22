"""CLI helpers powered by Typer."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from pydantic import BaseModel, Field

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


def main() -> None:
    """Entrypoint used by `python -m` invocations."""

    app()


if __name__ == "__main__":
    main()
