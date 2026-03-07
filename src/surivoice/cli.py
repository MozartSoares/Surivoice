"""Surivoice CLI — Typer application entry point.

This module defines the CLI interface using Typer. All arguments are
parsed here, validated, and converted into a PipelineConfig before
invoking the pipeline orchestrator.
"""

from pathlib import Path

import typer
from rich.console import Console

from surivoice import __version__
from surivoice.audio.validators import check_ffmpeg, validate_input_file
from surivoice.config import ComputeType, DeviceType, PipelineConfig, WhisperModel
from surivoice.diarization.validators import (
    resolve_hf_token,
)
from surivoice.diarization.validators import (
    save_token as _save_token_to_disk,
)
from surivoice.errors import SurivoiceError

console = Console(stderr=True)

app = typer.Typer(
    name="surivoice",
    help="Local-first speech transcription with speaker diarization.",
    no_args_is_help=True,
    rich_markup_mode="rich",
)


def print_error(msg: str) -> None:
    """Print a formatted error message."""
    console.print(f"[bold red]Error:[/] {msg}")


def print_success(msg: str) -> None:
    """Print a formatted success message."""
    console.print(f"[bold green]{msg}[/]")


def print_warning(msg: str) -> None:
    """Print a formatted warning message."""
    console.print(f"[yellow]{msg}[/]")


def version_callback(value: bool) -> None:
    """Print version and exit."""
    if value:
        console.print(f"surivoice {__version__}")
        raise typer.Exit()


@app.callback(invoke_without_command=True)
def main(
    version: bool = typer.Option(
        False,
        "--version",
        "-v",
        help="Show version and exit.",
        callback=version_callback,
        is_eager=True,
    ),
) -> None:
    """Local-first speech transcription with speaker diarization."""


def _cli_validate_input_file(path: Path) -> Path:
    """Validate the input file, exiting on error."""
    error = validate_input_file(path)
    if error is not None:
        print_error(error)
        raise typer.Exit(code=1)
    return path


def _cli_check_ffmpeg() -> None:
    """Check for FFmpeg, exiting on error."""
    error = check_ffmpeg()
    if error is not None:
        print_error(error)
        raise typer.Exit(code=2)


def _cli_resolve_hf_token(cli_token: str | None) -> str:
    """Resolve the HF token, exiting if not found.

    Resolution order:
        1. --hf-token flag (already includes HF_TOKEN env via Typer)
        2. HF_TOKEN environment variable
        3. Saved token file (~/.config/surivoice/token)
    """
    resolved = resolve_hf_token(cli_token)
    if resolved is not None:
        return resolved

    print_error(
        "Hugging Face token is required for speaker diarization.\n"
        "Provide it via one of:\n"
        "  1. surivoice save-token YOUR_TOKEN  (saves permanently)\n"
        "  2. --hf-token YOUR_TOKEN             (per-run)\n"
        "  3. export HF_TOKEN=YOUR_TOKEN        (env var)"
    )
    raise typer.Exit(code=1)


@app.command(name="save-token")
def save_token(
    token: str = typer.Argument(..., help="Hugging Face access token to save."),
) -> None:
    """Save your Hugging Face token locally for future runs."""
    saved_path = _save_token_to_disk(token)
    print_success(f"Token saved to {saved_path}")


@app.command()
def gui() -> None:
    """Launch the graphical interface for Surivoice."""
    try:
        from surivoice.gui import main as launch_gui
    except ImportError:
        print_error(
            "tkinter is not installed.\n"
            "Install it with:\n"
            "  Ubuntu/Debian: sudo apt install python3-tk\n"
            "  Fedora:        sudo dnf install python3-tkinter\n"
            "  macOS:         brew install python-tk\n"
            "  Arch:          sudo pacman -S tk"
        )
        raise typer.Exit(code=1) from None
    launch_gui()


@app.command()
def transcribe(
    input_file: Path = typer.Option(
        ...,
        "--input",
        "-i",
        help="Input video or audio file to transcribe.",
        exists=False,  # We do our own validation for better error messages
    ),
    output_file: Path = typer.Option(
        ...,
        "--output",
        "-o",
        help="Output Markdown file path.",
    ),
    model: WhisperModel = typer.Option(
        WhisperModel.LARGE_V3,
        "--model",
        "-m",
        help="Whisper model size.",
    ),
    device: DeviceType = typer.Option(
        DeviceType.AUTO,
        "--device",
        "-d",
        help="Compute device: auto, cpu, or cuda.",
    ),
    compute_type: ComputeType = typer.Option(
        ComputeType.INT8,
        "--compute-type",
        help="Quantization type for Whisper inference.",
    ),
    language: str | None = typer.Option(
        None,
        "--language",
        "-l",
        help="Language code (e.g. 'en', 'pt'). Auto-detected if omitted.",
    ),
    hf_token: str | None = typer.Option(
        None,
        "--hf-token",
        envvar="HF_TOKEN",
        help="Hugging Face access token for pyannote.audio models.",
    ),
    num_speakers: int | None = typer.Option(
        None,
        "--speakers",
        "-s",
        help="Exact number of speakers (optional).",
        min=1,
    ),
) -> None:
    """Transcribe an audio/video file with speaker diarization."""
    # Validate prerequisites (fail-fast)
    _cli_validate_input_file(input_file)
    _cli_check_ffmpeg()
    resolved_token = _cli_resolve_hf_token(hf_token)

    # Build pipeline configuration
    config = PipelineConfig(
        model=model,
        device=device,
        compute_type=compute_type,
        language=language,
        hf_token=resolved_token,
        num_speakers=num_speakers,
    )

    print_success(f"Surivoice v{__version__}")
    console.print(f"  Input:   {input_file}")
    console.print(f"  Output:  {output_file}")
    console.print(f"  Model:   {config.model.value}")
    console.print(f"  Device:  {config.device.value}")
    console.print()

    try:
        from surivoice.pipeline import run

        result = run(input_file, output_file, config)
        print_success(
            f"Done! {len(result.segments)} segments, "
            f"{result.speakers_count} speakers, "
            f"language={result.detected_language}"
        )
    except SurivoiceError as exc:
        print_error(str(exc))
        raise typer.Exit(code=1) from exc
