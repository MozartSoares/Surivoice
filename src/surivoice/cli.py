"""Surivoice CLI — Typer application entry point.

This module defines the CLI interface using Typer. All arguments are
parsed here, validated, and converted into a PipelineConfig before
invoking the pipeline orchestrator.
"""

import shutil
from pathlib import Path

import typer
from rich.console import Console

from surivoice import __version__
from surivoice.config import ComputeType, DeviceType, PipelineConfig, WhisperModel
from surivoice.errors import FFmpegError, InputFileError, SurivoiceError

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

# Supported input extensions (common audio/video formats)
SUPPORTED_EXTENSIONS: frozenset[str] = frozenset({
    ".mp4", ".mkv", ".avi", ".mov", ".webm",  # video
    ".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac", ".wma",  # audio
})


def version_callback(value: bool) -> None:
    """Print version and exit."""
    if value:
        console.print(f"surivoice {__version__}")
        raise typer.Exit()


def _validate_input_file(path: Path) -> Path:
    """Validate that the input file exists and has a supported extension."""
    if not path.exists():
        print_error(f"{InputFileError.NOT_FOUND}: {path}")
        raise typer.Exit(code=1)

    if not path.is_file():
        print_error(f"{InputFileError.NOT_A_FILE}: {path}")
        raise typer.Exit(code=1)

    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        print_error(
            f"{InputFileError.UNSUPPORTED_FORMAT}: '{path.suffix}'\n"
            f"Supported formats: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )
        raise typer.Exit(code=1)

    return path


def _check_ffmpeg() -> None:
    """Check that FFmpeg is available on PATH."""
    if shutil.which("ffmpeg") is None:
        print_error(
            f"{FFmpegError.NOT_INSTALLED}.\n"
            "Install it with:\n"
            "  Ubuntu/Debian: sudo apt install ffmpeg\n"
            "  macOS:         brew install ffmpeg\n"
            "  Arch:          sudo pacman -S ffmpeg"
        )
        raise typer.Exit(code=2)


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
        WhisperModel.MEDIUM,
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
    min_speakers: int | None = typer.Option(
        None,
        "--min-speakers",
        help="Minimum number of speakers (hint for diarization).",
        min=1,
    ),
    max_speakers: int | None = typer.Option(
        None,
        "--max-speakers",
        help="Maximum number of speakers (hint for diarization).",
        min=1,
    ),
    version: bool = typer.Option(
        False,
        "--version",
        "-v",
        help="Show version and exit.",
        callback=version_callback,
        is_eager=True,
    ),
) -> None:
    """Transcribe an audio/video file with speaker diarization."""
    # Validate prerequisites
    _validate_input_file(input_file)
    _check_ffmpeg()

    # Build pipeline configuration
    config = PipelineConfig(
        model=model,
        device=device,
        compute_type=compute_type,
        language=language,
        hf_token=hf_token,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
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
