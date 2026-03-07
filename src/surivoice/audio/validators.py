"""Input file and FFmpeg validation helpers.

These pure functions return error messages on failure (or None on success),
keeping validation logic free from any CLI/GUI framework dependency.
"""

import shutil
from pathlib import Path

from surivoice.errors import FFmpegError, InputFileError

# Supported input extensions (common audio/video formats)
SUPPORTED_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".mp4",
        ".mkv",
        ".avi",
        ".mov",
        ".webm",  # video
        ".mp3",
        ".wav",
        ".flac",
        ".ogg",
        ".m4a",
        ".aac",
        ".wma",  # audio
    }
)


def validate_input_file(path: Path) -> str | None:
    """Validate that the input file exists and has a supported extension.

    Returns:
        An error message string if validation fails, or None if the file is valid.
    """
    if not path.exists():
        return f"{InputFileError.NOT_FOUND}: {path}"

    if not path.is_file():
        return f"{InputFileError.NOT_A_FILE}: {path}"

    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        return (
            f"{InputFileError.UNSUPPORTED_FORMAT}: '{path.suffix}'\n"
            f"Supported formats: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    return None


def check_ffmpeg() -> str | None:
    """Check that FFmpeg is available on PATH.

    Returns:
        An error message string if FFmpeg is missing, or None if it is available.
    """
    if shutil.which("ffmpeg") is None:
        return (
            f"{FFmpegError.NOT_INSTALLED}.\n"
            "Install it with:\n"
            "  Ubuntu/Debian: sudo apt install ffmpeg\n"
            "  macOS:         brew install ffmpeg\n"
            "  Arch:          sudo pacman -S ffmpeg"
        )
    return None
