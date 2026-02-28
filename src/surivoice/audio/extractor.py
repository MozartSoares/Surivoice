"""Audio extraction from video/audio files via FFmpeg.

This module handles converting any supported input file into a WAV file
with the exact format required by Transcriber and Diarizer:
16kHz sample rate, mono channel, PCM s16le encoding.
"""

import logging
import shutil
import subprocess
from pathlib import Path

from surivoice.errors import FFmpegError

logger = logging.getLogger(__name__)

# Output WAV parameters matching Transcriber/Diarizer requirements
SAMPLE_RATE = 16000
CHANNELS = 1
AUDIO_CODEC = "pcm_s16le"


def _ensure_ffmpeg_available() -> str:
    """Verify FFmpeg is installed and return its path.

    Returns:
        Absolute path to the ffmpeg binary.

    Raises:
        FFmpegError: If FFmpeg is not found on PATH.
    """
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        raise FFmpegError(FFmpegError.NOT_INSTALLED)
    return ffmpeg_path


def extract_audio(input_path: Path, output_dir: Path) -> Path:
    """Extract audio from a video/audio file as WAV 16kHz mono.

    Runs FFmpeg as a subprocess to convert the input into an uncompressed
    WAV file suitable for speech processing pipelines.

    Args:
        input_path: Path to the input video/audio file.
        output_dir: Directory to write the extracted WAV file.

    Returns:
        Path to the extracted WAV file.

    Raises:
        FFmpegError: If FFmpeg is not installed or extraction fails.
    """
    ffmpeg_path = _ensure_ffmpeg_available()

    output_path = output_dir / f"{input_path.stem}.wav"

    cmd = [
        ffmpeg_path,
        "-i",
        str(input_path),
        "-vn",
        "-acodec",
        AUDIO_CODEC,
        "-ar",
        str(SAMPLE_RATE),
        "-ac",
        str(CHANNELS),
        "-y",
        str(output_path),
    ]

    logger.debug("Running FFmpeg: %s", " ".join(cmd))

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        stderr_snippet = result.stderr.strip().splitlines()[-3:] if result.stderr else []
        detail = "\n".join(stderr_snippet)
        raise FFmpegError(f"{FFmpegError.EXTRACTION_FAILED}: {input_path}\n{detail}")

    if not output_path.exists():
        raise FFmpegError(f"{FFmpegError.EXTRACTION_FAILED}: output file was not created")

    logger.info("Extracted audio to %s", output_path)
    return output_path
