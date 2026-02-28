"""Speech-to-text transcription via faster-whisper.

This module wraps the faster-whisper library to provide timestamped
transcription segments from a WAV audio file. It reads configuration
from PipelineConfig and returns a typed TranscribeResult.
"""

import logging
from pathlib import Path

from pydantic import BaseModel

from surivoice.config import PipelineConfig
from surivoice.errors import TranscriptionError
from surivoice.models import TranscriptionSegment

logger = logging.getLogger(__name__)


class TranscribeResult(BaseModel, frozen=True):
    """Result from the transcription stage (before merge with diarization)."""

    segments: tuple[TranscriptionSegment, ...]
    """Timestamped transcription segments in chronological order."""

    detected_language: str | None
    """ISO 639-1 language code detected by Whisper, or None."""

    language_probability: float
    """Confidence score for the detected language (0.0 to 1.0)."""

    duration_seconds: float
    """Total duration of the input audio in seconds."""


def transcribe(wav_path: Path, config: PipelineConfig) -> TranscribeResult:
    """Transcribe audio using faster-whisper.

    Loads the Whisper model, runs inference on the WAV file, and returns
    structured transcription segments with timestamps.

    Args:
        wav_path: Path to the extracted WAV file (16kHz, mono).
        config: Pipeline configuration with model size, device, etc.

    Returns:
        TranscribeResult with segments and metadata.

    Raises:
        TranscriptionError: If model loading or transcription fails.
    """
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise TranscriptionError(
            f"{TranscriptionError.MODEL_LOAD_FAILED}: faster-whisper is not installed. "
            "Install it with: pip install surivoice[ml]"
        ) from exc

    device = config.device.resolve()
    compute_type = config.compute_type.value

    logger.info(
        "Loading Whisper model '%s' on %s (%s)",
        config.model.value,
        device,
        compute_type,
    )

    try:
        model = WhisperModel(
            config.model.value,
            device=device,
            compute_type=compute_type,
        )
    except Exception as exc:
        raise TranscriptionError(
            f"{TranscriptionError.MODEL_LOAD_FAILED}: {exc}"
        ) from exc

    language = config.language

    logger.info("Transcribing %s", wav_path)

    try:
        segments_generator, info = model.transcribe(
            str(wav_path),
            language=language,
        )

        segments = tuple(
            TranscriptionSegment(
                start=segment.start,
                end=segment.end,
                text=segment.text.strip(),
            )
            for segment in segments_generator
        )
    except Exception as exc:
        raise TranscriptionError(
            f"{TranscriptionError.TRANSCRIPTION_FAILED}: {exc}"
        ) from exc

    logger.info(
        "Transcription complete: %d segments, language=%s (%.1f%%), duration=%.1fs",
        len(segments),
        info.language,
        info.language_probability * 100,
        info.duration,
    )

    return TranscribeResult(
        segments=segments,
        detected_language=info.language,
        language_probability=info.language_probability,
        duration_seconds=info.duration,
    )
