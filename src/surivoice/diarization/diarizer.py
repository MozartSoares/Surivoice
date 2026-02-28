"""Speaker diarization via pyannote.audio.

This module wraps the pyannote.audio Pipeline to identify
who spoke when in an audio file. It reads configuration from
PipelineConfig and returns a typed DiarizeResult.
"""

import logging
from pathlib import Path

from pydantic import BaseModel

from surivoice.config import PipelineConfig
from surivoice.errors import DiarizationError
from surivoice.models import DiarizationSegment

logger = logging.getLogger(__name__)

PIPELINE_MODEL = "pyannote/speaker-diarization-3.1"


class DiarizeResult(BaseModel, frozen=True):
    """Result from the diarization stage (before merge with transcription)."""

    segments: tuple[DiarizationSegment, ...]
    """Speaker-labeled time ranges in chronological order."""

    speakers_count: int
    """Number of distinct speakers identified."""


def diarize(wav_path: Path, config: PipelineConfig) -> DiarizeResult:
    """Run speaker diarization using pyannote.audio.

    Loads the pre-trained diarization pipeline and processes the WAV file
    to identify speaker turns.

    Args:
        wav_path: Path to the extracted WAV file (16kHz, mono).
        config: Pipeline configuration with HF token and speaker hints.

    Returns:
        DiarizeResult with speaker segments and count.

    Raises:
        DiarizationError: If token is missing, pipeline loading, or inference fails.
    """
    if config.hf_token is None:
        raise DiarizationError(
            f"{DiarizationError.MISSING_HF_TOKEN}. "
            "Provide it via --hf-token or the HF_TOKEN environment variable."
        )

    try:
        from pyannote.audio import Pipeline
    except ImportError as exc:
        raise DiarizationError(
            f"{DiarizationError.PIPELINE_LOAD_FAILED}: pyannote.audio is not installed. "
            "Install it with: pip install surivoice[ml]"
        ) from exc

    device = config.device.resolve()

    logger.info(
        "Loading diarization pipeline '%s' on %s",
        PIPELINE_MODEL,
        device,
    )

    try:
        pipeline = Pipeline.from_pretrained(
            PIPELINE_MODEL,
            use_auth_token=config.hf_token,
        )
        pipeline.to(device)
    except Exception as exc:
        raise DiarizationError(
            f"{DiarizationError.PIPELINE_LOAD_FAILED}: {exc}"
        ) from exc

    logger.info("Diarizing %s", wav_path)

    # Build kwargs for speaker count hints
    pipeline_kwargs: dict[str, int] = {}
    if config.min_speakers is not None:
        pipeline_kwargs["min_speakers"] = config.min_speakers
    if config.max_speakers is not None:
        pipeline_kwargs["max_speakers"] = config.max_speakers

    try:
        annotation = pipeline(str(wav_path), **pipeline_kwargs)
    except Exception as exc:
        raise DiarizationError(
            f"{DiarizationError.DIARIZATION_FAILED}: {exc}"
        ) from exc

    segments = tuple(
        DiarizationSegment(
            start=turn.start,
            end=turn.end,
            speaker=speaker,
        )
        for turn, _, speaker in annotation.itertracks(yield_label=True)
    )

    unique_speakers = {seg.speaker for seg in segments}

    logger.info(
        "Diarization complete: %d segments, %d speakers",
        len(segments),
        len(unique_speakers),
    )

    return DiarizeResult(
        segments=segments,
        speakers_count=len(unique_speakers),
    )
