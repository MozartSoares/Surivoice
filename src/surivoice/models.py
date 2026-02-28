"""Shared data structures for the Surivoice pipeline.

All models are frozen (immutable) Pydantic models that flow between
pipeline stages. Each stage produces and consumes well-typed data,
ensuring type safety throughout the entire processing pipeline.
"""

from pydantic import BaseModel


class TranscriptionSegment(BaseModel, frozen=True):
    """A segment of transcribed text with timestamps.

    Produced by the transcription stage.
    """

    start: float
    """Start time in seconds."""

    end: float
    """End time in seconds."""

    text: str
    """Transcribed text content."""


class DiarizationSegment(BaseModel, frozen=True):
    """A speaker-labeled time range.

    Produced by the diarization stage.
    """

    start: float
    """Start time in seconds."""

    end: float
    """End time in seconds."""

    speaker: str
    """Speaker identifier, e.g. 'SPEAKER_00', 'SPEAKER_01'."""


class MergedSegment(BaseModel, frozen=True):
    """A transcription segment attributed to a specific speaker.

    Produced by the merge stage, combining transcription and diarization results.
    """

    start: float
    """Start time in seconds."""

    end: float
    """End time in seconds."""

    speaker: str
    """Speaker identifier."""

    text: str
    """Transcribed text content."""


class TranscriptionResult(BaseModel, frozen=True):
    """Complete result of the transcription pipeline.

    This is the final output of the pipeline, containing all merged segments
    along with metadata about the transcription.
    """

    segments: tuple[MergedSegment, ...]
    """All merged segments in chronological order."""

    detected_language: str | None
    """ISO 639-1 language code detected by Whisper, or None if not detected."""

    duration_seconds: float
    """Total duration of the input audio in seconds."""

    speakers_count: int
    """Number of distinct speakers identified."""
