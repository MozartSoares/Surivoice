"""Custom exception hierarchy for Surivoice.

All exceptions inherit from SurivoiceError, making it easy to catch
any Surivoice-specific error at the CLI boundary while still allowing
granular handling per stage.
"""


class SurivoiceError(Exception):
    """Base exception for all Surivoice errors."""


class InputFileError(SurivoiceError):
    """File not found, unreadable, or unsupported format."""

    NOT_FOUND = "Input file not found"
    NOT_A_FILE = "Not a file"
    UNSUPPORTED_FORMAT = "Unsupported file format"


class FFmpegError(SurivoiceError):
    """FFmpeg not installed or audio extraction failed."""

    NOT_INSTALLED = "FFmpeg is not installed or not on PATH"
    EXTRACTION_FAILED = "Audio extraction failed"


class TranscriptionError(SurivoiceError):
    """Whisper model load or inference failure."""

    MODEL_LOAD_FAILED = "Failed to load Whisper model"
    TRANSCRIPTION_FAILED = "Transcription failed"


class DiarizationError(SurivoiceError):
    """Pyannote model load or inference failure."""

    PIPELINE_LOAD_FAILED = "Failed to load diarization pipeline"
    DIARIZATION_FAILED = "Diarization failed"
    MISSING_HF_TOKEN = "Hugging Face token is required for diarization"


class MergeError(SurivoiceError):
    """Segment alignment failure, e.g. empty inputs."""

    EMPTY_SEGMENTS = "No segments to merge"
    NO_OVERLAP = "No speaker overlap found for segment"


class OutputError(SurivoiceError):
    """Unable to write output file."""
