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


class DiarizationError(SurivoiceError):
    """Pyannote model load or inference failure."""


class MergeError(SurivoiceError):
    """Segment alignment failure, e.g. empty inputs."""


class OutputError(SurivoiceError):
    """Unable to write output file."""
