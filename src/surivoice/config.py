"""Pipeline configuration with sensible defaults.

All configuration flows through PipelineConfig, which is constructed
from CLI arguments and environment variables. Each pipeline stage
reads only the fields it needs from this config.
"""

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class DeviceType(str, Enum):
    """Compute device for ML inference."""

    AUTO = "auto"
    CPU = "cpu"
    CUDA = "cuda"

    def resolve(self) -> str:
        """Resolve the actual device string for ML frameworks."""
        if self == DeviceType.AUTO:
            # Lazy import: torch is only loaded when actually running inference,
            # keeping the CLI lightweight for help/validation commands.
            import torch

            has_cuda: bool = bool(torch.cuda.is_available())
            return "cuda" if has_cuda else "cpu"
        return str(self.value)


class WhisperModel(str, Enum):
    """Available Whisper model sizes."""

    TINY = "tiny"
    BASE = "base"
    SMALL = "small"
    MEDIUM = "medium"
    LARGE_V2 = "large-v2"
    LARGE_V3 = "large-v3"


class ComputeType(str, Enum):
    """Quantization level for Whisper inference."""

    FLOAT16 = "float16"
    INT8 = "int8"
    FLOAT32 = "float32"


class PipelineConfig(BaseModel, frozen=True):
    """Immutable configuration for a single pipeline run."""

    model: WhisperModel = WhisperModel.MEDIUM
    """Whisper model size to use for transcription."""

    device: DeviceType = DeviceType.AUTO
    """Compute device. 'auto' will prefer CUDA if available."""

    compute_type: ComputeType = ComputeType.INT8
    """Quantization type for Whisper inference."""

    language: str | None = None
    """ISO 639-1 language code. None for auto-detection."""

    hf_token: str | None = None
    """Hugging Face access token for pyannote.audio models."""

    min_speakers: int | None = Field(default=None, ge=1)
    """Minimum number of speakers (hint for diarization)."""

    max_speakers: int | None = Field(default=None, ge=1)
    """Maximum number of speakers (hint for diarization)."""

    output_format: Literal["markdown"] = "markdown"
    """Output format. Only 'markdown' in MVP; extensible later."""
