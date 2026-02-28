"""Tests for the transcription module.

All tests mock faster-whisper — no model downloads or GPU needed.
"""

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from surivoice.config import ComputeType, DeviceType, PipelineConfig, WhisperModel
from surivoice.errors import TranscriptionError
from surivoice.models import TranscriptionSegment
from surivoice.transcription.transcriber import TranscribeResult, transcribe


def _make_fake_segment(
    start: float, end: float, text: str
) -> SimpleNamespace:
    """Create a fake faster-whisper Segment object."""
    return SimpleNamespace(start=start, end=end, text=text)


def _make_fake_info(
    language: str = "en",
    language_probability: float = 0.95,
    duration: float = 10.0,
) -> SimpleNamespace:
    """Create a fake faster-whisper TranscriptionInfo object."""
    return SimpleNamespace(
        language=language,
        language_probability=language_probability,
        duration=duration,
    )


def _build_mock_faster_whisper(
    segments: list[SimpleNamespace],
    info: SimpleNamespace,
) -> MagicMock:
    """Build a mock faster_whisper module with a working WhisperModel."""
    mock_module = MagicMock()
    mock_model_instance = MagicMock()
    mock_model_instance.transcribe.return_value = (iter(segments), info)
    mock_module.WhisperModel.return_value = mock_model_instance
    return mock_module


class TestTranscribe:
    """Test transcription via mocked faster-whisper."""

    def test_transcribe_returns_result(self, tmp_path: Path) -> None:
        """Transcribing should return a TranscribeResult."""
        wav_file = tmp_path / "audio.wav"
        wav_file.touch()

        fake_segments = [
            _make_fake_segment(0.0, 1.5, "Hello world"),
            _make_fake_segment(1.5, 3.0, "How are you"),
        ]
        mock_fw = _build_mock_faster_whisper(fake_segments, _make_fake_info())
        config = PipelineConfig(device=DeviceType.CPU)

        with patch.dict(sys.modules, {"faster_whisper": mock_fw}):
            result = transcribe(wav_file, config)

        assert isinstance(result, TranscribeResult)
        assert len(result.segments) == 2

    def test_transcribe_maps_segments(self, tmp_path: Path) -> None:
        """Each faster-whisper Segment should map to a TranscriptionSegment."""
        wav_file = tmp_path / "audio.wav"
        wav_file.touch()

        fake_segments = [
            _make_fake_segment(0.0, 2.0, " Hello "),
            _make_fake_segment(2.0, 4.5, " World "),
        ]
        mock_fw = _build_mock_faster_whisper(fake_segments, _make_fake_info())
        config = PipelineConfig(device=DeviceType.CPU)

        with patch.dict(sys.modules, {"faster_whisper": mock_fw}):
            result = transcribe(wav_file, config)

        assert result.segments[0] == TranscriptionSegment(start=0.0, end=2.0, text="Hello")
        assert result.segments[1] == TranscriptionSegment(start=2.0, end=4.5, text="World")

    def test_transcribe_captures_language(self, tmp_path: Path) -> None:
        """Detected language and probability should come from info."""
        wav_file = tmp_path / "audio.wav"
        wav_file.touch()

        fake_info = _make_fake_info(language="pt", language_probability=0.87)
        mock_fw = _build_mock_faster_whisper([], fake_info)
        config = PipelineConfig(device=DeviceType.CPU)

        with patch.dict(sys.modules, {"faster_whisper": mock_fw}):
            result = transcribe(wav_file, config)

        assert result.detected_language == "pt"
        assert result.language_probability == pytest.approx(0.87)

    def test_transcribe_captures_duration(self, tmp_path: Path) -> None:
        """Duration should come from info.duration."""
        wav_file = tmp_path / "audio.wav"
        wav_file.touch()

        fake_info = _make_fake_info(duration=42.5)
        mock_fw = _build_mock_faster_whisper([], fake_info)
        config = PipelineConfig(device=DeviceType.CPU)

        with patch.dict(sys.modules, {"faster_whisper": mock_fw}):
            result = transcribe(wav_file, config)

        assert result.duration_seconds == pytest.approx(42.5)

    def test_transcribe_raises_on_model_load_failure(self, tmp_path: Path) -> None:
        """Model init error should raise TranscriptionError with MODEL_LOAD_FAILED."""
        wav_file = tmp_path / "audio.wav"
        wav_file.touch()

        mock_fw = MagicMock()
        mock_fw.WhisperModel.side_effect = RuntimeError("out of memory")
        config = PipelineConfig(device=DeviceType.CPU)

        with patch.dict(sys.modules, {"faster_whisper": mock_fw}):
            with pytest.raises(TranscriptionError, match=TranscriptionError.MODEL_LOAD_FAILED):
                transcribe(wav_file, config)


class TestDeviceResolve:
    """Test DeviceType.resolve() method."""

    def test_resolve_auto_cpu(self) -> None:
        """AUTO should resolve to 'cpu' when CUDA is unavailable."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False

        with patch.dict(sys.modules, {"torch": mock_torch}):
            assert DeviceType.AUTO.resolve() == "cpu"

    def test_resolve_auto_cuda(self) -> None:
        """AUTO should resolve to 'cuda' when CUDA is available."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True

        with patch.dict(sys.modules, {"torch": mock_torch}):
            assert DeviceType.AUTO.resolve() == "cuda"

    def test_resolve_explicit_cpu(self) -> None:
        """CPU should resolve to 'cpu' directly without torch."""
        assert DeviceType.CPU.resolve() == "cpu"

    def test_resolve_explicit_cuda(self) -> None:
        """CUDA should resolve to 'cuda' directly without torch."""
        assert DeviceType.CUDA.resolve() == "cuda"
