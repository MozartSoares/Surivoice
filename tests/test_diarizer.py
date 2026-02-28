"""Tests for the diarization module.

All tests mock pyannote.audio — no model downloads or GPU needed.
"""

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from surivoice.config import DeviceType, PipelineConfig
from surivoice.diarization.diarizer import DiarizeResult, diarize
from surivoice.errors import DiarizationError
from surivoice.models import DiarizationSegment


def _make_fake_turn(start: float, end: float) -> SimpleNamespace:
    """Create a fake pyannote Segment (turn) object."""
    return SimpleNamespace(start=start, end=end)


def _build_mock_pyannote(
    tracks: list[tuple[SimpleNamespace, str]],
) -> MagicMock:
    """Build a mock pyannote.audio module with a working Pipeline.

    Args:
        tracks: List of (turn, speaker) pairs to return from itertracks.
    """
    mock_module = MagicMock()

    mock_annotation = MagicMock()
    mock_annotation.itertracks.return_value = [(turn, None, speaker) for turn, speaker in tracks]

    mock_pipeline_output = MagicMock()
    mock_pipeline_output.speaker_diarization = mock_annotation

    mock_pipeline_instance = MagicMock()
    mock_pipeline_instance.return_value = mock_pipeline_output
    mock_pipeline_instance.to = MagicMock()

    mock_module.Pipeline.from_pretrained.return_value = mock_pipeline_instance

    return mock_module


class TestDiarize:
    """Test diarization via mocked pyannote.audio."""

    def test_diarize_returns_result(self, tmp_path: Path) -> None:
        """Diarizing should return a DiarizeResult."""
        wav_file = tmp_path / "audio.wav"
        wav_file.touch()

        tracks = [
            (_make_fake_turn(0.0, 2.0), "SPEAKER_00"),
            (_make_fake_turn(2.0, 4.0), "SPEAKER_01"),
        ]
        mock_pa = _build_mock_pyannote(tracks)
        config = PipelineConfig(device=DeviceType.CPU, hf_token="fake-token")

        with patch.dict(
            sys.modules,
            {
                "pyannote.audio": mock_pa,
                "pyannote": MagicMock(),
                "torch": MagicMock(),
            },
        ):
            result = diarize(wav_file, config)

        assert isinstance(result, DiarizeResult)
        assert len(result.segments) == 2

    def test_diarize_maps_segments(self, tmp_path: Path) -> None:
        """Each annotation turn should map to a DiarizationSegment."""
        wav_file = tmp_path / "audio.wav"
        wav_file.touch()

        tracks = [
            (_make_fake_turn(0.5, 1.5), "SPEAKER_00"),
            (_make_fake_turn(2.0, 3.5), "SPEAKER_01"),
        ]
        mock_pa = _build_mock_pyannote(tracks)
        config = PipelineConfig(device=DeviceType.CPU, hf_token="fake-token")

        with patch.dict(
            sys.modules,
            {
                "pyannote.audio": mock_pa,
                "pyannote": MagicMock(),
                "torch": MagicMock(),
            },
        ):
            result = diarize(wav_file, config)

        assert result.segments[0] == DiarizationSegment(start=0.5, end=1.5, speaker="SPEAKER_00")
        assert result.segments[1] == DiarizationSegment(start=2.0, end=3.5, speaker="SPEAKER_01")

    def test_diarize_counts_speakers(self, tmp_path: Path) -> None:
        """speakers_count should match unique speaker labels."""
        wav_file = tmp_path / "audio.wav"
        wav_file.touch()

        tracks = [
            (_make_fake_turn(0.0, 1.0), "SPEAKER_00"),
            (_make_fake_turn(1.0, 2.0), "SPEAKER_01"),
            (_make_fake_turn(2.0, 3.0), "SPEAKER_00"),
            (_make_fake_turn(3.0, 4.0), "SPEAKER_02"),
        ]
        mock_pa = _build_mock_pyannote(tracks)
        config = PipelineConfig(device=DeviceType.CPU, hf_token="fake-token")

        with patch.dict(
            sys.modules,
            {
                "pyannote.audio": mock_pa,
                "pyannote": MagicMock(),
                "torch": MagicMock(),
            },
        ):
            result = diarize(wav_file, config)

        assert result.speakers_count == 3

    def test_diarize_passes_speaker_hints(self, tmp_path: Path) -> None:
        """num_speakers should be forwarded to the pipeline call."""
        wav_file = tmp_path / "audio.wav"
        wav_file.touch()

        mock_pa = _build_mock_pyannote([])
        config = PipelineConfig(
            device=DeviceType.CPU,
            hf_token="fake-token",
            num_speakers=3,
        )

        with patch.dict(
            sys.modules,
            {
                "pyannote.audio": mock_pa,
                "pyannote": MagicMock(),
                "torch": MagicMock(),
            },
        ):
            diarize(wav_file, config)

        # The pipeline instance is called with the wav path + speaker hints
        pipeline_instance = mock_pa.Pipeline.from_pretrained.return_value
        pipeline_instance.assert_called_once_with(
            str(wav_file),
            num_speakers=3,
        )

    def test_diarize_raises_on_pipeline_load_failure(self, tmp_path: Path) -> None:
        """Pipeline init error should raise DiarizationError with PIPELINE_LOAD_FAILED."""
        wav_file = tmp_path / "audio.wav"
        wav_file.touch()

        mock_pa = MagicMock()
        mock_pa.Pipeline.from_pretrained.side_effect = RuntimeError("auth failed")
        config = PipelineConfig(device=DeviceType.CPU, hf_token="bad-token")

        with (
            patch.dict(
                sys.modules,
                {
                    "pyannote.audio": mock_pa,
                    "pyannote": MagicMock(),
                    "torch": MagicMock(),
                },
            ),
            pytest.raises(DiarizationError, match=DiarizationError.PIPELINE_LOAD_FAILED),
        ):
            diarize(wav_file, config)

    def test_diarize_raises_on_missing_hf_token(self, tmp_path: Path) -> None:
        """No HF token should raise DiarizationError with MISSING_HF_TOKEN."""
        wav_file = tmp_path / "audio.wav"
        wav_file.touch()

        config = PipelineConfig(device=DeviceType.CPU, hf_token=None)

        with pytest.raises(DiarizationError, match=DiarizationError.MISSING_HF_TOKEN):
            diarize(wav_file, config)
