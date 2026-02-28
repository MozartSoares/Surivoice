"""Tests for the pipeline orchestrator.

All external dependencies (FFmpeg, faster-whisper, pyannote) are mocked.
Tests verify that the pipeline wires stages correctly and handles errors.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from surivoice.config import DeviceType, PipelineConfig
from surivoice.errors import FFmpegError
from surivoice.models import (
    DiarizationSegment,
    TranscriptionResult,
    TranscriptionSegment,
)
from surivoice.pipeline import run


def _build_pipeline_mocks(
    transcription_segments: tuple[TranscriptionSegment, ...] | None = None,
    diarization_segments: tuple[DiarizationSegment, ...] | None = None,
    detected_language: str = "en",
    duration: float = 10.0,
    speakers_count: int = 2,
) -> dict[str, MagicMock]:
    """Build mocks for all pipeline stages.

    Returns short attribute names for use with patch.multiple("surivoice.pipeline", ...).
    """
    if transcription_segments is None:
        transcription_segments = (
            TranscriptionSegment(start=0.0, end=0.8, text="Hello"),
            TranscriptionSegment(start=0.9, end=2.0, text="World"),
        )

    if diarization_segments is None:
        diarization_segments = (
            DiarizationSegment(start=0.0, end=2.0, speaker="SPEAKER_00"),
            DiarizationSegment(start=2.0, end=4.0, speaker="SPEAKER_01"),
        )

    mock_extract = MagicMock(return_value=Path("/tmp/fake/audio.wav"))

    mock_transcribe_result = MagicMock()
    mock_transcribe_result.segments = transcription_segments
    mock_transcribe_result.detected_language = detected_language
    mock_transcribe_result.language_probability = 0.95
    mock_transcribe_result.duration_seconds = duration
    mock_transcribe = MagicMock(return_value=mock_transcribe_result)

    mock_diarize_result = MagicMock()
    mock_diarize_result.segments = diarization_segments
    mock_diarize_result.speakers_count = speakers_count
    mock_diarize = MagicMock(return_value=mock_diarize_result)

    return {
        "extract_audio": mock_extract,
        "transcribe": mock_transcribe,
        "diarize": mock_diarize,
    }


class TestPipelineRun:
    """Test the pipeline orchestrator with mocked stages."""

    def test_run_produces_output_file(self, tmp_path: Path) -> None:
        """Pipeline should produce an output Markdown file."""
        input_file = tmp_path / "input.wav"
        input_file.touch()
        output_file = tmp_path / "output.md"

        mocks = _build_pipeline_mocks()
        config = PipelineConfig(device=DeviceType.CPU, hf_token="fake")

        with patch.multiple("surivoice.pipeline", **mocks):
            run(input_file, output_file, config)

        assert output_file.exists()
        content = output_file.read_text(encoding="utf-8")
        assert "# Transcript" in content

    def test_run_returns_transcription_result(self, tmp_path: Path) -> None:
        """Pipeline should return a TranscriptionResult."""
        input_file = tmp_path / "input.wav"
        input_file.touch()
        output_file = tmp_path / "output.md"

        mocks = _build_pipeline_mocks()
        config = PipelineConfig(device=DeviceType.CPU, hf_token="fake")

        with patch.multiple("surivoice.pipeline", **mocks):
            result = run(input_file, output_file, config)

        assert isinstance(result, TranscriptionResult)
        assert len(result.segments) > 0
        assert result.detected_language == "en"

    def test_run_calls_all_stages(self, tmp_path: Path) -> None:
        """Pipeline should invoke extract, transcribe, diarize in order."""
        input_file = tmp_path / "input.wav"
        input_file.touch()
        output_file = tmp_path / "output.md"

        mocks = _build_pipeline_mocks()
        config = PipelineConfig(device=DeviceType.CPU, hf_token="fake")

        with patch.multiple("surivoice.pipeline", **mocks):
            run(input_file, output_file, config)

        mocks["extract_audio"].assert_called_once()
        mocks["transcribe"].assert_called_once()
        mocks["diarize"].assert_called_once()

    def test_run_propagates_stage_error(self, tmp_path: Path) -> None:
        """Pipeline should propagate errors from individual stages."""
        input_file = tmp_path / "input.wav"
        input_file.touch()
        output_file = tmp_path / "output.md"

        config = PipelineConfig(device=DeviceType.CPU, hf_token="fake")

        with (
            patch(
                "surivoice.pipeline.extract_audio",
                side_effect=FFmpegError("extraction failed"),
            ),
            pytest.raises(FFmpegError),
        ):
            run(input_file, output_file, config)

    def test_run_merges_segments_correctly(self, tmp_path: Path) -> None:
        """Pipeline should merge transcription with diarization segments."""
        input_file = tmp_path / "input.wav"
        input_file.touch()
        output_file = tmp_path / "output.md"

        t_segments = (TranscriptionSegment(start=0.0, end=3.0, text="Both speakers overlap"),)
        d_segments = (
            DiarizationSegment(start=0.0, end=1.0, speaker="SPEAKER_00"),
            DiarizationSegment(start=1.0, end=3.0, speaker="SPEAKER_01"),
        )

        mocks = _build_pipeline_mocks(
            transcription_segments=t_segments,
            diarization_segments=d_segments,
            speakers_count=2,
        )
        config = PipelineConfig(device=DeviceType.CPU, hf_token="fake")

        with patch.multiple("surivoice.pipeline", **mocks):
            result = run(input_file, output_file, config)

        # SPEAKER_01 has more overlap (2s vs 1s)
        assert result.segments[0].speaker == "SPEAKER_01"
