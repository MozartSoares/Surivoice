"""Tests for the audio extraction module."""

import shutil
import wave
from pathlib import Path
from unittest.mock import patch

import pytest

from surivoice.audio.extractor import CHANNELS, SAMPLE_RATE, extract_audio
from surivoice.errors import FFmpegError

requires_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None,
    reason="FFmpeg is not installed",
)


class TestExtractAudio:
    """Test audio extraction via FFmpeg."""

    @requires_ffmpeg
    def test_extract_audio_from_wav(self, wav_fixture: Path, tmp_path: Path) -> None:
        """Extracting from a valid WAV should produce an output file."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        result = extract_audio(wav_fixture, output_dir)

        assert result.exists()
        assert result.suffix == ".wav"

    @requires_ffmpeg
    def test_output_file_properties(self, wav_fixture: Path, tmp_path: Path) -> None:
        """Extracted WAV must be 16kHz, mono, 16-bit PCM."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        result = extract_audio(wav_fixture, output_dir)

        with wave.open(str(result), "rb") as wf:
            assert wf.getframerate() == SAMPLE_RATE
            assert wf.getnchannels() == CHANNELS
            assert wf.getsampwidth() == 2  # 16-bit PCM

    @requires_ffmpeg
    def test_extraction_creates_file_in_output_dir(self, wav_fixture: Path, tmp_path: Path) -> None:
        """Output WAV must land inside the specified output directory."""
        output_dir = tmp_path / "custom_output"
        output_dir.mkdir()

        result = extract_audio(wav_fixture, output_dir)

        assert result.parent == output_dir

    @requires_ffmpeg
    def test_extraction_preserves_stem(self, wav_fixture: Path, tmp_path: Path) -> None:
        """Output file should keep the original file's stem."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        result = extract_audio(wav_fixture, output_dir)

        assert result.stem == wav_fixture.stem

    @requires_ffmpeg
    def test_extraction_raises_on_invalid_file(self, tmp_path: Path) -> None:
        """Corrupt input file should raise FFmpegError with EXTRACTION_FAILED."""
        corrupt_file = tmp_path / "corrupt.mp4"
        corrupt_file.write_bytes(b"not a real media file")

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        with pytest.raises(FFmpegError, match=FFmpegError.EXTRACTION_FAILED):
            extract_audio(corrupt_file, output_dir)

    def test_extraction_raises_on_missing_ffmpeg(self, wav_fixture: Path, tmp_path: Path) -> None:
        """Missing FFmpeg should raise FFmpegError with NOT_INSTALLED."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        with (
            patch("surivoice.audio.extractor.shutil.which", return_value=None),
            pytest.raises(FFmpegError, match=FFmpegError.NOT_INSTALLED),
        ):
            extract_audio(wav_fixture, output_dir)
