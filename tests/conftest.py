"""Shared pytest fixtures for Surivoice tests."""

import struct
import wave
from pathlib import Path

import pytest
from typer.testing import CliRunner


@pytest.fixture
def cli_runner() -> CliRunner:
    """Provide a Typer CLI test runner."""
    return CliRunner()


@pytest.fixture
def tmp_audio_file(tmp_path: Path) -> Path:
    """Create a dummy audio file for CLI input validation tests."""
    audio_file = tmp_path / "test_audio.wav"
    audio_file.write_bytes(b"RIFF" + b"\x00" * 100)  # minimal WAV-like header
    return audio_file


@pytest.fixture
def tmp_output_file(tmp_path: Path) -> Path:
    """Provide a temporary output file path."""
    return tmp_path / "output.md"


@pytest.fixture
def wav_fixture(tmp_path: Path) -> Path:
    """Generate a valid 0.1-second silence WAV file (16kHz, mono, 16-bit PCM).

    This is a real WAV that FFmpeg can process, used by audio extraction tests.
    """
    filepath = tmp_path / "silence.wav"
    sample_rate = 16000
    n_frames = int(sample_rate * 0.1)  # 0.1 seconds of silence

    with wave.open(str(filepath), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(struct.pack(f"<{n_frames}h", *([0] * n_frames)))

    return filepath

