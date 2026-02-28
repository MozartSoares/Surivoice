"""Shared pytest fixtures for Surivoice tests."""

from pathlib import Path

import pytest
from typer.testing import CliRunner

from surivoice.cli import app


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
