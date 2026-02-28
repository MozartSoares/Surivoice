"""Tests for the Markdown output formatter."""

from pathlib import Path

import pytest

from surivoice.errors import OutputError
from surivoice.models import MergedSegment
from surivoice.output.formatter import (
    HEADER,
    LABEL_DURATION,
    LABEL_LANGUAGE,
    LABEL_SPEAKERS,
    LABEL_TOTAL_SPEAKERS,
    SEPARATOR,
    _format_speaker_line,
    _format_timestamp,
    format_transcript,
    write_transcript,
)


def _m(start: float, end: float, speaker: str, text: str) -> MergedSegment:
    """Shorthand to create a MergedSegment."""
    return MergedSegment(start=start, end=end, speaker=speaker, text=text)


class TestFormatTimestamp:
    """Test timestamp formatting."""

    def test_zero(self) -> None:
        assert _format_timestamp(0.0) == "00:00:00"

    def test_seconds_only(self) -> None:
        assert _format_timestamp(45.7) == "00:00:45"

    def test_minutes_and_seconds(self) -> None:
        assert _format_timestamp(125.0) == "00:02:05"

    def test_hours(self) -> None:
        assert _format_timestamp(3661.0) == "01:01:01"


class TestFormatTranscript:
    """Test Markdown transcript formatting."""

    def test_starts_with_header(self) -> None:
        """Output should start with the defined header constant."""
        segments = (_m(0.0, 1.0, "SPEAKER_00", "Hello"),)
        result = format_transcript(segments)
        assert result.startswith(HEADER)

    def test_contains_speaker_lines(self) -> None:
        """Each segment should render a speaker line using the formatting function."""
        segments = (
            _m(0.0, 2.0, "SPEAKER_00", "Hello there"),
            _m(2.0, 4.0, "SPEAKER_01", "Hi back"),
        )
        result = format_transcript(segments)
        assert _format_speaker_line("SPEAKER_00", 0.0, 2.0) in result
        assert _format_speaker_line("SPEAKER_01", 2.0, 4.0) in result

    def test_contains_text(self) -> None:
        """Segment text should appear in output."""
        segments = (_m(0.0, 1.0, "SPEAKER_00", "Specific test phrase"),)
        result = format_transcript(segments)
        assert "Specific test phrase" in result

    def test_metadata_language(self) -> None:
        """Language label and value should appear when provided."""
        segments = (_m(0.0, 1.0, "SPEAKER_00", "Hi"),)
        result = format_transcript(segments, detected_language="en")
        assert LABEL_LANGUAGE in result
        assert "en" in result

    def test_metadata_duration(self) -> None:
        """Duration label and formatted value should appear when provided."""
        segments = (_m(0.0, 1.0, "SPEAKER_00", "Hi"),)
        result = format_transcript(segments, duration_seconds=120.0)
        assert LABEL_DURATION in result
        assert _format_timestamp(120.0) in result

    def test_metadata_speakers(self) -> None:
        """Speakers label, individual names, and total should appear."""
        segments = (
            _m(0.0, 1.0, "SPEAKER_00", "Hi"),
            _m(1.0, 2.0, "SPEAKER_01", "Hey"),
        )
        result = format_transcript(segments, speakers_count=2)
        assert LABEL_SPEAKERS in result
        assert "SPEAKER_00" in result
        assert "SPEAKER_01" in result
        assert LABEL_TOTAL_SPEAKERS in result

    def test_metadata_omitted_when_none(self) -> None:
        """No metadata labels when all values are None."""
        segments = (_m(0.0, 1.0, "SPEAKER_00", "Hi"),)
        result = format_transcript(segments)
        assert LABEL_LANGUAGE not in result
        assert LABEL_DURATION not in result
        assert LABEL_TOTAL_SPEAKERS not in result

    def test_separator_present(self) -> None:
        """The defined separator should appear in output."""
        segments = (_m(0.0, 1.0, "SPEAKER_00", "Hi"),)
        result = format_transcript(segments)
        assert SEPARATOR in result

    def test_multiple_segments_order(self) -> None:
        """Segments should appear in the order provided."""
        segments = (
            _m(0.0, 1.0, "SPEAKER_00", "First"),
            _m(1.0, 2.0, "SPEAKER_01", "Second"),
            _m(2.0, 3.0, "SPEAKER_00", "Third"),
        )
        result = format_transcript(segments)
        first_pos = result.index("First")
        second_pos = result.index("Second")
        third_pos = result.index("Third")
        assert first_pos < second_pos < third_pos


class TestWriteTranscript:
    """Test file writing."""

    def test_writes_file(self, tmp_path: Path) -> None:
        """Content should be written to the specified path."""
        output = tmp_path / "transcript.md"
        write_transcript("# Test content", output)
        assert output.read_text(encoding="utf-8") == "# Test content"

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        """Parent directories should be created if missing."""
        output = tmp_path / "nested" / "dir" / "transcript.md"
        write_transcript("# Nested", output)
        assert output.exists()

    def test_raises_on_write_failure(self, tmp_path: Path) -> None:
        """Writing to an unwritable path should raise OutputError."""
        output = Path("/proc/fake/transcript.md")
        with pytest.raises(OutputError, match=OutputError.WRITE_FAILED):
            write_transcript("# Fail", output)
