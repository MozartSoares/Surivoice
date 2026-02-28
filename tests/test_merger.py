"""Tests for the segment merger module."""

import pytest

from surivoice.errors import MergeError
from surivoice.merge.merger import UNKNOWN_SPEAKER, merge_segments
from surivoice.models import (
    DiarizationSegment,
    MergedSegment,
    TranscriptionSegment,
)


def _t(start: float, end: float, text: str) -> TranscriptionSegment:
    """Shorthand to create a TranscriptionSegment."""
    return TranscriptionSegment(start=start, end=end, text=text)


def _d(start: float, end: float, speaker: str) -> DiarizationSegment:
    """Shorthand to create a DiarizationSegment."""
    return DiarizationSegment(start=start, end=end, speaker=speaker)


class TestMergeSegments:
    """Test the max-overlap merge algorithm."""

    def test_simple_one_to_one(self) -> None:
        """Each transcription segment maps to one speaker."""
        transcription = (
            _t(0.0, 2.0, "Hello"),
            _t(2.0, 4.0, "World"),
        )
        diarization = (
            _d(0.0, 2.0, "SPEAKER_00"),
            _d(2.0, 4.0, "SPEAKER_01"),
        )

        result = merge_segments(transcription, diarization)

        assert len(result) == 2
        assert result[0] == MergedSegment(start=0.0, end=2.0, speaker="SPEAKER_00", text="Hello")
        assert result[1] == MergedSegment(start=2.0, end=4.0, speaker="SPEAKER_01", text="World")

    def test_midpoint_picks_best_speaker(self) -> None:
        """Transcription segment intersecting a speaker at its midpoint gets that speaker."""
        # Midpoint of (1.0..4.0) is 2.5
        transcription = (_t(1.0, 4.0, "Overlapping speech"),)
        diarization = (
            _d(0.0, 2.0, "SPEAKER_00"),  # Does not cover 2.5
            _d(2.0, 5.0, "SPEAKER_01"),  # Covers 2.5
        )

        result = merge_segments(transcription, diarization)

        assert result[0].speaker == "SPEAKER_01"

    def test_coalesces_consecutive_same_speaker(self) -> None:
        """Consecutive segments from the same speaker are merged."""
        transcription = (
            _t(0.0, 1.0, "Hello"),
            _t(1.0, 2.0, "there"),
            _t(2.0, 3.0, "friend"),
        )
        diarization = (_d(0.0, 3.0, "SPEAKER_00"),)

        result = merge_segments(transcription, diarization)

        assert len(result) == 1
        assert result[0] == MergedSegment(
            start=0.0, end=3.0, speaker="SPEAKER_00", text="Hello there friend"
        )

    def test_alternating_speakers_no_coalesce(self) -> None:
        """Alternating speakers produce separate segments."""
        transcription = (
            _t(0.0, 1.0, "Hi"),
            _t(1.0, 2.0, "Hey"),
            _t(2.0, 3.0, "Bye"),
        )
        diarization = (
            _d(0.0, 1.0, "SPEAKER_00"),
            _d(1.0, 2.0, "SPEAKER_01"),
            _d(2.0, 3.0, "SPEAKER_00"),
        )

        result = merge_segments(transcription, diarization)

        assert len(result) == 3
        assert result[0].speaker == "SPEAKER_00"
        assert result[1].speaker == "SPEAKER_01"
        assert result[2].speaker == "SPEAKER_00"

    def test_nearest_neighbor_fallback(self) -> None:
        """Transcription with no strict midpoint overlap falls back to the nearest track."""
        # Midpoint is 2.5 (falls into the dead zone between 2.0 and 3.0)
        transcription = (_t(2.4, 2.6, "Gap word"),)
        diarization = (
            _d(0.0, 2.0, "SPEAKER_00"),  # Center: 1.0 (Distance: 1.5)
            _d(3.0, 5.0, "SPEAKER_01"),  # Center: 4.0 (Distance: 1.5)
            _d(2.7, 2.8, "SPEAKER_02"),  # Center: 2.75 (Distance: 0.25 -> Winner)
        )

        result = merge_segments(transcription, diarization)

        # Should fall back to SPEAKER_02 because its center is closest to 2.5
        assert result[0].speaker == "SPEAKER_02"

    def test_no_overlap_assigns_unknown_beyond_threshold(self) -> None:
        """Transcription beyond MAX_FALLBACK_GAP gets UNKNOWN_SPEAKER."""
        # Midpoint 12.0
        transcription = (_t(11.0, 13.0, "Isolated"),)
        diarization = (
            _d(0.0, 2.0, "SPEAKER_00"),  # Center 1.0, Distance 11.0 > MAX_FALLBACK_GAP
        )

        result = merge_segments(transcription, diarization)

        assert result[0].speaker == UNKNOWN_SPEAKER

    def test_raises_on_empty_transcription(self) -> None:
        """Empty transcription should raise MergeError."""
        diarization = (_d(0.0, 1.0, "SPEAKER_00"),)

        with pytest.raises(MergeError, match=MergeError.EMPTY_SEGMENTS):
            merge_segments((), diarization)

    def test_raises_on_empty_diarization(self) -> None:
        """Empty diarization should raise MergeError."""
        transcription = (_t(0.0, 1.0, "Hello"),)

        with pytest.raises(MergeError, match=MergeError.EMPTY_SEGMENTS):
            merge_segments(transcription, ())

    def test_coalesce_preserves_timestamps(self) -> None:
        """Coalesced segment should have start of first and end of last."""
        transcription = (
            _t(0.5, 1.5, "Part one"),
            _t(1.5, 3.0, "Part two"),
        )
        diarization = (_d(0.0, 4.0, "SPEAKER_00"),)

        result = merge_segments(transcription, diarization)

        assert len(result) == 1
        assert result[0].start == pytest.approx(0.5)
        assert result[0].end == pytest.approx(3.0)
