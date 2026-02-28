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

    def test_max_overlap_picks_best_speaker(self) -> None:
        """Transcription segment overlapping two speakers picks the one with more overlap."""
        transcription = (_t(1.0, 4.0, "Overlapping speech"),)
        diarization = (
            _d(0.0, 2.0, "SPEAKER_00"),  # overlap: 1.0s (1.0..2.0)
            _d(2.0, 5.0, "SPEAKER_01"),  # overlap: 2.0s (2.0..4.0)
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
        diarization = (
            _d(0.0, 3.0, "SPEAKER_00"),
        )

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

    def test_no_overlap_assigns_unknown(self) -> None:
        """Transcription with no diarization overlap gets UNKNOWN_SPEAKER."""
        transcription = (_t(10.0, 12.0, "Isolated"),)
        diarization = (
            _d(0.0, 2.0, "SPEAKER_00"),
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
        diarization = (
            _d(0.0, 4.0, "SPEAKER_00"),
        )

        result = merge_segments(transcription, diarization)

        assert len(result) == 1
        assert result[0].start == pytest.approx(0.5)
        assert result[0].end == pytest.approx(3.0)
