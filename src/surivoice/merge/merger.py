"""Merge transcription and diarization segments.

Uses a maximum-overlap strategy to assign a speaker label to each
transcription segment, then coalesces consecutive segments from the
same speaker into a single MergedSegment.

Algorithm complexity: O(T × D) where T = transcription segments,D = diarization segments.
This can be improved by using a sweep-line algorithm, but for typical
meetings (<2h) this is efficient enough.
"""

import logging

from surivoice.errors import MergeError
from surivoice.models import (
    DiarizationSegment,
    MergedSegment,
    TranscriptionSegment,
)

logger = logging.getLogger(__name__)

UNKNOWN_SPEAKER = "SPEAKER_UNKNOWN"


def _compute_overlap(
    t_start: float,
    t_end: float,
    d_start: float,
    d_end: float,
) -> float:
    """Compute the temporal overlap between two time ranges in seconds."""
    overlap_start = max(t_start, d_start)
    overlap_end = min(t_end, d_end)
    return max(0.0, overlap_end - overlap_start)


def _assign_speaker(
    segment: TranscriptionSegment,
    diarization: tuple[DiarizationSegment, ...],
) -> str:
    """Find the speaker with the greatest overlap for a transcription segment.

    Returns UNKNOWN_SPEAKER if no diarization segment overlaps.
    """
    best_speaker = UNKNOWN_SPEAKER
    best_overlap = 0.0

    for d_seg in diarization:
        overlap = _compute_overlap(segment.start, segment.end, d_seg.start, d_seg.end)
        if overlap > best_overlap:
            best_overlap = overlap
            best_speaker = d_seg.speaker

    return best_speaker


def _coalesce(segments: list[MergedSegment]) -> tuple[MergedSegment, ...]:
    """Coalesce consecutive segments from the same speaker.

    Adjacent segments with the same speaker label are merged into a
    single segment with combined text.
    """
    if not segments:
        return ()

    coalesced: list[MergedSegment] = [segments[0]]

    for seg in segments[1:]:
        prev = coalesced[-1]
        if seg.speaker == prev.speaker:
            # Merge with previous: extend end time and append text
            coalesced[-1] = MergedSegment(
                start=prev.start,
                end=seg.end,
                speaker=prev.speaker,
                text=f"{prev.text} {seg.text}",
            )
        else:
            coalesced.append(seg)

    return tuple(coalesced)


def merge_segments(
    transcription: tuple[TranscriptionSegment, ...],
    diarization: tuple[DiarizationSegment, ...],
) -> tuple[MergedSegment, ...]:
    """Merge transcription segments with speaker labels from diarization.

    For each transcription segment, the speaker with the greatest temporal
    overlap is assigned. Consecutive segments from the same speaker are
    then coalesced into a single segment.

    Args:
        transcription: Timestamped transcription segments.
        diarization: Speaker-labeled time ranges.

    Returns:
        Tuple of merged segments with speaker labels.

    Raises:
        MergeError: If either input is empty.
    """
    if not transcription:
        raise MergeError(f"{MergeError.EMPTY_SEGMENTS}: transcription is empty")

    if not diarization:
        raise MergeError(f"{MergeError.EMPTY_SEGMENTS}: diarization is empty")

    # Assign speaker labels via max overlap
    labeled: list[MergedSegment] = []
    for seg in transcription:
        speaker = _assign_speaker(seg, diarization)
        labeled.append(
            MergedSegment(
                start=seg.start,
                end=seg.end,
                speaker=speaker,
                text=seg.text,
            )
        )

    # Coalesce consecutive same-speaker segments
    result = _coalesce(labeled)

    logger.info(
        "Merged %d transcription + %d diarization -> %d segments",
        len(transcription),
        len(diarization),
        len(result),
    )

    return result
