"""Merge transcription and diarization segments.

Uses a midpoint-inclusion strategy to assign a speaker label to each
transcription segment, then coalesces consecutive segments from the
same speaker into a single MergedSegment.

Algorithm complexity: O(T + D) where T = transcription segments, D = diarization segments.
We use a high-performance sweep-line algorithm (since segments are strictly chronological)
coupled with a midpoint distance check to cleanly handle edge boundaries.
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

MAX_FALLBACK_GAP = 5.0  # seconds


def _assign_speaker_sweep(
    transcription: tuple[TranscriptionSegment, ...],
    diarization: tuple[DiarizationSegment, ...],
) -> list[MergedSegment]:
    """Assign speakers using an O(T + D) sweep-line algorithm with midpoint fallback.

    For each word (transcription segment), we find the diarization track that
    covers its midpoint. If no track strictly covers the midpoint (e.g., the word
    falls in a micro-pause between tracks), we fall back to the nearest track
    within MAX_FALLBACK_GAP seconds.
    """
    labeled: list[MergedSegment] = []
    d_idx = 0
    d_len = len(diarization)

    for seg in transcription:
        # 1. Calculate the temporal midpoint of the transcribed word
        mid = (seg.start + seg.end) / 2.0

        # 2. Advance the diarization pointer until the track ends *after* this word starts.
        #    We don't advance past `mid` entirely because we might need this track for fallback.
        while d_idx < d_len - 1 and diarization[d_idx].end < seg.start:
            d_idx += 1

        # 3. Look at nearby tracks (current and maybe a few ahead)
        best_speaker = UNKNOWN_SPEAKER

        # Pass A: Strict Midpoint inclusion
        for idx in range(d_idx, d_len):
            d_seg = diarization[idx]
            if d_seg.start > mid:
                break  # Tracks are chronological; future tracks won't contain `mid`
            if d_seg.start <= mid <= d_seg.end:
                best_speaker = d_seg.speaker
                break

        # Pass B: Nearest Neighbor Fallback (if no strict midpoint match)
        if best_speaker == UNKNOWN_SPEAKER:
            min_dist = float("inf")
            best_candidate = UNKNOWN_SPEAKER

            # Search backwards from d_idx
            for idx in range(min(d_idx, d_len - 1), -1, -1):
                d_seg = diarization[idx]
                dist = max(0.0, d_seg.start - mid, mid - d_seg.end)

                if dist < min_dist:
                    min_dist = dist
                    best_candidate = d_seg.speaker
                elif mid - d_seg.end > MAX_FALLBACK_GAP:
                    # Too far back, stop looking backwards
                    break

            # Search forwards
            for idx in range(min(d_idx + 1, d_len - 1), d_len):
                d_seg = diarization[idx]
                dist = max(0.0, d_seg.start - mid, mid - d_seg.end)

                if dist < min_dist:
                    min_dist = dist
                    best_candidate = d_seg.speaker
                elif d_seg.start - mid > MAX_FALLBACK_GAP:
                    # Too far forward, stop looking forwards
                    break

            if min_dist <= MAX_FALLBACK_GAP:
                best_speaker = best_candidate

        labeled.append(
            MergedSegment(
                start=seg.start,
                end=seg.end,
                speaker=best_speaker,
                text=seg.text,
            )
        )

    return labeled


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
    intersection logic. Consecutive segments from the same speaker are
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

    # Assign speaker labels via O(T+D) sweep
    labeled = _assign_speaker_sweep(transcription, diarization)

    # Coalesce consecutive same-speaker segments
    result = _coalesce(labeled)

    logger.info(
        "Merged %d transcription + %d diarization -> %d segments",
        len(transcription),
        len(diarization),
        len(result),
    )

    return result
