"""Markdown output formatter.

Renders merged segments into a structured Markdown transcript with
speaker labels and timestamps.
"""

import logging
from pathlib import Path

from surivoice.errors import OutputError
from surivoice.models import MergedSegment

logger = logging.getLogger(__name__)

# Markdown structure constants — tests validate against these
HEADER = "# Transcript"
LABEL_LANGUAGE = "**Language:**"
LABEL_DURATION = "**Duration:**"
LABEL_SPEAKERS = "**Speakers:**"
LABEL_TOTAL_SPEAKERS = "**Total Speakers:**"
SEPARATOR = "---"


def _format_timestamp(seconds: float) -> str:
    """Format seconds as HH:MM:SS.

    Examples:
        0.0     -> "00:00:00"
        65.5    -> "00:01:05"
        3661.0  -> "01:01:01"
    """
    total_seconds = int(seconds)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _format_speaker_line(speaker: str, start: float, end: float) -> str:
    """Format a speaker segment header line."""
    return f"**{speaker}** [{_format_timestamp(start)} - {_format_timestamp(end)}]"


def format_transcript(
    segments: tuple[MergedSegment, ...],
    detected_language: str | None = None,
    duration_seconds: float | None = None,
    speakers_count: int | None = None,
) -> str:
    """Render merged segments as a Markdown transcript.

    Args:
        segments: Merged segments with speaker labels.
        detected_language: Optional detected language code for metadata.
        duration_seconds: Optional total audio duration for metadata.
        speakers_count: Optional number of speakers for metadata.

    Returns:
        Formatted Markdown string.
    """
    lines: list[str] = []

    # Header
    lines.append(HEADER)
    lines.append("")

    # Metadata block
    metadata_items: list[str] = []
    if detected_language is not None:
        metadata_items.append(f"- {LABEL_LANGUAGE} {detected_language}")
    if duration_seconds is not None:
        metadata_items.append(f"- {LABEL_DURATION} {_format_timestamp(duration_seconds)}")
    if speakers_count is not None:
        # List individual speaker identifiers from the segments
        unique_speakers = sorted({seg.speaker for seg in segments})
        metadata_items.append(f"- {LABEL_SPEAKERS}")
        for speaker in unique_speakers:
          prefix =  "_0" if speaker < 10 else "_"
          metadata_items.append(f"  - SPEAKER{prefix}{speaker}")
        metadata_items.append(f"- {LABEL_TOTAL_SPEAKERS} {speakers_count}")

    if metadata_items:
        for item in metadata_items:
            lines.append(item)
        lines.append("")

    lines.append(SEPARATOR)
    lines.append("")

    # Segments
    for segment in segments:
        lines.append(_format_speaker_line(segment.speaker, segment.start, segment.end))
        lines.append("")
        lines.append(segment.text)
        lines.append("")

    return "\n".join(lines)


def write_transcript(
    content: str,
    output_path: Path,
) -> None:
    """Write the formatted transcript to a file.

    Args:
        content: Formatted Markdown string.
        output_path: Destination file path.

    Raises:
        OutputError: If writing fails.
    """
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(content, encoding="utf-8")
    except OSError as exc:
        raise OutputError(
            f"{OutputError.WRITE_FAILED}: {output_path}\n{exc}"
        ) from exc

    logger.info("Transcript written to %s", output_path)
