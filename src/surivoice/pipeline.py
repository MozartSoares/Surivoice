"""Pipeline orchestrator.

Wires together all processing stages to produce a complete
transcription with speaker diarization from a single input file.
"""

import logging
import tempfile
from pathlib import Path

from surivoice.audio.extractor import extract_audio
from surivoice.config import PipelineConfig
from surivoice.diarization.diarizer import diarize
from surivoice.merge.merger import merge_segments
from surivoice.models import TranscriptionResult
from surivoice.output.formatter import format_transcript, write_transcript
from surivoice.transcription.transcriber import transcribe

logger = logging.getLogger(__name__)


def run(
    input_path: Path,
    output_path: Path,
    config: PipelineConfig,
) -> TranscriptionResult:
    """Execute the full transcription pipeline.

    Stages:
        1. Extract audio → WAV 16kHz mono
        2. Transcribe → timestamped text segments
        3. Diarize → speaker-labeled time ranges
        4. Merge → assign speakers to transcription segments
        5. Format → render as Markdown
        6. Write → save to output file

    Args:
        input_path: Path to the input video/audio file.
        output_path: Path to the output Markdown file.
        config: Pipeline configuration.

    Returns:
        TranscriptionResult with all merged segments and metadata.

    Raises:
        SurivoiceError: If any stage fails.
    """
    with tempfile.TemporaryDirectory(prefix="surivoice_") as tmp_dir:
        tmp_path = Path(tmp_dir)

        # Stage 1: Audio extraction
        logger.info("Stage 1/6: Extracting audio")
        wav_path = extract_audio(input_path, tmp_path)

        # Stage 2: Transcription
        logger.info("Stage 2/6: Transcribing")
        transcribe_result = transcribe(wav_path, config)

        # Stage 3: Diarization
        logger.info("Stage 3/6: Diarizing")
        diarize_result = diarize(wav_path, config)

        # Stage 4: Merge
        logger.info("Stage 4/6: Merging segments")
        merged = merge_segments(
            transcribe_result.segments,
            diarize_result.segments,
        )

    # Stage 5: Format
    logger.info("Stage 5/6: Formatting transcript")
    content = format_transcript(
        segments=merged,
        detected_language=transcribe_result.detected_language,
        duration_seconds=transcribe_result.duration_seconds,
        speakers_count=diarize_result.speakers_count,
    )

    # Stage 6: Write
    logger.info("Stage 6/6: Writing output")
    write_transcript(content, output_path)

    # Build final result
    unique_speakers = {seg.speaker for seg in merged}
    result = TranscriptionResult(
        segments=merged,
        detected_language=transcribe_result.detected_language,
        duration_seconds=transcribe_result.duration_seconds,
        speakers_count=len(unique_speakers),
    )

    logger.info("Pipeline complete: %s", output_path)
    return result
