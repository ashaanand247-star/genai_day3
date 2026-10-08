import time
import tempfile
from pathlib import Path

from faster_whisper import WhisperModel


TRANSCRIPTION_MODEL = "tiny"
STT_CONFIDENCE_THRESHOLD = -0.6

_model = None


def get_model():
    global _model

    if _model is None:
        _model = WhisperModel(
            TRANSCRIPTION_MODEL,
            device="cpu",
            compute_type="int8",
            cpu_threads=1,
            num_workers=1
        )

    return _model


def transcribe_audio(audio_bytes: bytes, filename: str):
    start_time = time.perf_counter()

    suffix = Path(filename).suffix or ".ogg"

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix
    ) as temp_file:
        temp_file.write(audio_bytes)
        temp_path = temp_file.name

    try:
        model = get_model()

        segments, info = model.transcribe(
            temp_path,
            beam_size=5,
            language="en"
        )

        transcript_parts = []
        segment_confidences = []

        for segment in segments:
            text = segment.text.strip()

            if text:
                transcript_parts.append(text)

                # Whisper provides an average log probability
                # for each segment.
                segment_confidences.append(
                    segment.avg_logprob
                )

        transcript = " ".join(
            part for part in transcript_parts
            if part
        ).strip()

        # Calculate average confidence signal
        if segment_confidences:
            avg_logprob = (
                sum(segment_confidences)
                / len(segment_confidences)
            )
        else:
            avg_logprob = None

        # Decide whether the transcript is reliable enough
        is_reliable = (
            avg_logprob is not None
            and avg_logprob >= STT_CONFIDENCE_THRESHOLD
        )

        latency_ms = (
            time.perf_counter() - start_time
        ) * 1000

        return {
            "transcript": transcript,
            "latency_ms": latency_ms,
            "model": TRANSCRIPTION_MODEL,
            "language": info.language,
            "avg_logprob": avg_logprob,
            "is_reliable": is_reliable
        }

    finally:
        Path(temp_path).unlink(
            missing_ok=True
        )