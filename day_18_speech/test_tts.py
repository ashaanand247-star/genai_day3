import os
import time
from pathlib import Path

from openai import OpenAI


# TTS configuration
TTS_MODEL = "gpt-4o-mini-tts"
TTS_VOICE = "alloy"

# Where generated audio will be stored
AUDIO_DIR = Path(__file__).parent / "audio"
AUDIO_DIR.mkdir(exist_ok=True)


client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)


def text_to_speech(
    grounded_answer: str,
    filename: str = "generated_answer.mp3"
):
    """
    Convert a validated grounded answer into synthetic speech.

    IMPORTANT:
    This function should receive only the validated answer
    produced by the RAG pipeline.
    """

    if not grounded_answer or not grounded_answer.strip():
        raise ValueError(
            "Cannot generate audio from an empty answer."
        )

    start_time = time.perf_counter()

    output_path = AUDIO_DIR / filename

    response = client.audio.speech.create(
        model=TTS_MODEL,
        voice=TTS_VOICE,
        input=grounded_answer,
        response_format="mp3"
    )

    response.write_to_file(output_path)

    latency_ms = (
        time.perf_counter() - start_time
    ) * 1000

    return {
        "audio_path": str(output_path),
        "audio_format": "mp3",
        "synthetic_voice": True,
        "model": TTS_MODEL,
        "voice": TTS_VOICE,
        "latency_ms": latency_ms
    }