import os
import time
from pathlib import Path

import requests


# ============================================================
# OpenRouter TTS configuration
# ============================================================

TTS_MODEL = "fish-audio/s2.1-pro-free:free"

OPENROUTER_URL = "https://openrouter.ai/api/v1/audio/speech"


# ============================================================
# Audio output directory
# ============================================================

AUDIO_DIR = Path(__file__).parent / "audio"
AUDIO_DIR.mkdir(exist_ok=True)


# ============================================================
# Text-to-speech
# ============================================================

def text_to_speech(
    grounded_answer: str,
    filename: str = "generated_answer.mp3"
):
    

    if not grounded_answer or not grounded_answer.strip():
        raise ValueError(
            "Cannot generate audio from an empty answer."
        )

    api_key = os.getenv("OPENROUTER_API_KEY")

    if not api_key:
        raise ValueError(
            "OPENROUTER_API_KEY is missing."
        )

    start_time = time.perf_counter()

    output_path = AUDIO_DIR / filename

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": TTS_MODEL,
        "input": grounded_answer,
        "response_format": "mp3"
    }

    response = requests.post(
        OPENROUTER_URL,
        headers=headers,
        json=payload
    )

    if response.status_code != 200:
        raise RuntimeError(
            f"TTS request failed: "
            f"{response.status_code} - {response.text}"
        )

    output_path.write_bytes(response.content)

    latency_ms = (
        time.perf_counter() - start_time
    ) * 1000

    return {
        "audio_path": str(output_path),
        "audio_format": "mp3",
        "synthetic_voice": True,
        "model": TTS_MODEL,
        "latency_ms": latency_ms
    }