from pathlib import Path

from day_17_speech.audio_validation import validate_audio_file
from day_17_speech.transcription import transcribe_audio


def test_clear_audio_file_validation():
    audio_path = Path("day_17_speech/clear_question.wav")

    file_size = audio_path.stat().st_size

    result = validate_audio_file(
        filename=audio_path.name,
        content_type="audio/wav",
        file_size=file_size
    )

    assert result["valid"] is True
    assert result["category"] is None


def test_clear_audio_transcription():
    audio_path = Path("day_17_speech/clear_question.wav")

    audio_bytes = audio_path.read_bytes()

    result = transcribe_audio(
        audio_bytes=audio_bytes,
        filename=audio_path.name
    )

    assert result["transcript"]
    assert result["language"] == "en"
    assert result["model"] == "tiny"


def test_empty_audio_file():
    result = validate_audio_file(
        filename="empty.wav",
        content_type="audio/wav",
        file_size=0
    )

    assert result["valid"] is False
    assert result["category"] == "empty_audio"


def test_unsupported_audio_file():
    result = validate_audio_file(
        filename="document.txt",
        content_type="text/plain",
        file_size=1000
    )

    assert result["valid"] is False
    assert result["category"] == "unsupported_audio_type"


def test_oversized_audio_file():
    result = validate_audio_file(
        filename="large.wav",
        content_type="audio/wav",
        file_size=11 * 1024 * 1024
    )

    assert result["valid"] is False
    assert result["category"] == "file_too_large"