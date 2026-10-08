from pathlib import Path


MAX_AUDIO_SIZE = 10 * 1024 * 1024  # 10 MB


ALLOWED_AUDIO_TYPES = {
    "audio/wav",
    "audio/x-wav",
    "audio/mpeg",
    "audio/mp3",
    "audio/mp4",
    "audio/m4a",
    "audio/webm",
    "audio/ogg",
    "application/ogg",
}


ALLOWED_AUDIO_EXTENSIONS = {
    ".wav",
    ".mp3",
    ".mp4",
    ".m4a",
    ".webm",
    ".ogg",
}


def validate_audio_file(
    filename: str | None,
    content_type: str | None,
    file_size: int
):
    """
    Validate an uploaded audio file.
    """

    if not filename:
        return {
            "valid": False,
            "category": "empty_audio",
            "message": "No audio file was provided."
        }

    if file_size == 0:
        return {
            "valid": False,
            "category": "empty_audio",
            "message": "The uploaded audio file is empty."
        }

    if file_size > MAX_AUDIO_SIZE:
        return {
            "valid": False,
            "category": "file_too_large",
            "message": "Audio file exceeds the 10 MB limit."
        }

    extension = Path(filename).suffix.lower()

    if (
        content_type not in ALLOWED_AUDIO_TYPES
        and extension not in ALLOWED_AUDIO_EXTENSIONS
    ):
        return {
            "valid": False,
            "category": "unsupported_audio_type",
            "message": "Unsupported audio file type."
        }

    return {
        "valid": True,
        "category": None,
        "message": "Audio file accepted.",
        "filename": filename,
        "content_type": content_type,
        "file_size": file_size,
        "extension": extension
    }