import uuid

from day_17_speech.audio_validation import validate_audio_file
from day_17_speech.transcription import transcribe_audio
from day_7_rag.generate import answer_question

from day_18_speech.tts import text_to_speech


def process_voice_question(
    audio_bytes: bytes,
    filename: str,
    content_type: str
):
    """
    Complete voice-to-voice RAG pipeline.

    Audio
      ↓
    Validation
      ↓
    STT
      ↓
    RAG
      ↓
    Validated grounded answer
      ↓
    TTS
      ↓
    Audio
    """

    # --------------------------------------------------------
    # 1. Create one request ID
    # --------------------------------------------------------

    request_id = str(uuid.uuid4())

    # --------------------------------------------------------
    # 2. Validate audio
    # --------------------------------------------------------

    validation = validate_audio_file(
        filename=filename,
        content_type=content_type,
        file_size=len(audio_bytes)
    )

    if not validation["valid"]:

        return {
            "request_id": request_id,
            "status": "audio_validation_failed",
            "transcript": None,
            "answer": None,
            "sources": [],
            "audio": None,
            "error": validation["category"]
        }

    # --------------------------------------------------------
    # 3. Speech-to-Text
    # --------------------------------------------------------

    transcription = transcribe_audio(
        audio_bytes=audio_bytes,
        filename=filename
    )

    transcript = transcription["transcript"]

    # Do not send an empty transcript to RAG
    if not transcript:

        return {
            "request_id": request_id,
            "status": "stt_failed",
            "transcript": None,
            "answer": None,
            "sources": [],
            "audio": None,
            "error": "No reliable transcript was produced."
        }

    # --------------------------------------------------------
    # 4. RAG
    # --------------------------------------------------------

    rag_response = answer_question(
        transcript
    )

    # --------------------------------------------------------
    # 5. Check whether RAG produced a validated answer
    # --------------------------------------------------------

    if rag_response.status != "answered":

        return {
            "request_id": request_id,
            "status": rag_response.status,
            "transcript": transcript,
            "answer": rag_response.answer,
            "sources": rag_response.sources,
            "audio": None,
            "error": None
        }

    # --------------------------------------------------------
    # 6. Get validated grounded answer
    # --------------------------------------------------------

    grounded_answer = rag_response.answer

    # --------------------------------------------------------
    # 7. Convert grounded answer to speech
    # --------------------------------------------------------

    tts_result = text_to_speech(
        grounded_answer
    )

    # --------------------------------------------------------
    # 8. Return complete voice response
    # --------------------------------------------------------

    return {
        "request_id": request_id,
        "status": "answered",
        "transcript": transcript,
        "answer": grounded_answer,
        "sources": rag_response.sources,
        "audio": tts_result,
        "error": None
    }