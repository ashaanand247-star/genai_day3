import time

from datetime import datetime, timezone

from fastapi import (
    APIRouter,
    HTTPException,
    Request,
    UploadFile,
    File
)

from openai import APIError

from day_17_speech.audio_validation import validate_audio_file
from day_17_speech.transcription import transcribe_audio
from day_18_speech.tts import text_to_speech

from day_11_fastapi.models import (
    IngestRequest,
    IngestResponse,
    AskRequest,
    AskResponse,
    DocumentResponse
)

from day_7_rag.ingest import ingest_file
from day_7_rag.generate import answer_question, GENERATION_MODEL

from day_11_fastapi.documents import get_document

from day_11_fastapi.database import (
    log_request,
    log_retrieved_sources,
    log_voice_request
)


router = APIRouter()


# ============================================================
# Health check
# ============================================================

@router.get("/health")
def health_check():
    return {"status": "ok"}


# ============================================================
# Document ingestion
# ============================================================

@router.post("/ingest", response_model=IngestResponse)
def ingest_document(
    request: Request,
    body: IngestRequest
):
    start_time = time.perf_counter()

    request_id = request.state.request_id

    start_timestamp = datetime.now(
        timezone.utc
    ).isoformat()

    try:

        result = ingest_file(
            body.file_reference
        )

        total_latency_ms = (
            time.perf_counter() - start_time
        ) * 1000

        log_request(
            request_id=request_id,
            endpoint="/ingest",
            start_time=start_timestamp,
            total_latency_ms=total_latency_ms,
            model_version=None,
            prompt_version=None,
            outcome=result["status"],
            error_category=None
        )

        return IngestResponse(
            document_id=result["document_id"],
            chunk_count=result["chunk_count"],
            status=result["status"],
            request_id=request_id
        )

    except Exception:

        total_latency_ms = (
            time.perf_counter() - start_time
        ) * 1000

        log_request(
            request_id=request_id,
            endpoint="/ingest",
            start_time=start_timestamp,
            total_latency_ms=total_latency_ms,
            model_version=None,
            prompt_version=None,
            outcome="error",
            error_category="internal_error"
        )

        raise


# ============================================================
# Voice question - Day 18
# Audio -> Validation -> STT -> Reliability Check
# -> RAG -> TTS
# ============================================================

@router.post("/voice-ask")
async def voice_question(
    request: Request,
    audio: UploadFile = File(...)
):
    request_id = request.state.request_id

    # --------------------------------------------------------
    # Start overall voice request timer
    # --------------------------------------------------------

    start_time = time.perf_counter()

    start_timestamp = datetime.now(
        timezone.utc
    ).isoformat()

    # --------------------------------------------------------
    # Step 1: Read uploaded audio
    # --------------------------------------------------------

    audio_bytes = await audio.read()

    file_size = len(audio_bytes)

    # --------------------------------------------------------
    # Step 2: Validate audio
    # --------------------------------------------------------

    validation = validate_audio_file(
        filename=audio.filename,
        content_type=audio.content_type,
        file_size=file_size
    )

    if not validation["valid"]:

        total_latency_ms = (
            time.perf_counter() - start_time
        ) * 1000

        log_request(
            request_id=request_id,
            endpoint="/voice-ask",
            start_time=start_timestamp,
            total_latency_ms=total_latency_ms,
            model_version=None,
            prompt_version=None,
            outcome="error",
            error_category=validation["category"]
        )

        raise HTTPException(
            status_code=400,
            detail={
                "request_id": request_id,
                "category": validation["category"],
                "message": validation["message"]
            }
        )

    # --------------------------------------------------------
    # Step 3: Speech-to-text
    # --------------------------------------------------------

    try:

        transcription = transcribe_audio(
            audio_bytes=audio_bytes,
            filename=validation["filename"]
        )

    except Exception as exc:

        total_latency_ms = (
            time.perf_counter() - start_time
        ) * 1000

        log_request(
            request_id=request_id,
            endpoint="/voice-ask",
            start_time=start_timestamp,
            total_latency_ms=total_latency_ms,
            model_version="tiny",
            prompt_version=None,
            outcome="error",
            error_category="transcription_failure"
        )

        raise HTTPException(
            status_code=502,
            detail={
                "request_id": request_id,
                "category": "transcription_failure",
                "message": f"Speech-to-text failed: {str(exc)}"
            }
        )

    transcript = transcription["transcript"]

    # --------------------------------------------------------
    # Step 4: Make sure transcription is not empty
    # --------------------------------------------------------

    if not transcript:

        total_latency_ms = (
            time.perf_counter() - start_time
        ) * 1000

        log_request(
            request_id=request_id,
            endpoint="/voice-ask",
            start_time=start_timestamp,
            total_latency_ms=total_latency_ms,
            model_version=transcription["model"],
            prompt_version=None,
            outcome="error",
            error_category="empty_transcript"
        )

        raise HTTPException(
            status_code=400,
            detail={
                "request_id": request_id,
                "category": "empty_transcript",
                "message": "No speech could be detected in the audio"
            }
        )

    # --------------------------------------------------------
    # Step 5: Check transcription reliability
    # --------------------------------------------------------

    if not transcription["is_reliable"]:

        total_latency_ms = (
            time.perf_counter() - start_time
        ) * 1000

        log_request(
            request_id=request_id,
            endpoint="/voice-ask",
            start_time=start_timestamp,
            total_latency_ms=total_latency_ms,
            model_version=transcription["model"],
            prompt_version=None,
            outcome="error",
            error_category="unreliable_transcript"
        )

        raise HTTPException(
            status_code=400,
            detail={
                "request_id": request_id,
                "category": "unreliable_transcript",
                "message": (
                    "The speech transcription was not "
                    "reliable enough to process."
                ),
                "transcript": transcript,
                "avg_logprob": transcription["avg_logprob"]
            }
        )

    # --------------------------------------------------------
    # Step 6: Send transcript to existing RAG
    # --------------------------------------------------------

    rag_start_time = time.perf_counter()

    try:

        rag_response = answer_question(
            question=transcript,
            top_k=3,
            filters=None
        )

        rag_latency_ms = (
            time.perf_counter() - rag_start_time
        ) * 1000

    except APIError:

        total_latency_ms = (
            time.perf_counter() - start_time
        ) * 1000

        log_request(
            request_id=request_id,
            endpoint="/voice-ask",
            start_time=start_timestamp,
            total_latency_ms=total_latency_ms,
            model_version=transcription["model"],
            prompt_version="day_8_current",
            outcome="error",
            error_category="provider_failure"
        )

        raise HTTPException(
            status_code=502,
            detail={
                "request_id": request_id,
                "category": "provider_failure",
                "message": "Upstream AI provider failed"
            }
        )

    except Exception:

        total_latency_ms = (
            time.perf_counter() - start_time
        ) * 1000

        log_request(
            request_id=request_id,
            endpoint="/voice-ask",
            start_time=start_timestamp,
            total_latency_ms=total_latency_ms,
            model_version=transcription["model"],
            prompt_version="day_8_current",
            outcome="error",
            error_category="rag_failure"
        )

        raise HTTPException(
            status_code=500,
            detail={
                "request_id": request_id,
                "category": "rag_failure",
                "message": "Voice RAG processing failed"
            }
        )

    # --------------------------------------------------------
    # Step 7: Text-to-speech
    # --------------------------------------------------------

    tts_result = None

    tts_error = None

    if rag_response.status == "answered":

        try:

            tts_result = text_to_speech(
                grounded_answer=rag_response.answer,
                filename=f"{request_id}.mp3"
            )

        except Exception as exc:

            # TTS failure should NOT destroy
            # the validated text answer.

            tts_error = str(exc)

    # --------------------------------------------------------
    # Step 8: Calculate total latency
    # --------------------------------------------------------

    total_latency_ms = (
        time.perf_counter() - start_time
    ) * 1000

    # --------------------------------------------------------
    # Step 9: Log overall request
    # --------------------------------------------------------

    log_request(
        request_id=request_id,
        endpoint="/voice-ask",
        start_time=start_timestamp,
        total_latency_ms=total_latency_ms,
        model_version=transcription["model"],
        prompt_version="day_8_current",
        outcome=rag_response.status,
        error_category=None
    )

    # --------------------------------------------------------
    # Step 10: Log voice-specific observability
    # --------------------------------------------------------

    log_voice_request(
        request_id=request_id,
        stt_latency_ms=transcription["latency_ms"],
        rag_latency_ms=rag_latency_ms,
        transcription_model=transcription["model"],
        language=transcription["language"],
        transcript=transcript,
        outcome=rag_response.status,
        error_category=None
    )

    # --------------------------------------------------------
    # Step 11: Log retrieved sources
    # --------------------------------------------------------

    log_retrieved_sources(
        request_id=request_id,
        sources=rag_response.retrieved_sources,
        scores=rag_response.retrieval_scores
    )

    # --------------------------------------------------------
    # Step 12: Return voice + RAG + TTS result
    # --------------------------------------------------------

    return {
        "request_id": request_id,

        "status": rag_response.status,

        "transcript": transcript,

        "language": transcription["language"],

        "stt_latency_ms": round(
            transcription["latency_ms"],
            2
        ),

        "rag_latency_ms": round(
            rag_latency_ms,
            2
        ),

        "answer": rag_response.answer,

        "retrieved_sources": rag_response.retrieved_sources,

        "retrieval_scores": rag_response.retrieval_scores,

        "model": transcription["model"],

        "avg_logprob": transcription["avg_logprob"],

        # Day 18 TTS result
        "audio": tts_result,

        "tts_latency_ms": (
            round(
                tts_result["latency_ms"],
                2
            )
            if tts_result
            else None
        ),

        "total_latency_ms": round(
            total_latency_ms,
            2
        ),

        # Will contain an error only if TTS failed.
        # The text answer is still returned.
        "tts_error": tts_error
    }


# ============================================================
# Ask question
# ============================================================

@router.post(
    "/ask",
    response_model=AskResponse
)
def ask_question(
    request: Request,
    body: AskRequest
):
    start_time = time.perf_counter()

    request_id = request.state.request_id

    start_timestamp = datetime.now(
        timezone.utc
    ).isoformat()

    try:

        response = answer_question(
            question=body.question,
            top_k=body.top_k,
            filters=body.filters or None
        )

        total_latency_ms = (
            time.perf_counter() - start_time
        ) * 1000

        if response.status == "answered":

            outcome = "answered"

        else:

            outcome = "insufficient_evidence"

        log_request(
            request_id=request_id,
            endpoint="/ask",
            start_time=start_timestamp,
            total_latency_ms=total_latency_ms,
            model_version=GENERATION_MODEL,
            prompt_version="day_8_current",
            outcome=outcome,
            error_category=None
        )

        log_retrieved_sources(
            request_id=request_id,
            sources=response.retrieved_sources,
            scores=response.retrieval_scores
        )

        print(
            "RAG RESPONSE:",
            response
        )

        return AskResponse(
            **response.model_dump(),
            request_id=request_id
        )

    except APIError:

        total_latency_ms = (
            time.perf_counter() - start_time
        ) * 1000

        log_request(
            request_id=request_id,
            endpoint="/ask",
            start_time=start_timestamp,
            total_latency_ms=total_latency_ms,
            model_version=GENERATION_MODEL,
            prompt_version="day_8_current",
            outcome="error",
            error_category="provider_failure"
        )

        raise HTTPException(
            status_code=502,
            detail={
                "request_id": request_id,
                "category": "provider_failure",
                "message": "Upstream AI provider failed"
            }
        )

    except Exception:

        total_latency_ms = (
            time.perf_counter() - start_time
        ) * 1000

        log_request(
            request_id=request_id,
            endpoint="/ask",
            start_time=start_timestamp,
            total_latency_ms=total_latency_ms,
            model_version=GENERATION_MODEL,
            prompt_version="day_8_current",
            outcome="error",
            error_category="internal_error"
        )

        raise


# ============================================================
# Get document details
# ============================================================

@router.get(
    "/documents/{document_id}",
    response_model=DocumentResponse
)
def get_document_details(
    request: Request,
    document_id: str
):
    start_time = time.perf_counter()

    request_id = request.state.request_id

    start_timestamp = datetime.now(
        timezone.utc
    ).isoformat()

    document = get_document(
        document_id
    )

    if document is None:

        total_latency_ms = (
            time.perf_counter() - start_time
        ) * 1000

        log_request(
            request_id=request_id,
            endpoint=f"/documents/{document_id}",
            start_time=start_timestamp,
            total_latency_ms=total_latency_ms,
            model_version=None,
            prompt_version=None,
            outcome="error",
            error_category="document_not_found"
        )

        raise HTTPException(
            status_code=404,
            detail={
                "request_id": request_id,
                "category": "document_not_found",
                "message": "Document not found"
            }
        )

    total_latency_ms = (
        time.perf_counter() - start_time
    ) * 1000

    log_request(
        request_id=request_id,
        endpoint=f"/documents/{document_id}",
        start_time=start_timestamp,
        total_latency_ms=total_latency_ms,
        model_version=None,
        prompt_version=None,
        outcome="found",
        error_category=None
    )

    return DocumentResponse(
        **document
    )