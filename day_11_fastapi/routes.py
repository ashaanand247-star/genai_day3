import time
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Request
from openai import APIError


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
    log_retrieved_sources
)


router = APIRouter()


# Health check

@router.get("/health")
def health_check():
    return {"status": "ok"}


# Document ingestion

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

# Ask question

@router.post("/ask", response_model=AskResponse)
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


# Get document details

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