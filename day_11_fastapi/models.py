from typing import Any

from pydantic import BaseModel, Field

from day_8.models import AnswerResponse


class IngestRequest(BaseModel):
    file_reference: str


class IngestResponse(BaseModel):
    document_id: list[str]
    chunk_count: int
    status: str
    request_id: str


class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    top_k: int = Field(default=3, ge=1, le=10)
    filters: dict[str, Any] | None = None


class AskResponse(AnswerResponse):
    request_id: str


class DocumentResponse(BaseModel):
    document_id: str
    title: str
    source_path: str
    updated_at: str
    chunk_count: int
    status: str


class ErrorDetails(BaseModel):
    category: str
    message: str


class ErrorResponse(BaseModel):
    request_id: str
    error: ErrorDetails