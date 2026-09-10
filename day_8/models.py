from pydantic import BaseModel
from typing import List, Literal


class AnswerResponse(BaseModel):
    answer: str
    sources: List[str]
    retrieved_sources: List[str]
    chunk_previews: List[str]
    retrieval_scores: List[float]
    status: Literal[
        "answered",
        "insufficient_evidence"
    ]