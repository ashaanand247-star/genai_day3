from typing import Literal
from pydantic import BaseModel, Field


class EvaluationCase(BaseModel):
    case_id: str = Field(min_length=1)
    question: str = Field(min_length=1)

    category: Literal[
        "answerable",
        "unanswerable",
        "ambiguous",
        "multi-document",
        "adversarial"
    ]

    expected_source_ids: list[str] = Field(default_factory=list)

    answerability: Literal[
        "answerable",
        "unanswerable"
    ]

    expected_facts: list[str] = Field(default_factory=list)

    answer_notes: str | None = None