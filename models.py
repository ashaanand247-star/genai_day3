from pydantic import BaseModel
from typing import Literal


class SummarizationOutput(BaseModel):
    summary: str


class ClassificationOutput(BaseModel):
    category: Literal["Health", "Sports", "Technology", "Business"]


class ExtractionOutput(BaseModel):
    name: str
    age: int
    department: str