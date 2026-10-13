"""Request/response shapes. They match what ResumeChat.jsx already sends and expects."""
from pydantic import BaseModel, Field, field_validator


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=500)

    @field_validator("question")
    @classmethod
    def _strip(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("question must not be empty")
        return v


class Source(BaseModel):
    section: str
    text: str


class ChatResponse(BaseModel):
    answer: str
    found: bool  # False -> the panda shrugs in the UI
    sources: list[Source] = []
