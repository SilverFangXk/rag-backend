"""Step 7 of RAG: GENERATE the answer (Gemini, structured JSON output)."""
from __future__ import annotations

import logging

from google import genai
from google.genai import types
from pydantic import BaseModel, Field, ValidationError

from .prompts import system_prompt, user_message
from .vector_store import Hit

log = logging.getLogger(__name__)

ANSWER_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "found": {"type": "BOOLEAN"},
        "answer": {"type": "STRING"},
        "source_ids": {"type": "ARRAY", "items": {"type": "INTEGER"}},
    },
    "required": ["found", "answer", "source_ids"],
}


class AnswerResult(BaseModel):
    found: bool
    answer: str
    source_ids: list[int] = Field(default_factory=list)


class LLM:
    def __init__(self, settings, client=None):
        self.s = settings
        if client is not None:
            self.client = client
        else:
            if not settings.gemini_api_key:
                raise RuntimeError("GEMINI_API_KEY is not set.")
            self.client = genai.Client(api_key=settings.gemini_api_key)

    def answer(self, question: str, hits: list[Hit]) -> AnswerResult:
        config = types.GenerateContentConfig(
            system_instruction=system_prompt(self.s.owner_full_name, self.s.owner_name),
            max_output_tokens=self.s.llm_max_tokens,
            response_mime_type="application/json",
            response_schema=ANSWER_SCHEMA,
        )
        if self.s.llm_temperature is not None:
            config.temperature = self.s.llm_temperature

        resp = self.client.models.generate_content(
            model=self.s.llm_model,
            contents=user_message(question, hits),
            config=config,
        )
        try:
            return AnswerResult.model_validate_json(resp.text or "")
        except (ValidationError, ValueError):
            log.exception("model returned malformed JSON")
        # Safe default: never show unvalidated text as a grounded answer.
        return AnswerResult(found=False, answer="", source_ids=[])