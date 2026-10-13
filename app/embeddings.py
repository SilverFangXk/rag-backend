"""Step 3 of RAG: EMBED text.

An embedding is a list of numbers (here 768) that captures the MEANING of a text.
Texts with similar meaning get vectors that point in similar directions, so
"Where did she intern?" lands near "Experience: Data Engineering Intern at Nexteer"
even though they share almost no words. That is the whole trick behind semantic search.

Details worth knowing:
  * task_type: the model embeds slightly differently for a DOCUMENT (stored text) and a
    QUERY (the user's question). Using the matching pair improves retrieval.
  * output_dimensionality: gemini-embedding-001 natively makes 3072 numbers; we ask for
    768 (smaller file, faster search, nearly as good).
  * normalization: for dims other than 3072 the vectors are not unit length, so we
    normalize them; cosine distance then behaves consistently.
  * Auth: a Gemini API key from https://aistudio.google.com (free tier, no billing account).
"""
from __future__ import annotations

import logging
import math
import time

from google import genai
from google.genai import types

log = logging.getLogger(__name__)


def _normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


class Embedder:
    def __init__(self, settings, client=None):
        self.model = settings.embedding_model
        self.dim = settings.embedding_dim
        if client is not None:
            self._client = client
        else:
            if not settings.gemini_api_key:
                raise RuntimeError("GEMINI_API_KEY is not set (get a free key at aistudio.google.com).")
            self._client = genai.Client(api_key=settings.gemini_api_key)

    @property
    def salt(self) -> str:
        """Part of every chunk's fingerprint: changing model/dim forces a re-embed."""
        return f"{self.model}:{self.dim}"

    def embed_document(self, text: str) -> list[float]:
        return self._embed(text, "RETRIEVAL_DOCUMENT")

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text, "RETRIEVAL_QUERY")

    def _embed(self, text: str, task_type: str) -> list[float]:
        config = types.EmbedContentConfig(task_type=task_type, output_dimensionality=self.dim)
        last_exc: Exception | None = None
        for attempt in range(3):  # tiny retry for 429 / transient errors
            try:
                resp = self._client.models.embed_content(model=self.model, contents=text, config=config)
                vec = list(resp.embeddings[0].values)
                break
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                log.warning("embedding attempt %d failed: %s", attempt + 1, exc)
                time.sleep(2**attempt)
        else:
            raise RuntimeError("Embedding request failed") from last_exc

        if len(vec) != self.dim:
            raise ValueError(f"Expected {self.dim} dims, got {len(vec)}")
        return vec if self.dim == 3072 else _normalize(vec)
