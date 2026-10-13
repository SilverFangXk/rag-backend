"""Step 5 of RAG: RETRIEVE the chunks relevant to a question.

question -> embedding -> nearest chunks -> drop the ones that are not close enough.

The distance cut-off is the FIRST anti-hallucination gate: if nothing in the portfolio
is reasonably similar to the question (e.g. "what's the capital of France?"), we return
no chunks, and the API answers "not found" WITHOUT even calling the LLM. Cheaper, faster
and impossible to hallucinate. Tune MAX_DISTANCE with `python -m scripts.ask`.
"""
from __future__ import annotations

import logging

from .vector_store import Hit

log = logging.getLogger(__name__)


class Retriever:
    def __init__(self, embedder, store, top_k: int, max_distance: float):
        self.embedder, self.store = embedder, store
        self.top_k, self.max_distance = top_k, max_distance

    def retrieve(self, question: str) -> list[Hit]:
        if hasattr(self.store, "__len__") and len(self.store) == 0:
            log.warning("The index is empty: run `python -m scripts.ingest` and redeploy.")
            return []  # nothing to search: skip the embedding call entirely
        qvec = self.embedder.embed_query(question)
        hits = self.store.search(qvec, self.top_k)
        kept = [h for h in hits if h.distance <= self.max_distance]
        log.info("retrieved=%d kept=%d best=%.3f", len(hits), len(kept), hits[0].distance if hits else -1)
        return kept
