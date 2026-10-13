"""Step 4 of RAG: STORE and SEARCH vectors.

For ~50-100 chunks you do not need a vector database. We keep everything in ONE json file
(`data/index.json`), load it into memory when the app starts, and compare the question's
vector with every chunk (a "brute-force" search). With 60 vectors that takes milliseconds.

Why this is nice:
  * zero infrastructure, zero cost, nothing to secure
  * the deployed app needs NO database credentials
  * the index is a normal file: you can open it, diff it in git, and see exactly what changed

When would you outgrow it? Around tens of thousands of chunks, where brute force gets slow
and you would switch to pgvector / Firestore vector search / Pinecone. The rest of the app
would not change: only this class is replaced (retrieval talks to `.search()`).

Similarity: COSINE distance = 1 - cos(angle). 0 = same direction, ~1 = unrelated.
"""
from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Hit:
    id: str
    section: str
    title: str
    text: str
    distance: float  # cosine distance, lower = more similar


def _cosine_distance(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return 1.0 - (dot / (na * nb)) if na and nb else 1.0


class JsonVectorStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._rows: dict[str, dict] = {}
        self._load()

    def __len__(self) -> int:
        return len(self._rows)

    def _load(self) -> None:
        if not self.path.exists():
            log.warning("Index file %s not found: run `python -m scripts.ingest` first.", self.path)
            return
        data = json.loads(self.path.read_text(encoding="utf-8"))
        self._rows = {r["id"]: r for r in data.get("chunks", [])}
        log.info("Loaded %d chunks from %s", len(self._rows), self.path)

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"version": 1, "chunks": [self._rows[k] for k in sorted(self._rows)]}
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
        os.replace(tmp, self.path)  # atomic: never leaves a half-written index

    # ---- write side (used by the indexer, on your laptop) ----
    def existing_hashes(self) -> dict[str, str]:
        return {cid: r.get("content_hash", "") for cid, r in self._rows.items()}

    def upsert_many(self, rows: list[tuple]) -> None:
        """rows = [(Chunk, embedding, content_hash)]. Same id => overwrite, never duplicate."""
        for chunk, vec, h in rows:
            self._rows[chunk.id] = {
                "id": chunk.id,
                "section": chunk.section,
                "title": chunk.title,
                "text": chunk.text,
                "content_hash": h,
                "embedding": [round(x, 6) for x in vec],  # 6 decimals: smaller file, same results
            }
        self._save()

    def delete_many(self, ids: list[str]) -> None:
        for cid in ids:
            self._rows.pop(cid, None)
        self._save()

    # ---- read side (used per question) ----
    def search(self, query_vec: list[float], k: int) -> list[Hit]:
        scored = []
        for r in self._rows.values():
            emb = r["embedding"]
            if len(emb) != len(query_vec):
                continue  # index built with a different embedding size: re-run ingest
            scored.append((_cosine_distance(query_vec, emb), r))
        scored.sort(key=lambda t: t[0])
        return [Hit(r["id"], r["section"], r["title"], r["text"], d) for d, r in scored[:k]]
