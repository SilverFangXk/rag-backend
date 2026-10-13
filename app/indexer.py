"""INDEXING pipeline: Firestore data -> chunks -> embeddings -> data/index.json.

Runs on YOUR laptop (scripts/ingest.py), never per user request and never on the server.
It is INCREMENTAL and IDEMPOTENT:
  * each chunk has a stable id and a content hash
  * unchanged chunks are skipped (no embedding cost)
  * changed chunks are overwritten, removed ones are deleted
  * running it twice in a row does nothing the second time
"""
from __future__ import annotations

import logging
from pathlib import Path

from .chunking import Chunk, build_chunks
from .firestore_client import COLLECTIONS, fetch_portfolio

log = logging.getLogger(__name__)


def read_extra_facts(path: str) -> str:
    p = Path(path)
    return p.read_text(encoding="utf-8") if p.exists() else ""


def data_warnings(data: dict) -> list[str]:
    """Catch the classic trap: the site shows built-in defaults until 'Import current data' is clicked."""
    imports = data["settings"].get("imports", {})
    warns = []
    for name in COLLECTIONS:
        if not data.get(name):
            warns.append(f"'{name}' is EMPTY in Firestore (the website shows built-in defaults until you "
                         f"click 'Import current data' in /admin).")
        elif not imports.get(name):
            warns.append(f"'{name}' has documents but is not flagged as imported; the website may still "
                         f"show the built-in defaults.")
    return warns


def run_index(*, settings, db, embedder, store, force: bool = False, dry_run: bool = False) -> dict:
    data = fetch_portfolio(db)
    warnings = data_warnings(data)
    chunks: list[Chunk] = build_chunks(data, settings.owner_full_name, read_extra_facts(settings.extra_facts_path))
    if not chunks:
        # Guard: a failed/empty read must never wipe the index.
        raise RuntimeError("No chunks were built (is Firestore empty?). Refusing to touch the index.")

    salt = embedder.salt
    existing = {} if dry_run else store.existing_hashes()
    wanted = {c.id: c.fingerprint(salt) for c in chunks}
    todo = [c for c in chunks if force or existing.get(c.id) != wanted[c.id]]
    stale = [cid for cid in existing if cid not in wanted]

    report = {
        "total_chunks": len(chunks),
        "to_embed": len(todo),
        "unchanged": len(chunks) - len(todo),
        "to_delete": len(stale),
        "warnings": warnings,
        "dry_run": dry_run,
    }
    if dry_run:
        return report

    rows = []
    for n, c in enumerate(todo, start=1):
        rows.append((c, embedder.embed_document(c.text), wanted[c.id]))
        log.info("embedded %d/%d: %s", n, len(todo), c.id)
    store.upsert_many(rows)
    store.delete_many(stale)
    return report
