"""Build / refresh the vector index (data/index.json).   Run from the backend folder:

    python -m scripts.ingest --dry-run     # show what would happen, call no AI APIs
    python -m scripts.ingest               # embed new/changed chunks, drop stale ones
    python -m scripts.ingest --force       # re-embed everything
    python -m scripts.ingest --print       # also print every chunk (great for learning)

Needs, on YOUR laptop only: GOOGLE_APPLICATION_CREDENTIALS (Firebase service-account key,
to read Firestore) and GEMINI_API_KEY. After it runs, commit data/index.json and push.
"""
import argparse
import json
import logging

from app.chunking import build_chunks
from app.config import get_settings
from app.firestore_client import fetch_portfolio, get_db
from app.indexer import data_warnings, read_extra_facts, run_index


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--print", dest="show", action="store_true", help="print all chunks")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    s = get_settings()
    db = get_db(s.gcp_project_id, s.firestore_database)
    data = fetch_portfolio(db)
    chunks = build_chunks(data, s.owner_full_name, read_extra_facts(s.extra_facts_path))

    if args.show:
        for c in chunks:
            print(f"\n=== {c.id}  [{c.section}]  ({len(c.text)} chars)\n{c.text}")
        print(f"\n{len(chunks)} chunks")

    if args.dry_run:  # needs neither embeddings nor the index
        print(json.dumps({"total_chunks": len(chunks), "warnings": data_warnings(data)}, indent=2))
        return

    from app.embeddings import Embedder
    from app.vector_store import JsonVectorStore

    report = run_index(settings=s, db=db, embedder=Embedder(s), store=JsonVectorStore(s.index_path), force=args.force)
    print(json.dumps(report, indent=2))
    print(f"\nIndex written to {s.index_path}. Commit it and push to deploy the new knowledge.")


if __name__ == "__main__":
    main()
