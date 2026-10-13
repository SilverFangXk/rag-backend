import pytest

from app import indexer
from app.chunking import Chunk
from app.config import get_settings
from app.vector_store import JsonVectorStore
from tests.test_chunking import DATA


def chunk(i, text="t"):
    return Chunk(f"c{i}", "Sec", f"title{i}", text)


def test_search_orders_by_cosine_distance(tmp_path):
    s = JsonVectorStore(tmp_path / "i.json")
    s.upsert_many([(chunk(1), [1, 0, 0], "h1"), (chunk(2), [0, 1, 0], "h2"), (chunk(3), [0.9, 0.1, 0], "h3")])
    hits = s.search([1, 0, 0], 3)
    assert [h.id for h in hits] == ["c1", "c3", "c2"]
    assert hits[0].distance == pytest.approx(0.0, abs=1e-9)
    assert hits[2].distance == pytest.approx(1.0)
    assert len(s.search([1, 0, 0], 2)) == 2


def test_store_persists_and_reloads(tmp_path):
    path = tmp_path / "data" / "i.json"
    s = JsonVectorStore(path)
    s.upsert_many([(chunk(1), [1, 0], "h1"), (chunk(2), [0, 1], "h2")])
    s.delete_many(["c2"])
    again = JsonVectorStore(path)
    assert len(again) == 1 and again.existing_hashes() == {"c1": "h1"}


def test_missing_index_file_is_an_empty_store(tmp_path):
    s = JsonVectorStore(tmp_path / "nope.json")
    assert len(s) == 0 and s.search([1, 0], 3) == []


def test_dimension_mismatch_is_ignored_not_crashing(tmp_path):
    s = JsonVectorStore(tmp_path / "i.json")
    s.upsert_many([(chunk(1), [1, 0, 0], "h")])
    assert s.search([1, 0], 3) == []


class FakeEmbedder:
    salt = "fake:3"

    def __init__(self):
        self.calls = 0

    def embed_document(self, text):
        self.calls += 1
        return [float(len(text) % 7 + 1), 1.0, 0.5]


def run(tmp_path, monkeypatch, data, embedder, **kw):
    monkeypatch.setattr(indexer, "fetch_portfolio", lambda db: data)
    store = JsonVectorStore(tmp_path / "index.json")
    s = get_settings().model_copy(update={"extra_facts_path": str(tmp_path / "none.md")})
    return indexer.run_index(settings=s, db=None, embedder=embedder, store=store, **kw), store


def test_indexing_is_incremental_and_idempotent(tmp_path, monkeypatch):
    emb = FakeEmbedder()
    r1, store = run(tmp_path, monkeypatch, DATA, emb)
    assert r1["to_embed"] == r1["total_chunks"] > 0 and len(store) == r1["total_chunks"]

    before = emb.calls
    r2, _ = run(tmp_path, monkeypatch, DATA, emb)  # nothing changed
    assert r2["to_embed"] == 0 and emb.calls == before

    changed = {**DATA, "certifications": [{"id": "c1", "name": "Azure Data Fundamentals", "issuer": "Microsoft", "date": "Jan 2027"}]}
    r3, _ = run(tmp_path, monkeypatch, changed, emb)  # one chunk edited
    assert r3["to_embed"] == 1 and r3["to_delete"] == 0

    removed = {**changed, "certifications": []}
    r4, store = run(tmp_path, monkeypatch, removed, emb)  # one chunk removed
    assert r4["to_delete"] == 1 and "certifications--c1" not in store.existing_hashes()


def test_empty_firestore_never_wipes_the_index(tmp_path, monkeypatch):
    emb = FakeEmbedder()
    run(tmp_path, monkeypatch, DATA, emb)
    empty = {k: [] for k in ("projects", "skills", "experience", "education", "certifications")}
    empty["settings"] = {}
    with pytest.raises(RuntimeError):
        run(tmp_path, monkeypatch, empty, emb)
    assert len(JsonVectorStore(tmp_path / "index.json")) > 0


def test_dry_run_changes_nothing(tmp_path, monkeypatch):
    emb = FakeEmbedder()
    r, store = run(tmp_path, monkeypatch, DATA, emb, dry_run=True)
    assert r["dry_run"] and emb.calls == 0 and len(store) == 0


def test_retriever_skips_embedding_when_index_is_empty(tmp_path):
    from app.retriever import Retriever

    class Boom:
        def embed_query(self, q):
            raise AssertionError("must not be called")

    assert Retriever(Boom(), JsonVectorStore(tmp_path / "none.json"), 6, 0.55).retrieve("anything") == []


def test_retriever_applies_distance_cutoff(tmp_path):
    from app.retriever import Retriever

    class E:
        def embed_query(self, q):
            return [1.0, 0.0]

    s = JsonVectorStore(tmp_path / "i.json")
    s.upsert_many([(chunk(1), [1, 0], "a"), (chunk(2), [0, 1], "b")])  # distances 0 and 1
    assert [h.id for h in Retriever(E(), s, 6, 0.55).retrieve("q")] == ["c1"]
