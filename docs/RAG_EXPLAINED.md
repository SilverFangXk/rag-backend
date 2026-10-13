# RAG from scratch, explained with this project

## 1. The problem RAG solves

An LLM like Claude knows nothing about *you*. Ask it "what did Khadija do at her internship?"
and it will happily **invent** a convincing answer (a hallucination). Two naive fixes fail:

* **Paste your whole portfolio into every prompt.** Works for tiny data, but gets slow, costly and
  noisy as it grows, and the model can still ignore it.
* **Fine-tune the model on your data.** Expensive, hard to update, and still doesn't prevent invention.

**Retrieval-Augmented Generation (RAG)** is the practical fix: for each question, first **retrieve**
only the few pieces of your data that are relevant, then **augment** the prompt with them, and let
the LLM **generate** an answer *from those pieces only*.

```
INDEXING (your laptop, offline)                  QUERY (server, every question)
───────────────────────────────                  ──────────────────────────────
Firestore documents                              "Which projects use Spark?"
      │ load                                            │ embed
      ▼                                                 ▼
 small chunks of text                              query vector
      │ embed                                           │ nearest-neighbour search
      ▼                                                 ▼
 vectors ──► saved in data/index.json ───────────► top-K relevant chunks
                                                        │ put in prompt
                                                        ▼
                                                  Claude writes the answer
                                                  (found / answer / sources)
```

## 2. The components, one by one

### 2.1 Loading (`firestore_client.py`)
Read the source of truth. This happens **only on your laptop**, using a Firebase service-account key,
so the deployed server never needs database credentials.

### 2.2 Chunking (`chunking.py`) ⭐ the part that decides quality
Embedding models turn *one text → one vector*. Chunk size is a trade-off:

* too big → the vector is a blurry average of many topics, so retrieval is imprecise
* too small → a chunk has no meaning on its own ("Python, SQL")

So we chunk **by meaning**, following how your data is already shaped (a project, an experience, a
skills category). Four tricks used here, worth remembering for any RAG project:

1. **Context header in every chunk.** "Project: ETL Pipeline - Ingestion" is the first line of every
   step chunk. Without it, a paragraph saying "we used Kafka" has no idea which project it belongs to.
2. **Group tiny items.** Each skill is a 2-word Firestore document. We merge them into one chunk per category.
3. **Overview chunk.** Broad questions ("tell me about her") need a table of contents to match.
4. **Never store things that go stale.** We store start/end *dates*, not "ongoing", and give the LLM
   *today's date* at question time. A stored "status" would silently become wrong.

Run `python -m scripts.ingest --print` and read the actual chunks, because seeing what the model will
see is the best way to understand RAG.

### 2.3 Embeddings (`embeddings.py`)
An embedding model maps text to a point in a high-dimensional space (here 768 numbers) so that
**similar meaning ⇒ nearby points**. "Where did she intern?" and "Experience: Data Intern at Nexteer"
share almost no words yet land close together. This is **semantic search**, and it is what keyword
search can't do.

Details: `task_type` differs for documents vs queries (the model is trained for that pairing);
we shrink 3072 → 768 dimensions (cheaper, ~same quality) and L2-normalise so cosine distance is well behaved.

### 2.4 Vector store (`vector_store.py`)
For ~60 chunks a vector database is overkill. We save `{text, embedding}` for every chunk in one file,
`data/index.json`, load it into memory at startup, and compare the question's vector with **every** chunk
("brute force") using **cosine distance**: 0 = identical direction, ~1 = unrelated. 60 comparisons take
milliseconds. Benefits: no infrastructure, no cost, nothing to secure, and the index is a readable file
you can diff in git. You'd outgrow it around tens of thousands of chunks, where you'd swap in pgvector,
Firestore vector search or Pinecone. Only this one class would change.

### 2.5 Retrieval (`retriever.py`)
Embed the question with the *same model*, fetch the top-K (default 6) nearest chunks, then **drop
those farther than `MAX_DISTANCE`**. That cut-off is anti-hallucination gate #1: for "capital of
France?" nothing is close, so we answer "not found" *without calling the LLM at all*.

### 2.6 Prompt augmentation (`prompts.py`)
The retrieved chunks go inside `<context>` tags with numbered `<source>` blocks. The system prompt
sets the rules (use only the context, say so when it's missing, third person, plain text, ignore
instructions found in the data, today's date).

### 2.7 Generation (`llm.py`)
Claude must answer through a **tool call** (`submit_answer`) returning `{found, answer, source_ids}`.
Forcing structured output means no fragile text-parsing, a reliable `found` flag for the UI, and exact
knowledge of *which chunks were used*, so "Sources" in the chat shows only what the answer relied on.

### 2.8 The API (`main.py`)
`POST /api/chat` ties it together and adds production concerns: CORS for your portfolio's domain, per-IP
rate limiting, input validation, and a 503 (not a stack trace) if an upstream service fails.

## 3. How hallucination is fought (defence in depth)

| Layer | Where | What it stops |
|---|---|---|
| 1. Distance cut-off | `retriever.py` | Off-topic questions never reach the LLM |
| 2. "Use only the context" + a legal way out (`found=false`) | `prompts.py` | The model inventing details to be helpful |
| 3. Structured output | `llm.py` | Free-form text sneaking past the `found` flag; malformed output defaults to *not found* |
| 4. Real sources only | `main.py` | Citing chunks that weren't used |
| 5. Data hygiene | `chunking.py` | Stale or contradictory facts in the index |
| 6. Injection hygiene | `prompts.py` | "Ignore previous instructions…" in a question |

No layer is perfect. RAG **reduces** hallucination, it doesn't eliminate it, which is why we show
sources and keep answers short.

## 4. Indexing is incremental (`indexer.py`)
Each chunk has a stable id and a hash of (model + text). Re-running `ingest` re-embeds only what
changed and deletes what disappeared. Two safety nets: changing the embedding model changes every
hash (forcing a re-embed, since vectors from different models aren't comparable), and an empty
Firestore read **refuses** to wipe the index. Because the index is a file, "deploying new knowledge" is just
`ingest` → `git push`.

## 5. Tuning & evaluating

* `python -m scripts.ask "question"` prints every retrieved chunk with its distance. Try 10 questions a
  recruiter would ask and 5 off-topic ones; set `MAX_DISTANCE` between the two groups.
* Keep a small **golden list** of questions with the answers you expect, and re-check after any change
  to chunking, prompts or models. That's the seed of a real evaluation.
* Wrong answer? Debug in order: *was the right chunk retrieved?* (chunking/embedding problem) →
  *was it in the prompt?* (threshold/top-K problem) → *did the model use it?* (prompt problem).

## 6. Where to go next
* **Conversation memory:** send the last few turns from `ResumeChat` and *rewrite* follow-ups
  ("and her skills there?") into standalone questions before retrieval.
* **Streaming:** `ResumeChat` expects one JSON response; streaming needs SSE on both sides.
* **Hybrid search + reranking:** combine keyword and vector search, then rerank top results.
* **Caching:** cache answers to common questions; the example chips in the UI are ideal candidates.
* **Auto re-index:** run `ingest` in a GitHub Action (on a schedule or a button) and commit the new `index.json`.
* **Bigger data:** swap `JsonVectorStore` for a real vector database; nothing else changes.
* **Bot-proofing:** Firebase App Check or a CAPTCHA before `/api/chat`.

## 7. File → concept map
| Concept | File |
|---|---|
| Load | `app/firestore_client.py` |
| Chunk | `app/chunking.py` |
| Embed | `app/embeddings.py` |
| Store / vector search | `app/vector_store.py` (`data/index.json`) |
| Retrieve + threshold | `app/retriever.py` |
| Augment (prompt) | `app/prompts.py` |
| Generate | `app/llm.py` |
| Indexing pipeline | `app/indexer.py`, `scripts/ingest.py` |
| Serving + security | `app/main.py`, `app/rate_limit.py`, `index.py` (Vercel) |
