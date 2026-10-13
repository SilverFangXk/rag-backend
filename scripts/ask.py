"""Debug retrieval from the terminal: see WHAT is retrieved and HOW close it is.

    python -m scripts.ask "What is her experience?"
    python -m scripts.ask "What is the capital of France?"   # should be far away
    python -m scripts.ask "Which projects use Spark?" --llm   # also run the LLM

Use the distances to calibrate MAX_DISTANCE: relevant questions should sit below it,
off-topic ones above it. (Needs only GEMINI_API_KEY + ANTHROPIC_API_KEY; no Firestore.)
"""
import argparse

from app.config import get_settings
from app.embeddings import Embedder
from app.vector_store import JsonVectorStore


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--llm", action="store_true")
    ap.add_argument("-k", type=int, default=None)
    args = ap.parse_args()

    s = get_settings()
    store = JsonVectorStore(s.index_path)
    hits = store.search(Embedder(s).embed_query(args.question), args.k or s.top_k)

    print(f"{len(store)} chunks in index | MAX_DISTANCE = {s.max_distance}\n")
    for h in hits:
        print(f"{h.distance:.3f}  {'KEEP' if h.distance <= s.max_distance else 'drop'}  {h.id}")
    kept = [h for h in hits if h.distance <= s.max_distance]

    if args.llm:
        from app.llm import LLM

        if not kept:
            print("\n(no chunk close enough -> the API answers 'not found' without calling the LLM)")
            return
        r = LLM(s).answer(args.question, kept)
        print(f"\nfound={r.found} sources={r.source_ids}\n{r.answer}")


if __name__ == "__main__":
    main()
