# Portfolio RAG backend (FastAPI + Claude, host it on Vercel OR Render, no Google Cloud)

A chatbot backend that answers recruiters' questions **only from your portfolio data**.

```
 INDEXING (on your laptop, when your content changes)
   Firestore ──► chunk ──► embed (Gemini) ──► data/index.json ──► git push

 QUERY (on Vercel, for every question)
   question ──► embed ──► compare with index.json ──► top chunks ──► Claude ──► answer
```

* **No Google Cloud account or billing.** The deployed app never talks to Firestore. It only reads
  `data/index.json`, a file you build locally and deploy with the code.
* **Embeddings:** free Gemini API key. **Answers:** Claude (Anthropic API, pay per use).
* **Hosting:** your choice of **Vercel** or **Render** (both have free plans). Config for both is included.

New to RAG? Read **[docs/RAG_EXPLAINED.md](docs/RAG_EXPLAINED.md)**.

## Project layout

```
index.py             Vercel entry point (exports the FastAPI app)
render.yaml          Render config (Blueprint)         Dockerfile  optional, for any container host
.python-version      pins Python 3.12
app/
  config.py          settings from environment variables
  firestore_client.py  load your portfolio data (used only by ingest, on your laptop)
  chunking.py        cut the data into small, self-contained pieces   ← read this one first
  embeddings.py      text -> vector (Gemini)
  vector_store.py    index.json loaded in memory + cosine search
  retriever.py       question -> relevant chunks (+ distance cut-off)
  prompts.py         system prompt + context formatting
  llm.py             Claude call with structured output (found / answer / sources)
  indexer.py         incremental, idempotent indexing
  rate_limit.py      per-IP limiter
  main.py            FastAPI app: /api/chat, /health, CORS
scripts/ingest.py    build/refresh data/index.json       scripts/ask.py  debug retrieval
knowledge/extra_facts.md   facts that live in your JSX rather than Firestore
data/index.json      the vector index (created by ingest, committed to git)
tests/               29 offline tests
frontend/            the one-line change for your React app
```

---

## 1. Local setup

```bash
cd rag-backend
python3.12 -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
pytest -q                                                  # expect: 29 passed
cp .env.example .env
```

Fill in `.env`:

| Variable | Where to get it |
|---|---|
| `ANTHROPIC_API_KEY` | console.anthropic.com → API keys (also set a **monthly spend limit** there) |
| `GEMINI_API_KEY` | https://aistudio.google.com → "Get API key" (free, no billing account) |
| `ALLOWED_ORIGINS` | `http://localhost:5173` for now (your Vite dev address) |
| `GOOGLE_APPLICATION_CREDENTIALS` | see below |

**Firestore read access (only for building the index).** Firebase console → ⚙️ Project settings →
*Service accounts* → **Generate new private key**. Save the file as `rag-backend/serviceAccountKey.json`
(it is git-ignored). This key stays on your laptop and is **never deployed**. Treat it like a password.
(It is free: no billing account is needed for the Spark plan.)

## 2. Build the index

> ⚠️ Your site shows built-in defaults until you click **"Import current data"** in `/admin` for each
> section. The bot only sees Firestore, so import every section first.

```bash
python -m scripts.ingest --dry-run      # no AI calls: shows chunk count + warns about empty sections
python -m scripts.ingest --print        # builds data/index.json and prints every chunk
python -m scripts.ask "Which projects use Spark?" --llm      # test retrieval + answer
python -m scripts.ask "What is the capital of France?"       # should show everything as 'drop'
```

If real questions get `drop`ped, raise `MAX_DISTANCE` in `.env` (e.g. 0.65).
Before indexing, review `knowledge/extra_facts.md`: **every line must be true**.

Run the API and try your site locally:
```bash
uvicorn app.main:app --reload --port 8000
```
Then `npm run dev` in your portfolio (with the new `Home.jsx`) and chat with the panda.

## 3. Deploy the backend (pick ONE host)

| | **Vercel** | **Render** |
|---|---|---|
| Cost | free Hobby plan | free plan |
| Idle behaviour | wakes up fast | free instance **sleeps when idle; the first message can take about a minute** |
| Best for | this chatbot (light, fast) | heavier agents, long jobs, bigger libraries, local models |
| Config file | `index.py` (+ `.python-version`) | `render.yaml` |
| Same code? | yes, nothing to change | yes, nothing to change |

**Both start the same way**: put `rag-backend/` in its **own Git repo** (private is fine; `data/index.json` is committed on purpose):
```bash
cd rag-backend
git init && git add . && git commit -m "RAG backend"
# create an empty repo on GitHub, then:
git remote add origin https://github.com/YOU/portfolio-rag.git && git push -u origin main
```
Check that `git status` shows no `.env` and no `serviceAccountKey.json` (both are git-ignored), and that `data/index.json` **is** in the commit.

You will enter the same three values on either host:

| Name | Value |
|---|---|
| `ANTHROPIC_API_KEY` | your key |
| `GEMINI_API_KEY` | your key |
| `ALLOWED_ORIGINS` | your portfolio's exact URL, e.g. `https://yourname.vercel.app` (https, no trailing `/`) |

### 3A. Vercel
1. Vercel dashboard → **Add New → Project** → import the repo. Vercel detects FastAPI by itself; leave build settings alone.
2. Add the three environment variables **before** the first deploy, then **Deploy**.
3. Your URL looks like `https://portfolio-rag-xxxx.vercel.app`.

> I could not deploy to Vercel from my environment. If the build complains about the entry point, read the build log: Vercel looks for an `app` object in root files like `index.py` (included). Guide: https://vercel.com/docs/frameworks/backend/fastapi

### 3B. Render
1. Render dashboard → **New → Blueprint** → connect GitHub → pick the repo. Render reads `render.yaml`.
2. It asks for the three secret values (they are not stored in git). Fill them in and **Apply**.
3. Wait until the service shows **Live**. Your URL is `https://portfolio-rag.onrender.com` (if the name is taken, change `name:` in `render.yaml` and push).
4. Keep `plan: free` in `render.yaml`: if that line is removed, Render creates a paid plan.

### 3C. Test it, then connect your portfolio
```bash
curl https://YOUR-BACKEND/health          # {"status":"ok","chunks":NN}   (NN must be > 0)
curl -X POST https://YOUR-BACKEND/api/chat -H 'Content-Type: application/json' \
     -d '{"question":"What projects has she built?"}'
```
(On Render's free plan the first call may take about a minute while the service wakes up.)

Then, in your **portfolio's** Vercel project add `VITE_CHAT_API_URL = https://YOUR-BACKEND/api/chat` and **redeploy** the portfolio (see `frontend/README.md`).

**Rate limiting note:** the limiter identifies visitors by the `X-Forwarded-For` header. If you notice all visitors being throttled together, your host has one more proxy in front, so set `CLIENT_IP_HOPS=2`.

## 4. Updating the bot when your portfolio changes

The index is a snapshot, so after editing something in `/admin`:

```bash
python -m scripts.ingest        # only changed chunks are re-embedded (fast, almost free)
git add data/index.json && git commit -m "refresh index" && git push    # Vercel / Render redeploy themselves
```

## 5. Security notes

| Concern | What this project does |
|---|---|
| Firestore | The deployed app has **no** database access or credentials at all. |
| Secrets | API keys live only in `.env` (git-ignored) and Vercel's encrypted env vars. |
| Service-account key | Used on your laptop only, git-ignored. If it ever leaks: Firebase console → delete the key. |
| CORS | Only `ALLOWED_ORIGINS` (+ optional regex). `allow_credentials=False`. |
| Abuse / cost | Per-IP rate limit (forgery-resistant), 500-char question cap. **The limit is in memory and per instance (serverless may run several), so treat it as a speed bump**: the real cap is your Anthropic spend limit. |
| Prompt injection | Question + context are escaped and treated as data; the model's only tool is `submit_answer`. |
| Privacy | Questions are sent to Anthropic (answer) and Google (Gemini embedding). Gemini's free tier may use submitted content to improve Google products: fine for a public portfolio, but tell recruiters not to type anything private. A paid Gemini key avoids that. |

CORS is not authentication: anyone can call a public URL with `curl`, which is why the spend limit matters.

## 6. Troubleshooting

| Symptom | Fix |
|---|---|
| `/health` shows `"chunks": 0` | `data/index.json` wasn't committed/deployed. Run ingest, commit, push. |
| First message is very slow (Render) | The free instance was asleep. Normal; later messages are fast. |
| Render service is billed | `plan: free` was removed from `render.yaml`. Put it back. |
| Bot always shrugs | Index empty, or `MAX_DISTANCE` too strict: use `scripts.ask` and compare distances. |
| Bot knows nothing about a section | That section wasn't imported in `/admin` (dry run warns), or you didn't re-run ingest + push. |
| Browser CORS error | `ALLOWED_ORIGINS` must match the page's origin exactly. Redeploy after changing env vars. |
| HTML / 401 page instead of JSON | Vercel *Deployment Protection* is blocking the backend project: Settings → Deployment Protection. |
| 503 from `/api/chat` | Check the host's Logs (bad API key, Gemini rate limit, ...). |
| Gemini 429 during ingest | Free-tier rate limit: wait a minute and run ingest again; finished chunks are kept. |
