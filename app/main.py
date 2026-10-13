import os
_k = os.environ.get("GEMINI_API_KEY")
print("GEMINI_API_KEY set:", _k is not None, "| length:", len(_k) if _k else 0, flush=True)
print("env names containing KEY/GEMINI:", [n for n in os.environ if "GEMINI" in n.upper() or "KEY" in n.upper()], flush=True)

"""The FastAPI app: wires the RAG pipeline to HTTP.

POST /api/chat   {question}  ->  {answer, found, sources[]}     (public, CORS + rate limited)
GET  /health
"""
from __future__ import annotations

import logging
import re
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from .config import Settings, get_settings
from .rate_limit import SlidingWindowLimiter
from .schemas import ChatRequest, ChatResponse, Source

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("app")

GREETING = re.compile(r"^\s*(hi|hello|hey|hola|bonjour|salut|good (morning|afternoon|evening))\b[\s!.?]*$", re.I)
THANKS = re.compile(r"^\s*(thanks|thank you|thx|merci|bye|goodbye)\b[\s!.?]*$", re.I)


@dataclass
class Services:
    settings: Settings
    embedder: Any
    store: Any
    retriever: Any
    llm: Any


def build_services(settings: Settings) -> Services:
    # imported here so unit tests (which inject fakes) never touch the real SDKs
    from .embeddings import Embedder
    from .llm import LLM
    from .retriever import Retriever
    from .vector_store import JsonVectorStore

    embedder = Embedder(settings)
    store = JsonVectorStore(settings.index_path)
    retriever = Retriever(embedder, store, settings.top_k, settings.max_distance)
    return Services(settings, embedder, store, retriever, LLM(settings))


def _snippet(text: str, n: int = 240) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 1].rstrip() + "…"


def _client_ip(request: Request, hops: int = 1) -> str:
    # Behind a proxy, X-Forwarded-For is "client, proxy1, proxy2...". Entries on the LEFT can
    # be forged by the caller, so we count from the RIGHT: hops=1 is the last entry (added by
    # the platform's proxy). If every visitor ends up sharing one IP in your logs/limits,
    # your host has an extra proxy in front: set CLIENT_IP_HOPS=2.
    parts = [p.strip() for p in request.headers.get("x-forwarded-for", "").split(",") if p.strip()]
    if parts:
        return parts[-hops] if len(parts) >= hops else parts[0]
    return request.client.host if request.client else "unknown"


def create_app(services: Services | None = None) -> FastAPI:
    settings = services.settings if services else get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if getattr(app.state, "services", None) is None:
            app.state.services = build_services(settings)
        if not settings.origins and not settings.allowed_origin_regex:
            log.warning("ALLOWED_ORIGINS is empty: browsers will be blocked by CORS.")
        yield

    docs = {} if settings.enable_docs else {"docs_url": None, "redoc_url": None, "openapi_url": None}
    app = FastAPI(title="Portfolio RAG API", lifespan=lifespan, **docs)
    app.state.services = services

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.origins,
        allow_origin_regex=settings.allowed_origin_regex,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type"],
        allow_credentials=False,
        max_age=3600,
    )

    limiter = SlidingWindowLimiter(settings.rate_limit_per_minute)

    def rate_limit(request: Request) -> None:
        wait = limiter.check(_client_ip(request, settings.client_ip_hops))
        if wait:
            raise HTTPException(429, "Too many questions, please slow down.", headers={"Retry-After": str(wait)})

    def get_services(request: Request) -> Services:
        # Normally built at startup (lifespan). Some serverless hosts skip lifespan,
        # so build lazily on first use as a safety net.
        if request.app.state.services is None:
            request.app.state.services = build_services(settings)
        return request.app.state.services

    not_found = (
        f"I couldn't find that in {settings.owner_name}'s portfolio, so I don't want to guess. "
        "Try asking about her projects, skills, experience or availability."
    )

    @app.get("/health")
    def health(request: Request):
        svc = request.app.state.services
        return {"status": "ok", "chunks": len(svc.store) if svc and hasattr(svc.store, "__len__") else None}

    # NOTE: plain `def` (not async) -> FastAPI runs it in a thread pool, so the blocking
    # Gemini / Anthropic SDK calls don't freeze the event loop.
    @app.post("/api/chat", response_model=ChatResponse)
    def chat(body: ChatRequest, svc: Services = Depends(get_services), _: None = Depends(rate_limit)):
        q = body.question

        if GREETING.match(q):
            return ChatResponse(answer=f"Hi! Ask me about {settings.owner_name}'s projects, skills, "
                                       "experience or availability.", found=True)
        if THANKS.match(q):
            return ChatResponse(answer="You're welcome! Ask me anything else about her profile.", found=True)

        try:
            hits = svc.retriever.retrieve(q)
            if not hits:  # gate 1: nothing similar enough -> no LLM call at all
                return ChatResponse(answer=not_found, found=False)
            result = svc.llm.answer(q, hits)  # gate 2: the LLM can also say "not in context"
        except Exception:  # noqa: BLE001
            log.exception("chat pipeline failed")
            raise HTTPException(503, "The assistant is temporarily unavailable.")

        if not result.found or not result.answer.strip():
            return ChatResponse(answer=result.answer.strip() or not_found, found=False)

        cited = [i for i in dict.fromkeys(result.source_ids) if 1 <= i <= len(hits)] or [1, 2]
        sources = [Source(section=f"{hits[i-1].section} · {hits[i-1].title}", text=_snippet(hits[i-1].text))
                   for i in cited if i <= len(hits)]
        return ChatResponse(answer=result.answer.strip(), found=True, sources=sources)

    return app


app = create_app()
