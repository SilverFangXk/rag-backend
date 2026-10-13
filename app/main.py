
"""The FastAPI app: wires the RAG pipeline to HTTP.

POST /api/chat   {question}  ->  {answer, found, sources[]} (public, CORS + rate limited)
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

GREETING = re.compile(
    r"^\s*(hi|hello|hey|hola|bonjour|salut|good (morning|afternoon|evening))\b[\s!.?]*$",
    re.I,
)
THANKS = re.compile(r"^\s*(thanks|thank you|thx|merci|bye|goodbye)\b[\s!.?]*$", re.I)

# Exact frontend origin allowed to call the API.
FRONTEND_ORIGINS = [
    "https://portfolio-zar628yov-silverfangxks-projects.vercel.app",
]


@dataclass
class Services:
    settings: Settings
    embedder: Any
    store: Any
    retriever: Any
    llm: Any


def build_services(settings: Settings) -> Services:
    # Imported here so unit tests with fake services do not load real SDKs.
    from .embeddings import Embedder
    from .llm import LLM
    from .retriever import Retriever
    from .vector_store import JsonVectorStore

    embedder = Embedder(settings)
    store = JsonVectorStore(settings.index_path)
    retriever = Retriever(
        embedder,
        store,
        settings.top_k,
        settings.max_distance,
    )
    return Services(settings, embedder, store, retriever, LLM(settings))


def _snippet(text: str, n: int = 240) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 1].rstrip() + "…"


def _client_ip(request: Request, hops: int = 1) -> str:
    # Count from the right because entries on the left may be forged.
    parts = [
        p.strip()
        for p in request.headers.get("x-forwarded-for", "").split(",")
        if p.strip()
    ]
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
            log.warning("Configured ALLOWED_ORIGINS is empty.")

        yield

    docs = (
        {}
        if settings.enable_docs
        else {
            "docs_url": None,
            "redoc_url": None,
            "openapi_url": None,
        }
    )

    app = FastAPI(
        title="Portfolio RAG API",
        lifespan=lifespan,
        **docs,
    )
    app.state.services = services

    # Combine configured origins with the portfolio's frontend origin.
    allowed_origins = list(
        dict.fromkeys([*settings.origins, *FRONTEND_ORIGINS])
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_origin_regex=settings.allowed_origin_regex,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type"],
        allow_credentials=False,
        max_age=3600,
    )

    limiter = SlidingWindowLimiter(settings.rate_limit_per_minute)

    def rate_limit(request: Request) -> None:
        wait = limiter.check(
            _client_ip(request, settings.client_ip_hops)
        )
        if wait:
            raise HTTPException(
                429,
                "Too many questions, please slow down.",
                headers={"Retry-After": str(wait)},
            )

    def get_services(request: Request) -> Services:
        # Build lazily if the hosting platform skips lifespan startup.
        if request.app.state.services is None:
            request.app.state.services = build_services(settings)
        return request.app.state.services

    not_found = (
        f"I couldn't find that in {settings.owner_name}'s portfolio, "
        "so I don't want to guess. Try asking about her projects, "
        "skills, experience or availability."
    )

    @app.get("/health")
    def health(request: Request):
        svc = request.app.state.services
        return {
            "status": "ok",
            "chunks": (
                len(svc.store)
                if svc and hasattr(svc.store, "__len__")
                else None
            ),
        }

    # Plain def lets FastAPI run blocking SDK calls in a thread pool.
    @app.post("/api/chat", response_model=ChatResponse)
    def chat(
        body: ChatRequest,
        svc: Services = Depends(get_services),
        _: None = Depends(rate_limit),
    ):
        q = body.question

        if GREETING.match(q):
            return ChatResponse(
                answer=(
                    f"Hi! Ask me about {settings.owner_name}'s projects, "
                    "skills, experience or availability."
                ),
                found=True,
            )

        if THANKS.match(q):
            return ChatResponse(
                answer="You're welcome! Ask me anything else about her profile.",
                found=True,
            )

        try:
            hits = svc.retriever.retrieve(q)

            if not hits:
                return ChatResponse(answer=not_found, found=False)

            result = svc.llm.answer(q, hits)

        except Exception:  # noqa: BLE001
            log.exception("chat pipeline failed")
            raise HTTPException(
                503,
                "The assistant is temporarily unavailable.",
            )

        if not result.found or not result.answer.strip():
            return ChatResponse(
                answer=result.answer.strip() or not_found,
                found=False,
            )

        cited = [
            i
            for i in dict.fromkeys(result.source_ids)
            if 1 <= i <= len(hits)
        ] or [1, 2]

        sources = [
            Source(
                section=f"{hits[i - 1].section} · {hits[i - 1].title}",
                text=_snippet(hits[i - 1].text),
            )
            for i in cited
            if i <= len(hits)
        ]

        return ChatResponse(
            answer=result.answer.strip(),
            found=True,
            sources=sources,
        )

    return app


app = create_app()
