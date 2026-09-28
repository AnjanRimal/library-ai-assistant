"""FastAPI service exposing the assistant."""

from __future__ import annotations

import logging
from functools import lru_cache

from fastapi import Depends, FastAPI, HTTPException

from . import __version__
from .agent import AgentError, LibraryAgent, make_client
from .catalog import build_catalog
from .config import get_settings
from .embeddings import build_embedder
from .models import AskRequest, AskResponse
from .retrieval import VectorIndex

logging.basicConfig(level=logging.INFO, format="%(message)s")

app = FastAPI(
    title="Library AI Assistant",
    version=__version__,
    description="Claude-powered library assistant with RAG, tool calling, guardrails and evals.",
)


@lru_cache(maxsize=1)
def _build_agent() -> LibraryAgent:
    settings = get_settings()
    catalog = build_catalog(settings.catalog_backend, settings.library_api_url)
    index = VectorIndex(build_embedder(settings.embedder)).build(catalog.all_books())
    return LibraryAgent(make_client(), catalog, index, settings)


def get_agent() -> LibraryAgent:
    try:
        return _build_agent()
    except AgentError as exc:  # e.g. missing API key: report it instead of a bare 500
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "version": __version__}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest, agent: LibraryAgent = Depends(get_agent)) -> AskResponse:
    try:
        return agent.ask(request.question, member_id=request.member_id)
    except ValueError as exc:  # blocked by an input guardrail
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except AgentError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.get("/metrics")
def metrics(agent: LibraryAgent = Depends(get_agent)) -> dict:
    return agent.metrics.snapshot()
