from __future__ import annotations

from typing import Dict

from fastapi import APIRouter, Header, HTTPException
from fastapi.responses import HTMLResponse
from pathlib import Path

from app.embeddings.factory import EmbeddingFactory
from app.generation.factory import GenerationFactory
from app.models.config import Settings, get_settings
from app.retrieval.service import HybridRetrievalService


admin_router = APIRouter(prefix="/admin")


def _auth(api_key: str | None) -> None:
    s = get_settings()
    if s.admin_api_key and api_key != s.admin_api_key:
        raise HTTPException(status_code=401, detail="Unauthorized")


@admin_router.post("/hot-swap/embeddings")
def hot_swap_embeddings(backend: str, model: str, x_api_key: str | None = Header(default=None)) -> Dict[str, str]:
    _auth(x_api_key)
    s = get_settings()
    s.embedding_backend = backend
    s.embedding_model = model
    return {"status": "ok", "embedding_backend": backend, "embedding_model": model}


@admin_router.post("/hot-swap/generation")
def hot_swap_generation(backend: str, model: str, x_api_key: str | None = Header(default=None)) -> Dict[str, str]:
    _auth(x_api_key)
    s = get_settings()
    s.generation_backend = backend
    s.generation_model = model
    return {"status": "ok", "generation_backend": backend, "generation_model": model}


@admin_router.post("/hot-swap/retriever")
def hot_swap_retriever(backend: str, x_api_key: str | None = Header(default=None)) -> Dict[str, str]:
    _auth(x_api_key)
    s = get_settings()
    s.retriever_backend = backend
    return {"status": "ok", "retriever_backend": backend}


@admin_router.get("/dashboard", response_class=HTMLResponse)
def dashboard() -> HTMLResponse:
    html_path = Path(__file__).parent / "templates" / "dashboard.html"
    return HTMLResponse(html_path.read_text())


