"""FastAPI app: serves the single-page UI and the search / RAG endpoints."""
from __future__ import annotations

import logging
from collections import Counter
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import llm, rag
from .config import settings
from .corpus import load_corpus
from .index import Index
from .search import Context

STATIC = Path(__file__).resolve().parent / "static"
STATE: dict = {}
log = logging.getLogger("uvicorn.error")


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not settings.corpus_path.exists():
        raise SystemExit(f"corpus not found: {settings.corpus_path}\n"
                         "Set CORPUS_PATH to a corpus jsonl.")
    docs = load_corpus(settings.corpus_path)
    log.info("corpus %s: %s documents", settings.corpus_path, len(docs))
    log.info("embeddings: %s via %s", settings.embed_model, settings.embed_backend)
    STATE["index"] = Index(docs)
    log.info("ready: fields %s", STATE["index"].fields)
    yield
    STATE.clear()


app = FastAPI(title="ragplus", lifespan=lifespan)


class QueryReq(BaseModel):
    query: str
    mode: str = "explore"
    text_field: str = "text"
    year_from: int | None = None
    year_to: int | None = None
    sources: list[str] = Field(default_factory=list)
    languages: list[str] = Field(default_factory=list)
    regions: list[str] = Field(default_factory=list)
    genres: list[str] = Field(default_factory=list)
    k: int = 10
    alpha: float = 0.6
    diversity: float = 0.3
    serendipity: float = 0.3
    balance_sources: bool = True
    selected_ids: list[str] = Field(default_factory=list)

    def to_context(self) -> Context:
        return Context(**self.model_dump(exclude={"selected_ids"}))


@app.get("/api/meta")
def meta() -> dict:
    idx: Index = STATE["index"]
    docs = idx.docs
    def facet(name):
        return sorted(Counter(getattr(d, name) for d in docs).items(),
                      key=lambda t: (-t[1], t[0]))
    return {
        "corpus_size": len(docs),
        "year_min": min(d.year for d in docs),
        "year_max": max(d.year for d in docs),
        "sources": [{"value": v, "count": c} for v, c in facet("source")],
        "languages": [{"value": v, "count": c} for v, c in facet("language")],
        "regions": [{"value": v, "count": c} for v, c in facet("region")],
        "genres": [{"value": v, "count": c} for v, c in facet("genre")],
        "topics": sorted({t for d in docs for t in d.topics}),
        "text_fields": idx.fields,
        "embed_model": settings.embed_model,
        "embed_backend": settings.embed_backend,
        "embed_on_gpu": idx.on_gpu,
        "embed_device": idx.device,
        "embed_gpu": idx.gpu_name,
        "llm_model": settings.llm_model if llm.available() else None,
        "llm_available": llm.available(),
    }


@app.post("/api/search")
def search_endpoint(req: QueryReq) -> dict:
    from . import search as search_mod
    return search_mod.run(STATE["index"], req.to_context())


@app.post("/api/rag")
def rag_endpoint(req: QueryReq) -> dict:
    return rag.answer(STATE["index"], req.to_context(), req.selected_ids)


@app.get("/")
def home() -> FileResponse:
    return FileResponse(STATIC / "index.html")


app.mount("/static", StaticFiles(directory=STATIC), name="static")
