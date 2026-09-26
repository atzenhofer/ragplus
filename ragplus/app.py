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

from . import llm, rag, search
from .config import settings
from .corpus import FACETS, load_corpus
from .index import Index
from .search import Context

STATIC = Path(__file__).resolve().parent / "static"
STATE: dict = {}
log = logging.getLogger("uvicorn.error")


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not settings.corpus_path.exists():
        raise SystemExit(f"corpus not found: {settings.corpus_path}\n"
                         "Build one with `python -m ragplus.ddp`, or set CORPUS_PATH.")
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
    filters: dict[str, list[str]] = Field(default_factory=dict)
    k: int = 10
    alpha: float = 0.6
    diversity: float = 0.3
    serendipity: float = 0.3
    balance_sources: bool = True
    selected_ids: list[str] = Field(default_factory=list)

    def to_context(self) -> Context:
        return Context(**self.model_dump(exclude={"selected_ids"}))


def _facet_counts(docs, name: str) -> list[dict]:
    """Values of one facet with their document counts, most common first."""
    counts = Counter(value for doc in docs for value in doc.facets.get(name, ()))
    return [{"value": value, "count": count}
            for value, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))]


@app.get("/api/meta")
def meta() -> dict:
    index: Index = STATE["index"]
    docs = index.docs
    return {
        "corpus_size": len(docs),
        "year_min": min(doc.year for doc in docs),
        "year_max": max(doc.year for doc in docs),
        "facets": [{"name": name, "label": label, "values": values}
                   for name, label in FACETS.items()
                   if (values := _facet_counts(docs, name))],
        "text_fields": index.fields,
        "embed_model": settings.embed_model,
        "embed_backend": settings.embed_backend,
        "embed_on_gpu": index.on_gpu,
        "embed_device": index.device,
        "embed_gpu": index.gpu_name,
        "llm_model": settings.llm_model if llm.available() else None,
        "llm_available": llm.available(),
    }


@app.post("/api/search")
def search_endpoint(request: QueryReq) -> dict:
    return search.run(STATE["index"], request.to_context())


@app.post("/api/rag")
def rag_endpoint(request: QueryReq) -> dict:
    return rag.answer(STATE["index"], request.to_context(), request.selected_ids)


@app.get("/")
def home() -> FileResponse:
    return FileResponse(STATIC / "index.html")


app.mount("/static", StaticFiles(directory=STATIC), name="static")
