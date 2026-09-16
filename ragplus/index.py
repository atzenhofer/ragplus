"""Indexing: dense embeddings (local or remote encoder) + lexical BM25, with an on-disk
cache that later starts load."""
from __future__ import annotations

import hashlib
import logging
import re
import time

import numpy as np
from rank_bm25 import BM25Okapi

from .config import CACHE_DIR, settings
from .corpus import Document
from .embed import build_encoder

_TOKEN = re.compile(r"[A-Za-zÀ-ÿ0-9]+")
CHUNK = 512  # documents per checkpoint
log = logging.getLogger("uvicorn.error")


def _tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class Index:
    """Documents, their unit-normalised dense embeddings, and a BM25 index."""

    def __init__(self, docs: list[Document]):
        self.docs = docs
        self.encoder = build_encoder()
        self.device = self.encoder.device
        self.on_gpu = self.encoder.on_gpu
        self.gpu_name = self.encoder.gpu_name
        self.fields = ["text"] + (["htr"] if any(d.htr for d in docs) else [])
        self.embeddings_by_field = {f: self._build_embeddings(docs, f) for f in self.fields}
        self.embeddings = self.embeddings_by_field["text"]      # (N, D), unit rows
        self.bm25 = BM25Okapi([_tokenize(f"{d.title} {d.text}") for d in docs])

    def _embed_texts(self, texts: list[str]) -> np.ndarray:
        return self.encoder.encode(texts)

    def _field_text(self, doc: Document, field: str) -> str:
        return doc.htr if field == "htr" else f"{doc.title}. {doc.text}"

    def _build_embeddings(self, docs: list[Document], field: str) -> np.ndarray:
        texts = [self._field_text(d, field) for d in docs]
        key = hashlib.sha1(
            (f"{settings.embed_backend}|{settings.embed_model}|{field}|"
             + "\n".join(texts)).encode("utf-8")
        ).hexdigest()[:16]
        CACHE_DIR.mkdir(exist_ok=True)
        cache = CACHE_DIR / f"emb-{key}.npy"
        if cache.exists():
            log.info("field %s: cached", field)
            return np.load(cache)
        # Each chunk is saved as it finishes, so an interrupted run resumes at the next one.
        chunks = range(0, len(texts), CHUNK)
        parts = [CACHE_DIR / f"emb-{key}.part{i // CHUNK:04d}.npy" for i in chunks]
        todo = [(i, part) for i, part in zip(chunks, parts) if not part.exists()]
        log.info("field %s: embedding %s documents in %s chunks, %s already done",
                 field, len(texts), len(parts), len(parts) - len(todo))
        started = time.monotonic()
        for i, part in todo:
            np.save(part, self._embed_texts(texts[i:i + CHUNK]))
        emb = np.concatenate([np.load(part) for part in parts]) if parts else np.zeros((0, 0))
        np.save(cache, emb)
        for part in parts:
            part.unlink()
        log.info("field %s: embedded in %.0fs", field, time.monotonic() - started)
        return emb

    def matrix(self, field: str) -> np.ndarray:
        """Embeddings for `field`, falling back to the text when absent."""
        return self.embeddings_by_field.get(field, self.embeddings)

    def encode_query(self, query: str) -> np.ndarray:
        """Unit-normalised embedding for a query string."""
        # bge v1.5 wants a query instruction; bge-m3 does not.
        if "bge-" in settings.embed_model.lower() and settings.embed_model.endswith("v1.5"):
            query = "Represent this sentence for searching relevant passages: " + query
        return self._embed_texts([query])[0]

    def dense_scores(self, query: str, field: str = "text") -> np.ndarray:
        """Cosine similarity of the query to every document, over `field`."""
        q = self.encode_query(query)
        return self.matrix(field) @ q  # cosine, since rows and q are unit vectors

    def lexical_scores(self, query: str) -> np.ndarray:
        """BM25 score of the query against every document."""
        return np.asarray(self.bm25.get_scores(_tokenize(query)), dtype=np.float32)

    def pairwise(self, idx: list[int], field: str = "text") -> np.ndarray:
        """Cosine similarity matrix among the documents at the given indices."""
        sub = self.matrix(field)[idx]
        return sub @ sub.T
