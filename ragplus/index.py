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
        self.fields = ["text"] + (["htr"] if any(doc.htr for doc in docs) else [])
        self.embeddings_by_field = {field: self._build_embeddings(docs, field)
                                    for field in self.fields}
        self.embeddings = self.embeddings_by_field["text"]      # (N, D), unit rows
        self.bm25 = BM25Okapi([_tokenize(f"{doc.title} {doc.text}") for doc in docs])

    def _embed_texts(self, texts: list[str]) -> np.ndarray:
        return self.encoder.encode(texts)

    def _field_text(self, doc: Document, field: str) -> str:
        return doc.htr if field == "htr" else f"{doc.title}. {doc.text}"

    def _build_embeddings(self, docs: list[Document], field: str) -> np.ndarray:
        texts = [self._field_text(doc, field) for doc in docs]
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
        starts = range(0, len(texts), CHUNK)
        parts = [CACHE_DIR / f"emb-{key}.part{start // CHUNK:04d}.npy" for start in starts]
        todo = [(start, part) for start, part in zip(starts, parts) if not part.exists()]
        log.info("field %s: embedding %s documents in %s chunks, %s already done",
                 field, len(texts), len(parts), len(parts) - len(todo))
        started = time.monotonic()
        for start, part in todo:
            np.save(part, self._embed_texts(texts[start:start + CHUNK]))
        embeddings = (np.concatenate([np.load(part) for part in parts]) if parts
                      else np.zeros((0, 0)))
        np.save(cache, embeddings)
        for part in parts:
            part.unlink()
        log.info("field %s: embedded in %.0fs", field, time.monotonic() - started)
        return embeddings

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
        query_vector = self.encode_query(query)
        return self.matrix(field) @ query_vector  # cosine: rows and query are unit vectors

    def lexical_scores(self, query: str) -> np.ndarray:
        """BM25 score of the query against every document."""
        return np.asarray(self.bm25.get_scores(_tokenize(query)), dtype=np.float32)

    def pairwise(self, indices: list[int], field: str = "text") -> np.ndarray:
        """Cosine similarity matrix among the documents at the given indices."""
        vectors = self.matrix(field)[indices]
        return vectors @ vectors.T
