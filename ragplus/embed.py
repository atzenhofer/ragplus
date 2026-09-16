"""Dense encoders: a remote OpenAI-compatible endpoint or local sentence-transformers,
selected by `EMBED_BACKEND`. The remote one needs no GPU and no torch.
"""
from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor

import httpx
import numpy as np

from .config import settings

log = logging.getLogger("uvicorn.error")

def _unit(vectors: list[list[float]]) -> np.ndarray:
    matrix = np.asarray(vectors, dtype=np.float32)
    return matrix / np.linalg.norm(matrix, axis=1, keepdims=True)


class ApiEncoder:
    """Embeddings from an OpenAI-compatible /embeddings endpoint."""

    def __init__(self):
        self.model = settings.embed_model
        self.device = f"api:{settings.embed_api_base}"
        self.on_gpu = False
        self.gpu_name = None
        if not settings.embed_api_key:
            raise RuntimeError("EMBED_BACKEND=api needs an API key "
                               "(DHINFRA_API_KEY / LLM_API_KEY / EMBED_API_KEY)")

    def encode(self, texts: list[str]) -> np.ndarray:
        """Unit-normalised embeddings. Each text is cut at EMBED_MAX_CHARS; requests hold at
        most EMBED_BATCH texts and about EMBED_BATCH_CHARS characters, run EMBED_CONCURRENCY
        at a time, retry on 429/503 and timeouts, and come back in input order."""
        payload = [text[:settings.embed_max_chars] for text in texts]
        batches = _batches(payload, settings.embed_batch, settings.embed_batch_chars)
        workers = max(1, settings.embed_concurrency)
        vectors: list[list[float]] = []
        with httpx.Client(timeout=settings.embed_timeout,
                          limits=httpx.Limits(max_connections=workers)) as client, \
             ThreadPoolExecutor(max_workers=workers) as pool:
            self._post(client, [payload[0][:64]] if payload else ["warm-up"])  # loads the model
            started = time.monotonic()
            for done, rows in enumerate(pool.map(lambda batch: self._post(client, batch), batches), 1):
                vectors += rows
                if done % 25 == 0 or done == len(batches):
                    log.info("embedded %d/%d texts in %.0fs", len(vectors), len(payload),
                             time.monotonic() - started)
        return _unit(vectors)

    def _post(self, client: httpx.Client, batch: list[str]) -> list[list[float]]:
        """One embeddings request, retried on 429, 503 and transport errors."""
        url = settings.embed_api_base.rstrip("/") + "/embeddings"
        headers = {"Authorization": f"Bearer {settings.embed_api_key}"}
        for attempt in range(settings.embed_retries + 1):
            last_try = attempt == settings.embed_retries
            backoff = min(60, 5 * 2 ** attempt)
            try:
                response = client.post(url, headers=headers,
                                       json={"model": self.model, "input": batch})
            except (httpx.TimeoutException, httpx.TransportError):
                if last_try:
                    raise
                time.sleep(backoff)
                continue
            if response.status_code in (429, 503) and not last_try:
                retry_after = response.headers.get("Retry-After")
                time.sleep(float(retry_after) if retry_after else backoff)
                continue
            response.raise_for_status()
            rows = sorted(response.json()["data"], key=lambda item: item["index"])
            return [item["embedding"] for item in rows]
        raise RuntimeError("unreachable")


def _batches(texts: list[str], max_count: int, max_chars: int) -> list[list[str]]:
    """Consecutive groups of at most max_count texts and about max_chars characters;
    one over-long text still forms its own batch."""
    batches: list[list[str]] = []
    current: list[str] = []
    size = 0
    for text in texts:
        if current and (len(current) >= max_count or size + len(text) > max_chars):
            batches.append(current)
            current, size = [], 0
        current.append(text)
        size += len(text)
    if current:
        batches.append(current)
    return batches


class LocalEncoder:
    """Embeddings from sentence-transformers, on the local GPU when one is available."""

    def __init__(self):
        import torch
        from sentence_transformers import SentenceTransformer

        use_gpu = settings.embed_device.lower() != "cpu" and torch.cuda.is_available()
        self.model = settings.embed_model
        self.device = "cuda" if use_gpu else "cpu"
        self.on_gpu = use_gpu
        self.gpu_name = torch.cuda.get_device_name(0) if use_gpu else None
        self._model = SentenceTransformer(self.model, device=self.device)

    def encode(self, texts: list[str]) -> np.ndarray:
        """Unit-normalised embeddings."""
        vectors = self._model.encode(texts, normalize_embeddings=True, convert_to_numpy=True,
                                     batch_size=64, show_progress_bar=False)
        return vectors.astype(np.float32)


def build_encoder():
    """Return the encoder named by EMBED_BACKEND."""
    if settings.embed_backend.lower() == "api":
        return ApiEncoder()
    return LocalEncoder()
