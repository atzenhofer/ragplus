"""Dense encoders: a remote OpenAI-compatible endpoint or local sentence-transformers,
selected by `EMBED_BACKEND`. The remote one needs no GPU and no torch.
"""
from __future__ import annotations

import httpx
import numpy as np

from .config import settings

MAX_CHARS = 2000


def _unit(vectors: list[list[float]]) -> np.ndarray:
    v = np.asarray(vectors, dtype=np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


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
        """Unit-normalised embeddings; over-long inputs are truncated."""
        payload = [t[:MAX_CHARS] for t in texts]
        url = settings.embed_api_base.rstrip("/") + "/embeddings"
        headers = {"Authorization": f"Bearer {settings.embed_api_key}"}
        vectors: list[list[float]] = []
        with httpx.Client(timeout=settings.embed_timeout) as client:
            for i in range(0, len(payload), settings.embed_batch):
                r = client.post(url, headers=headers,
                                json={"model": self.model,
                                      "input": payload[i:i + settings.embed_batch]})
                r.raise_for_status()
                rows = sorted(r.json()["data"], key=lambda d: d["index"])
                vectors += [d["embedding"] for d in rows]
        return _unit(vectors)


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
        vecs = self._model.encode(texts, normalize_embeddings=True, convert_to_numpy=True,
                                  batch_size=64, show_progress_bar=False)
        return vecs.astype(np.float32)


def build_encoder():
    """Return the encoder named by EMBED_BACKEND."""
    if settings.embed_backend.lower() == "api":
        return ApiEncoder()
    return LocalEncoder()
