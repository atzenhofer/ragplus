"""Runtime configuration, sourced from environment (.env is loaded if present)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CACHE_DIR = ROOT / ".cache"


def _bool(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    llm_base_url: str = os.environ.get("LLM_BASE_URL", "https://api.dhinfra.uni-graz.at/v1")
    llm_model: str = os.environ.get("LLM_MODEL", "qwen3.5-397b")
    llm_api_key: str = (os.environ.get("DHINFRA_API_KEY") or os.environ.get("LLM_API_KEY") or "")
    llm_disable_thinking: bool = _bool("LLM_DISABLE_THINKING", True)
    llm_timeout: float = float(os.environ.get("LLM_TIMEOUT", "180"))

    embed_backend: str = os.environ.get("EMBED_BACKEND", "api")  # api | local
    embed_model: str = os.environ.get("EMBED_MODEL", "qwen3-embedding-8b")
    embed_device: str = os.environ.get("EMBED_DEVICE", "auto")  # auto | cpu
    embed_api_base: str = os.environ.get("EMBED_API_BASE") or os.environ.get(
        "LLM_BASE_URL", "https://api.dhinfra.uni-graz.at/v1")
    embed_api_key: str = (os.environ.get("EMBED_API_KEY") or os.environ.get("DHINFRA_API_KEY")
                          or os.environ.get("LLM_API_KEY") or "")
    embed_batch: int = int(os.environ.get("EMBED_BATCH", "128"))
    embed_timeout: float = float(os.environ.get("EMBED_TIMEOUT", "180"))

    corpus_path: Path = Path(os.environ.get("CORPUS_PATH", str(DATA_DIR / "corpus.jsonl")))


settings = Settings()
