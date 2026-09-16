"""OpenAI-compatible LLM client.

Defaults to the DHInfra Qwen endpoint. `qwen3.5-397b` is a reasoning model whose thinking
mode is disabled via `chat_template_kwargs.enable_thinking=false` so the answer comes back
in `content` rather than `reasoning`.
"""
from __future__ import annotations

import httpx

from .config import settings


class LLMUnavailable(RuntimeError):
    pass


def available() -> bool:
    """Whether an endpoint and a key are configured."""
    return bool(settings.llm_api_key) and bool(settings.llm_base_url)


def chat(messages: list[dict], *, max_tokens: int = 1200, temperature: float = 0.2) -> str:
    """Return the assistant message content. Raises LLMUnavailable on any failure."""
    if not available():
        raise LLMUnavailable("no LLM endpoint / API key configured")

    body: dict = {
        "model": settings.llm_model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if settings.llm_disable_thinking:
        body["chat_template_kwargs"] = {"enable_thinking": False}

    url = settings.llm_base_url.rstrip("/") + "/chat/completions"
    headers = {"Authorization": f"Bearer {settings.llm_api_key}",
               "Content-Type": "application/json"}
    try:
        response = httpx.post(url, json=body, headers=headers, timeout=settings.llm_timeout)
        response.raise_for_status()
        message = response.json()["choices"][0]["message"]
    except (httpx.HTTPError, KeyError, IndexError, ValueError) as error:
        raise LLMUnavailable(str(error)) from error

    # A reasoning model that was not told to stop thinking puts the text in `reasoning`.
    content = message.get("content") or message.get("reasoning") or ""
    return content.strip()
