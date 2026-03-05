"""Anthropic Claude chat provider."""
from __future__ import annotations

from typing import Iterator

from llm.base_provider import BaseLLMProvider


class ClaudeProvider(BaseLLMProvider):
    MODELS = [
        "claude-opus-4-6",
        "claude-sonnet-4-6",
        "claude-haiku-4-5-20251001",
    ]

    def __init__(self, api_key: str) -> None:
        try:
            from anthropic import Anthropic
        except ImportError:
            raise ImportError("anthropic 패키지가 필요합니다: uv add anthropic")
        self._client = Anthropic(api_key=api_key)

    # ------------------------------------------------------------------

    def generate(self, system, user, model, temperature, max_tokens=2048) -> str:
        resp = self._client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
            temperature=temperature,
        )
        return resp.content[0].text

    def stream(self, system, user, model, temperature, max_tokens=2048) -> Iterator[str]:
        with self._client.messages.stream(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
            temperature=temperature,
        ) as stream_ctx:
            yield from stream_ctx.text_stream
