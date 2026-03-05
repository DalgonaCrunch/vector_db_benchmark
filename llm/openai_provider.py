"""OpenAI chat provider."""
from __future__ import annotations

from typing import Iterator

from llm.base_provider import BaseLLMProvider


class OpenAIProvider(BaseLLMProvider):
    MODELS = ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"]

    def __init__(self, api_key: str, base_url: str | None = None) -> None:
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai 패키지가 필요합니다: uv add openai")
        kwargs: dict = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self._client = OpenAI(**kwargs)

    # ------------------------------------------------------------------

    def generate(self, system, user, model, temperature, max_tokens=2048) -> str:
        resp = self._client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content or ""

    def stream(self, system, user, model, temperature, max_tokens=2048) -> Iterator[str]:
        for chunk in self._client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        ):
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta
