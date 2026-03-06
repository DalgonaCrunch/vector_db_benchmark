"""Google Gemini chat provider (google-genai SDK)."""
from __future__ import annotations

from typing import Iterator

from llm.base_provider import BaseLLMProvider


class GeminiProvider(BaseLLMProvider):
    MODELS = [
        "gemini-2.5-flash-lite",
        "gemini-2.5-flash",
        "gemini-3.1-flash-lite-preview",
    ]

    def __init__(self, api_key: str) -> None:
        try:
            from google import genai
            from google.genai import types as genai_types
        except ImportError:
            raise ImportError(
                "google-genai 패키지가 필요합니다: uv add google-genai"
            )
        self._client = genai.Client(api_key=api_key)
        self._types = genai_types

    def _config(self, system: str, temperature: float, max_tokens: int):
        return self._types.GenerateContentConfig(
            system_instruction=system,
            temperature=temperature,
            max_output_tokens=max_tokens,
        )

    def generate(self, system, user, model, temperature, max_tokens=2048) -> str:
        resp = self._client.models.generate_content(
            model=model,
            contents=user,
            config=self._config(system, temperature, max_tokens),
        )
        return resp.text

    def stream(self, system, user, model, temperature, max_tokens=2048) -> Iterator[str]:
        for chunk in self._client.models.generate_content_stream(
            model=model,
            contents=user,
            config=self._config(system, temperature, max_tokens),
        ):
            if chunk.text:
                yield chunk.text
