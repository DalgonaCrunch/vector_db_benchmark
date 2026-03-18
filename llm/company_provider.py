"""Company internal LLM provider (vLLM / OpenAI-compatible endpoint).

Endpoint  : http://182.162.73.81:8000/v1  (override via COMPANY_LLM_URL env var)
Auth      : api_key = "EMPTY"  (no real key required)
Model list: fetched dynamically from GET /v1/models at init time
"""
from __future__ import annotations

import os
from typing import Iterator

from llm.base_provider import BaseLLMProvider

_DEFAULT_BASE_URL = "http://182.162.73.81:8000/v1"
_FALLBACK_MODELS = [
    "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct-AWQ",
    "LGAI-EXAONE/EXAONE-4.0-32B-GPTQ",
    "mistralai/Devstral-Small-2507-unsloth-bnb-4bit",
    "mistralai/Devstral-Small-2-24B-Instruct-2512",
]


def _fetch_company_models(base_url: str) -> list[str]:
    """Query /v1/models and return model ids. Falls back to _FALLBACK_MODELS."""
    try:
        from openai import OpenAI

        client = OpenAI(api_key="EMPTY", base_url=base_url)
        model_list = client.models.list()
        ids = [m.id for m in model_list.data]
        return ids if ids else _FALLBACK_MODELS
    except Exception:
        return _FALLBACK_MODELS


class CompanyProvider(BaseLLMProvider):
    """OpenAI-compatible provider for the company's on-premise vLLM server."""

    # MODELS is populated lazily at first access to avoid blocking import time.
    MODELS: list[str] = _FALLBACK_MODELS

    def __init__(self) -> None:
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai 패키지가 필요합니다: uv add openai")

        self._base_url: str = os.getenv("COMPANY_LLM_URL", _DEFAULT_BASE_URL)
        self._client = OpenAI(api_key="EMPTY", base_url=self._base_url)

        # Refresh available models from server
        fetched = _fetch_company_models(self._base_url)
        CompanyProvider.MODELS = fetched

        # Default model = first model served
        self._default_model: str = fetched[0]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve_model(self, model: str) -> str:
        """Use provided model name; fall back to whatever the server reports."""
        if model and model in CompanyProvider.MODELS:
            return model
        return self._default_model

    # ------------------------------------------------------------------
    # Interface
    # ------------------------------------------------------------------

    def generate(
        self,
        system: str,
        user: str,
        model: str,
        temperature: float,
        max_tokens: int = 2048,
    ) -> str:
        resolved = self._resolve_model(model)
        resp = self._client.chat.completions.create(
            model=resolved,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content or ""

    def stream(
        self,
        system: str,
        user: str,
        model: str,
        temperature: float,
        max_tokens: int = 2048,
    ) -> Iterator[str]:
        resolved = self._resolve_model(model)
        for chunk in self._client.chat.completions.create(
            model=resolved,
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
