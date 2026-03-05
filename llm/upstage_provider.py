"""Upstage Solar chat provider (OpenAI-compatible API)."""
from __future__ import annotations

from typing import Iterator

from llm.openai_provider import OpenAIProvider

_UPSTAGE_BASE_URL = "https://api.upstage.ai/v1"


class UpstageProvider(OpenAIProvider):
    """Upstage Solar via OpenAI-compatible endpoint.

    Uses the same UPSTAGE_API_KEY as the embedder.
    Base URL: https://api.upstage.ai/v1
    """

    MODELS = ["solar-pro", "solar-1-mini-chat"]

    def __init__(self, api_key: str) -> None:
        super().__init__(api_key=api_key, base_url=_UPSTAGE_BASE_URL)
