"""Abstract base for LLM providers used in the RAG chatbot."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterator


class BaseLLMProvider(ABC):
    """Common interface for OpenAI / Upstage / Claude providers."""

    @abstractmethod
    def generate(
        self,
        system: str,
        user: str,
        model: str,
        temperature: float,
        max_tokens: int = 2048,
    ) -> str:
        """Return full response as a string (non-streaming)."""

    @abstractmethod
    def stream(
        self,
        system: str,
        user: str,
        model: str,
        temperature: float,
        max_tokens: int = 2048,
    ) -> Iterator[str]:
        """Yield response tokens one by one (streaming)."""
