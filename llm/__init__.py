"""LLM provider factory for the RAG chatbot.

Usage::

    from llm import get_provider, PROVIDER_MODELS

    provider = get_provider("Upstage")          # reads UPSTAGE_API_KEY from env
    answer = provider.generate(system, user, model="solar-pro", temperature=0)

    # or streaming (for st.write_stream)
    for token in provider.stream(...):
        print(token, end="", flush=True)
"""
from __future__ import annotations

import os

from dotenv import load_dotenv

from llm.base_provider import BaseLLMProvider
from llm.claude_provider import ClaudeProvider
from llm.openai_provider import OpenAIProvider
from llm.upstage_provider import UpstageProvider

load_dotenv()

# Ordered: Upstage first (UPSTAGE_API_KEY already configured for embeddings)
PROVIDER_MODELS: dict[str, list[str]] = {
    "Upstage": UpstageProvider.MODELS,
    "OpenAI": OpenAIProvider.MODELS,
    "Claude": ClaudeProvider.MODELS,
}

_PROVIDER_ENV: dict[str, str] = {
    "Upstage": "UPSTAGE_API_KEY",
    "OpenAI": "OPENAI_API_KEY",
    "Claude": "ANTHROPIC_API_KEY",
}


def get_provider(name: str) -> BaseLLMProvider:
    """Instantiate and return a provider, reading API key from env.

    Raises
    ------
    ValueError
        If the required API key env var is not set.
    ImportError
        If the underlying SDK package is not installed.
    """
    env_key = _PROVIDER_ENV.get(name)
    if env_key is None:
        raise ValueError(f"Unknown provider '{name}'. Available: {list(PROVIDER_MODELS)}")

    api_key = os.getenv(env_key, "")
    if not api_key:
        raise ValueError(
            f"API 키가 설정되지 않았습니다. "
            f"`.env` 파일에 `{env_key}=<your-key>`를 추가하세요."
        )

    if name == "Upstage":
        return UpstageProvider(api_key)
    if name == "OpenAI":
        return OpenAIProvider(api_key)
    if name == "Claude":
        return ClaudeProvider(api_key)
    raise ValueError(f"Unknown provider '{name}'")


def available_providers() -> list[str]:
    """Return providers whose API key env var is currently set."""
    return [name for name, env in _PROVIDER_ENV.items() if os.getenv(env)]
