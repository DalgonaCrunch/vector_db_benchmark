"""
Embedding router — multi-model embedding dispatcher.

Supported models
----------------
upstage  → Upstage Embedding REST API (passage / query asymmetric)
openai   → OpenAI Embeddings API (text-embedding-3-small by default)
bge      → BAAI/bge-large-en-v1.5 via sentence-transformers (local GPU/CPU)
e5       → intfloat/e5-large-v2 via sentence-transformers (local GPU/CPU)

All public functions accept an IngestionConfig and return plain
``list[list[float]]`` so callers are completely model-agnostic.

Optional L2 normalisation is applied when config["normalize"] is True.
"""
from __future__ import annotations

import math
import os
from typing import Iterator

from ingestion.ingestion_config import IngestionConfig

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_CHARS_PER_TOKEN = 2.0  # conservative estimate for Korean/English mixed text


def _iter_batches(texts: list[str], batch_size: int) -> Iterator[list[str]]:
    for i in range(0, len(texts), batch_size):
        yield texts[i : i + batch_size]


def _l2_normalize(vectors: list[list[float]]) -> list[list[float]]:
    """L2-normalise each vector in-place and return the list."""
    result: list[list[float]] = []
    for vec in vectors:
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0.0:
            result.append([v / norm for v in vec])
        else:
            result.append(list(vec))
    return result


# ---------------------------------------------------------------------------
# Upstage
# ---------------------------------------------------------------------------


def _upstage_embed(
    texts: list[str],
    batch_size: int,
    mode: str = "passage",   # "passage" | "query"
) -> list[list[float]]:
    """Call the Upstage Embedding REST API."""
    import requests  # already in requirements

    api_key = os.getenv("UPSTAGE_API_KEY", "")
    api_url = os.getenv(
        "UPSTAGE_EMBEDDING_URL",
        "https://api.upstage.ai/v1/solar/embeddings",
    )
    model = (
        "embedding-passage" if mode == "passage" else "embedding-query"
    )

    session = requests.Session()
    session.headers.update(
        {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    )

    all_vectors: list[list[float]] = []
    for batch in _iter_batches(texts, batch_size):
        for attempt in range(3):
            import time
            try:
                resp = session.post(
                    api_url,
                    json={"model": model, "input": batch},
                    timeout=60,
                )
                resp.raise_for_status()
                items = sorted(resp.json()["data"], key=lambda x: x["index"])
                all_vectors.extend(item["embedding"] for item in items)
                break
            except Exception as exc:
                if attempt == 2:
                    raise RuntimeError(
                        f"[Upstage] Embedding failed after 3 attempts: {exc}"
                    ) from exc
                time.sleep(2 ** attempt)
    return all_vectors


# ---------------------------------------------------------------------------
# OpenAI
# ---------------------------------------------------------------------------


def _openai_embed(
    texts: list[str],
    batch_size: int,
    model: str = "text-embedding-3-small",
) -> list[list[float]]:
    """Call the OpenAI Embeddings API."""
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError(
            "openai 패키지가 필요합니다: pip install openai"
        ) from exc

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))
    all_vectors: list[list[float]] = []

    for batch in _iter_batches(texts, batch_size):
        response = client.embeddings.create(model=model, input=batch)
        items = sorted(response.data, key=lambda x: x.index)
        all_vectors.extend(item.embedding for item in items)

    return all_vectors


# ---------------------------------------------------------------------------
# BGE (BAAI/bge-large-en-v1.5)
# ---------------------------------------------------------------------------


def _bge_embed(
    texts: list[str],
    batch_size: int,
    model_name: str = "BAAI/bge-large-en-v1.5",
) -> list[list[float]]:
    """Embed using sentence-transformers BGE model (local)."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "sentence-transformers 패키지가 필요합니다: "
            "pip install sentence-transformers"
        ) from exc

    model = SentenceTransformer(model_name)
    # BGE recommends adding instruction prefix for retrieval
    prefixed = [f"Represent this sentence for searching relevant passages: {t}" for t in texts]
    all_vectors: list[list[float]] = []
    for batch in _iter_batches(prefixed, batch_size):
        vecs = model.encode(batch, normalize_embeddings=False, batch_size=len(batch))
        all_vectors.extend(vecs.tolist())
    return all_vectors


# ---------------------------------------------------------------------------
# E5 (intfloat/e5-large-v2)
# ---------------------------------------------------------------------------


def _e5_embed(
    texts: list[str],
    batch_size: int,
    model_name: str = "intfloat/e5-large-v2",
) -> list[list[float]]:
    """Embed using sentence-transformers E5 model (local)."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "sentence-transformers 패키지가 필요합니다: "
            "pip install sentence-transformers"
        ) from exc

    model = SentenceTransformer(model_name)
    # E5 requires "passage: " prefix for documents, "query: " for queries
    prefixed = [f"passage: {t}" for t in texts]
    all_vectors: list[list[float]] = []
    for batch in _iter_batches(prefixed, batch_size):
        vecs = model.encode(batch, normalize_embeddings=False, batch_size=len(batch))
        all_vectors.extend(vecs.tolist())
    return all_vectors


def _e5_embed_query(
    text: str,
    batch_size: int,
    model_name: str = "intfloat/e5-large-v2",
) -> list[float]:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "sentence-transformers 패키지가 필요합니다: "
            "pip install sentence-transformers"
        ) from exc

    model = SentenceTransformer(model_name)
    vec = model.encode([f"query: {text}"], normalize_embeddings=False, batch_size=1)
    return vec[0].tolist()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def embed_passages(texts: list[str], config: IngestionConfig) -> list[list[float]]:
    """
    Embed a list of passage texts using the model specified in *config*.

    Parameters
    ----------
    texts:
        Non-empty list of document chunk texts.
    config:
        IngestionConfig with ``embedding_model``, ``batch_size``,
        and ``normalize`` fields.

    Returns
    -------
    list[list[float]]
        One vector per input text, in the same order.

    Raises
    ------
    ValueError
        For unknown ``embedding_model`` values.
    ImportError
        When a required third-party package is missing.
    RuntimeError
        When the embedding API call fails.
    """
    if not texts:
        return []

    model = config["embedding_model"]
    batch_size = config["batch_size"]

    if model == "upstage":
        vectors = _upstage_embed(texts, batch_size, mode="passage")
    elif model == "openai":
        vectors = _openai_embed(texts, batch_size)
    elif model == "bge":
        vectors = _bge_embed(texts, batch_size)
    elif model == "e5":
        vectors = _e5_embed(texts, batch_size)
    else:
        raise ValueError(
            f"지원하지 않는 embedding_model: '{model}'. "
            "upstage | openai | bge | e5 중 선택하세요."
        )

    if config.get("normalize", False):
        vectors = _l2_normalize(vectors)

    return vectors


def embed_single_query(text: str, config: IngestionConfig) -> list[float]:
    """
    Embed a single search query using the model specified in *config*.

    For asymmetric models (upstage, e5) the query-specific endpoint/prefix
    is used automatically.

    Parameters
    ----------
    text:
        Query string.
    config:
        IngestionConfig (only ``embedding_model``, ``batch_size``,
        ``normalize`` are used).

    Returns
    -------
    list[float]
        Single query vector.
    """
    model = config["embedding_model"]
    batch_size = config["batch_size"]

    if model == "upstage":
        vectors = _upstage_embed([text], batch_size, mode="query")
        vec = vectors[0]
    elif model == "openai":
        vectors = _openai_embed([text], batch_size)
        vec = vectors[0]
    elif model == "bge":
        # BGE is symmetric — same prefix for queries as passages
        vectors = _bge_embed([text], batch_size)
        vec = vectors[0]
    elif model == "e5":
        vec = _e5_embed_query(text, batch_size)
    else:
        raise ValueError(
            f"지원하지 않는 embedding_model: '{model}'."
        )

    if config.get("normalize", False):
        vec = _l2_normalize([vec])[0]

    return vec


def get_actual_dimension(config: IngestionConfig, sample_text: str = "dimension probe") -> int:
    """
    Embed a single probe text and return the actual vector dimension.

    Use this to verify that *config[\"embedding_dimension\"]* matches
    reality before building a vector store index.
    """
    vec = embed_single_query(sample_text, config)
    return len(vec)
