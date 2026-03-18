"""
IngestionConfig — ingestion pipeline configuration object and validation.

All ingestion-related options (DB targets, embedding model, chunk params,
batch size, duplicate policy, extra metadata) are bundled into one
TypedDict so every component receives the same source of truth.

MODEL_DIMENSIONS maps each supported embedding model to its known vector
dimension.  Validation warns when the user-supplied dimension deviates.

DISTANCE_TO_SPACE_TYPE maps the human-readable metric names that appear
in the UI to the internal string understood by OpenSearch's KNN engine.
"""
from __future__ import annotations

from typing import Any, TypedDict

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_DIMENSIONS: dict[str, int] = {
    "upstage": 4096,
    "openai": 1536,
    "bge": 1024,
    "e5": 1024,
}

DISTANCE_TO_SPACE_TYPE: dict[str, str] = {
    "cosine": "cosinesimil",
    "dot": "innerproduct",
    "euclidean": "l2",
}

DISTANCE_TO_QDRANT_METRIC: dict[str, str] = {
    "cosine": "Cosine",
    "dot": "Dot",
    "euclidean": "Euclid",
}


# ---------------------------------------------------------------------------
# Config TypedDicts
# ---------------------------------------------------------------------------


class IngestionConfig(TypedDict):
    """Full set of options for one ingestion run."""

    dbs: list[str]            # subset of ["qdrant", "opensearch"]
    embedding_model: str      # upstage | openai | bge | e5
    embedding_dimension: int  # expected vector dim; validated against model default
    normalize: bool           # L2-normalise vectors before storing
    chunk_size: int           # approximate tokens per chunk
    chunk_overlap: int        # approximate token overlap between consecutive chunks
    batch_size: int           # embedding API batch size
    duplicate_policy: str     # skip | overwrite
    metadata: dict[str, Any]  # extra key-value pairs merged into each chunk's metadata


class IndexCreateConfig(TypedDict):
    """Options used when creating a new KB index / vector store."""

    index_name: str           # sanitised name (shared for Qdrant + OpenSearch)
    dimension: int            # vector dimension
    distance_metric: str      # cosine | dot | euclidean

    # Qdrant-specific (currently dimension + distance are sufficient)
    qdrant_collection: str    # collection name (defaults to index_name)

    # OpenSearch-specific
    os_index: str             # index name (defaults to index_name)
    os_vector_field: str      # field name for the knn_vector  (default: "vector")
    os_analyzer: str          # text analyzer: standard | keyword
    os_shards: int            # number of primary shards (default 1)
    os_replicas: int          # number of replicas (default 1)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_ingestion_config(config: IngestionConfig) -> list[str]:
    """
    Return a list of human-readable error/warning strings.

    An empty list means the config is valid and ready to use.

    Checks
    ------
    * At least one DB selected
    * chunk_size > chunk_overlap
    * embedding_dimension matches the model's known default
    * batch_size ≥ 1
    """
    errors: list[str] = []

    # ── DB selection ──────────────────────────────────────────────────────
    if not config["dbs"]:
        errors.append("저장할 DB를 최소 하나 선택하세요.")

    # ── Chunk params ──────────────────────────────────────────────────────
    if config["chunk_size"] <= config["chunk_overlap"]:
        errors.append(
            f"chunk_size({config['chunk_size']})는 "
            f"chunk_overlap({config['chunk_overlap']})보다 커야 합니다."
        )

    # ── Batch size ────────────────────────────────────────────────────────
    if config["batch_size"] < 1:
        errors.append("batch_size는 1 이상이어야 합니다.")

    # ── Embedding dimension check ─────────────────────────────────────────
    model = config["embedding_model"]
    expected = MODEL_DIMENSIONS.get(model)
    given = config["embedding_dimension"]
    if expected is not None and given != expected:
        errors.append(
            f"⚠️ '{model}' 모델의 기본 dimension은 {expected}입니다. "
            f"입력값({given})이 다릅니다. 불일치 시 저장 오류가 발생합니다."
        )

    return errors


def default_ingestion_config() -> IngestionConfig:
    """Return a sensible default IngestionConfig."""
    return IngestionConfig(
        dbs=["qdrant", "opensearch"],
        embedding_model="upstage",
        embedding_dimension=MODEL_DIMENSIONS["upstage"],
        normalize=False,
        chunk_size=650,
        chunk_overlap=100,
        batch_size=32,
        duplicate_policy="skip",
        metadata={},
    )
