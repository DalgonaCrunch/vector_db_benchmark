"""
IngestionConfig — full ingestion pipeline configuration.

Extended fields (v2):
  loader_strategy     — extraction strategy per file type
  splitter_strategy   — text splitting strategy
  sentence_count      — for sentence splitter
  semantic_threshold  — for semantic splitter
  prepend_section_title — prefix each chunk with its section label
  min_chunk_length    — discard chunks shorter than this (chars)
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

# Default loader strategy per extension
DEFAULT_LOADER_STRATEGY: dict[str, str] = {
    "pdf": "page",
    "docx": "heading",
    "txt": "fulltext",
    "md": "heading",
    "html": "tag",
    "htm": "tag",
    "csv": "row_batch",
}

# Loader strategies available per file type
LOADER_STRATEGIES: dict[str, list[str]] = {
    "pdf": ["page", "pdfplumber", "pymupdf", "fulltext", "ocr"],
    "docx": ["heading", "paragraph", "fulltext"],
    "txt": ["fulltext", "paragraph"],
    "md": ["heading", "fulltext"],
    "html": ["tag", "fulltext"],
    "htm": ["tag", "fulltext"],
    "csv": ["row_batch"],
}

SPLITTER_STRATEGIES: list[str] = [
    "sliding_window",
    "recursive",
    "sentence",
    "semantic",
]


# ---------------------------------------------------------------------------
# Config TypedDicts
# ---------------------------------------------------------------------------


class IngestionConfig(TypedDict):
    """Full set of options for one ingestion run."""

    # DB targets
    dbs: list[str]               # subset of ["qdrant", "opensearch"]

    # Embedding
    embedding_model: str         # upstage | openai | bge | e5
    embedding_dimension: int
    normalize: bool

    # Loader
    loader_strategy: str         # per-file-type strategy key

    # Splitter
    splitter_strategy: str       # sliding_window | recursive | sentence | semantic
    chunk_size: int              # tokens (sliding_window / recursive)
    chunk_overlap: int           # tokens (sliding_window / recursive)
    sentence_count: int          # sentences per chunk (sentence strategy)
    sentence_overlap: int        # overlap in sentences (sentence strategy)
    semantic_threshold: float    # cosine-drop threshold (semantic strategy)

    # Post-processing
    prepend_section_title: bool  # prefix chunk text with section label
    min_chunk_length: int        # discard chunks shorter than this (chars)

    # Runtime
    batch_size: int
    duplicate_policy: str        # skip | overwrite
    metadata: dict[str, Any]


class IndexCreateConfig(TypedDict):
    index_name: str
    dimension: int
    distance_metric: str
    qdrant_collection: str
    os_index: str
    os_vector_field: str
    os_analyzer: str
    os_shards: int
    os_replicas: int


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_ingestion_config(config: IngestionConfig) -> list[str]:
    errors: list[str] = []

    if not config["dbs"]:
        errors.append("저장할 DB를 최소 하나 선택하세요.")

    if config["splitter_strategy"] in ("sliding_window", "recursive"):
        if config["chunk_size"] <= config["chunk_overlap"]:
            errors.append(
                f"chunk_size({config['chunk_size']})는 "
                f"chunk_overlap({config['chunk_overlap']})보다 커야 합니다."
            )

    if config["batch_size"] < 1:
        errors.append("batch_size는 1 이상이어야 합니다.")

    model = config["embedding_model"]
    expected = MODEL_DIMENSIONS.get(model)
    given = config["embedding_dimension"]
    if expected is not None and given != expected:
        errors.append(
            f"⚠️ '{model}' 모델의 기본 dimension은 {expected}입니다. "
            f"입력값({given})이 다릅니다."
        )

    return errors


def default_ingestion_config() -> IngestionConfig:
    return IngestionConfig(
        dbs=["qdrant", "opensearch"],
        embedding_model="upstage",
        embedding_dimension=MODEL_DIMENSIONS["upstage"],
        normalize=False,
        loader_strategy="page",
        splitter_strategy="sliding_window",
        chunk_size=650,
        chunk_overlap=100,
        sentence_count=5,
        sentence_overlap=1,
        semantic_threshold=0.5,
        prepend_section_title=False,
        min_chunk_length=50,
        batch_size=32,
        duplicate_policy="skip",
        metadata={},
    )
