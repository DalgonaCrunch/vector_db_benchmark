"""
Central configuration module.
All settings are loaded from environment variables via .env file.
No global variables; use load_config() to get a Config instance.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


def _require_env(key: str) -> str:
    """Return env var value or raise if missing/empty."""
    value = os.getenv(key)
    if not value:
        raise ValueError(
            f"Required environment variable '{key}' is not set. "
            f"Check your .env file."
        )
    return value


def _optional_env(key: str, default: str) -> str:
    """Return env var value or a default."""
    return os.getenv(key, default)


@dataclass(frozen=True)
class EmbedderConfig:
    api_key: str
    api_url: str
    # Upstage uses separate models for passages (docs) and queries
    passage_model: str   # used during ingestion
    query_model: str     # used during search
    batch_size: int


@dataclass(frozen=True)
class OpenSearchConfig:
    host: str
    port: int
    index_name: str
    engine: str
    space_type: str
    ef_construction: int
    m: int
    username: str | None
    password: str | None
    use_ssl: bool
    verify_certs: bool


@dataclass(frozen=True)
class QdrantConfig:
    host: str
    port: int
    collection_name: str
    # "memory"  → in-process, no server required (default)
    # "local"   → persistent on-disk, no server required
    # "server"  → remote Qdrant server (host:port)
    mode: str
    local_path: str  # used only when mode == "local"


@dataclass(frozen=True)
class IngestConfig:
    pdf_dir: str
    chunk_size_tokens: int   # target chunk size in approximate tokens
    chunk_overlap_tokens: int  # overlap between consecutive chunks


@dataclass(frozen=True)
class BenchmarkConfig:
    top_k: int
    num_repeats: int
    output_csv: str


@dataclass(frozen=True)
class Config:
    embedder: EmbedderConfig
    opensearch: OpenSearchConfig
    qdrant: QdrantConfig
    ingest: IngestConfig
    benchmark: BenchmarkConfig


def load_config() -> Config:
    """Load and validate all configuration from environment."""
    load_dotenv()

    return Config(
        embedder=EmbedderConfig(
            api_key=_require_env("UPSTAGE_API_KEY"),
            api_url=_optional_env(
                "UPSTAGE_API_URL",
                "https://api.upstage.ai/v1/embeddings",
            ),
            passage_model=_optional_env("UPSTAGE_PASSAGE_MODEL", "embedding-passage"),
            query_model=_optional_env("UPSTAGE_QUERY_MODEL", "embedding-query"),
            batch_size=int(_optional_env("EMBEDDING_BATCH_SIZE", "32")),
        ),
        opensearch=OpenSearchConfig(
            host=_optional_env("OPENSEARCH_HOST", "localhost"),
            port=int(_optional_env("OPENSEARCH_PORT", "9200")),
            index_name=_optional_env("OPENSEARCH_INDEX", "benchmark_index"),
            engine=_optional_env("OPENSEARCH_KNN_ENGINE", "nmslib"),
            space_type=_optional_env("OPENSEARCH_SPACE_TYPE", "cosinesimil"),
            ef_construction=int(_optional_env("OPENSEARCH_EF_CONSTRUCTION", "128")),
            m=int(_optional_env("OPENSEARCH_M", "16")),
            username=os.getenv("OPENSEARCH_USERNAME") or None,
            password=os.getenv("OPENSEARCH_PASSWORD") or None,
            use_ssl=_optional_env("OPENSEARCH_USE_SSL", "false").lower() == "true",
            verify_certs=_optional_env("OPENSEARCH_VERIFY_CERTS", "false").lower() == "true",
        ),
        qdrant=QdrantConfig(
            host=_optional_env("QDRANT_HOST", "localhost"),
            port=int(_optional_env("QDRANT_PORT", "6333")),
            collection_name=_optional_env("QDRANT_COLLECTION", "benchmark_collection"),
            mode=_optional_env("QDRANT_MODE", "memory"),
            local_path=_optional_env("QDRANT_LOCAL_PATH", "./qdrant_data"),
        ),
        ingest=IngestConfig(
            pdf_dir=_optional_env("PDF_DIR", "./data/pdfs"),
            chunk_size_tokens=int(_optional_env("CHUNK_SIZE_TOKENS", "650")),
            chunk_overlap_tokens=int(_optional_env("CHUNK_OVERLAP_TOKENS", "100")),
        ),
        benchmark=BenchmarkConfig(
            top_k=int(_optional_env("TOP_K", "5")),
            num_repeats=int(_optional_env("NUM_REPEATS", "20")),
            output_csv=_optional_env("OUTPUT_CSV", "benchmark_results.csv"),
        ),
    )
