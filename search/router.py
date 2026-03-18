"""Retrieval router — db + strategy-based search dispatcher.

DB:
  qdrant       → dense_vector only (KNN)
  opensearch   → dense_vector | keyword | hybrid

Strategies:
  dense_vector → KNN semantic search
  keyword      → BM25 exact match
  hybrid       → KNN + BM25 via Reciprocal Rank Fusion

Optional:
  score_threshold → post-filter by minimum score
  use_rerank      → Upstage solar-reranker re-scores top candidates
"""
from __future__ import annotations

import os
from typing import TypedDict

import requests

from config import load_config
from stores.base_store import SearchResult
from stores.opensearch_store import OpenSearchStoreConfig, OpenSearchVectorStore
from stores.qdrant_store import QdrantStoreConfig, QdrantVectorStore


class SearchConfig(TypedDict):
    db: str               # qdrant | opensearch
    strategy: str         # dense_vector | keyword | hybrid
    top_k: int
    use_rerank: bool
    score_threshold: float


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

_VALID_STRATEGIES: dict[str, list[str]] = {
    "qdrant": ["dense_vector"],
    "opensearch": ["dense_vector", "keyword", "hybrid"],
}


def validate_config(config: SearchConfig) -> SearchConfig:
    """Coerce invalid db+strategy combinations and return corrected config."""
    db = config["db"]
    strategy = config["strategy"]
    allowed = _VALID_STRATEGIES.get(db, ["dense_vector"])
    if strategy not in allowed:
        config = dict(config)  # type: ignore[assignment]
        config["strategy"] = "dense_vector"
    return config  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Store factories
# ---------------------------------------------------------------------------


def _make_qdrant(index_name: str) -> QdrantVectorStore:
    cfg = load_config()
    return QdrantVectorStore(
        QdrantStoreConfig(
            collection_name=index_name,
            mode=cfg.qdrant.mode,
            local_path=cfg.qdrant.local_path,
            host=cfg.qdrant.host,
            port=cfg.qdrant.port,
        )
    )


def _make_opensearch(index_name: str) -> OpenSearchVectorStore:
    cfg = load_config()
    return OpenSearchVectorStore(
        OpenSearchStoreConfig(
            host=cfg.opensearch.host,
            port=cfg.opensearch.port,
            index_name=index_name,
            engine=cfg.opensearch.engine,
            space_type=cfg.opensearch.space_type,
            ef_construction=cfg.opensearch.ef_construction,
            m=cfg.opensearch.m,
            username=cfg.opensearch.username,
            password=cfg.opensearch.password,
            use_ssl=cfg.opensearch.use_ssl,
            verify_certs=cfg.opensearch.verify_certs,
        )
    )


# ---------------------------------------------------------------------------
# Search functions
# ---------------------------------------------------------------------------


def qdrant_search(
    query_vector: list[float],
    index_name: str,
    top_k: int,
) -> list[SearchResult]:
    store = _make_qdrant(index_name)
    if not store.exists():
        return []
    try:
        return store.search(query_vector, top_k=top_k)
    except Exception:
        return []


def opensearch_vector_search(
    query_vector: list[float],
    index_name: str,
    top_k: int,
) -> list[SearchResult]:
    store = _make_opensearch(index_name)
    if not store.exists():
        return []
    try:
        return store.search(query_vector, top_k=top_k)
    except Exception:
        return []


def opensearch_keyword_search(
    query_text: str,
    index_name: str,
    top_k: int,
) -> list[SearchResult]:
    store = _make_opensearch(index_name)
    if not store.exists():
        return []
    try:
        body = {
            "size": top_k,
            "_source": ["id", "text", "metadata"],
            "query": {"match": {"text": {"query": query_text}}},
        }
        response = store._client.search(index=index_name, body=body)
        hits = response["hits"]["hits"]
        max_score = float(response["hits"]["max_score"] or 1.0)
        return [
            SearchResult(
                id=hit["_source"]["id"],
                text=hit["_source"]["text"],
                score=float(hit["_score"]) / max_score,
                rank=i + 1,
                metadata=hit["_source"].get("metadata", {}),
            )
            for i, hit in enumerate(hits)
        ]
    except Exception:
        return []


def opensearch_hybrid_search(
    query_vector: list[float],
    query_text: str,
    index_name: str,
    top_k: int,
    alpha: float = 0.5,
) -> list[SearchResult]:
    """Merge KNN and BM25 results via Reciprocal Rank Fusion (RRF).

    alpha controls KNN weight; (1 - alpha) = BM25 weight.
    """
    store = _make_opensearch(index_name)
    if not store.exists():
        return []
    try:
        knn_results = store.search(query_vector, top_k=top_k)

        body = {
            "size": top_k,
            "_source": ["id", "text", "metadata"],
            "query": {"match": {"text": {"query": query_text}}},
        }
        response = store._client.search(index=index_name, body=body)
        hits = response["hits"]["hits"]
        max_score = float(response["hits"]["max_score"] or 1.0)
        bm25_results = [
            SearchResult(
                id=hit["_source"]["id"],
                text=hit["_source"]["text"],
                score=float(hit["_score"]) / max_score,
                rank=i + 1,
                metadata=hit["_source"].get("metadata", {}),
            )
            for i, hit in enumerate(hits)
        ]

        # Reciprocal Rank Fusion
        k = 60
        scores: dict[str, float] = {}
        docs: dict[str, SearchResult] = {}

        for rank, r in enumerate(knn_results, 1):
            scores[r.id] = scores.get(r.id, 0.0) + alpha / (k + rank)
            docs[r.id] = r

        for rank, r in enumerate(bm25_results, 1):
            scores[r.id] = scores.get(r.id, 0.0) + (1 - alpha) / (k + rank)
            if r.id not in docs:
                docs[r.id] = r

        sorted_ids = sorted(scores, key=lambda x: scores[x], reverse=True)[:top_k]
        return [
            SearchResult(
                id=doc_id,
                text=docs[doc_id].text,
                score=scores[doc_id],
                rank=i + 1,
                metadata=docs[doc_id].metadata,
            )
            for i, doc_id in enumerate(sorted_ids)
        ]
    except Exception:
        return []


def rerank(
    query: str,
    results: list[SearchResult],
    top_k: int,
) -> list[SearchResult]:
    """Re-score results using Upstage solar-reranker API."""
    if not results:
        return results

    api_key = os.getenv("UPSTAGE_API_KEY", "")
    if not api_key:
        return results

    url = "https://api.upstage.ai/v1/rerank"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "solar-reranker",
        "query": query,
        "documents": [r.text for r in results],
        "top_n": min(top_k, len(results)),
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=15)
        response.raise_for_status()
        data = response.json()
        reranked: list[SearchResult] = []
        for item in data.get("results", []):
            idx = item["index"]
            original = results[idx]
            reranked.append(
                SearchResult(
                    id=original.id,
                    text=original.text,
                    score=float(item["relevance_score"]),
                    rank=len(reranked) + 1,
                    metadata=original.metadata,
                )
            )
        return reranked
    except Exception:
        return results


# ---------------------------------------------------------------------------
# Main search entry point
# ---------------------------------------------------------------------------


def search(
    query_text: str,
    query_vector: list[float],
    index_name: str,
    config: SearchConfig,
) -> list[SearchResult]:
    """Dispatch retrieval based on db + strategy config.

    Steps:
      1. Validate and coerce config
      2. Fetch candidates (fetch_k ≥ top_k for rerank headroom)
      3. Apply score_threshold filter
      4. Optionally rerank
      5. Return top_k results
    """
    config = validate_config(config)

    db = config["db"]
    strategy = config["strategy"]
    top_k = config["top_k"]
    use_rerank = config["use_rerank"]
    score_threshold = config["score_threshold"]

    fetch_k = min(top_k * 3, 20) if use_rerank else top_k

    # ── Dispatch ──────────────────────────────────────────────────────────
    if db == "qdrant":
        results = qdrant_search(query_vector, index_name, fetch_k)
    else:  # opensearch
        if strategy == "dense_vector":
            results = opensearch_vector_search(query_vector, index_name, fetch_k)
        elif strategy == "keyword":
            results = opensearch_keyword_search(query_text, index_name, fetch_k)
        else:  # hybrid
            results = opensearch_hybrid_search(
                query_vector, query_text, index_name, fetch_k
            )

    # ── Score threshold filter ────────────────────────────────────────────
    if score_threshold > 0.0:
        results = [r for r in results if r.score >= score_threshold]

    # ── Rerank ───────────────────────────────────────────────────────────
    if use_rerank and results:
        results = rerank(query_text, results, top_k)
    elif len(results) > top_k:
        results = results[:top_k]

    return results
