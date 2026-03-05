"""
Upstage Embedding API client with batching, retry, and dimension auto-detection.

Upstage provides two asymmetric embedding models:
  - embedding-passage : optimised for indexing documents
  - embedding-query   : optimised for encoding search queries

Both produce 4096-dim vectors and are used together for best retrieval quality.
"""
from __future__ import annotations

import time
from typing import Iterator

import requests


class UpstageEmbedder:
    """
    Wraps the Upstage Embedding REST API.

    Supports:
    - Separate passage (document) and query models
    - Batch embedding with configurable batch size
    - Automatic vector dimension detection from first API response
    - Exponential back-off on rate-limit (HTTP 429) errors
    - Retryable connection errors
    """

    def __init__(
        self,
        api_key: str,
        api_url: str,
        passage_model: str = "embedding-passage",
        query_model: str = "embedding-query",
        batch_size: int = 32,
    ) -> None:
        self._passage_model = passage_model
        self._query_model = query_model
        self._api_url = api_url
        self._batch_size = batch_size
        self._dimension: int | None = None

        self._session = requests.Session()
        self._session.headers.update(
            {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def dimension(self) -> int:
        """Vector dimension detected from the first embed call."""
        if self._dimension is None:
            raise RuntimeError(
                "Vector dimension is not yet known. "
                "Call embed_passages() or embed_queries() at least once."
            )
        return self._dimension

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        """Embed document texts using the passage model (for ingestion)."""
        return self._embed_all(texts, self._passage_model)

    def embed_queries(self, texts: list[str]) -> list[list[float]]:
        """Embed query texts using the query model (for search)."""
        return self._embed_all(texts, self._query_model)

    def embed_single_passage(self, text: str) -> list[float]:
        """Convenience: embed one passage text."""
        return self.embed_passages([text])[0]

    def embed_single_query(self, text: str) -> list[float]:
        """Convenience: embed one query text."""
        return self.embed_queries([text])[0]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _embed_all(self, texts: list[str], model: str) -> list[list[float]]:
        if not texts:
            return []
        all_vectors: list[list[float]] = []
        for batch in self._iter_batches(texts):
            all_vectors.extend(self._embed_batch(batch, model))
        return all_vectors

    def _iter_batches(self, texts: list[str]) -> Iterator[list[str]]:
        for i in range(0, len(texts), self._batch_size):
            yield texts[i : i + self._batch_size]

    def _embed_batch(
        self, texts: list[str], model: str, max_retries: int = 3
    ) -> list[list[float]]:
        """
        Call the Upstage API for one batch.
        Retries on rate-limit (429) and transient connection errors.
        """
        payload = {"model": model, "input": texts}

        for attempt in range(max_retries):
            try:
                response = self._session.post(
                    self._api_url, json=payload, timeout=60
                )
                response.raise_for_status()
                data = response.json()

                # Preserve original order (API may return in any order)
                items = sorted(data["data"], key=lambda x: x["index"])
                vectors: list[list[float]] = [item["embedding"] for item in items]

                if self._dimension is None and vectors:
                    self._dimension = len(vectors[0])

                return vectors

            except requests.exceptions.HTTPError as exc:
                status = exc.response.status_code if exc.response is not None else None
                body = ""
                if exc.response is not None:
                    try:
                        body = exc.response.json().get("error", {}).get("message", "")
                    except Exception:
                        body = exc.response.text[:200]
                if status == 429 and attempt < max_retries - 1:
                    time.sleep(2**attempt)
                    continue
                raise RuntimeError(
                    f"Upstage API error (status={status}, model={model}): {body or exc}"
                ) from exc

            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as exc:
                if attempt < max_retries - 1:
                    time.sleep(1)
                    continue
                raise RuntimeError(
                    f"Upstage API connection error after {max_retries} attempts: {exc}"
                ) from exc

        raise RuntimeError("Embedding failed after maximum retries.")
