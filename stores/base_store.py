"""
Abstract base class for all vector store backends.
Every concrete store (OpenSearch, Qdrant, …) must inherit VectorStore.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class Document:
    """A document chunk with its text content and pre-computed embedding."""

    id: str
    text: str
    vector: list[float]
    metadata: dict = field(default_factory=dict)


@dataclass
class SearchResult:
    """A single ranked result returned by a vector search."""

    id: str
    text: str
    score: float
    rank: int  # 1-based
    metadata: dict = field(default_factory=dict)


class VectorStore(ABC):
    """
    Minimal contract that every vector store backend must satisfy.

    Lifecycle:
      1. initialize(vector_dim)  → create (or re-create) the store
      2. insert(documents)       → add documents
      3. search(query_vector)    → retrieve top-k
      4. count()                 → verify stored count
      5. delete()                → tear down the store
    """

    @abstractmethod
    def initialize(self, vector_dim: int) -> None:
        """Create (or recreate) the store for the given embedding dimension."""
        ...

    @abstractmethod
    def insert(self, documents: list[Document]) -> None:
        """Bulk-insert documents into the store."""
        ...

    @abstractmethod
    def search(
        self,
        query_vector: list[float],
        top_k: int,
        filter_doc_id: str | None = None,
    ) -> list[SearchResult]:
        """Return the top_k most similar documents to query_vector.

        Parameters
        ----------
        filter_doc_id:
            When set, restrict results to chunks whose ``metadata.doc_id``
            matches this value exactly.  ``None`` means no filter (default).
        """
        ...

    @abstractmethod
    def count(self) -> int:
        """Return the total number of stored documents."""
        ...

    @abstractmethod
    def delete(self) -> None:
        """Permanently remove the index / collection."""
        ...
