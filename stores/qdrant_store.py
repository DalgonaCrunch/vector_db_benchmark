"""
Qdrant vector store implementation.

Collection design:
- Vector size = embedding dimension (auto-detected)
- Distance metric: Cosine
- Payload fields: id (string), text (string)
- Integer point IDs (0-based index); original string ID stored in payload
- Collection is always recreated fresh in initialize()

Connection modes (QDRANT_MODE env var):
  "memory"  – in-process, no server required (QdrantClient(":memory:"))
  "local"   – persistent on-disk, no server required (QdrantClient(path=…))
  "server"  – remote Qdrant server via host:port
"""
from __future__ import annotations

from dataclasses import dataclass

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, FieldCondition, Filter, MatchValue, PointStruct, VectorParams

from stores.base_store import Document, SearchResult, VectorStore

_VALID_MODES = ("memory", "local", "server")


@dataclass(frozen=True)
class QdrantStoreConfig:
    collection_name: str
    # Connection mode: "memory" | "local" | "server"
    mode: str = "memory"
    # Used when mode == "local"
    local_path: str = "./qdrant_data"
    # Used when mode == "server"
    host: str = "localhost"
    port: int = 6333


class QdrantVectorStore(VectorStore):
    """
    Concrete VectorStore backed by Qdrant (Cosine similarity).

    Score note:
      Qdrant returns cosine similarity directly in [-1, 1].
      For well-formed unit vectors the score typically falls in [0, 1].
    """

    def __init__(self, config: QdrantStoreConfig) -> None:
        if config.mode not in _VALID_MODES:
            raise ValueError(
                f"[Qdrant] Unknown mode '{config.mode}'. "
                f"Valid options: {_VALID_MODES}"
            )
        self._cfg = config
        self._client = self._build_client()
        self._point_id_counter: int = 0  # monotonically increasing; reset on initialize()

    def _build_client(self) -> QdrantClient:
        mode = self._cfg.mode
        if mode == "memory":
            return QdrantClient(":memory:")
        elif mode == "local":
            return QdrantClient(path=self._cfg.local_path)
        else:  # "server"
            return QdrantClient(host=self._cfg.host, port=self._cfg.port)

    # ------------------------------------------------------------------
    # VectorStore interface
    # ------------------------------------------------------------------

    def exists(self) -> bool:
        """Return True if this collection exists on the backend."""
        result = self._client.get_collections()
        return any(c.name == self._cfg.collection_name for c in result.collections)

    def initialize_if_not_exists(self, vector_dim: int) -> None:
        """Create the collection only if it does not already exist."""
        if not self.exists():
            self.initialize(vector_dim)

    def initialize(self, vector_dim: int) -> None:
        """Drop existing collection (if any) and create a fresh one."""
        self._point_id_counter = 0
        self._delete_if_exists()
        self._client.create_collection(
            collection_name=self._cfg.collection_name,
            vectors_config=VectorParams(
                size=vector_dim,
                distance=Distance.COSINE,
            ),
        )
        print(
            f"[Qdrant] Collection '{self._cfg.collection_name}' created "
            f"(dim={vector_dim}, distance=Cosine, mode={self._cfg.mode})"
        )

    # Batch size for upsert: keep each request under ~20 MB for server mode.
    # 4096-dim float32 ≈ 16 KB/vector → 50 vectors ≈ 800 KB per request.
    _UPSERT_BATCH = 50

    def insert(self, documents: list[Document]) -> None:
        """Upsert documents in batches to avoid server-mode request timeouts."""
        if not documents:
            return

        # Sync the ID counter with the existing point count on first use so that
        # appending to an existing collection does not overwrite earlier points.
        if self._point_id_counter == 0:
            try:
                existing = self._client.count(
                    collection_name=self._cfg.collection_name, exact=True
                ).count
                self._point_id_counter = existing
            except Exception:
                pass

        total = 0
        for batch_start in range(0, len(documents), self._UPSERT_BATCH):
            batch = documents[batch_start : batch_start + self._UPSERT_BATCH]
            points = [
                PointStruct(
                    id=self._point_id_counter + i,
                    vector=doc.vector,
                    payload={"id": doc.id, "text": doc.text, **doc.metadata},
                )
                for i, doc in enumerate(batch)
            ]
            self._point_id_counter += len(batch)
            self._client.upsert(
                collection_name=self._cfg.collection_name,
                points=points,
            )
            total += len(batch)

        print(f"[Qdrant] Inserted {total} documents.")

    def search(
        self,
        query_vector: list[float],
        top_k: int,
        filter_doc_id: str | None = None,
    ) -> list[SearchResult]:
        """Run a cosine similarity search and return ranked results.

        Uses query_points() (qdrant-client >= 1.7.4).
        When ``filter_doc_id`` is set, only points whose payload field
        ``doc_id`` matches that value are considered.
        """
        query_filter = (
            Filter(must=[FieldCondition(key="doc_id", match=MatchValue(value=filter_doc_id))])
            if filter_doc_id
            else None
        )
        response = self._client.query_points(
            collection_name=self._cfg.collection_name,
            query=query_vector,
            limit=top_k,
            with_payload=True,
            query_filter=query_filter,
        )
        return [
            SearchResult(
                id=str(hit.payload["id"]),
                text=str(hit.payload["text"]),
                score=float(hit.score),
                rank=i + 1,
                metadata={
                    k: v
                    for k, v in hit.payload.items()
                    if k not in ("id", "text")
                },
            )
            for i, hit in enumerate(response.points)
        ]

    def source_file_exists(self, source_file: str) -> bool:
        """Return True if any point has payload.source_file == source_file."""
        try:
            points, _ = self._client.scroll(
                collection_name=self._cfg.collection_name,
                scroll_filter=Filter(
                    must=[FieldCondition(key="source_file", match=MatchValue(value=source_file))]
                ),
                limit=1,
                with_payload=False,
                with_vectors=False,
            )
            return len(points) > 0
        except Exception:
            return False

    def count(self) -> int:
        result = self._client.count(
            collection_name=self._cfg.collection_name,
            exact=True,
        )
        return result.count

    def list_collections(self) -> list[str]:
        """Return names of all Qdrant collections visible to this client."""
        result = self._client.get_collections()
        return [c.name for c in result.collections]

    def delete(self) -> None:
        self._delete_if_exists()
        print(f"[Qdrant] Collection '{self._cfg.collection_name}' deleted.")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _delete_if_exists(self) -> None:
        try:
            self._client.delete_collection(
                collection_name=self._cfg.collection_name
            )
        except Exception:
            # Collection may not exist yet; ignore
            pass
