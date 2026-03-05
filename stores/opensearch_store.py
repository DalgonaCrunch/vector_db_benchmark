"""
OpenSearch vector store implementation.

Index design:
- Dense vector field mapped as knn_vector (HNSW)
- Configurable engine (nmslib / faiss / lucene) and space type
- Cosine similarity by default (cosinesimil)
- Index is always recreated fresh in initialize()
"""
from __future__ import annotations

from dataclasses import dataclass

from opensearchpy import NotFoundError, OpenSearch, RequestError
from opensearchpy.helpers import bulk as opensearch_bulk

from stores.base_store import Document, SearchResult, VectorStore


@dataclass(frozen=True)
class OpenSearchStoreConfig:
    host: str
    port: int
    index_name: str
    engine: str          # nmslib | faiss | lucene
    space_type: str      # cosinesimil | l2 | innerproduct
    ef_construction: int
    m: int
    username: str | None = None
    password: str | None = None
    use_ssl: bool = False
    verify_certs: bool = False


class OpenSearchVectorStore(VectorStore):
    """
    Concrete VectorStore backed by OpenSearch KNN (HNSW).

    Score note:
      With nmslib + cosinesimil, OpenSearch returns a score in [0, 1]
      where 1.0 = identical vectors.  The exact formula is:
        score = (1 + cosine_sim) / 2
    """

    def __init__(self, config: OpenSearchStoreConfig) -> None:
        self._cfg = config
        self._client: OpenSearch = self._build_client()

    # ------------------------------------------------------------------
    # VectorStore interface
    # ------------------------------------------------------------------

    def exists(self) -> bool:
        """Return True if this index exists on the backend."""
        return bool(self._client.indices.exists(index=self._cfg.index_name))

    def initialize_if_not_exists(self, vector_dim: int) -> None:
        """Create the index only if it does not already exist."""
        if not self.exists():
            self.initialize(vector_dim)

    def initialize(self, vector_dim: int) -> None:
        """Drop existing index (if any) and create a fresh KNN index."""
        self._delete_if_exists()
        self._create_index(vector_dim)
        print(
            f"[OpenSearch] Index '{self._cfg.index_name}' created "
            f"(dim={vector_dim}, engine={self._cfg.engine}, "
            f"space={self._cfg.space_type})"
        )

    def insert(self, documents: list[Document]) -> None:
        """Bulk-insert documents and force a refresh for immediate searchability."""
        if not documents:
            return

        actions = [
            {
                "_index": self._cfg.index_name,
                "_id": doc.id,
                "_source": {
                    "id": doc.id,
                    "text": doc.text,
                    "vector": doc.vector,
                    "metadata": doc.metadata,
                },
            }
            for doc in documents
        ]

        success_count, errors = opensearch_bulk(self._client, actions, raise_on_error=False)
        if errors:
            raise RuntimeError(
                f"[OpenSearch] Bulk insert had {len(errors)} error(s): {errors[:3]}"
            )

        # Refresh so documents are immediately searchable
        self._client.indices.refresh(index=self._cfg.index_name)
        print(f"[OpenSearch] Inserted {success_count} documents.")

    def search(
        self,
        query_vector: list[float],
        top_k: int,
        filter_doc_id: str | None = None,
    ) -> list[SearchResult]:
        """Run a KNN search and return ranked SearchResult objects.

        When ``filter_doc_id`` is set, only chunks whose
        ``metadata.doc_id`` matches that value are returned.
        The filter is applied as a lucene pre-filter inside the knn query,
        which is the most efficient approach for the lucene KNN engine.
        """
        knn_clause: dict = {"vector": query_vector, "k": top_k}
        if filter_doc_id:
            knn_clause["filter"] = {"term": {"metadata.doc_id": filter_doc_id}}

        query_body = {
            "size": top_k,
            "_source": ["id", "text", "metadata"],
            "query": {"knn": {"vector": knn_clause}},
        }
        response = self._client.search(
            index=self._cfg.index_name, body=query_body
        )
        hits = response["hits"]["hits"]
        return [
            SearchResult(
                id=hit["_source"]["id"],
                text=hit["_source"]["text"],
                score=float(hit["_score"]),
                rank=i + 1,
                metadata=hit["_source"].get("metadata", {}),
            )
            for i, hit in enumerate(hits)
        ]

    def source_file_exists(self, source_file: str) -> bool:
        """Return True if any document has metadata.source_file == source_file."""
        try:
            resp = self._client.search(
                index=self._cfg.index_name,
                body={
                    "size": 1,
                    "_source": False,
                    "query": {"term": {"metadata.source_file": source_file}},
                },
            )
            return resp["hits"]["total"]["value"] > 0
        except NotFoundError:
            return False
        except Exception:
            return False

    def count(self) -> int:
        response = self._client.count(index=self._cfg.index_name)
        return int(response["count"])

    def list_indices(self, pattern: str = "*") -> list[str]:
        """Return names of all OpenSearch indices matching ``pattern``."""
        try:
            response = self._client.indices.get(index=pattern)
            return sorted(response.keys())
        except NotFoundError:
            return []

    def delete(self) -> None:
        self._delete_if_exists()
        print(f"[OpenSearch] Index '{self._cfg.index_name}' deleted.")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_client(self) -> OpenSearch:
        http_auth = (
            (self._cfg.username, self._cfg.password)
            if self._cfg.username and self._cfg.password
            else None
        )
        return OpenSearch(
            hosts=[{"host": self._cfg.host, "port": self._cfg.port}],
            http_auth=http_auth,
            use_ssl=self._cfg.use_ssl,
            verify_certs=self._cfg.verify_certs,
            ssl_show_warn=False,
        )

    def _delete_if_exists(self) -> None:
        try:
            self._client.indices.delete(index=self._cfg.index_name)
        except NotFoundError:
            pass

    def _create_index(self, vector_dim: int) -> None:
        mapping = {
            "settings": {
                "index.knn": True,
            },
            "mappings": {
                "properties": {
                    "id": {"type": "keyword"},
                    "text": {"type": "text"},
                    "vector": {
                        "type": "knn_vector",
                        "dimension": vector_dim,
                        "method": {
                            "name": "hnsw",
                            "space_type": self._cfg.space_type,
                            "engine": self._cfg.engine,
                            "parameters": {
                                "ef_construction": self._cfg.ef_construction,
                                "m": self._cfg.m,
                            },
                        },
                    },
                    "metadata": {
                        "type": "object",
                        "properties": {
                            "index_name":   {"type": "keyword"},
                            "doc_id":       {"type": "keyword"},
                            "source_file":  {"type": "keyword"},
                            "file_type":    {"type": "keyword"},
                            "section":      {"type": "keyword"},
                            "chunk_index":  {"type": "integer"},
                            "upload_time":  {"type": "keyword"},
                            "filepath":     {"type": "keyword"},
                        },
                    },
                }
            },
        }
        try:
            self._client.indices.create(
                index=self._cfg.index_name, body=mapping
            )
        except RequestError as exc:
            raise RuntimeError(
                f"[OpenSearch] Failed to create index '{self._cfg.index_name}': {exc}"
            ) from exc
