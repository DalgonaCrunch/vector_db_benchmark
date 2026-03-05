"""
Index-based ingestion: adds documents into a shared KB Index.

Unlike the old per-document architecture (one collection per file), here:
  • A KB Index (index_name) maps to exactly ONE Qdrant collection and ONE
    OpenSearch index.
  • Multiple documents can be appended into the same index over time.
  • Existing data is never deleted; new chunks are always appended.

Usage::

    registry = IndexRegistry()
    registry.create("my_kb")

    ingestor = IndexIngestor(embedder, registry)
    result = ingestor.ingest(pdf_bytes, "report.pdf", "my_kb")
    print(result.chunk_count)

    ingestor.delete_index("my_kb")   # removes both stores + registry entry
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from config import load_config
from embedder.upstage_embedder import UpstageEmbedder
from ingestion.chunker import TextChunker
from ingestion.doc_registry import make_doc_id
from ingestion.index_registry import DocEntry, IndexRegistry
from ingestion.loaders import get_loader
from ingestion.pdf_loader import _sections_to_raw_doc
from stores.base_store import Document
from stores.opensearch_store import OpenSearchStoreConfig, OpenSearchVectorStore
from stores.qdrant_store import QdrantStoreConfig, QdrantVectorStore


@dataclass
class IngestResult:
    """Statistics returned after ingesting a document into a KB index."""

    index_name: str
    doc_id: str
    source_file: str
    file_type: str
    chunk_count: int
    vector_dim: int
    total_text_length: int
    avg_chunk_length: float

    def print_summary(self) -> None:
        sep = "-" * 60
        print(f"\n{sep}")
        print(f"  INGEST → index: '{self.index_name}'")
        print(sep)
        print(f"  source_file  : {self.source_file}")
        print(f"  file_type    : {self.file_type}")
        print(f"  doc_id       : {self.doc_id}")
        print(f"  chunks       : {self.chunk_count}")
        print(f"  vector_dim   : {self.vector_dim}")
        print(f"  total_text   : {self.total_text_length:,} chars")
        print(f"  avg_chunk    : {self.avg_chunk_length:.0f} chars")
        print(sep)


class IndexIngestor:
    """
    Appends documents into an existing KB index.

    The KB index's Qdrant collection and OpenSearch index are created on the
    first ``ingest()`` call (lazy initialisation), then reused for subsequent
    documents.

    Parameters
    ----------
    embedder:
        Configured UpstageEmbedder.
    registry:
        Shared IndexRegistry instance.
    """

    def __init__(self, embedder: UpstageEmbedder, registry: IndexRegistry) -> None:
        self._embedder = embedder
        self._registry = registry
        self._cfg = load_config()
        self._chunker = TextChunker(
            chunk_size_tokens=self._cfg.ingest.chunk_size_tokens,
            chunk_overlap_tokens=self._cfg.ingest.chunk_overlap_tokens,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ingest(
        self, file_bytes: bytes, filename: str, index_name: str
    ) -> IngestResult:
        """
        Load, chunk, embed and append a document to *index_name*.

        If *index_name* does not yet exist in the registry it is created
        automatically (useful for CLI / scripted use).

        The underlying Qdrant collection and OpenSearch index are created on
        the first call; subsequent calls append without touching existing data.

        Parameters
        ----------
        file_bytes:
            Raw bytes of the uploaded file.
        filename:
            Original filename, used to determine the file type and doc_id.
        index_name:
            Name of the target KB index (must be sanitised before calling).

        Raises
        ------
        UnsupportedFileTypeError
            If the file extension is not registered.
        ValueError
            If the document has no extractable text.
        """
        # Auto-create index if not present
        if not self._registry.get(index_name):
            self._registry.create(index_name)

        file_type = Path(filename).suffix.lstrip(".").lower()
        loader = get_loader(file_type)
        doc_id = make_doc_id(filename)

        # Load → raw_doc
        print(f"[IndexIngestor] Loading '{filename}' → index '{index_name}' …")
        loaded_sections = loader.load_bytes(file_bytes, filename)
        raw_doc = _sections_to_raw_doc(loaded_sections, filename, file_type)

        total_text_length = sum(len(s) for s in raw_doc.sections)
        print(
            f"[IndexIngestor] {len(loaded_sections)} sections, "
            f"{total_text_length:,} chars"
        )

        # Chunk
        chunks = self._chunker.chunk_document(raw_doc)
        if not chunks:
            raise ValueError(f"No chunks produced from '{filename}'.")

        # Enrich metadata
        now_iso = datetime.now().isoformat()
        for chunk in chunks:
            chunk.metadata["index_name"] = index_name
            chunk.metadata["doc_id"] = doc_id
            chunk.metadata["source_file"] = filename
            chunk.metadata["upload_time"] = now_iso

        avg_chunk_len = sum(len(c.text) for c in chunks) / len(chunks)

        # Embed
        print(f"[IndexIngestor] Embedding {len(chunks)} chunks …")
        vectors = self._embedder.embed_passages([c.text for c in chunks])
        vector_dim = self._embedder.dimension

        documents = [
            Document(
                id=chunk.chunk_id,
                text=chunk.text,
                vector=vec,
                metadata=chunk.metadata,
            )
            for chunk, vec in zip(chunks, vectors)
        ]

        # Get or create stores (lazy init on first doc, append on subsequent)
        qd_store, os_store = self._get_or_create_stores(index_name, vector_dim)
        qd_store.insert(documents)
        os_store.insert(documents)

        # Update registry
        doc_entry = DocEntry(
            doc_id=doc_id,
            source_file=filename,
            file_type=file_type,
            upload_time=now_iso,
            chunk_count=len(chunks),
        )
        self._registry.add_document(index_name, doc_entry, vector_dim)

        result = IngestResult(
            index_name=index_name,
            doc_id=doc_id,
            source_file=filename,
            file_type=file_type,
            chunk_count=len(chunks),
            vector_dim=vector_dim,
            total_text_length=total_text_length,
            avg_chunk_length=avg_chunk_len,
        )
        result.print_summary()
        return result

    def is_duplicate(self, filename: str, index_name: str) -> bool:
        """Return True if *filename* is already ingested in *index_name*.

        Checks Qdrant first (local/fast), falls back to OpenSearch.
        Returns False if neither store exists yet.
        """
        cfg = self._cfg
        try:
            qd_store = QdrantVectorStore(
                QdrantStoreConfig(
                    collection_name=index_name,
                    mode=cfg.qdrant.mode,
                    local_path=cfg.qdrant.local_path,
                    host=cfg.qdrant.host,
                    port=cfg.qdrant.port,
                )
            )
            if qd_store.exists():
                return qd_store.source_file_exists(filename)
        except Exception:
            pass

        try:
            os_store = OpenSearchVectorStore(
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
            if os_store.exists():
                return os_store.source_file_exists(filename)
        except Exception:
            pass

        return False

    def delete_index(self, index_name: str) -> None:
        """
        Delete the Qdrant collection, OpenSearch index, and registry entry.

        Parameters
        ----------
        index_name:
            Must be a registered index name.

        Raises
        ------
        ValueError
            If *index_name* is not found in the registry.
        """
        kb = self._registry.get(index_name)
        if not kb:
            raise ValueError(f"Index '{index_name}' not found.")

        cfg = self._cfg

        try:
            qd_store = QdrantVectorStore(
                QdrantStoreConfig(
                    collection_name=kb.qdrant_collection,
                    mode=cfg.qdrant.mode,
                    local_path=cfg.qdrant.local_path,
                    host=cfg.qdrant.host,
                    port=cfg.qdrant.port,
                )
            )
            qd_store.delete()
        except Exception as exc:
            print(
                f"[IndexIngestor] WARNING: Qdrant delete failed "
                f"for '{index_name}': {exc}"
            )

        try:
            os_store = OpenSearchVectorStore(
                OpenSearchStoreConfig(
                    host=cfg.opensearch.host,
                    port=cfg.opensearch.port,
                    index_name=kb.os_index,
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
            os_store.delete()
        except Exception as exc:
            print(
                f"[IndexIngestor] WARNING: OpenSearch delete failed "
                f"for '{index_name}': {exc}"
            )

        self._registry.delete(index_name)
        print(f"[IndexIngestor] Deleted index '{index_name}'.")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_or_create_stores(
        self, index_name: str, vector_dim: int
    ) -> tuple[QdrantVectorStore, OpenSearchVectorStore]:
        """
        Return store objects for *index_name*, initialising them if they
        do not yet exist on the backend.  Existing data is never deleted.
        """
        cfg = self._cfg

        qd_store = QdrantVectorStore(
            QdrantStoreConfig(
                collection_name=index_name,
                mode=cfg.qdrant.mode,
                local_path=cfg.qdrant.local_path,
                host=cfg.qdrant.host,
                port=cfg.qdrant.port,
            )
        )
        qd_store.initialize_if_not_exists(vector_dim)

        os_store = OpenSearchVectorStore(
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
        os_store.initialize_if_not_exists(vector_dim)

        return qd_store, os_store
