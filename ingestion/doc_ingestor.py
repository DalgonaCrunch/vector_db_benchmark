"""
Per-document ingestion: creates independent Qdrant collection + OpenSearch index per document.

Supports any file format registered in ``ingestion.loaders.FILE_LOADER_REGISTRY``
(currently PDF and DOCX; TXT/HTML/Markdown can be added by registering a loader).

Each call to ``ingest()`` produces:
  - A new Qdrant collection named ``{doc_id}_qdrant``
  - A new OpenSearch index named ``{doc_id}_os``
  - A DocRecord written to the DocRegistry

Usage::

    ingestor = DocIngestor(embedder, registry)
    result = ingestor.ingest(pdf_bytes, "report_2024.pdf")
    print(result.doc_id, result.chunk_count, result.file_type)

    ingestor.delete(result.doc_id)
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from config import load_config
from embedder.upstage_embedder import UpstageEmbedder
from ingestion.chunker import TextChunker
from ingestion.doc_registry import DocRecord, DocRegistry, make_doc_id
from ingestion.loaders import FILE_LOADER_REGISTRY, UnsupportedFileTypeError, get_loader
from ingestion.pdf_loader import RawDocument, _sections_to_raw_doc
from stores.base_store import Document
from stores.opensearch_store import OpenSearchStoreConfig, OpenSearchVectorStore
from stores.qdrant_store import QdrantStoreConfig, QdrantVectorStore


@dataclass
class IngestResult:
    """Result returned by DocIngestor.ingest()."""

    doc_id: str
    source_file: str
    file_type: str
    qdrant_collection: str
    os_index: str
    chunk_count: int
    vector_dim: int
    total_text_length: int
    avg_chunk_length: float

    def print_summary(self) -> None:
        sep = "-" * 60
        print(f"\n{sep}")
        print(f"  INGESTION RESULT")
        print(sep)
        print(f"  source_file  : {self.source_file}")
        print(f"  file_type    : {self.file_type}")
        print(f"  doc_id       : {self.doc_id}")
        print(f"  chunks       : {self.chunk_count}")
        print(f"  vector_dim   : {self.vector_dim}")
        print(f"  total_text   : {self.total_text_length:,} chars")
        print(f"  avg_chunk    : {self.avg_chunk_length:.0f} chars"
              f"  (~{self.avg_chunk_length / 2:.0f} tokens)")
        print(f"  qdrant       : {self.qdrant_collection}")
        print(f"  opensearch   : {self.os_index}")
        print(sep)


class DocIngestor:
    """
    Ingests a single document (PDF, DOCX, …) into its own vector stores.

    Parameters
    ----------
    embedder:
        Configured UpstageEmbedder instance.
    registry:
        Shared DocRegistry for persistence.
    """

    def __init__(self, embedder: UpstageEmbedder, registry: DocRegistry) -> None:
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

    @staticmethod
    def supported_extensions() -> list[str]:
        """Return the list of file extensions this ingestor can handle."""
        return sorted(FILE_LOADER_REGISTRY.keys())

    def ingest(self, file_bytes: bytes, filename: str) -> IngestResult:
        """
        Process a document and store it in independent per-document stores.

        Parameters
        ----------
        file_bytes:
            Raw bytes of the uploaded file.
        filename:
            Original filename (e.g. ``"report_2024.pdf"``).  Used to determine
            the file type and to derive the timestamped doc_id.

        Returns
        -------
        IngestResult
            Statistics and store identifiers for the ingested document.

        Raises
        ------
        UnsupportedFileTypeError
            If the file extension is not registered in FILE_LOADER_REGISTRY.
        ValueError
            If the document contains no extractable text.
        RuntimeError
            If embedding or store insertion fails.
        """
        # Determine file type from extension
        file_type = Path(filename).suffix.lstrip(".").lower()
        loader = get_loader(file_type)   # raises UnsupportedFileTypeError if unknown

        doc_id = make_doc_id(filename)
        qdrant_name = f"{doc_id}_qdrant"
        os_name = f"{doc_id}_os"

        # Load → convert to RawDocument
        print(f"[DocIngestor] Loading '{filename}' (type={file_type}) …")
        loaded_sections = loader.load_bytes(file_bytes, filename)
        raw_doc = _sections_to_raw_doc(loaded_sections, filename, file_type)

        total_text_length = sum(len(s) for s in raw_doc.sections)
        print(
            f"[DocIngestor] Loaded {len(loaded_sections)} sections, "
            f"{total_text_length:,} chars total"
        )

        # Chunk
        chunks = self._chunker.chunk_document(raw_doc)
        if not chunks:
            raise ValueError(f"No chunks produced from '{filename}'.")

        # Enrich metadata: override doc_id with timestamped version, add upload_time
        now_iso = datetime.now().isoformat()
        for chunk in chunks:
            chunk.metadata["doc_id"] = doc_id
            chunk.metadata["source_file"] = filename
            chunk.metadata["upload_time"] = now_iso

        avg_chunk_len = sum(len(c.text) for c in chunks) / len(chunks)

        # Embed
        print(f"[DocIngestor] Embedding {len(chunks)} chunks …")
        texts = [c.text for c in chunks]
        vectors = self._embedder.embed_passages(texts)
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

        # Create Qdrant collection and insert
        qd_store = self._make_qdrant(qdrant_name, vector_dim)
        qd_store.insert(documents)

        # Create OpenSearch index and insert
        os_store = self._make_opensearch(os_name, vector_dim)
        os_store.insert(documents)

        # Persist to registry
        record = DocRecord(
            doc_id=doc_id,
            source_file=filename,
            upload_time=now_iso,
            qdrant_collection=qdrant_name,
            os_index=os_name,
            chunk_count=len(chunks),
            vector_dim=vector_dim,
        )
        self._registry.add(record)

        result = IngestResult(
            doc_id=doc_id,
            source_file=filename,
            file_type=file_type,
            qdrant_collection=qdrant_name,
            os_index=os_name,
            chunk_count=len(chunks),
            vector_dim=vector_dim,
            total_text_length=total_text_length,
            avg_chunk_length=avg_chunk_len,
        )
        result.print_summary()
        return result

    def delete(self, doc_id: str) -> None:
        """
        Delete both stores for a given doc_id and remove from registry.

        Raises
        ------
        ValueError
            If doc_id is not found in the registry.
        """
        record = self._registry.get(doc_id)
        if not record:
            raise ValueError(f"doc_id '{doc_id}' not found in registry.")

        cfg = self._cfg

        try:
            qd_store = QdrantVectorStore(
                QdrantStoreConfig(
                    collection_name=record.qdrant_collection,
                    mode=cfg.qdrant.mode,
                    local_path=cfg.qdrant.local_path,
                    host=cfg.qdrant.host,
                    port=cfg.qdrant.port,
                )
            )
            qd_store.delete()
        except Exception as exc:
            print(f"[DocIngestor] WARNING: Qdrant delete failed for '{doc_id}': {exc}")

        try:
            os_store = OpenSearchVectorStore(
                OpenSearchStoreConfig(
                    host=cfg.opensearch.host,
                    port=cfg.opensearch.port,
                    index_name=record.os_index,
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
            print(f"[DocIngestor] WARNING: OpenSearch delete failed for '{doc_id}': {exc}")

        self._registry.remove(doc_id)
        print(f"[DocIngestor] Deleted doc_id='{doc_id}' from both stores.")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _make_qdrant(self, collection_name: str, vector_dim: int) -> QdrantVectorStore:
        cfg = self._cfg
        store = QdrantVectorStore(
            QdrantStoreConfig(
                collection_name=collection_name,
                mode=cfg.qdrant.mode,
                local_path=cfg.qdrant.local_path,
                host=cfg.qdrant.host,
                port=cfg.qdrant.port,
            )
        )
        store.initialize(vector_dim)
        return store

    def _make_opensearch(self, index_name: str, vector_dim: int) -> OpenSearchVectorStore:
        cfg = self._cfg
        store = OpenSearchVectorStore(
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
        store.initialize(vector_dim)
        return store
