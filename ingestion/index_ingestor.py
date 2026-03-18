"""
Index-based ingestion: adds documents into a shared KB Index.

Unlike the old per-document architecture (one collection per file), here:
  • A KB Index (index_name) maps to exactly ONE Qdrant collection and ONE
    OpenSearch index.
  • Multiple documents can be appended into the same index over time.
  • Existing data is never deleted; new chunks are always appended.

Extended API (ingest_with_config):
  • DB targets are controlled by IngestionConfig["dbs"]
  • Embedding model is dispatched via embedder.embedding_router
  • Chunk size / overlap are overridden per-run
  • duplicate_policy controls skip vs overwrite behaviour
  • Extra metadata is merged into every chunk's metadata dict

Usage::

    registry = IndexRegistry()
    registry.create("my_kb")

    ingestor = IndexIngestor(embedder, registry)

    # Legacy: always stores in both Qdrant + OpenSearch, uses UpstageEmbedder
    result = ingestor.ingest(pdf_bytes, "report.pdf", "my_kb")

    # Extended: full config control
    from ingestion.ingestion_config import IngestionConfig
    cfg = IngestionConfig(
        dbs=["opensearch"],
        embedding_model="openai",
        embedding_dimension=1536,
        normalize=True,
        chunk_size=512,
        chunk_overlap=64,
        batch_size=16,
        duplicate_policy="overwrite",
        metadata={"project": "demo"},
    )
    result = ingestor.ingest_with_config(pdf_bytes, "report.pdf", "my_kb", cfg)

    ingestor.delete_index("my_kb")   # removes both stores + registry entry
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from config import load_config
from embedder.upstage_embedder import UpstageEmbedder
from ingestion.chunker import Chunk, TextChunker, get_chunker
from ingestion.strategy_info import DEFAULT_LOADER_STRATEGY
from ingestion.doc_registry import make_doc_id
from ingestion.index_registry import DocEntry, IndexRegistry
from ingestion.ingestion_config import (
    DISTANCE_TO_SPACE_TYPE,
    IngestionConfig,
)
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
    stored_in: list[str]         # which DBs were written to
    embedding_model: str         # which model was used
    elapsed_s: float             # wall-clock seconds for the full run

    def print_summary(self) -> None:
        sep = "-" * 60
        print(f"\n{sep}")
        print(f"  INGEST → index: '{self.index_name}'")
        print(sep)
        print(f"  source_file    : {self.source_file}")
        print(f"  file_type      : {self.file_type}")
        print(f"  doc_id         : {self.doc_id}")
        print(f"  chunks         : {self.chunk_count}")
        print(f"  vector_dim     : {self.vector_dim}")
        print(f"  total_text     : {self.total_text_length:,} chars")
        print(f"  avg_chunk      : {self.avg_chunk_length:.0f} chars")
        print(f"  embedding      : {self.embedding_model}")
        print(f"  stored_in      : {', '.join(self.stored_in)}")
        print(f"  elapsed        : {self.elapsed_s:.2f}s")
        print(sep)


class IndexIngestor:
    """
    Appends documents into an existing KB index.

    The KB index's Qdrant collection and OpenSearch index are created on the
    first ``ingest()`` or ``ingest_with_config()`` call (lazy initialisation),
    then reused for subsequent documents.

    Parameters
    ----------
    embedder:
        Configured UpstageEmbedder (used by the legacy ``ingest()`` method).
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
    # Public API — legacy (Upstage only, both DBs always)
    # ------------------------------------------------------------------

    def ingest(
        self, file_bytes: bytes, filename: str, index_name: str
    ) -> IngestResult:
        """
        Legacy ingestion: Upstage embedding, both Qdrant + OpenSearch.

        Delegates to ``ingest_with_config`` with a default upstage config.
        """
        from ingestion.ingestion_config import default_ingestion_config
        config = default_ingestion_config()
        return self.ingest_with_config(file_bytes, filename, index_name, config)

    # ------------------------------------------------------------------
    # Public API — extended (full config control)
    # ------------------------------------------------------------------

    def ingest_with_config(
        self,
        file_bytes: bytes,
        filename: str,
        index_name: str,
        config: IngestionConfig,
    ) -> IngestResult:
        """
        Load, chunk, embed, and append a document to *index_name*.

        Parameters
        ----------
        file_bytes:
            Raw bytes of the uploaded file.
        filename:
            Original filename, used to determine the file type and doc_id.
        index_name:
            Name of the target KB index (must be sanitised before calling).
            Auto-created if not present in the registry.
        config:
            Full IngestionConfig controlling DB targets, embedding model,
            chunk params, batch size, duplicate policy, and extra metadata.

        Returns
        -------
        IngestResult

        Raises
        ------
        UnsupportedFileTypeError
            If the file extension is not registered.
        ValueError
            If the document has no extractable text.
        RuntimeError
            If embedding or store insertion fails.
        """
        t_start = time.perf_counter()

        # Auto-create index if not present
        if not self._registry.get(index_name):
            self._registry.create(index_name)

        file_type = Path(filename).suffix.lstrip(".").lower()
        # Per-file-type strategy map takes priority over the single loader_strategy field
        loader_strategy_map: dict = config.get("metadata", {}).get("_loader_strategy_map", {})
        loader_strategy = (
            loader_strategy_map.get(file_type)
            or config.get("loader_strategy")
            or DEFAULT_LOADER_STRATEGY.get(file_type)
        )
        print(f"[IndexIngestor] loader_strategy='{loader_strategy}' for '{file_type}'")
        loader = get_loader(file_type, strategy=loader_strategy)
        doc_id = make_doc_id(filename)

        # ── Duplicate check / overwrite ───────────────────────────────
        if config["duplicate_policy"] == "skip":
            if self._is_duplicate_in_any_db(filename, index_name, config["dbs"]):
                raise ValueError(
                    f"'{filename}' 은(는) '{index_name}'에 이미 존재합니다. "
                    "(duplicate_policy=skip)"
                )

        # ── Load → RawDocument ─────────────────────────────────────────
        print(f"[IndexIngestor] Loading '{filename}' → index '{index_name}' …")
        loaded_sections = loader.load_bytes(file_bytes, filename)
        raw_doc = _sections_to_raw_doc(loaded_sections, filename, file_type)

        total_text_length = sum(len(s) for s in raw_doc.sections)
        print(
            f"[IndexIngestor] {len(loaded_sections)} sections, "
            f"{total_text_length:,} chars"
        )

        # ── Chunk ─────────────────────────────────────────────────────
        splitter_strategy = config.get("splitter_strategy", "sliding_window")

        # semantic chunker needs an embed_fn
        _embed_fn = None
        if splitter_strategy == "semantic":
            def _embed_fn(texts: list[str]) -> list[list[float]]:
                return self._embed_texts(texts, config)

        chunker = get_chunker(
            strategy=splitter_strategy,
            chunk_size_tokens=config.get("chunk_size", 650),
            chunk_overlap_tokens=config.get("chunk_overlap", 100),
            sentence_count=config.get("sentence_count", 5),
            sentence_overlap=config.get("sentence_overlap", 1),
            semantic_threshold=config.get("semantic_threshold", 0.5),
            embed_fn=_embed_fn,
        )
        chunks = chunker.chunk_document(raw_doc)
        if not chunks:
            raise ValueError(f"No chunks produced from '{filename}'.")

        # ── Enrich metadata ───────────────────────────────────────────
        now_iso = datetime.now().isoformat()
        extra_meta: dict[str, Any] = config.get("metadata") or {}
        for chunk in chunks:
            chunk.metadata.update(
                {
                    "index_name": index_name,
                    "doc_id": doc_id,
                    "source_file": filename,
                    "upload_time": now_iso,
                    "embedding_model": config["embedding_model"],
                    **extra_meta,
                }
            )

        # ── Post-processing ───────────────────────────────────────────
        # Prepend section title
        if config.get("prepend_section_title", False):
            for chunk in chunks:
                section_label = chunk.metadata.get("section", "")
                if section_label and section_label not in ("?", "fulltext"):
                    chunk.text = f"[{section_label}] {chunk.text}"

        # Filter short chunks
        min_len = config.get("min_chunk_length", 0)
        if min_len > 0:
            chunks = [c for c in chunks if len(c.text) >= min_len]

        if not chunks:
            raise ValueError(
                f"Post-processing 후 청크가 없습니다. "
                f"min_chunk_length={min_len}를 낮추거나 전략을 변경하세요."
            )

        avg_chunk_len = sum(len(c.text) for c in chunks) / len(chunks)

        # ── Embed ─────────────────────────────────────────────────────
        print(
            f"[IndexIngestor] Embedding {len(chunks)} chunks "
            f"(model={config['embedding_model']}) …"
        )
        vectors = self._embed(chunks, config)
        vector_dim = len(vectors[0]) if vectors else config["embedding_dimension"]

        # Validate actual dimension matches config
        if vectors and vector_dim != config["embedding_dimension"]:
            print(
                f"[IndexIngestor] WARNING: actual dim={vector_dim} "
                f"≠ config dim={config['embedding_dimension']}"
            )

        # ── Build Document objects ─────────────────────────────────────
        documents = [
            Document(
                id=chunk.chunk_id,
                text=chunk.text,
                vector=vec,
                metadata=chunk.metadata,
            )
            for chunk, vec in zip(chunks, vectors)
        ]

        # ── Store in selected DBs ──────────────────────────────────────
        stored_in: list[str] = []

        if "qdrant" in config["dbs"]:
            self._upsert_qdrant(
                index_name,
                vector_dim,
                documents,
                overwrite=(config["duplicate_policy"] == "overwrite"),
            )
            stored_in.append("qdrant")

        if "opensearch" in config["dbs"]:
            self._index_opensearch(
                index_name,
                vector_dim,
                documents,
                overwrite=(config["duplicate_policy"] == "overwrite"),
            )
            stored_in.append("opensearch")

        # ── Update registry ────────────────────────────────────────────
        doc_entry = DocEntry(
            doc_id=doc_id,
            source_file=filename,
            file_type=file_type,
            upload_time=now_iso,
            chunk_count=len(chunks),
        )
        self._registry.add_document(index_name, doc_entry, vector_dim)

        elapsed = time.perf_counter() - t_start

        result = IngestResult(
            index_name=index_name,
            doc_id=doc_id,
            source_file=filename,
            file_type=file_type,
            chunk_count=len(chunks),
            vector_dim=vector_dim,
            total_text_length=total_text_length,
            avg_chunk_length=avg_chunk_len,
            stored_in=stored_in,
            embedding_model=config["embedding_model"],
            elapsed_s=elapsed,
        )
        result.print_summary()
        return result

    # ------------------------------------------------------------------
    # Duplicate detection
    # ------------------------------------------------------------------

    def is_duplicate(self, filename: str, index_name: str) -> bool:
        """Return True if *filename* is already ingested in *index_name*."""
        return self._is_duplicate_in_any_db(
            filename, index_name, ["qdrant", "opensearch"]
        )

    def _is_duplicate_in_any_db(
        self, filename: str, index_name: str, dbs: list[str]
    ) -> bool:
        """Check only the DBs listed in *dbs*."""
        cfg = self._cfg

        if "qdrant" in dbs:
            try:
                store = QdrantVectorStore(
                    QdrantStoreConfig(
                        collection_name=index_name,
                        mode=cfg.qdrant.mode,
                        local_path=cfg.qdrant.local_path,
                        host=cfg.qdrant.host,
                        port=cfg.qdrant.port,
                    )
                )
                if store.exists():
                    return store.source_file_exists(filename)
            except Exception:
                pass

        if "opensearch" in dbs:
            try:
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
                if store.exists():
                    return store.source_file_exists(filename)
            except Exception:
                pass

        return False

    # ------------------------------------------------------------------
    # Delete
    # ------------------------------------------------------------------

    def delete_index(self, index_name: str) -> None:
        """
        Delete the Qdrant collection, OpenSearch index, and registry entry.

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
    # Private — embedding
    # ------------------------------------------------------------------

    def _embed(
        self, chunks: list[Chunk], config: IngestionConfig
    ) -> list[list[float]]:
        """Dispatch to the correct embedding backend."""
        texts = [c.text for c in chunks]
        model = config["embedding_model"]

        if model == "upstage":
            # Use the pre-built UpstageEmbedder (connection pooled)
            self._embedder._batch_size = config["batch_size"]
            vectors = self._embedder.embed_passages(texts)
        else:
            from embedder.embedding_router import embed_passages
            vectors = embed_passages(texts, config)

        return vectors

    def _embed_texts(self, chunks_or_texts, config: IngestionConfig) -> list[list[float]]:
        """Alias that accepts either chunks or plain text strings."""
        if chunks_or_texts and isinstance(chunks_or_texts[0], str):
            texts = chunks_or_texts
        else:
            texts = [c.text for c in chunks_or_texts]
        model = config["embedding_model"]
        if model == "upstage":
            self._embedder._batch_size = config["batch_size"]
            return self._embedder.embed_passages(texts)
        from embedder.embedding_router import embed_passages
        return embed_passages(texts, config)

    # ------------------------------------------------------------------
    # Private — store helpers
    # ------------------------------------------------------------------

    def _upsert_qdrant(
        self,
        index_name: str,
        vector_dim: int,
        documents: list[Document],
        overwrite: bool = False,
    ) -> None:
        """Create-or-append into a Qdrant collection."""
        cfg = self._cfg
        store = QdrantVectorStore(
            QdrantStoreConfig(
                collection_name=index_name,
                mode=cfg.qdrant.mode,
                local_path=cfg.qdrant.local_path,
                host=cfg.qdrant.host,
                port=cfg.qdrant.port,
            )
        )
        if overwrite and store.exists():
            store.delete()
            store.initialize(vector_dim)
        else:
            store.initialize_if_not_exists(vector_dim)
        store.insert(documents)
        print(f"[IndexIngestor] Qdrant '{index_name}': {store.count()} vectors total.")

    def _index_opensearch(
        self,
        index_name: str,
        vector_dim: int,
        documents: list[Document],
        overwrite: bool = False,
    ) -> None:
        """Create-or-append into an OpenSearch index."""
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
        if overwrite and store.exists():
            store.delete()
            store.initialize(vector_dim)
        else:
            store.initialize_if_not_exists(vector_dim)
        store.insert(documents)
        print(
            f"[IndexIngestor] OpenSearch '{index_name}': {store.count()} vectors total."
        )

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
