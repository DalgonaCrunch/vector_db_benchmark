"""
PDF ingestion pipeline: Chunk → Embed → Store.

Responsibilities:
  1. Receive pre-split Chunk objects (from PDFLoader + TextChunker)
  2. Embed chunk texts via UpstageEmbedder (passage model)
  3. Bulk-insert Document objects into every registered VectorStore
  4. Return IngestionStats with counts, sample chunks, and averages

Usage::

    from ingestion.pdf_loader import PDFLoader
    from ingestion.chunker import TextChunker
    from ingestion.ingest_pipeline import IngestPipeline

    raw_docs = PDFLoader(pdf_dir).load_all()
    chunker  = TextChunker(chunk_size_tokens=650, chunk_overlap_tokens=100)
    chunks   = [c for doc in raw_docs for c in chunker.chunk_document(doc)]

    pipeline = IngestPipeline(embedder)
    stats    = pipeline.run(chunks=chunks, stores={"opensearch": os_store, "qdrant": qd_store})
    stats.verify()
    stats.print_summary()
"""
from __future__ import annotations

from dataclasses import dataclass, field

from embedder.upstage_embedder import UpstageEmbedder
from ingestion.chunker import Chunk
from stores.base_store import Document, VectorStore


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class IngestionStats:
    """Statistics produced after a completed ingestion run."""

    total_chunks: int
    vector_dim: int
    avg_chunk_length: float         # chars
    store_counts: dict[str, int]
    sample_chunks: list[Chunk] = field(default_factory=list)  # up to 3

    def verify(self) -> None:
        """Raise RuntimeError if any store count ≠ total_chunks."""
        for name, cnt in self.store_counts.items():
            if cnt != self.total_chunks:
                raise RuntimeError(
                    f"Count mismatch in '{name}': "
                    f"expected {self.total_chunks}, got {cnt}"
                )

    def print_summary(self) -> None:
        """Print a human-readable ingestion summary to stdout."""
        sep = "=" * 70
        print(f"\n{sep}")
        print("  INGESTION SUMMARY")
        print(sep)
        print(f"  Total chunks ingested : {self.total_chunks}")
        print(f"  Vector dimension      : {self.vector_dim}")
        print(f"  Avg chunk length      : {self.avg_chunk_length:.0f} chars"
              f"  (~{self.avg_chunk_length / 2:.0f} tokens)")
        for name, cnt in self.store_counts.items():
            status = "OK" if cnt == self.total_chunks else "MISMATCH"
            print(f"  {name:<22}: {cnt} vectors  [{status}]")

        if self.sample_chunks:
            print(f"\n  --- Sample Chunks (first {len(self.sample_chunks)}) ---")
            for chunk in self.sample_chunks:
                print(f"\n  chunk_id   : {chunk.chunk_id}")
                print(f"  doc_id     : {chunk.metadata.get('doc_id', '?')}")
                print(f"  section    : {chunk.metadata.get('section', '?')}")
                print(f"  length     : {len(chunk.text)} chars")
                preview = chunk.text[:140].replace("\n", " ")
                print(f"  preview    : {preview}…")
        print(sep)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class IngestPipeline:
    """
    Orchestrates chunk embedding and multi-store insertion.

    Parameters
    ----------
    embedder:
        Configured UpstageEmbedder instance.
    """

    def __init__(self, embedder: UpstageEmbedder) -> None:
        self._embedder = embedder

    def run(
        self,
        chunks: list[Chunk],
        stores: dict[str, VectorStore],
    ) -> IngestionStats:
        """
        Embed ``chunks`` with the passage model and insert into all stores.

        Parameters
        ----------
        chunks:
            Pre-split document chunks (from TextChunker.chunk_document).
            Must be non-empty.
        stores:
            Mapping of store_name → VectorStore.
            Each store must already be initialised (initialize() called).

        Returns
        -------
        IngestionStats
            Call .verify() to assert all stores have the expected count,
            and .print_summary() for a human-readable report.

        Raises
        ------
        ValueError
            If ``chunks`` is empty.
        """
        if not chunks:
            raise ValueError(
                "No chunks to ingest. "
                "Check that PDF files exist in the configured PDF_DIR and are readable."
            )

        # ---- Embed -------------------------------------------------------
        print(f"\n[Ingest] Embedding {len(chunks)} chunks (passage model) …")
        texts = [c.text for c in chunks]
        vectors = self._embedder.embed_passages(texts)
        vector_dim = self._embedder.dimension
        print(f"[Ingest] Embedding complete. Dimension = {vector_dim}")

        # ---- Build Document objects ---------------------------------------
        doc_objects: list[Document] = [
            Document(
                id=chunk.chunk_id,
                text=chunk.text,
                vector=vec,
                metadata=chunk.metadata,
            )
            for chunk, vec in zip(chunks, vectors)
        ]

        # ---- Insert into each store --------------------------------------
        store_counts: dict[str, int] = {}
        for name, store in stores.items():
            print(f"[Ingest] Inserting into '{name}' …")
            store.insert(doc_objects)
            cnt = store.count()
            store_counts[name] = cnt
            print(f"[Ingest] '{name}' now holds {cnt} vectors.")

        # ---- Statistics --------------------------------------------------
        avg_len = sum(len(c.text) for c in chunks) / len(chunks)
        return IngestionStats(
            total_chunks=len(chunks),
            vector_dim=vector_dim,
            avg_chunk_length=avg_len,
            store_counts=store_counts,
            sample_chunks=chunks[:3],
        )
