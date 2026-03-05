"""
Vector-DB Benchmark – entry point.

Flow:
  1.  Load configuration from .env
  2.  Build Upstage embedder and vector stores
  3.  Detect embedding dimension
  4.  Initialise stores (drop + recreate)
  5.  Load PDFs from PDF_DIR → extract text per page
  6.  Chunk documents (sliding-window, ~650 tokens, 100-token overlap)
  7.  Embed chunks (passage model) and insert into both stores
  8.  Print ingestion summary (count, sample chunks, avg length)
  9.  Warm-up pass (cold-start measurement)
  10. N repeated benchmark passes
  11. Compute and print comparison report
  12. Save benchmark_results.csv
"""
from __future__ import annotations

import sys

from benchmark.comparison import BenchmarkComparison
from config import load_config
from embedder.upstage_embedder import UpstageEmbedder
from ingestion.chunker import TextChunker
from ingestion.ingest_pipeline import IngestPipeline
from ingestion.pdf_loader import PDFLoader
from query.queries import CLI_BENCHMARK_QUERIES
from query.query_runner import QueryRunner
from stores.opensearch_store import OpenSearchStoreConfig, OpenSearchVectorStore
from stores.qdrant_store import QdrantStoreConfig, QdrantVectorStore


def build_stores(cfg) -> tuple[OpenSearchVectorStore, QdrantVectorStore]:
    """Instantiate both vector stores from config (no I/O yet)."""
    os_store = OpenSearchVectorStore(
        OpenSearchStoreConfig(
            host=cfg.opensearch.host,
            port=cfg.opensearch.port,
            index_name=cfg.opensearch.index_name,
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

    qd_store = QdrantVectorStore(
        QdrantStoreConfig(
            collection_name=cfg.qdrant.collection_name,
            mode=cfg.qdrant.mode,
            local_path=cfg.qdrant.local_path,
            host=cfg.qdrant.host,
            port=cfg.qdrant.port,
        )
    )

    return os_store, qd_store


def main() -> int:
    # ------------------------------------------------------------------
    # 1. Configuration
    # ------------------------------------------------------------------
    try:
        cfg = load_config()
    except ValueError as exc:
        print(f"[ERROR] Configuration error: {exc}", file=sys.stderr)
        return 1

    print("=" * 70)
    print("  Vector-DB Benchmark: OpenSearch vs Qdrant")
    print(f"  Passage model  : {cfg.embedder.passage_model}")
    print(f"  Query model    : {cfg.embedder.query_model}")
    print(f"  PDF directory  : {cfg.ingest.pdf_dir}")
    print(f"  Chunk size     : ~{cfg.ingest.chunk_size_tokens} tokens")
    print(f"  Chunk overlap  : ~{cfg.ingest.chunk_overlap_tokens} tokens")
    print(f"  Top-K          : {cfg.benchmark.top_k}")
    print(f"  Repeats        : {cfg.benchmark.num_repeats}")
    print(f"  Qdrant mode    : {cfg.qdrant.mode}")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 2. Embedder
    # ------------------------------------------------------------------
    embedder = UpstageEmbedder(
        api_key=cfg.embedder.api_key,
        api_url=cfg.embedder.api_url,
        passage_model=cfg.embedder.passage_model,
        query_model=cfg.embedder.query_model,
        batch_size=cfg.embedder.batch_size,
    )

    # ------------------------------------------------------------------
    # 3. Vector stores
    # ------------------------------------------------------------------
    os_store, qd_store = build_stores(cfg)

    # ------------------------------------------------------------------
    # 4. Detect embedding dimension then initialise stores
    # ------------------------------------------------------------------
    print("\n[Setup] Detecting embedding dimension …")
    embedder.embed_single_passage("dimension detection probe")
    vector_dim = embedder.dimension
    print(f"[Setup] Vector dimension = {vector_dim}")

    print("\n[Setup] Initialising vector stores …")
    os_store.initialize(vector_dim)
    qd_store.initialize(vector_dim)

    # ------------------------------------------------------------------
    # 5–7. Load PDFs → Chunk → Embed → Insert
    # ------------------------------------------------------------------
    try:
        loader = PDFLoader(cfg.ingest.pdf_dir)
        raw_docs = loader.load_all()
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"[ERROR] PDF loading failed: {exc}", file=sys.stderr)
        return 1

    chunker = TextChunker(
        chunk_size_tokens=cfg.ingest.chunk_size_tokens,
        chunk_overlap_tokens=cfg.ingest.chunk_overlap_tokens,
    )
    chunks = [c for doc in raw_docs for c in chunker.chunk_document(doc)]
    print(
        f"\n[Chunker] {len(raw_docs)} PDF(s) → {len(chunks)} chunks "
        f"(size≈{cfg.ingest.chunk_size_tokens} tok, "
        f"overlap≈{cfg.ingest.chunk_overlap_tokens} tok)"
    )

    stores_map = {"opensearch": os_store, "qdrant": qd_store}
    pipeline = IngestPipeline(embedder)

    try:
        stats = pipeline.run(chunks=chunks, stores=stores_map)
    except ValueError as exc:
        print(f"[ERROR] Ingestion failed: {exc}", file=sys.stderr)
        return 1

    # ------------------------------------------------------------------
    # 8. Ingestion summary + verification
    # ------------------------------------------------------------------
    try:
        stats.verify()
    except RuntimeError as exc:
        print(f"[ERROR] Ingestion verification failed: {exc}", file=sys.stderr)
        return 1

    stats.print_summary()

    # ------------------------------------------------------------------
    # 9. Query runner
    # ------------------------------------------------------------------
    queries = CLI_BENCHMARK_QUERIES
    runner = QueryRunner(embedder=embedder, stores=stores_map)

    # ------------------------------------------------------------------
    # 10. Warm-up (cold-start measurement)
    # ------------------------------------------------------------------
    warmup_results = runner.run_warmup(queries, top_k=cfg.benchmark.top_k)

    # ------------------------------------------------------------------
    # 11. Benchmark runs
    # ------------------------------------------------------------------
    print(f"\n[Benchmark] Starting {cfg.benchmark.num_repeats} benchmark runs …")
    benchmark_results = runner.run_repeated(
        queries, top_k=cfg.benchmark.top_k, num_repeats=cfg.benchmark.num_repeats
    )
    print(
        f"[Benchmark] Completed {cfg.benchmark.num_repeats} runs × "
        f"{len(queries)} queries × {len(stores_map)} stores = "
        f"{len(benchmark_results)} result records."
    )

    # ------------------------------------------------------------------
    # 12. Analysis & report
    # ------------------------------------------------------------------
    comparison = BenchmarkComparison()
    report = comparison.analyze(
        warmup_results=warmup_results,
        benchmark_results=benchmark_results,
        top_k=cfg.benchmark.top_k,
    )
    comparison.print_report(report)

    # ------------------------------------------------------------------
    # 13. CSV export
    # ------------------------------------------------------------------
    comparison.save_csv(
        warmup_results=warmup_results,
        benchmark_results=benchmark_results,
        output_path=cfg.benchmark.output_csv,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
