"""
Query runner: embeds each query, searches all stores, and measures latency.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from embedder.upstage_embedder import UpstageEmbedder
from stores.base_store import SearchResult, VectorStore


@dataclass
class QueryResult:
    """Holds the outcome of a single store search for one query."""

    query_text: str
    store_name: str
    run_id: int              # 0-based (0 = warm-up, 1+ = benchmark)
    latency_ms: float
    results: list[SearchResult] = field(default_factory=list)

    @property
    def result_ids(self) -> list[str]:
        return [r.id for r in self.results]

    @property
    def result_scores(self) -> list[float]:
        return [r.score for r in self.results]

    @property
    def mean_score(self) -> float:
        if not self.result_scores:
            return 0.0
        return sum(self.result_scores) / len(self.result_scores)


class QueryRunner:
    """
    Runs a list of query strings against multiple vector stores.

    Usage::

        runner = QueryRunner(
            embedder=embedder,
            stores={"opensearch": os_store, "qdrant": qd_store},
        )
        results = runner.run_queries(queries, top_k=5, run_id=1)
    """

    def __init__(
        self,
        embedder: UpstageEmbedder,
        stores: dict[str, VectorStore],
    ) -> None:
        self._embedder = embedder
        self._stores = stores

    def run_queries(
        self,
        queries: list[str],
        top_k: int,
        run_id: int = 0,
    ) -> list[QueryResult]:
        """
        Embed each query then search every registered store.

        Returns a flat list of QueryResult objects
        (len = len(queries) × len(stores)).
        """
        print(f"\n[QueryRunner] Embedding {len(queries)} queries (run_id={run_id}) …")
        query_vectors = self._embedder.embed_queries(queries)

        all_results: list[QueryResult] = []
        for query_text, query_vec in zip(queries, query_vectors):
            for store_name, store in self._stores.items():
                start = time.perf_counter()
                results = store.search(query_vec, top_k=top_k)
                elapsed_ms = (time.perf_counter() - start) * 1000.0

                all_results.append(
                    QueryResult(
                        query_text=query_text,
                        store_name=store_name,
                        run_id=run_id,
                        latency_ms=elapsed_ms,
                        results=results,
                    )
                )
        return all_results

    def run_warmup(self, queries: list[str], top_k: int) -> list[QueryResult]:
        """
        Execute one warm-up pass (run_id = 0).
        Results are returned but not included in benchmark statistics.
        """
        print("\n[QueryRunner] === Warm-up Run ===")
        return self.run_queries(queries, top_k=top_k, run_id=0)

    def run_repeated(
        self,
        queries: list[str],
        top_k: int,
        num_repeats: int,
    ) -> list[QueryResult]:
        """
        Execute `num_repeats` benchmark passes (run_id = 1 … num_repeats).
        """
        all_results: list[QueryResult] = []
        for run_id in range(1, num_repeats + 1):
            results = self.run_queries(queries, top_k=top_k, run_id=run_id)
            all_results.extend(results)
        return all_results
