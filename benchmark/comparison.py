"""
Benchmark comparison module.

Computes per-query metrics between two vector stores:
  - Jaccard similarity of top-K result sets
  - Average rank-position difference for overlapping documents
  - Mean score difference
  - Latency statistics (mean, std, diff) across N runs
  - Cold-start vs warm-search latency comparison

Produces:
  - Console report
  - benchmark_results.csv
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import groupby
from pathlib import Path

import numpy as np
import pandas as pd

from query.query_runner import QueryResult


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class QueryMetrics:
    """Aggregated metrics for one query across all benchmark runs."""

    query_text: str

    # Result-set overlap (first run used for display)
    first_run_os_ids: list[str]
    first_run_qdrant_ids: list[str]
    first_run_os_latency_ms: float
    first_run_qdrant_latency_ms: float

    # Overlap metrics (mean across runs)
    mean_jaccard: float
    std_jaccard: float

    # Rank-difference metrics (mean across runs)
    mean_avg_rank_diff: float
    std_avg_rank_diff: float

    # Score metrics (mean across runs)
    mean_os_score: float
    mean_qdrant_score: float
    mean_score_diff: float

    # Latency metrics
    os_latency_mean: float
    os_latency_std: float
    qdrant_latency_mean: float
    qdrant_latency_std: float
    latency_diff_mean: float  # os - qdrant; positive = OS is slower


@dataclass
class ColdWarmMetrics:
    """Cold-start vs warm-search latency comparison."""

    store_name: str
    cold_latency_ms: float       # run_id = 1 (first benchmark run)
    warm_latency_mean_ms: float  # run_id = 2+ mean
    warm_latency_std_ms: float


@dataclass
class ComparisonReport:
    """Full benchmark report."""

    store_names: list[str]
    query_metrics: list[QueryMetrics]
    cold_warm_metrics: list[ColdWarmMetrics]
    num_runs: int
    top_k: int


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------


def _jaccard(ids_a: list[str], ids_b: list[str]) -> float:
    set_a = set(ids_a)
    set_b = set(ids_b)
    union = set_a | set_b
    if not union:
        return 0.0
    return len(set_a & set_b) / len(union)


def _avg_rank_diff(ids_a: list[str], ids_b: list[str]) -> float | None:
    """
    Mean absolute rank difference for documents that appear in both lists.
    Returns None if no overlap.
    """
    overlap = set(ids_a) & set(ids_b)
    if not overlap:
        return None
    diffs = [abs(ids_a.index(d) - ids_b.index(d)) for d in overlap]
    return sum(diffs) / len(diffs)


# ---------------------------------------------------------------------------
# BenchmarkComparison
# ---------------------------------------------------------------------------


class BenchmarkComparison:
    """
    Analyses a collection of QueryResult objects produced by QueryRunner.

    Expected store names: "opensearch" and "qdrant".
    """

    STORE_OS = "opensearch"
    STORE_QD = "qdrant"

    def analyze(
        self,
        warmup_results: list[QueryResult],
        benchmark_results: list[QueryResult],
        top_k: int,
    ) -> ComparisonReport:
        """
        Parameters
        ----------
        warmup_results:
            Results from the warm-up pass (run_id = 0).  Used for
            cold-start latency reference.
        benchmark_results:
            Results from run_id = 1 … N.
        """
        num_runs = max((r.run_id for r in benchmark_results), default=0)

        # Group benchmark results by (query_text, store_name)
        by_query_store: dict[tuple[str, str], list[QueryResult]] = {}
        for qr in benchmark_results:
            key = (qr.query_text, qr.store_name)
            by_query_store.setdefault(key, []).append(qr)

        # Unique query texts (preserve order)
        seen: set[str] = set()
        query_texts: list[str] = []
        for qr in benchmark_results:
            if qr.query_text not in seen:
                seen.add(qr.query_text)
                query_texts.append(qr.query_text)

        store_names = sorted(
            {qr.store_name for qr in benchmark_results}
        )

        query_metrics: list[QueryMetrics] = []
        for q in query_texts:
            os_runs = by_query_store.get((q, self.STORE_OS), [])
            qd_runs = by_query_store.get((q, self.STORE_QD), [])

            if not os_runs or not qd_runs:
                continue  # Skip if one store is missing

            # Per-run metrics
            jaccards: list[float] = []
            rank_diffs: list[float] = []
            for os_r, qd_r in zip(os_runs, qd_runs):
                j = _jaccard(os_r.result_ids, qd_r.result_ids)
                jaccards.append(j)
                rd = _avg_rank_diff(os_r.result_ids, qd_r.result_ids)
                if rd is not None:
                    rank_diffs.append(rd)

            os_latencies = [r.latency_ms for r in os_runs]
            qd_latencies = [r.latency_ms for r in qd_runs]
            os_scores = [r.mean_score for r in os_runs]
            qd_scores = [r.mean_score for r in qd_runs]

            # First run for display
            first_os = os_runs[0]
            first_qd = qd_runs[0]

            qm = QueryMetrics(
                query_text=q,
                first_run_os_ids=first_os.result_ids,
                first_run_qdrant_ids=first_qd.result_ids,
                first_run_os_latency_ms=first_os.latency_ms,
                first_run_qdrant_latency_ms=first_qd.latency_ms,
                mean_jaccard=float(np.mean(jaccards)),
                std_jaccard=float(np.std(jaccards)),
                mean_avg_rank_diff=(
                    float(np.mean(rank_diffs)) if rank_diffs else 0.0
                ),
                std_avg_rank_diff=(
                    float(np.std(rank_diffs)) if rank_diffs else 0.0
                ),
                mean_os_score=float(np.mean(os_scores)),
                mean_qdrant_score=float(np.mean(qd_scores)),
                mean_score_diff=float(
                    np.mean([abs(o - q) for o, q in zip(os_scores, qd_scores)])
                ),
                os_latency_mean=float(np.mean(os_latencies)),
                os_latency_std=float(np.std(os_latencies)),
                qdrant_latency_mean=float(np.mean(qd_latencies)),
                qdrant_latency_std=float(np.std(qd_latencies)),
                latency_diff_mean=float(
                    np.mean([o - q for o, q in zip(os_latencies, qd_latencies)])
                ),
            )
            query_metrics.append(qm)

        # Cold-start vs warm-search
        cold_warm: list[ColdWarmMetrics] = []
        for store in store_names:
            warmup_store = [r for r in warmup_results if r.store_name == store]
            bench_store = [r for r in benchmark_results if r.store_name == store]
            if not warmup_store or not bench_store:
                continue

            cold_lat = float(np.mean([r.latency_ms for r in warmup_store]))
            warm_lats = [r.latency_ms for r in bench_store if r.run_id >= 2]
            if warm_lats:
                warm_mean = float(np.mean(warm_lats))
                warm_std = float(np.std(warm_lats))
            else:
                warm_mean = cold_lat
                warm_std = 0.0

            cold_warm.append(
                ColdWarmMetrics(
                    store_name=store,
                    cold_latency_ms=cold_lat,
                    warm_latency_mean_ms=warm_mean,
                    warm_latency_std_ms=warm_std,
                )
            )

        return ComparisonReport(
            store_names=store_names,
            query_metrics=query_metrics,
            cold_warm_metrics=cold_warm,
            num_runs=num_runs,
            top_k=top_k,
        )

    # ------------------------------------------------------------------
    # Console output
    # ------------------------------------------------------------------

    def print_report(self, report: ComparisonReport) -> None:
        sep = "=" * 70

        print(f"\n{sep}")
        print("  VECTOR DB BENCHMARK REPORT")
        print(f"  Stores: {', '.join(report.store_names)}")
        print(f"  Top-K: {report.top_k}   Runs: {report.num_runs}")
        print(sep)

        for qm in report.query_metrics:
            print(f'\n=== Query: "{qm.query_text[:60]}" ===')

            # OpenSearch
            latency_flag = (
                "(slower)" if qm.first_run_os_latency_ms > qm.first_run_qdrant_latency_ms
                else ""
            )
            print(f"\nOpenSearch {latency_flag}")
            print(f"  latency : {qm.first_run_os_latency_ms:.1f} ms")
            print(f"  results : {qm.first_run_os_ids}")

            # Qdrant
            latency_flag = (
                "(slower)" if qm.first_run_qdrant_latency_ms > qm.first_run_os_latency_ms
                else ""
            )
            print(f"\nQdrant {latency_flag}")
            print(f"  latency : {qm.first_run_qdrant_latency_ms:.1f} ms")
            print(f"  results : {qm.first_run_qdrant_ids}")

            diff_sign = "+" if qm.latency_diff_mean >= 0 else ""
            print(f"\nMetrics (over {report.num_runs} runs):")
            print(f"  Jaccard overlap       : {qm.mean_jaccard:.3f}  ± {qm.std_jaccard:.3f}")
            print(f"  Avg rank diff         : {qm.mean_avg_rank_diff:.2f} ± {qm.std_avg_rank_diff:.2f}")
            print(f"  OpenSearch score mean : {qm.mean_os_score:.4f}")
            print(f"  Qdrant     score mean : {qm.mean_qdrant_score:.4f}")
            print(f"  Score diff (|OS-QD|)  : {qm.mean_score_diff:.4f}")
            print(
                f"  OS   latency mean     : {qm.os_latency_mean:.1f} ± {qm.os_latency_std:.1f} ms"
            )
            print(
                f"  QD   latency mean     : {qm.qdrant_latency_mean:.1f} ± {qm.qdrant_latency_std:.1f} ms"
            )
            print(f"  Latency diff (OS-QD)  : {diff_sign}{qm.latency_diff_mean:.1f} ms")
            print("-" * 70)

        # Cold vs warm
        if report.cold_warm_metrics:
            print(f"\n{sep}")
            print("  COLD-START vs WARM-SEARCH LATENCY")
            print(sep)
            for cw in report.cold_warm_metrics:
                print(f"\n  Store: {cw.store_name}")
                print(f"    Cold start (warm-up) : {cw.cold_latency_ms:.1f} ms")
                print(
                    f"    Warm search (run 2+) : {cw.warm_latency_mean_ms:.1f} ± "
                    f"{cw.warm_latency_std_ms:.1f} ms"
                )
                speedup = (
                    cw.cold_latency_ms / cw.warm_latency_mean_ms
                    if cw.warm_latency_mean_ms > 0
                    else float("inf")
                )
                print(f"    Warm speedup         : {speedup:.2f}x")

        # Summary table
        print(f"\n{sep}")
        print("  SUMMARY TABLE")
        print(sep)
        rows = []
        for qm in report.query_metrics:
            rows.append(
                {
                    "Query (short)": qm.query_text[:40] + ("…" if len(qm.query_text) > 40 else ""),
                    "Jaccard": f"{qm.mean_jaccard:.3f}",
                    "RankDiff": f"{qm.mean_avg_rank_diff:.2f}",
                    "ScoreDiff": f"{qm.mean_score_diff:.4f}",
                    "OS lat(ms)": f"{qm.os_latency_mean:.1f}±{qm.os_latency_std:.1f}",
                    "QD lat(ms)": f"{qm.qdrant_latency_mean:.1f}±{qm.qdrant_latency_std:.1f}",
                    "LatDiff": f"{'+' if qm.latency_diff_mean >= 0 else ''}{qm.latency_diff_mean:.1f}",
                }
            )
        df = pd.DataFrame(rows)
        print(df.to_string(index=False))
        print()

    # ------------------------------------------------------------------
    # CSV export
    # ------------------------------------------------------------------

    def save_csv(
        self,
        warmup_results: list[QueryResult],
        benchmark_results: list[QueryResult],
        output_path: str,
    ) -> None:
        """
        Write per-run results to a CSV file.
        Columns: run_type, run_id, query, store, latency_ms,
                 result_ids, result_scores, mean_score
        """
        rows: list[dict] = []

        def _build_rows(results: list[QueryResult], run_type: str) -> None:
            for qr in results:
                rows.append(
                    {
                        "run_type": run_type,
                        "run_id": qr.run_id,
                        "query": qr.query_text,
                        "store": qr.store_name,
                        "latency_ms": round(qr.latency_ms, 3),
                        "result_ids": "|".join(qr.result_ids),
                        "result_scores": "|".join(
                            f"{s:.6f}" for s in qr.result_scores
                        ),
                        "mean_score": round(qr.mean_score, 6),
                    }
                )

        _build_rows(warmup_results, "warmup")
        _build_rows(benchmark_results, "benchmark")

        df = pd.DataFrame(rows)
        path = Path(output_path)
        df.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"\n[Benchmark] Results saved to '{path.resolve()}'")
