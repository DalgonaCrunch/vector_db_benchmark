"""
RAG pipeline — retry loop with optional query transform, document grading,
and hallucination check.

Replaces the LangGraph workflow with a plain Python loop that calls
the same logical nodes as ordinary functions.

Pipeline flow
-------------
for i in range(max_loop):
    1. embed(current_query)
    2. search(current_query, query_vector, index_name, search_cfg)
    3. [use_doc_grading]      grade_documents(query, docs)
    4. if len(docs) < min_docs:
           [use_query_transform] current_query = transform_query(current_query)
           continue
    5. answer = llm.generate(context_from_docs, current_query)
    6. [use_hallucination_check] check_hallucination(query, answer, docs)
           if fail → continue
    7. return PipelineResult(answer, docs, trace, …)
return PipelineResult with fallback_answer

Config flags
------------
All advanced features are individually togglable via PipelineConfig,
making it trivially easy to A/B test with/without each node.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Iterator, TypedDict

from embedder.upstage_embedder import UpstageEmbedder
from llm.base_provider import BaseLLMProvider
from rag.nodes import check_hallucination, grade_documents, transform_query
from search.router import SearchConfig, search, validate_config
from stores.base_store import SearchResult


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class PipelineConfig(TypedDict):
    """Feature flags and loop parameters for the RAG pipeline."""

    use_query_transform: bool     # rewrite query when results are insufficient
    use_doc_grading: bool         # LLM-grade each retrieved document
    use_hallucination_check: bool # verify answer is grounded in docs
    max_loop: int                 # maximum retry iterations (1–5)
    min_docs: int                 # minimum docs required before generating
    grading_threshold: float      # vector score floor before LLM grading (0 = off)
    grade_model: str              # model used for all grading/transform calls


def default_pipeline_config() -> PipelineConfig:
    return PipelineConfig(
        use_query_transform=False,
        use_doc_grading=False,
        use_hallucination_check=False,
        max_loop=3,
        min_docs=1,
        grading_threshold=0.0,
        grade_model="solar-1-mini-chat",
    )


# ---------------------------------------------------------------------------
# Result + Trace
# ---------------------------------------------------------------------------


@dataclass
class LoopTrace:
    """Diagnostic record for one pipeline iteration."""

    loop_idx: int
    query_used: str
    raw_doc_count: int
    graded_doc_count: int
    query_transformed: bool
    hallucination_passed: bool
    hallucination_reason: str
    elapsed_ms: float


@dataclass
class PipelineResult:
    """Full output of a completed ``rag_pipeline`` call."""

    answer: str
    docs: list[SearchResult]
    original_query: str
    final_query: str
    total_loops: int
    is_fallback: bool
    trace: list[LoopTrace] = field(default_factory=list)
    elapsed_s: float = 0.0

    @property
    def succeeded(self) -> bool:
        return not self.is_fallback and bool(self.docs)


# ---------------------------------------------------------------------------
# Generation helpers (non-streaming and streaming)
# ---------------------------------------------------------------------------

_SYSTEM_TEMPLATE = """\
너는 회사 문서 기반 Q&A 시스템이다.
다음 문서 내용에 포함된 정보만을 근거로 답변하라.
문서에 없는 정보는 절대로 추측하거나 생성하지 말라.
답을 찾을 수 없으면 반드시 "문서에서 해당 내용을 찾을 수 없습니다."라고 답하라.

--- 문서 컨텍스트 시작 ---
{context}
--- 문서 컨텍스트 끝 ---"""

_NO_DOCS_ANSWER = "문서에서 해당 내용을 찾을 수 없습니다."


def _build_context(docs: list[SearchResult]) -> str:
    parts: list[str] = []
    for i, r in enumerate(docs, 1):
        meta = r.metadata
        source = meta.get("source_file", "unknown")
        section = meta.get("section", "")
        header = f"[문서 {i}] {source}" + (f" | {section}" if section else "")
        parts.extend([header, r.text.strip(), ""])
    return "\n".join(parts)


def _generate(
    query: str,
    docs: list[SearchResult],
    provider: BaseLLMProvider,
    model: str,
    temperature: float,
    max_tokens: int = 2048,
) -> str:
    """Non-streaming generation used inside the pipeline loop."""
    context = _build_context(docs)
    system_prompt = _SYSTEM_TEMPLATE.format(context=context)
    return provider.generate(
        system=system_prompt,
        user=query,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def build_stream(
    query: str,
    docs: list[SearchResult],
    provider: BaseLLMProvider,
    model: str,
    temperature: float,
    max_tokens: int = 2048,
) -> Iterator[str]:
    """Streaming generation — for use with ``st.write_stream``."""
    context = _build_context(docs)
    system_prompt = _SYSTEM_TEMPLATE.format(context=context)
    yield from provider.stream(
        system=system_prompt,
        user=query,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def rag_pipeline(
    query: str,
    index_name: str,
    search_cfg: SearchConfig,
    pipeline_cfg: PipelineConfig,
    embedder: UpstageEmbedder,
    provider: BaseLLMProvider,
    gen_model: str,
    temperature: float = 0.0,
) -> PipelineResult:
    """
    Execute the full RAG pipeline with optional retry loop.

    Parameters
    ----------
    query:
        Original user question (never mutated; internal current_query may differ).
    index_name:
        Target KB index name.
    search_cfg:
        SearchConfig for the retrieval step (db, strategy, top_k, rerank, threshold).
    pipeline_cfg:
        PipelineConfig controlling which nodes are active and loop limits.
    embedder:
        Configured UpstageEmbedder for query vectorisation.
    provider:
        LLM provider used for generation AND all grading/transform calls.
    gen_model:
        Model used for the final answer generation step.
    temperature:
        Temperature for the generation step (grading always uses 0).

    Returns
    -------
    PipelineResult
    """
    t_pipeline_start = time.perf_counter()

    search_cfg = validate_config(search_cfg)

    max_loop = max(1, min(pipeline_cfg["max_loop"], 5))
    min_docs = max(1, pipeline_cfg["min_docs"])
    grade_model = pipeline_cfg.get("grade_model", gen_model)

    current_query = query
    trace: list[LoopTrace] = []

    for loop_idx in range(max_loop):
        t_loop_start = time.perf_counter()
        query_transformed = False

        # ── Step 1: Embed current query ──────────────────────────────────
        try:
            query_vector = embedder.embed_single_query(current_query)
        except Exception as exc:
            raise RuntimeError(f"임베딩 실패: {exc}") from exc

        # ── Step 2: Retrieve ─────────────────────────────────────────────
        raw_docs = search(
            query_text=current_query,
            query_vector=query_vector,
            index_name=index_name,
            config=search_cfg,
        )
        raw_count = len(raw_docs)

        # ── Step 3: Document grading ─────────────────────────────────────
        if pipeline_cfg["use_doc_grading"] and raw_docs:
            graded_docs = grade_documents(
                query=current_query,
                docs=raw_docs,
                provider=provider,
                model=grade_model,
                relevance_threshold=pipeline_cfg["grading_threshold"],
            )
        else:
            graded_docs = raw_docs

        graded_count = len(graded_docs)

        # ── Step 4: Insufficient docs → transform query and retry ────────
        if graded_count < min_docs:
            if pipeline_cfg["use_query_transform"] and loop_idx < max_loop - 1:
                current_query = transform_query(
                    query=current_query,
                    provider=provider,
                    model=grade_model,
                )
                query_transformed = True

            trace.append(
                LoopTrace(
                    loop_idx=loop_idx,
                    query_used=current_query,
                    raw_doc_count=raw_count,
                    graded_doc_count=graded_count,
                    query_transformed=query_transformed,
                    hallucination_passed=False,
                    hallucination_reason="문서 부족으로 생성 건너뜀",
                    elapsed_ms=(time.perf_counter() - t_loop_start) * 1000,
                )
            )
            continue

        # ── Step 5: Generate answer (non-streaming) ──────────────────────
        try:
            answer = _generate(
                query=query,        # always use original query for generation
                docs=graded_docs,
                provider=provider,
                model=gen_model,
                temperature=temperature,
            )
        except Exception as exc:
            trace.append(
                LoopTrace(
                    loop_idx=loop_idx,
                    query_used=current_query,
                    raw_doc_count=raw_count,
                    graded_doc_count=graded_count,
                    query_transformed=query_transformed,
                    hallucination_passed=False,
                    hallucination_reason=f"생성 오류: {exc}",
                    elapsed_ms=(time.perf_counter() - t_loop_start) * 1000,
                )
            )
            continue

        # ── Step 6: Hallucination check ──────────────────────────────────
        is_grounded = True
        h_reason = ""
        if pipeline_cfg["use_hallucination_check"]:
            is_grounded, h_reason = check_hallucination(
                query=query,
                answer=answer,
                docs=graded_docs,
                provider=provider,
                model=grade_model,
            )

        trace.append(
            LoopTrace(
                loop_idx=loop_idx,
                query_used=current_query,
                raw_doc_count=raw_count,
                graded_doc_count=graded_count,
                query_transformed=query_transformed,
                hallucination_passed=is_grounded,
                hallucination_reason=h_reason,
                elapsed_ms=(time.perf_counter() - t_loop_start) * 1000,
            )
        )

        if not is_grounded:
            # Transform query and retry on hallucination
            if pipeline_cfg["use_query_transform"] and loop_idx < max_loop - 1:
                current_query = transform_query(
                    query=current_query,
                    provider=provider,
                    model=grade_model,
                )
            continue

        # ── Step 7: Success — return result ──────────────────────────────
        return PipelineResult(
            answer=answer,
            docs=graded_docs,
            original_query=query,
            final_query=current_query,
            total_loops=loop_idx + 1,
            is_fallback=False,
            trace=trace,
            elapsed_s=time.perf_counter() - t_pipeline_start,
        )

    # ── All loops exhausted → return fallback ────────────────────────────
    # Try to return the last retrieved docs even on fallback
    fallback_docs: list[SearchResult] = []
    if trace:
        last = trace[-1]
        # Re-retrieve one last time using the latest query
        try:
            fv = embedder.embed_single_query(current_query)
            fallback_docs = search(
                query_text=current_query,
                query_vector=fv,
                index_name=index_name,
                config=search_cfg,
            )
        except Exception:
            pass

    return PipelineResult(
        answer=_NO_DOCS_ANSWER,
        docs=fallback_docs,
        original_query=query,
        final_query=current_query,
        total_loops=max_loop,
        is_fallback=True,
        trace=trace,
        elapsed_s=time.perf_counter() - t_pipeline_start,
    )
