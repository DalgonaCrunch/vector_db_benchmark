"""
RAG Chatbot — document-grounded Q&A over a selected KB Index.

Flow:
  user question
    → rag_pipeline()
        ├─ embed-query
        ├─ search (db + strategy + params)
        ├─ [optional] grade_documents  — LLM relevance filter
        ├─ [optional] transform_query  — rewrite on insufficient results
        ├─ generate answer             — strict context-only prompt
        └─ [optional] check_hallucination — verify answer is grounded
    → streaming display of final answer + source citations

Pipeline options (sidebar):
  use_query_transform    — rewrite query when docs are insufficient
  use_doc_grading        — LLM binary relevance filter per document
  use_hallucination_check— verify answer grounding before display
  max_loop               — maximum retry iterations
  grade_model            — model used for all grading / transform calls

Hallucination guards (always on):
  • temperature defaults to 0
  • system prompt explicitly forbids external knowledge
  • if no docs → skip LLM, return canned message
  • max_tokens capped at 2048
"""
from __future__ import annotations

import os

import streamlit as st
from dotenv import load_dotenv

from config import load_config
from embedder.upstage_embedder import UpstageEmbedder
from ingestion.index_registry import IndexRegistry
from llm import PROVIDER_MODELS, _PROVIDER_ENV, available_providers, get_provider
from rag.pipeline import (
    PipelineConfig,
    PipelineResult,
    build_stream,
    default_pipeline_config,
    rag_pipeline,
)
from search.router import SearchConfig, _VALID_STRATEGIES, validate_config
from stores.base_store import SearchResult

load_dotenv()

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)

_REGISTRY_PATH = "./data/index_registry.json"
_MAX_TOKENS = 2048
_NO_DOCS_ANSWER = "문서에서 해당 내용을 찾을 수 없습니다."

# ---------------------------------------------------------------------------
# Cached resources
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner="Upstage 임베더 로딩 중…")
def _load_embedder() -> UpstageEmbedder:
    cfg = load_config()
    return UpstageEmbedder(
        api_key=cfg.embedder.api_key,
        api_url=cfg.embedder.api_url,
        passage_model=cfg.embedder.passage_model,
        query_model=cfg.embedder.query_model,
        batch_size=cfg.embedder.batch_size,
    )


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------


def _render_sources(results: list[SearchResult]) -> None:
    with st.expander("📚 참고 문서", expanded=False):
        for i, r in enumerate(results, 1):
            meta = r.metadata
            source = meta.get("source_file", "?")
            section = meta.get("section", "")
            chunk_idx = meta.get("chunk_index", "")
            if r.score >= 0.7:
                color = "#2ecc71"
            elif r.score >= 0.4:
                color = "#f39c12"
            else:
                color = "#e74c3c"
            score_html = (
                f'<span style="color:{color};font-weight:bold">{r.score:.4f}</span>'
            )
            parts = [f"**[{i}]** 📄 `{source}`"]
            if section:
                parts.append(f"· 📑 {section}")
            if chunk_idx != "":
                parts.append(f"· 🔢 chunk #{chunk_idx}")
            parts.append(f"· score: {score_html}")
            st.markdown(" ".join(parts), unsafe_allow_html=True)


def _render_pipeline_trace(result: PipelineResult) -> None:
    """Render debug panel showing full pipeline execution trace."""
    with st.expander("🔬 Pipeline Trace", expanded=True):
        # Summary row
        status = "✅ 성공" if result.succeeded else "⚠️ Fallback"
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("상태", status)
        c2.metric("총 루프", f"{result.total_loops}회")
        c3.metric("최종 문서 수", f"{len(result.docs)}개")
        c4.metric("처리 시간", f"{result.elapsed_s:.2f}s")

        if result.original_query != result.final_query:
            st.info(
                f"**원본 질문:** {result.original_query}  \n"
                f"**최종 질문:** {result.final_query}"
            )

        # Per-loop breakdown
        for t in result.trace:
            loop_color = "🟢" if t.hallucination_passed else "🔴"
            transform_badge = " ✏️ 쿼리변환" if t.query_transformed else ""
            with st.expander(
                f"{loop_color} Loop {t.loop_idx + 1}{transform_badge} "
                f"— {t.graded_doc_count}/{t.raw_doc_count} docs "
                f"({t.elapsed_ms:.0f} ms)",
                expanded=False,
            ):
                st.markdown(f"**사용된 쿼리:** `{t.query_used}`")
                dc1, dc2 = st.columns(2)
                dc1.metric("검색 원본", f"{t.raw_doc_count}개")
                dc2.metric("Grading 통과", f"{t.graded_doc_count}개")
                if t.hallucination_reason:
                    label = "Hallucination 판정" if t.hallucination_passed else "⚠️ Hallucination 이유"
                    st.caption(f"{label}: {t.hallucination_reason}")


def _render_debug_search(search_cfg: SearchConfig, results: list[SearchResult]) -> None:
    with st.expander("🐛 Search Debug", expanded=True):
        st.markdown("**Search Config**")
        st.json(dict(search_cfg))
        st.markdown(f"**결과 수:** {len(results)}")
        if results:
            raw = [
                {
                    "rank": r.rank,
                    "id": r.id,
                    "score": r.score,
                    "text_preview": r.text[:120] + ("…" if len(r.text) > 120 else ""),
                    "metadata": r.metadata,
                }
                for r in results
            ]
            st.json(raw)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------


def _render_sidebar() -> dict:
    with st.sidebar:
        st.header("⚙️ 챗봇 설정")

        registry = IndexRegistry(_REGISTRY_PATH)
        names = registry.index_names()

        if not names:
            st.warning("KB Index가 없습니다.\n관리 페이지에서 문서를 먼저 추가하세요.")
            st.stop()

        selected_index = st.selectbox("📚 Knowledge Base", names, key="cb_index")
        kb = registry.get(selected_index)
        if kb:
            st.caption(
                f"문서 {len(kb.documents)}개 · "
                f"청크 {kb.total_chunks():,}개 · "
                f"dim {kb.vector_dim or '?'}"
            )

        st.divider()

        # LLM provider + model
        all_providers = list(PROVIDER_MODELS.keys())
        ready = available_providers()

        provider = st.selectbox(
            "LLM Provider",
            all_providers,
            key="cb_provider",
            format_func=lambda p: f"{'✅' if p in ready else '⚠️'} {p}",
        )
        model = st.selectbox("모델", PROVIDER_MODELS[provider], key="cb_model")

        env_key = _PROVIDER_ENV[provider]
        if not os.getenv(env_key):
            st.warning(
                f"`{env_key}` 환경변수가 없습니다.\n`.env` 파일에 추가 후 재시작하세요."
            )

        st.divider()

        temperature = st.slider(
            "Temperature",
            0.0, 1.0, 0.0, 0.05,
            key="cb_temp",
            help="0 = 결정적·사실 기반 (권장)",
        )

        # ── Search options ────────────────────────────────────────────────
        st.divider()
        st.subheader("🔍 검색 설정")

        db = st.selectbox(
            "Vector DB",
            ["qdrant", "opensearch"],
            key="cb_db",
            help="qdrant: dense_vector만 지원 | opensearch: 모든 전략 지원",
        )

        allowed_strategies = _VALID_STRATEGIES.get(db, ["dense_vector"])
        current_strategy = st.session_state.get("cb_strategy", "dense_vector")
        strategy_index = (
            allowed_strategies.index(current_strategy)
            if current_strategy in allowed_strategies
            else 0
        )
        strategy = st.selectbox(
            "Retrieval Strategy",
            allowed_strategies,
            index=strategy_index,
            key="cb_strategy",
            help="dense_vector: KNN | keyword: BM25 | hybrid: KNN+BM25 (RRF)",
        )

        if db == "qdrant" and strategy != "dense_vector":
            st.warning("Qdrant는 dense_vector만 지원합니다.")

        top_k = st.slider("Top-K", 1, 20, 5, key="cb_topk")

        score_threshold = st.slider(
            "Score Threshold",
            0.0, 1.0, 0.0, 0.05,
            key="cb_threshold",
            help="이 값 미만의 결과는 필터링합니다 (0 = 비활성화)",
        )

        use_rerank = st.checkbox(
            "Rerank 사용",
            value=False,
            key="cb_rerank",
            help="Upstage solar-reranker로 재정렬합니다 (속도 저하)",
        )

        # ── Pipeline options ──────────────────────────────────────────────
        st.divider()
        st.subheader("🔄 Pipeline 옵션")

        use_query_transform = st.checkbox(
            "Query Transform",
            value=False,
            key="cb_qt",
            help="검색 결과가 부족하면 LLM으로 쿼리를 재작성합니다",
        )

        use_doc_grading = st.checkbox(
            "Document Grading",
            value=False,
            key="cb_dg",
            help="LLM이 각 문서의 관련성을 이진 판정합니다 (속도 저하)",
        )

        use_hallucination_check = st.checkbox(
            "Hallucination Check",
            value=False,
            key="cb_hc",
            help="LLM이 생성된 답변의 근거 여부를 검증합니다 (속도 저하)",
        )

        if use_query_transform or use_doc_grading or use_hallucination_check:
            max_loop = st.slider(
                "Max Loop",
                min_value=1,
                max_value=5,
                value=3,
                key="cb_max_loop",
                help="최대 재시도 횟수 (query transform / hallucination retry)",
            )
            min_docs = st.number_input(
                "Min Docs (재검색 기준)",
                min_value=1,
                max_value=10,
                value=1,
                key="cb_min_docs",
                help="이 수 미만의 문서가 남으면 query transform을 시도합니다",
            )
            grading_threshold = st.slider(
                "Grading Score Threshold",
                0.0, 1.0, 0.0, 0.05,
                key="cb_grade_thresh",
                help="LLM 판정 전 벡터 점수 사전 필터 (0 = 비활성화)",
            )

            # Grade model — can differ from generation model
            grade_provider_models = [
                m for prov_models in PROVIDER_MODELS.values() for m in prov_models
            ]
            all_models_flat = []
            for prov, models in PROVIDER_MODELS.items():
                for m in models:
                    all_models_flat.append(f"{prov}/{m}")

            grade_model_label = st.selectbox(
                "Grading Model",
                all_models_flat,
                key="cb_grade_model",
                help="Query Transform / Document Grading / Hallucination Check에 사용할 모델",
            )
            # parse "Provider/model-name"
            if "/" in grade_model_label:
                _, grade_model = grade_model_label.split("/", 1)
            else:
                grade_model = model
        else:
            max_loop = 1
            min_docs = 1
            grading_threshold = 0.0
            grade_model = model

        # ── Debug ─────────────────────────────────────────────────────────
        st.divider()
        debug_mode = st.checkbox(
            "Debug Mode",
            value=False,
            key="cb_debug",
            help="검색 config, raw 결과, pipeline trace를 출력합니다",
        )

        if st.button("🗑 대화 초기화", key="cb_clear", use_container_width=True):
            st.session_state["chatbot_messages"] = []
            st.rerun()

        st.caption("✅ API 키 설정됨 · ⚠️ 키 없음")

    search_cfg = SearchConfig(
        db=db,
        strategy=strategy,
        top_k=top_k,
        use_rerank=use_rerank,
        score_threshold=score_threshold,
    )

    pipeline_cfg = PipelineConfig(
        use_query_transform=use_query_transform,
        use_doc_grading=use_doc_grading,
        use_hallucination_check=use_hallucination_check,
        max_loop=max_loop,
        min_docs=int(min_docs),
        grading_threshold=grading_threshold,
        grade_model=grade_model,
    )

    return {
        "index": selected_index,
        "provider": provider,
        "model": model,
        "temperature": temperature,
        "search": search_cfg,
        "pipeline": pipeline_cfg,
        "debug": debug_mode,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    st.title("💬 RAG Chatbot")
    st.caption(
        "선택된 KB Index 기반 · 문서 외 정보 생성 금지 · "
        "Query Transform / Doc Grading / Hallucination Check 지원"
    )

    cfg = _render_sidebar()

    # ── Chat history ──────────────────────────────────────────────────────
    if "chatbot_messages" not in st.session_state:
        st.session_state["chatbot_messages"] = []

    for msg in st.session_state["chatbot_messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                _render_sources(msg["sources"])

    # ── Chat input ────────────────────────────────────────────────────────
    user_input = st.chat_input("질문을 입력하세요…")
    if not user_input:
        return

    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state["chatbot_messages"].append(
        {"role": "user", "content": user_input}
    )

    # ── Load resources ────────────────────────────────────────────────────
    try:
        provider = get_provider(cfg["provider"])
    except (ValueError, ImportError) as exc:
        st.error(str(exc))
        return

    embedder = _load_embedder()
    search_cfg: SearchConfig = validate_config(cfg["search"])
    pipeline_cfg: PipelineConfig = cfg["pipeline"]

    any_pipeline_active = (
        pipeline_cfg["use_query_transform"]
        or pipeline_cfg["use_doc_grading"]
        or pipeline_cfg["use_hallucination_check"]
    )

    # ── RAG pipeline ──────────────────────────────────────────────────────
    pipeline_label = (
        f"[{search_cfg['db']}/{search_cfg['strategy']}]"
        + (" 🔄" if any_pipeline_active else "")
    )

    with st.spinner(f"관련 문서 검색 및 답변 생성 중… {pipeline_label}"):
        try:
            result: PipelineResult = rag_pipeline(
                query=user_input,
                index_name=cfg["index"],
                search_cfg=search_cfg,
                pipeline_cfg=pipeline_cfg,
                embedder=embedder,
                provider=provider,
                gen_model=cfg["model"],
                temperature=cfg["temperature"],
            )
        except Exception as exc:
            st.error(f"Pipeline 오류: {exc}")
            return

    # ── Debug output ──────────────────────────────────────────────────────
    if cfg["debug"]:
        _render_debug_search(search_cfg, result.docs)
        _render_pipeline_trace(result)

    # ── Fallback / no docs guard ──────────────────────────────────────────
    if result.is_fallback or not result.docs:
        with st.chat_message("assistant"):
            st.warning(_NO_DOCS_ANSWER)
            if result.trace and cfg["debug"]:
                st.caption(f"총 {result.total_loops}회 시도 후 문서 없음")
        st.session_state["chatbot_messages"].append(
            {"role": "assistant", "content": _NO_DOCS_ANSWER, "sources": []}
        )
        return

    # ── Streaming generation using final docs ─────────────────────────────
    with st.chat_message("assistant"):
        try:
            # Re-stream using the final graded docs and (possibly transformed) query
            full_answer = st.write_stream(
                build_stream(
                    query=user_input,          # display with original question
                    docs=result.docs,
                    provider=provider,
                    model=cfg["model"],
                    temperature=cfg["temperature"],
                    max_tokens=_MAX_TOKENS,
                )
            )
        except Exception as exc:
            st.error(f"LLM 응답 실패: {exc}")
            return

        _render_sources(result.docs)

        # Pipeline summary badge
        if any_pipeline_active:
            badges: list[str] = []
            if pipeline_cfg["use_query_transform"] and result.final_query != user_input:
                badges.append("✏️ 쿼리 변환됨")
            if pipeline_cfg["use_doc_grading"]:
                badges.append(f"✅ Grading ({len(result.docs)}개 통과)")
            if pipeline_cfg["use_hallucination_check"]:
                badges.append("🔍 Hallucination 통과")
            if badges:
                st.caption(f"Pipeline: {' · '.join(badges)} · {result.total_loops}회 루프 · {result.elapsed_s:.1f}s")

    st.session_state["chatbot_messages"].append(
        {
            "role": "assistant",
            "content": full_answer,
            "sources": result.docs,
        }
    )


if __name__ == "__main__":
    main()
