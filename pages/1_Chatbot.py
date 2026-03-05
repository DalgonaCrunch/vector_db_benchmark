"""
RAG Chatbot — document-grounded Q&A over a selected KB Index.

Flow:
  user question
    → embed-query (Upstage embedding-query)
    → Qdrant top-k retrieval
    → strict system prompt + retrieved context
    → LLM (Upstage / OpenAI / Claude)   [streaming]
    → answer + source citations

Hallucination guards:
  • temperature defaults to 0
  • system prompt explicitly forbids external knowledge
  • if retrieval returns 0 results → skip LLM, return canned message
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
from stores.base_store import SearchResult
from stores.qdrant_store import QdrantStoreConfig, QdrantVectorStore

load_dotenv()

# ---------------------------------------------------------------------------
# Page config  (must be the first Streamlit call in this file)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_REGISTRY_PATH = "./data/index_registry.json"
_MAX_TOKENS = 2048
_NO_DOCS_ANSWER = "문서에서 해당 내용을 찾을 수 없습니다."

_SYSTEM_TEMPLATE = """\
너는 회사 문서 기반 Q&A 시스템이다.
다음 문서 내용에 포함된 정보만을 근거로 답변하라.
문서에 없는 정보는 절대로 추측하거나 생성하지 말라.
답을 찾을 수 없으면 반드시 "{no_docs}"라고 답하라.

--- 문서 컨텍스트 시작 ---
{{context}}
--- 문서 컨텍스트 끝 ---""".format(no_docs=_NO_DOCS_ANSWER)

# ---------------------------------------------------------------------------
# Cached resources  (shared across reruns within the same session)
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
# Lightweight helpers
# ---------------------------------------------------------------------------


def _make_qdrant(index_name: str) -> QdrantVectorStore:
    cfg = load_config()
    return QdrantVectorStore(
        QdrantStoreConfig(
            collection_name=index_name,
            mode=cfg.qdrant.mode,
            local_path=cfg.qdrant.local_path,
            host=cfg.qdrant.host,
            port=cfg.qdrant.port,
        )
    )


def _retrieve(query_vector: list[float], index_name: str, top_k: int) -> list[SearchResult]:
    store = _make_qdrant(index_name)
    if not store.exists():
        return []
    try:
        return store.search(query_vector, top_k=top_k)
    except Exception:
        return []


def _build_context(results: list[SearchResult]) -> str:
    """Format retrieved chunks into a numbered context block."""
    parts: list[str] = []
    for i, r in enumerate(results, 1):
        meta = r.metadata
        source = meta.get("source_file", "unknown")
        section = meta.get("section", "")
        header = f"[문서 {i}] {source}" + (f" | {section}" if section else "")
        parts.extend([header, r.text.strip(), ""])
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# UI widgets
# ---------------------------------------------------------------------------


def _render_sources(results: list[SearchResult]) -> None:
    """Collapsible source citation block below an assistant message."""
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
            score_html = f'<span style="color:{color};font-weight:bold">{r.score:.4f}</span>'
            parts = [f"**[{i}]** 📄 `{source}`"]
            if section:
                parts.append(f"· 📑 {section}")
            if chunk_idx != "":
                parts.append(f"· 🔢 chunk #{chunk_idx}")
            parts.append(f"· score: {score_html}")
            st.markdown(" ".join(parts), unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------


def _render_sidebar() -> dict:
    """Render chatbot configuration. Returns settings dict."""
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

        # Provider + model
        all_providers = list(PROVIDER_MODELS.keys())
        ready = available_providers()

        provider = st.selectbox(
            "LLM Provider",
            all_providers,
            key="cb_provider",
            format_func=lambda p: f"{'✅' if p in ready else '⚠️'} {p}",
        )
        model = st.selectbox(
            "모델",
            PROVIDER_MODELS[provider],
            key="cb_model",
        )

        # Key status warning
        env_key = _PROVIDER_ENV[provider]
        if not os.getenv(env_key):
            st.warning(
                f"`{env_key}` 환경변수가 없습니다.\n"
                f"`.env` 파일에 추가 후 재시작하세요."
            )

        st.divider()

        temperature = st.slider(
            "Temperature",
            0.0, 1.0, 0.0, 0.05,
            key="cb_temp",
            help="0 = 결정적·사실 기반 (권장), 높을수록 창의적·불확실",
        )
        top_k = st.slider("검색 Top-K", 1, 10, 5, key="cb_topk")

        st.divider()

        if st.button("🗑 대화 초기화", key="cb_clear", use_container_width=True):
            st.session_state["chatbot_messages"] = []
            st.rerun()

        st.caption("✅ API 키 설정됨 · ⚠️ 키 없음")

    return {
        "index": selected_index,
        "provider": provider,
        "model": model,
        "temperature": temperature,
        "top_k": top_k,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    st.title("💬 RAG Chatbot")
    st.caption(
        "선택된 KB Index 기반 · 문서 외 정보 생성 금지 · "
        "컨텍스트 없으면 LLM 미호출"
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

    # ── Step 1: Embed query ───────────────────────────────────────────────
    try:
        embedder = _load_embedder()
        query_vector = embedder.embed_single_query(user_input)
    except Exception as exc:
        st.error(f"임베딩 실패: {exc}")
        return

    # ── Step 2: Retrieve ──────────────────────────────────────────────────
    with st.spinner("관련 문서 검색 중…"):
        results = _retrieve(query_vector, cfg["index"], cfg["top_k"])

    # ── Step 3: Guard — empty retrieval → no LLM call ────────────────────
    if not results:
        with st.chat_message("assistant"):
            st.warning(_NO_DOCS_ANSWER)
        st.session_state["chatbot_messages"].append(
            {"role": "assistant", "content": _NO_DOCS_ANSWER, "sources": []}
        )
        return

    # ── Step 4: Build strict prompt ───────────────────────────────────────
    context = _build_context(results)
    system_prompt = _SYSTEM_TEMPLATE.format(context=context)

    # ── Step 5: LLM streaming call ────────────────────────────────────────
    try:
        provider = get_provider(cfg["provider"])
    except (ValueError, ImportError) as exc:
        st.error(str(exc))
        return

    with st.chat_message("assistant"):
        try:
            full_answer = st.write_stream(
                provider.stream(
                    system=system_prompt,
                    user=user_input,
                    model=cfg["model"],
                    temperature=cfg["temperature"],
                    max_tokens=_MAX_TOKENS,
                )
            )
        except Exception as exc:
            st.error(f"LLM 응답 실패: {exc}")
            return
        _render_sources(results)

    st.session_state["chatbot_messages"].append(
        {
            "role": "assistant",
            "content": full_answer,
            "sources": results,
        }
    )


if __name__ == "__main__":
    main()
