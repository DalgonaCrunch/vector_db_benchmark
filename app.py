"""
Streamlit UI: Knowledge Base Index-based Vector DB Benchmark.

Architecture:
  KB Index  (= Qdrant collection + OpenSearch index)
    └─ Document A (pdf), Document B (docx), …  → shared vector space

Tabs:
  📤 문서 추가   — 선택된 Index에 PDF/DOCX 업로드
  🔍 검색        — 선택된 Index에서 Qdrant vs OpenSearch 비교
  ⚖️ Index 비교  — 두 Index를 동일 쿼리로 비교
  🗂 관리        — Index 목록 · 문서 목록 · 삭제

Sidebar:
  • 기존 Index 드롭다운 선택
  • 새 Index 생성

Run:
    make start-opensearch && make app
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

import streamlit as st

from config import load_config
from embedder.upstage_embedder import UpstageEmbedder
from ingestion.index_ingestor import IndexIngestor, IngestResult
from ingestion.index_registry import IndexRegistry, KBIndex, sanitize_index_name
from ingestion.loaders import FILE_LOADER_REGISTRY
from stores.base_store import SearchResult
from stores.opensearch_store import OpenSearchStoreConfig, OpenSearchVectorStore
from stores.qdrant_store import QdrantStoreConfig, QdrantVectorStore

# ---------------------------------------------------------------------------
# Page config (must be first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Vector DB Benchmark",
    page_icon="🗄️",
    layout="wide",
    initial_sidebar_state="expanded",
)

_REGISTRY_PATH = "./data/index_registry.json"
_FILE_TYPE_ICON: dict[str, str] = {"pdf": "📕", "docx": "📘"}
_PREVIEW_LEN = 300


# ---------------------------------------------------------------------------
# Cached heavy resources
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner="Upstage embedder 로딩 중…")
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
# Lightweight helpers (cheap to recreate — not cached)
# ---------------------------------------------------------------------------


def _get_registry() -> IndexRegistry:
    return IndexRegistry(_REGISTRY_PATH)


def _load_ingestor() -> IndexIngestor:
    return IndexIngestor(_load_embedder(), IndexRegistry(_REGISTRY_PATH))


def _make_qdrant(index_name: str) -> QdrantVectorStore:
    """Connect to an existing Qdrant collection (no initialize())."""
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


def _make_opensearch(index_name: str) -> OpenSearchVectorStore:
    """Connect to an existing OpenSearch index (no initialize())."""
    cfg = load_config()
    return OpenSearchVectorStore(
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


def _active_index() -> str | None:
    """Return the currently selected index name from session state."""
    return st.session_state.get("active_index_name")


def _set_active_index(name: str) -> None:
    st.session_state["active_index_name"] = name


# ---------------------------------------------------------------------------
# Domain types
# ---------------------------------------------------------------------------


@dataclass
class SearchOutcome:
    store_name: str
    results: list[SearchResult]
    latency_ms: float
    vector_count: int = 0
    error: str | None = None


# ---------------------------------------------------------------------------
# Search logic
# ---------------------------------------------------------------------------


def _jaccard(a: list[str], b: list[str]) -> float:
    sa, sb = set(a), set(b)
    union = sa | sb
    return round(len(sa & sb) / len(union), 4) if union else 0.0


def run_search(
    query: str,
    index_name: str,
    top_k: int,
    use_qdrant: bool,
    use_opensearch: bool,
    filter_doc_id: str | None = None,
) -> dict[str, SearchOutcome]:
    """Embed *query* and search within the given KB index."""
    embedder = _load_embedder()
    try:
        query_vector = embedder.embed_single_query(query)
    except Exception as exc:
        st.error(f"임베딩 실패: {exc}")
        return {}

    outcomes: dict[str, SearchOutcome] = {}

    if use_qdrant:
        try:
            store = _make_qdrant(index_name)
            count = store.count()
            t0 = time.perf_counter()
            results = store.search(query_vector, top_k=top_k, filter_doc_id=filter_doc_id)
            outcomes["qdrant"] = SearchOutcome(
                "qdrant", results, (time.perf_counter() - t0) * 1000.0, count
            )
        except Exception as exc:
            outcomes["qdrant"] = SearchOutcome("qdrant", [], 0.0, error=str(exc))

    if use_opensearch:
        try:
            store = _make_opensearch(index_name)
            count = store.count()
            t0 = time.perf_counter()
            results = store.search(query_vector, top_k=top_k, filter_doc_id=filter_doc_id)
            outcomes["opensearch"] = SearchOutcome(
                "opensearch", results, (time.perf_counter() - t0) * 1000.0, count
            )
        except Exception as exc:
            outcomes["opensearch"] = SearchOutcome("opensearch", [], 0.0, error=str(exc))

    return outcomes


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

_SCORE_COLORS = {"high": "#2ecc71", "mid": "#f39c12", "low": "#e74c3c"}


def _score_color(score: float) -> str:
    if score >= 0.7:
        return _SCORE_COLORS["high"]
    if score >= 0.4:
        return _SCORE_COLORS["mid"]
    return _SCORE_COLORS["low"]


def _file_icon(filename: str) -> str:
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    return _FILE_TYPE_ICON.get(ext, "📄")


def _highlight(text: str, query: str) -> str:
    """Bold-wrap query tokens in *text* for markdown rendering."""
    tokens = [re.escape(w) for w in re.split(r"\s+", query.strip()) if len(w) > 1]
    if not tokens:
        return text
    return re.compile("|".join(tokens), re.IGNORECASE).sub(
        lambda m: f"**{m.group()}**", text
    )


def render_result_card(result: SearchResult, common_ids: set[str]) -> None:
    badge = "✅" if result.id in common_ids else "🔹"
    meta = result.metadata
    color = _score_color(result.score)
    score_html = f'<span style="color:{color};font-weight:bold">{result.score:.4f}</span>'

    full_text = result.text
    is_long = len(full_text) > _PREVIEW_LEN
    # pick up the active search query for keyword highlighting (both tabs)
    query = st.session_state.get("search_query", "") or st.session_state.get("cmp_query", "")

    with st.container(border=True):
        col_rank, col_info = st.columns([1, 9])
        with col_rank:
            st.markdown(f"### {badge}{result.rank}")
        with col_info:
            st.markdown(
                f"**`{result.id}`** &nbsp;|&nbsp; score: {score_html}",
                unsafe_allow_html=True,
            )
            # ── Metadata row ──────────────────────────────────────────────
            m_cols = st.columns(4)
            file_icon = _FILE_TYPE_ICON.get(meta.get("file_type", ""), "📄")
            if "source_file" in meta:
                m_cols[0].caption(f"{file_icon} {meta['source_file']}")
            if "section" in meta:
                m_cols[1].caption(f"📑 {meta['section']}")
            if "chunk_index" in meta:
                m_cols[2].caption(f"🔢 chunk #{meta['chunk_index']}")
            if "file_type" in meta:
                m_cols[3].caption(f"🗂 {meta['file_type'].upper()}")
            # ── Preview (300 chars) ───────────────────────────────────────
            preview = full_text[:_PREVIEW_LEN]
            suffix = "…" if is_long else ""
            if query:
                st.markdown(_highlight(preview, query) + suffix)
            else:
                st.text(preview + suffix)
            # ── Full text expander (st.code has built-in copy button) ─────
            if is_long:
                with st.expander("전체보기"):
                    st.code(full_text, language=None)


def render_store_column(
    outcome: SearchOutcome | None,
    label: str,
    common_ids: set[str],
) -> None:
    if outcome is None:
        st.info(f"{label} 비활성")
        return
    if outcome.error:
        st.error(f"오류: {outcome.error}")
        if "opensearch" in label.lower():
            st.caption("`make start-opensearch` 실행 후 재시도하세요.")
        return

    c1, c2 = st.columns(2)
    c1.metric("응답 시간", f"{outcome.latency_ms:.1f} ms")
    c2.metric("총 벡터", f"{outcome.vector_count:,}개")

    if not outcome.results:
        st.warning("검색 결과 없음")
        return

    st.caption(f"Top-{len(outcome.results)} 결과")
    for result in outcome.results:
        render_result_card(result, common_ids)


def render_comparison_panel(
    outcomes: dict[str, SearchOutcome],
    title: str = "📊 비교 메트릭",
) -> None:
    qd = outcomes.get("qdrant")
    os = outcomes.get("opensearch")
    if not qd or not os or qd.error or os.error:
        return

    qd_ids = [r.id for r in qd.results]
    os_ids = [r.id for r in os.results]
    jaccard = _jaccard(qd_ids, os_ids)
    latency_diff = os.latency_ms - qd.latency_ms
    common = set(qd_ids) & set(os_ids)

    avg_rank_diff: float | None = None
    if common:
        diffs = [abs(qd_ids.index(d) - os_ids.index(d)) for d in common]
        avg_rank_diff = sum(diffs) / len(diffs)

    st.divider()
    st.subheader(title)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Jaccard Overlap", f"{jaccard:.3f}", help="1.0 = 결과 완전 동일")
    sign = "+" if latency_diff >= 0 else ""
    m2.metric(
        "Latency (OS−QD)",
        f"{sign}{latency_diff:.1f} ms",
        delta=f"{sign}{latency_diff:.1f} ms",
        delta_color="inverse",
    )
    m3.metric("공통 결과", f"{len(common)} / {max(len(qd_ids), len(os_ids))}")
    if avg_rank_diff is not None:
        m4.metric("Avg 순위 차이", f"{avg_rank_diff:.2f}")
    else:
        m4.metric("Avg 순위 차이", "N/A")

    only_qd = set(qd_ids) - set(os_ids)
    only_os = set(os_ids) - set(qd_ids)
    with st.expander("결과 집합 상세", expanded=(len(common) < len(qd_ids))):
        if common:
            st.success(f"✅ 공통 ({len(common)}개): {', '.join(sorted(common))}")
        if only_qd:
            st.info(f"🟦 Qdrant 전용 ({len(only_qd)}개): {', '.join(sorted(only_qd))}")
        if only_os:
            st.warning(
                f"🟧 OpenSearch 전용 ({len(only_os)}개): {', '.join(sorted(only_os))}"
            )


# ---------------------------------------------------------------------------
# Sidebar: Index selector + creation
# ---------------------------------------------------------------------------


def render_sidebar() -> None:
    """Render the KB Index selector in the sidebar."""
    with st.sidebar:
        st.header("📚 Knowledge Base")

        registry = _get_registry()
        indices = registry.all_indices()
        names = [kb.index_name for kb in indices]

        # ── Active index selector ────────────────────────────────────────
        if names:
            current = _active_index()
            if current not in names:
                current = names[0]
                _set_active_index(current)

            selected = st.selectbox(
                "활성 Index",
                names,
                index=names.index(current),
                key="sidebar_index_sel",
            )
            if selected != _active_index():
                _set_active_index(selected)
                st.rerun()

            kb = registry.get(selected)
            if kb:
                st.metric("총 청크", f"{kb.total_chunks():,}개")
                st.caption(
                    f"문서 {len(kb.documents)}개 · "
                    f"dim {kb.vector_dim or '?'}"
                )
        else:
            st.info("Index가 없습니다.\n아래에서 새로 생성하세요.")

        st.divider()

        # ── Create new index ─────────────────────────────────────────────
        st.subheader("➕ 새 Index 생성")
        raw_name = st.text_input(
            "Index 이름",
            key="new_index_name_input",
            placeholder="예: medical_kb, project_docs",
        )
        if st.button("생성", key="create_index_btn", type="primary"):
            if not raw_name.strip():
                st.warning("이름을 입력하세요.")
            else:
                sanitized = sanitize_index_name(raw_name.strip())
                reg = _get_registry()
                if reg.get(sanitized):
                    st.error(f"'{sanitized}' 이미 존재합니다.")
                else:
                    reg.create(sanitized)
                    _set_active_index(sanitized)
                    st.success(f"Index **'{sanitized}'** 생성됨")
                    st.rerun()
            if raw_name and raw_name != sanitize_index_name(raw_name):
                st.caption(
                    f"사용될 이름: **`{sanitize_index_name(raw_name)}`**"
                )

        st.divider()
        st.caption("실행 순서")
        st.code("make start-opensearch\nmake app", language="bash")


# ---------------------------------------------------------------------------
# Tab: Upload
# ---------------------------------------------------------------------------

_BATCH_RESULT_KEY = "upload_batch_result"


@dataclass
class _BatchResult:
    total: int
    successes: list = field(default_factory=list)   # list[tuple[str, IngestResult]]
    duplicates: list = field(default_factory=list)  # list[str]
    failures: list = field(default_factory=list)    # list[tuple[str, str]]
    elapsed_s: float = 0.0


def _run_batch_ingest(uploaded_files: list, active: str) -> _BatchResult:
    """Duplicate-check then sequentially ingest all new files."""
    ingestor = _load_ingestor()
    br = _BatchResult(total=len(uploaded_files))

    # ── Duplicate pre-check ───────────────────────────────────────────────
    dup_status_text = st.empty()
    dup_status_text.text("중복 파일 확인 중…")
    dup_flags: dict[str, bool] = {}
    for uf in uploaded_files:
        try:
            dup_flags[uf.name] = ingestor.is_duplicate(uf.name, active)
        except Exception:
            dup_flags[uf.name] = False
    dup_status_text.empty()

    new_files = [uf for uf in uploaded_files if not dup_flags[uf.name]]
    br.duplicates = [uf.name for uf in uploaded_files if dup_flags[uf.name]]

    # ── Sequential ingest ─────────────────────────────────────────────────
    t_start = time.perf_counter()
    if new_files:
        progress = st.progress(0.0, text="인제스트 준비 중…")
        for i, uf in enumerate(new_files):
            progress.progress(
                i / len(new_files),
                text=f"처리 중 ({i + 1}/{len(new_files)}): {uf.name}",
            )
            try:
                res = ingestor.ingest(uf.getvalue(), uf.name, active)
                br.successes.append((uf.name, res))
            except Exception as exc:
                br.failures.append((uf.name, str(exc)))
        progress.progress(1.0, text="완료")

    br.elapsed_s = time.perf_counter() - t_start
    return br


def _render_batch_result(br: _BatchResult) -> None:
    st.subheader("📊 인제스트 결과")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("총 선택 파일", f"{br.total}개")
    c2.metric("신규 인제스트", f"{len(br.successes)}개")
    c3.metric("중복으로 건너뜀", f"{len(br.duplicates)}개")
    c4.metric("총 처리 시간", f"{br.elapsed_s:.1f}s")

    if br.successes:
        st.success(f"✅ 신규 인제스트 완료 — {len(br.successes)}개 파일")
        for filename, res in br.successes:
            with st.expander(f"{_file_icon(filename)} {filename} — {res.chunk_count}청크"):
                cc1, cc2, cc3, cc4 = st.columns(4)
                cc1.metric("청크 수", f"{res.chunk_count}개")
                cc2.metric("파일 타입", res.file_type.upper())
                cc3.metric("총 텍스트", f"{res.total_text_length:,}")
                cc4.metric("평균 청크", f"{res.avg_chunk_length:.0f}")
                st.caption(f"doc_id: `{res.doc_id}` · dim: {res.vector_dim}")

    if br.duplicates:
        st.warning(f"⚠️ 중복으로 건너뜀 — {len(br.duplicates)}개 파일")
        for fname in br.duplicates:
            st.markdown(f"- {_file_icon(fname)} `{fname}`")

    if br.failures:
        st.error(f"❌ 인제스트 실패 — {len(br.failures)}개 파일")
        for fname, err in br.failures:
            st.markdown(f"- **{_file_icon(fname)} {fname}**: {err}")


def render_upload_tab() -> None:
    active = _active_index()
    supported = sorted(FILE_LOADER_REGISTRY.keys())

    if not active:
        st.warning("사이드바에서 Index를 선택하거나 새로 생성하세요.")
        return

    st.subheader(f"문서 추가 → **`{active}`**")
    st.caption(
        f"지원 포맷: {', '.join(f.upper() for f in supported)} — "
        "업로드된 문서는 동일 Index에 누적 저장됩니다."
    )

    # Live chunk count
    try:
        qd = _make_qdrant(active)
        if qd.exists():
            st.info(f"현재 Index 내 벡터: **{qd.count():,}개**")
    except Exception:
        pass

    # ── Previous batch result (persists across rerun) ─────────────────────
    if _BATCH_RESULT_KEY in st.session_state:
        _render_batch_result(st.session_state[_BATCH_RESULT_KEY])
        st.divider()

    # ── File uploader ─────────────────────────────────────────────────────
    uploaded_files = st.file_uploader(
        "문서를 업로드하세요",
        type=supported,
        accept_multiple_files=True,
        key="doc_uploader",
    )

    if not uploaded_files:
        return

    # ── Selected file list ────────────────────────────────────────────────
    st.markdown(f"**선택된 파일 {len(uploaded_files)}개**")
    for uf in uploaded_files:
        st.markdown(f"- {_file_icon(uf.name)} **{uf.name}** &nbsp;·&nbsp; {uf.size / 1024:.1f} KB")

    # ── Batch ingest button ───────────────────────────────────────────────
    if st.button(
        f"📥 선택한 모든 파일 인제스트 ({len(uploaded_files)}개)",
        type="primary",
        key="batch_ingest_btn",
    ):
        st.session_state.pop(_BATCH_RESULT_KEY, None)
        br = _run_batch_ingest(uploaded_files, active)
        st.session_state[_BATCH_RESULT_KEY] = br
        st.rerun()


# ---------------------------------------------------------------------------
# Tab: Search
# ---------------------------------------------------------------------------


def render_search_tab() -> None:
    active = _active_index()

    if not active:
        st.warning("사이드바에서 Index를 선택하세요.")
        return

    registry = _get_registry()
    kb = registry.get(active)

    st.subheader(f"검색 — Index: **`{active}`**")
    if kb:
        st.caption(
            f"문서 {len(kb.documents)}개 · "
            f"총 청크 {kb.total_chunks():,}개 · "
            f"dim {kb.vector_dim or '?'}"
        )

    # Query input
    query = st.text_input(
        "질문 입력",
        placeholder="이 Knowledge Base에 대한 질문을 입력하세요",
        key="search_query_input",
    )

    # Controls
    c1, c2, c3 = st.columns([3, 1, 1])
    top_k = c1.slider("Top-K", 1, 20, 5, key="search_top_k")
    use_qdrant = c2.checkbox("🟦 Qdrant", value=True, key="search_use_qdrant")
    use_opensearch = c3.checkbox("🟧 OpenSearch", value=True, key="search_use_os")

    # Document filter within the index
    filter_doc_id: str | None = None
    if kb and kb.documents:
        doc_options = {"전체 문서 검색": None} | {
            f"{d.source_file} ({d.doc_id[:20]}…)": d.doc_id
            for d in kb.documents
        }
        sel_doc = st.selectbox(
            "문서 필터 (선택적)",
            list(doc_options.keys()),
            key="search_doc_filter",
        )
        filter_doc_id = doc_options[sel_doc]

    if filter_doc_id:
        st.caption(f"필터 적용: doc_id = `{filter_doc_id}`")

    search_btn = st.button(
        "🔍 검색",
        type="primary",
        disabled=not query.strip() or (not use_qdrant and not use_opensearch),
        key="search_btn",
    )

    if search_btn and query.strip():
        with st.spinner("검색 중…"):
            outcomes = run_search(
                query.strip(), active, top_k, use_qdrant, use_opensearch, filter_doc_id
            )
        st.session_state["search_outcomes"] = outcomes
        st.session_state["search_query"] = query.strip()

    if "search_outcomes" not in st.session_state:
        return

    outcomes = st.session_state["search_outcomes"]
    st.markdown(f"### `{st.session_state.get('search_query', '')}`")

    qd_ids = [r.id for r in outcomes["qdrant"].results] if "qdrant" in outcomes else []
    os_ids = [r.id for r in outcomes["opensearch"].results] if "opensearch" in outcomes else []
    common_ids: set[str] = set(qd_ids) & set(os_ids)

    col_qd, col_os = st.columns(2)
    with col_qd:
        st.subheader("🟦 Qdrant")
        render_store_column(outcomes.get("qdrant"), "Qdrant", common_ids)
    with col_os:
        st.subheader("🟧 OpenSearch")
        render_store_column(outcomes.get("opensearch"), "OpenSearch", common_ids)

    if "qdrant" in outcomes and "opensearch" in outcomes:
        render_comparison_panel(outcomes)


# ---------------------------------------------------------------------------
# Tab: Compare (two indices)
# ---------------------------------------------------------------------------


def render_compare_tab() -> None:
    st.subheader("Index 간 비교")
    st.caption("동일한 질문으로 두 KB Index를 비교합니다.")

    registry = _get_registry()
    indices = registry.all_indices()

    if len(indices) < 2:
        st.info("비교하려면 최소 2개 Index가 필요합니다.")
        return

    names = [kb.index_name for kb in indices]
    active = _active_index() or names[0]

    col_a, col_b = st.columns(2)
    sel_a = col_a.selectbox(
        "Index A",
        names,
        index=names.index(active) if active in names else 0,
        key="cmp_sel_a",
    )
    default_b_idx = 1 if names[0] == sel_a else 0
    sel_b = col_b.selectbox(
        "Index B",
        names,
        index=default_b_idx,
        key="cmp_sel_b",
    )

    query = st.text_input(
        "비교 질문", placeholder="두 Index에 동일하게 적용할 질문", key="cmp_query"
    )

    c1, c2, c3 = st.columns([3, 1, 1])
    top_k = c1.slider("Top-K", 1, 10, 5, key="cmp_top_k")
    use_qdrant = c2.checkbox("🟦 Qdrant", value=True, key="cmp_use_qdrant")
    use_opensearch = c3.checkbox("🟧 OpenSearch", value=True, key="cmp_use_os")

    cmp_btn = st.button(
        "⚖️ 비교 검색",
        type="primary",
        disabled=(
            not query.strip()
            or sel_a == sel_b
            or (not use_qdrant and not use_opensearch)
        ),
        key="cmp_btn",
    )

    if cmp_btn and query.strip():
        with st.spinner("두 Index 동시 검색 중…"):
            out_a = run_search(query.strip(), sel_a, top_k, use_qdrant, use_opensearch)
            out_b = run_search(query.strip(), sel_b, top_k, use_qdrant, use_opensearch)
        st.session_state.update(
            cmp_out_a=out_a, cmp_out_b=out_b,
            cmp_query=query.strip(), cmp_sel_a=sel_a, cmp_sel_b=sel_b,
        )

    if "cmp_out_a" not in st.session_state:
        return

    out_a = st.session_state["cmp_out_a"]
    out_b = st.session_state["cmp_out_b"]
    la = st.session_state.get("cmp_sel_a", sel_a)
    lb = st.session_state.get("cmp_sel_b", sel_b)

    st.markdown(f"### `{st.session_state.get('cmp_query', '')}`")

    # Cross-index Jaccard
    st.subheader("📊 Index 간 결과 중복도")
    cross_cols: list[tuple[str, str]] = []
    for key, label in [("qdrant", "Qdrant"), ("opensearch", "OpenSearch")]:
        oa, ob = out_a.get(key), out_b.get(key)
        if oa and ob and not oa.error and not ob.error:
            cross_cols.append((label, _jaccard([r.id for r in oa.results],
                                               [r.id for r in ob.results])))
    if cross_cols:
        cols = st.columns(len(cross_cols))
        for col, (label, j) in zip(cols, cross_cols):
            col.metric(f"{label}: A∩B Jaccard", f"{j:.3f}",
                       help="0 = 두 Index에서 완전히 다른 결과")
    st.divider()

    left, right = st.columns(2)

    def _doc_results_col(outcomes: dict[str, SearchOutcome], label: str) -> None:
        qd_ids = [r.id for r in outcomes.get("qdrant", SearchOutcome("", [], 0)).results]
        os_ids = [r.id for r in outcomes.get("opensearch", SearchOutcome("", [], 0)).results]
        common = set(qd_ids) & set(os_ids)
        tab_qd, tab_os = st.tabs(["🟦 Qdrant", "🟧 OpenSearch"])
        with tab_qd:
            render_store_column(outcomes.get("qdrant"), "Qdrant", common)
        with tab_os:
            render_store_column(outcomes.get("opensearch"), "OpenSearch", common)

    with left:
        st.markdown(f"#### 🗄️ Index A: `{la}`")
        _doc_results_col(out_a, la)

    with right:
        st.markdown(f"#### 🗄️ Index B: `{lb}`")
        _doc_results_col(out_b, lb)


# ---------------------------------------------------------------------------
# Tab: Manage
# ---------------------------------------------------------------------------


def render_manage_tab() -> None:
    st.subheader("Index 관리")

    registry = _get_registry()
    indices = registry.all_indices()

    if not indices:
        st.info("등록된 Index가 없습니다. 사이드바에서 새로 생성하세요.")
        return

    st.caption(f"총 **{len(indices)}**개 Index")

    for kb in indices:
        with st.container(border=True):
            col_info, col_del = st.columns([6, 1])

            with col_info:
                st.markdown(f"### 🗄️ `{kb.index_name}`")

                m1, m2, m3 = st.columns(3)
                m1.metric("문서", f"{len(kb.documents)}개")
                m2.metric("총 청크", f"{kb.total_chunks():,}개")
                m3.metric("Dim", str(kb.vector_dim or "?"))
                st.caption(f"생성: {kb.created_at[:19]}")

                # Live store counts
                cnt_cols = st.columns(2)
                with cnt_cols[0]:
                    try:
                        store = _make_qdrant(kb.index_name)
                        if store.exists():
                            cnt_cols[0].success(f"🟦 Qdrant: {store.count():,} vectors")
                        else:
                            cnt_cols[0].warning("🟦 Qdrant: 컬렉션 없음")
                    except Exception as e:
                        cnt_cols[0].error(f"🟦 Qdrant 연결 실패")

                with cnt_cols[1]:
                    try:
                        store = _make_opensearch(kb.index_name)
                        if store.exists():
                            cnt_cols[1].success(f"🟧 OS: {store.count():,} vectors")
                        else:
                            cnt_cols[1].warning("🟧 OpenSearch: 인덱스 없음")
                    except Exception as e:
                        cnt_cols[1].error(f"🟧 OpenSearch 연결 실패")

                # Document list
                if kb.documents:
                    with st.expander(f"📄 문서 목록 ({len(kb.documents)}개)"):
                        for doc in kb.documents:
                            icon = _FILE_TYPE_ICON.get(doc.file_type, "📄")
                            st.markdown(
                                f"**{icon} {doc.source_file}** — "
                                f"`{doc.doc_id}` · "
                                f"{doc.chunk_count}청크 · "
                                f"{doc.upload_time[:19]}"
                            )

            with col_del:
                st.write("")
                if st.button(
                    "🗑 삭제",
                    key=f"del_idx_{kb.index_name}",
                    type="secondary",
                    help=f"'{kb.index_name}' 삭제 (Qdrant + OpenSearch 동시)",
                ):
                    with st.spinner(f"'{kb.index_name}' 삭제 중…"):
                        try:
                            ingestor = _load_ingestor()
                            ingestor.delete_index(kb.index_name)
                            if _active_index() == kb.index_name:
                                st.session_state.pop("active_index_name", None)
                            st.success(f"'{kb.index_name}' 삭제 완료")
                            st.rerun()
                        except Exception as exc:
                            st.error(f"삭제 실패: {exc}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    st.title("🗄️ Vector DB Benchmark")
    st.caption(
        "KB Index 기반 · 문서 다중 누적 저장 · "
        "**Qdrant** vs **OpenSearch** 실시간 시맨틱 검색"
    )

    render_sidebar()

    tab_upload, tab_search, tab_compare, tab_manage = st.tabs([
        "📤 문서 추가",
        "🔍 검색",
        "⚖️ Index 비교",
        "🗂 관리",
    ])

    with tab_upload:
        render_upload_tab()
    with tab_search:
        render_search_tab()
    with tab_compare:
        render_compare_tab()
    with tab_manage:
        render_manage_tab()


if __name__ == "__main__":
    main()
