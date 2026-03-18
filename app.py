"""
Streamlit UI: Knowledge Base Index-based Vector DB Benchmark.

Architecture:
  KB Index  (= Qdrant collection + OpenSearch index)
    └─ Document A (pdf), Document B (docx), …  → shared vector space

Tabs:
  📤 문서 추가   — 선택된 Index에 PDF/DOCX 업로드 (full ingestion config)
  🔍 검색        — 선택된 Index에서 Qdrant vs OpenSearch 비교
  ⚖️ Index 비교  — 두 Index를 동일 쿼리로 비교
  🗂 관리        — Index 목록 · 문서 목록 · 삭제

Sidebar:
  • 기존 Index 드롭다운 선택 (변경 시 관련 상태 자동 초기화)
  • 새 Index 생성 (dimension / distance metric / OS shards+replicas 설정 포함)

Run:
    make start-opensearch && make app
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field

import streamlit as st

from config import load_config
from embedder.upstage_embedder import UpstageEmbedder
from ingestion.index_ingestor import IndexIngestor, IngestResult
from ingestion.index_registry import IndexRegistry, KBIndex, sanitize_index_name
from ingestion.ingestion_config import (
    DISTANCE_TO_SPACE_TYPE,
    LOADER_STRATEGIES,
    MODEL_DIMENSIONS,
    SPLITTER_STRATEGIES,
    IngestionConfig,
    validate_ingestion_config,
)
from ingestion.loaders import FILE_LOADER_REGISTRY
from ingestion.pkg_installer import check_strategy_deps, install_package
from ingestion.strategy_info import LOADER_STRATEGY_INFO, SPLITTER_STRATEGY_INFO
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
# Session state keys that must be reset when the active index changes
# ---------------------------------------------------------------------------
_INDEX_BOUND_KEYS = [
    "upload_batch_result",
    "search_outcomes",
    "search_query",
    "doc_uploader",
]

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
# Lightweight helpers
# ---------------------------------------------------------------------------


def _get_registry() -> IndexRegistry:
    return IndexRegistry(_REGISTRY_PATH)


def _load_ingestor() -> IndexIngestor:
    return IndexIngestor(_load_embedder(), IndexRegistry(_REGISTRY_PATH))


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


def _make_opensearch(index_name: str) -> OpenSearchVectorStore:
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
    return st.session_state.get("active_index_name")


def _set_active_index(name: str) -> None:
    prev = st.session_state.get("active_index_name")
    if prev != name:
        for key in _INDEX_BOUND_KEYS:
            st.session_state.pop(key, None)
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
            preview = full_text[:_PREVIEW_LEN]
            suffix = "…" if is_long else ""
            if query:
                st.markdown(_highlight(preview, query) + suffix)
            else:
                st.text(preview + suffix)
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
    os_out = outcomes.get("opensearch")
    if not qd or not os_out or qd.error or os_out.error:
        return

    qd_ids = [r.id for r in qd.results]
    os_ids = [r.id for r in os_out.results]
    jaccard = _jaccard(qd_ids, os_ids)
    latency_diff = os_out.latency_ms - qd.latency_ms
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
    m4.metric(
        "Avg 순위 차이",
        f"{avg_rank_diff:.2f}" if avg_rank_diff is not None else "N/A",
    )

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
# Sidebar: Index selector + advanced creation
# ---------------------------------------------------------------------------


def render_sidebar() -> None:
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
                    f"문서 {len(kb.documents)}개 · dim {kb.vector_dim or '?'}"
                )
        else:
            st.info("Index가 없습니다.\n아래에서 새로 생성하세요.")

        st.divider()

        # ── Create new index (advanced) ───────────────────────────────────
        st.subheader("➕ 새 Index 생성")

        raw_name = st.text_input(
            "Index 이름",
            key="new_index_name_input",
            placeholder="예: medical_kb, project_docs",
        )

        with st.expander("⚙️ Index 상세 옵션", expanded=False):
            idx_dim = st.number_input(
                "Embedding Dimension",
                min_value=64,
                max_value=8192,
                value=4096,
                step=64,
                key="new_idx_dim",
                help="Upstage=4096, OpenAI=1536, BGE/E5=1024",
            )
            idx_metric = st.selectbox(
                "Distance Metric",
                ["cosine", "dot", "euclidean"],
                key="new_idx_metric",
                help="코사인 유사도(권장) | 내적 | 유클리드 거리",
            )

            st.caption("🟧 OpenSearch 전용")
            idx_os_field = st.text_input(
                "Vector Field Name",
                value="vector",
                key="new_idx_os_field",
            )
            idx_os_analyzer = st.selectbox(
                "Text Analyzer",
                ["standard", "keyword"],
                key="new_idx_os_analyzer",
                help="standard: 형태소 분리 | keyword: 원문 그대로",
            )
            c_sh, c_rp = st.columns(2)
            idx_os_shards = c_sh.number_input(
                "Shards", min_value=1, max_value=10, value=1, key="new_idx_os_shards"
            )
            idx_os_replicas = c_rp.number_input(
                "Replicas", min_value=0, max_value=5, value=1, key="new_idx_os_replicas"
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
                    space_type = DISTANCE_TO_SPACE_TYPE.get(idx_metric, "cosinesimil")
                    st.success(
                        f"Index **'{sanitized}'** 생성됨  \n"
                        f"dim={idx_dim} · metric={idx_metric} · "
                        f"space={space_type}  \n"
                        f"shards={idx_os_shards} · replicas={idx_os_replicas}"
                    )
                    # Stash advanced params in session_state for the ingestor to use
                    st.session_state[f"idx_cfg_{sanitized}"] = {
                        "dimension": idx_dim,
                        "distance_metric": idx_metric,
                        "os_vector_field": idx_os_field,
                        "os_analyzer": idx_os_analyzer,
                        "os_shards": idx_os_shards,
                        "os_replicas": idx_os_replicas,
                    }
                    st.rerun()

            if raw_name and raw_name != sanitize_index_name(raw_name):
                st.caption(f"사용될 이름: **`{sanitize_index_name(raw_name)}`**")

        st.divider()
        st.caption("실행 순서")
        st.code("make start-opensearch\nmake app", language="bash")


# ---------------------------------------------------------------------------
# Tab: Upload (full IngestionConfig)
# ---------------------------------------------------------------------------

_BATCH_RESULT_KEY = "upload_batch_result"


@dataclass
class _BatchResult:
    total: int
    successes: list = field(default_factory=list)   # list[tuple[str, IngestResult]]
    skipped: list = field(default_factory=list)     # list[str]  (duplicates or policy=skip)
    failures: list = field(default_factory=list)    # list[tuple[str, str]]
    elapsed_s: float = 0.0
    config: dict = field(default_factory=dict)      # snapshot of IngestionConfig


def _build_ingestion_config() -> IngestionConfig:
    """Read all ingestion widgets from session_state and build an IngestionConfig."""
    dbs: list[str] = st.session_state.get("ing_dbs", ["qdrant", "opensearch"])
    emb_model: str = st.session_state.get("ing_emb_model", "upstage")

    return IngestionConfig(
        dbs=dbs,
        embedding_model=emb_model,
        embedding_dimension=int(
            st.session_state.get("ing_emb_dim", MODEL_DIMENSIONS.get(emb_model, 4096))
        ),
        normalize=bool(st.session_state.get("ing_normalize", False)),
        loader_strategy=st.session_state.get("ing_loader_strategy", "page"),
        splitter_strategy=st.session_state.get("ing_splitter_strategy", "sliding_window"),
        chunk_size=int(st.session_state.get("ing_chunk_size", 650)),
        chunk_overlap=int(st.session_state.get("ing_chunk_overlap", 100)),
        sentence_count=int(st.session_state.get("ing_sentence_count", 5)),
        sentence_overlap=int(st.session_state.get("ing_sentence_overlap", 1)),
        semantic_threshold=float(st.session_state.get("ing_semantic_threshold", 0.5)),
        prepend_section_title=bool(st.session_state.get("ing_prepend_title", False)),
        min_chunk_length=int(st.session_state.get("ing_min_chunk_len", 50)),
        batch_size=int(st.session_state.get("ing_batch_size", 32)),
        duplicate_policy=st.session_state.get("ing_dup_policy", "skip"),
        metadata={
            **json.loads(st.session_state.get("ing_metadata_json", "{}") or "{}"),
            # Pass per-type loader strategy map so ingestor can dispatch correctly
            "_loader_strategy_map": st.session_state.get("ing_loader_strategy_map", {}),
        },
    )


# ---------------------------------------------------------------------------
# Strategy info popup helpers
# ---------------------------------------------------------------------------


def _render_strategy_info_popup(info: dict) -> None:
    """Render a strategy's detail card inside a st.popover."""
    st.markdown(f"### {info.get('icon', '')} {info['name']}")
    st.markdown(info["description"])
    st.divider()

    col_pro, col_con = st.columns(2)
    with col_pro:
        st.markdown("**✅ 장점**")
        for p in info["pros"]:
            st.markdown(f"- {p}")
    with col_con:
        st.markdown("**❌ 단점**")
        for c in info["cons"]:
            st.markdown(f"- {c}")

    st.divider()
    st.markdown(f"**🎯 추천 상황:** {info['recommended']}")

    tags = "  ".join(f"`{t}`" for t in info.get("best_for", []))
    if tags:
        st.markdown(f"**🏷 Best for:** {tags}")

    # Dependency status
    requires = info.get("requires", [])
    import_check = info.get("import_check")
    if requires:
        st.divider()
        st.markdown("**📦 의존성**")
        available, missing = check_strategy_deps(requires, import_check)
        if available:
            st.success(f"✅ 설치됨: {', '.join(requires)}")
        else:
            for pkg in missing:
                c1, c2 = st.columns([3, 1])
                c1.warning(f"⚠️ `{pkg}` 미설치")
                if c2.button(f"설치", key=f"install_{pkg}_{info['name']}"):
                    with st.spinner(f"`{pkg}` 설치 중…"):
                        ok, msg = install_package(pkg)
                    if ok:
                        st.success(msg)
                        st.rerun()
                    else:
                        st.error(f"설치 실패: {msg}")


def _strategy_selector_with_info(
    label: str,
    options: list[str],
    info_dict: dict[str, dict],
    session_key: str,
    default_idx: int = 0,
) -> str:
    """
    Render a selectbox + ℹ️ popover button side-by-side.
    Returns the selected option string.
    """
    col_sel, col_info = st.columns([5, 1])
    current = st.session_state.get(session_key, options[default_idx])
    idx = options.index(current) if current in options else 0

    selected = col_sel.selectbox(label, options, index=idx, key=session_key)

    info = info_dict.get(selected, {})
    icon = info.get("icon", "ℹ️")
    with col_info:
        st.write("")  # spacing
        with st.popover(f"{icon} 정보"):
            if info:
                _render_strategy_info_popup(info)
            else:
                st.info("전략 정보가 없습니다.")

    # Availability badge
    requires = info.get("requires", [])
    import_check = info.get("import_check")
    if requires:
        available, missing = check_strategy_deps(requires, import_check)
        if not available:
            st.warning(
                f"⚠️ `{selected}` 전략에 필요한 패키지가 없습니다: "
                f"{', '.join(missing)}  — ℹ️ 버튼을 눌러 설치하세요."
            )

    return selected


def _run_batch_ingest(
    uploaded_files: list,
    active: str,
    config: IngestionConfig,
    debug: bool,
) -> _BatchResult:
    ingestor = _load_ingestor()
    br = _BatchResult(total=len(uploaded_files), config=dict(config))

    # ── Duplicate pre-check (only when policy == skip) ────────────────
    dup_flags: dict[str, bool] = {}
    if config["duplicate_policy"] == "skip":
        dup_text = st.empty()
        dup_text.text("중복 파일 확인 중…")
        for uf in uploaded_files:
            try:
                dup_flags[uf.name] = ingestor._is_duplicate_in_any_db(
                    uf.name, active, config["dbs"]
                )
            except Exception:
                dup_flags[uf.name] = False
        dup_text.empty()

    new_files = [uf for uf in uploaded_files if not dup_flags.get(uf.name, False)]
    br.skipped = [uf.name for uf in uploaded_files if dup_flags.get(uf.name, False)]

    # ── Sequential ingest ─────────────────────────────────────────────
    t_start = time.perf_counter()
    if new_files:
        progress = st.progress(0.0, text="인제스트 준비 중…")
        for i, uf in enumerate(new_files):
            progress.progress(
                i / len(new_files),
                text=f"처리 중 ({i + 1}/{len(new_files)}): {uf.name}",
            )
            try:
                res = ingestor.ingest_with_config(
                    uf.getvalue(), uf.name, active, config
                )
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
    c3.metric("건너뜀/중복", f"{len(br.skipped)}개")
    c4.metric("총 처리 시간", f"{br.elapsed_s:.1f}s")

    if br.successes:
        st.success(f"✅ 인제스트 완료 — {len(br.successes)}개 파일")
        for filename, res in br.successes:
            stored_label = " · ".join(
                f"{'🟦 Qdrant' if db == 'qdrant' else '🟧 OpenSearch'}"
                for db in res.stored_in
            )
            with st.expander(
                f"{_file_icon(filename)} {filename} — {res.chunk_count}청크"
            ):
                r1, r2, r3, r4, r5 = st.columns(5)
                r1.metric("청크 수", f"{res.chunk_count}개")
                r2.metric("Embedding Dim", str(res.vector_dim))
                r3.metric("파일 타입", res.file_type.upper())
                r4.metric("총 텍스트", f"{res.total_text_length:,}")
                r5.metric("처리 시간", f"{res.elapsed_s:.2f}s")

                st.caption(
                    f"📌 저장된 DB: {stored_label}  |  "
                    f"모델: `{res.embedding_model}`  |  "
                    f"doc_id: `{res.doc_id}`"
                )

    if br.skipped:
        st.warning(f"⚠️ 중복으로 건너뜀 — {len(br.skipped)}개 파일")
        for fname in br.skipped:
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
            st.info(f"현재 Qdrant 벡터: **{qd.count():,}개**")
    except Exception:
        pass

    # ── Previous batch result ─────────────────────────────────────────
    if _BATCH_RESULT_KEY in st.session_state:
        _render_batch_result(st.session_state[_BATCH_RESULT_KEY])
        st.divider()

    # ── Ingestion options ─────────────────────────────────────────────
    with st.expander("⚙️ Ingestion 옵션", expanded=True):

        # ── DB selection ──────────────────────────────────────────────
        st.markdown("**💾 저장 DB 선택**")
        dbs = st.multiselect(
            "저장 대상 DB",
            ["qdrant", "opensearch"],
            default=["qdrant", "opensearch"],
            key="ing_dbs",
            help="선택한 DB에만 저장합니다.",
        )

        st.divider()

        # ── Embedding model ───────────────────────────────────────────
        st.markdown("**🧠 Embedding 모델**")
        ec1, ec2 = st.columns([2, 1])
        emb_model = ec1.selectbox(
            "Embedding Model",
            ["upstage", "openai", "bge", "e5"],
            key="ing_emb_model",
            help=(
                "upstage: Solar 4096-dim | "
                "openai: text-embedding-3-small 1536-dim | "
                "bge/e5: sentence-transformers 1024-dim (로컬 GPU)"
            ),
        )
        default_dim = MODEL_DIMENSIONS.get(emb_model, 1024)
        ec2.number_input(
            "Embedding Dimension",
            min_value=64,
            max_value=8192,
            value=default_dim,
            step=64,
            key="ing_emb_dim",
        )
        st.checkbox(
            "L2 Normalize",
            value=False,
            key="ing_normalize",
            help="벡터를 단위 구면에 사영합니다. dot product = cosine similarity",
        )

        st.divider()

        # ── Loader strategy ───────────────────────────────────────────
        st.markdown("**📂 Document Loader 전략**")
        st.caption(
            "업로드할 파일 형식에 맞는 로딩 전략을 선택하세요. "
            "ℹ️ 버튼으로 각 전략의 장단점을 확인할 수 있습니다."
        )

        loader_tabs = st.tabs(["📕 PDF", "📘 DOCX", "📄 TXT", "📝 MD", "🌐 HTML"])
        _loader_strategy_per_type: dict[str, str] = {}

        for tab, (ftype, tab_label) in zip(
            loader_tabs,
            [("pdf", "PDF"), ("docx", "DOCX"), ("txt", "TXT"), ("md", "MD"), ("html", "HTML")],
        ):
            with tab:
                type_strategies = LOADER_STRATEGIES.get(ftype, ["fulltext"])
                type_info = LOADER_STRATEGY_INFO.get(ftype, {})
                sel = _strategy_selector_with_info(
                    label=f"{tab_label} Loader 전략",
                    options=type_strategies,
                    info_dict=type_info,
                    session_key=f"ing_loader_{ftype}",
                    default_idx=0,
                )
                _loader_strategy_per_type[ftype] = sel

        # Determine effective loader_strategy from uploaded file type (or use pdf default)
        # Stored as a JSON map in session_state; read by _build_ingestion_config
        st.session_state["ing_loader_strategy_map"] = _loader_strategy_per_type

        # Use the most common or default for the config key
        # (index_ingestor dispatches by file_type; we pass the per-type map via metadata)
        # For now, store the PDF strategy as default (it drives the config field)
        st.session_state["ing_loader_strategy"] = _loader_strategy_per_type.get("pdf", "page")

        st.divider()

        # ── Splitter strategy ─────────────────────────────────────────
        st.markdown("**✂️ Splitter (Chunking) 전략**")
        st.caption(
            "텍스트 분할 방식을 선택하세요. "
            "ℹ️ 버튼으로 각 전략의 상세 설명을 확인할 수 있습니다."
        )

        splitter_strategy = _strategy_selector_with_info(
            label="Splitter 전략",
            options=SPLITTER_STRATEGIES,
            info_dict=SPLITTER_STRATEGY_INFO,
            session_key="ing_splitter_strategy",
            default_idx=0,
        )

        # Show relevant params per strategy
        if splitter_strategy in ("sliding_window", "recursive"):
            ck1, ck2 = st.columns(2)
            ck1.slider(
                "Chunk Size (tokens)",
                min_value=100, max_value=2000, value=650, step=50,
                key="ing_chunk_size",
                help="1 token ≈ 2자 (한영 혼합 기준)",
            )
            ck2.slider(
                "Chunk Overlap (tokens)",
                min_value=0, max_value=500, value=100, step=25,
                key="ing_chunk_overlap",
            )
            cs = st.session_state.get("ing_chunk_size", 650)
            co = st.session_state.get("ing_chunk_overlap", 100)
            if co >= cs:
                st.warning("chunk_overlap이 chunk_size 이상입니다.")

        elif splitter_strategy == "sentence":
            sc1, sc2 = st.columns(2)
            sc1.number_input(
                "Sentences per Chunk",
                min_value=1, max_value=30, value=5,
                key="ing_sentence_count",
                help="청크당 포함할 문장 수",
            )
            sc2.number_input(
                "Sentence Overlap",
                min_value=0, max_value=10, value=1,
                key="ing_sentence_overlap",
                help="이전 청크와 공유할 문장 수",
            )

        elif splitter_strategy == "semantic":
            sem1, sem2 = st.columns(2)
            sem1.slider(
                "Semantic Threshold",
                min_value=0.0, max_value=1.0, value=0.5, step=0.05,
                key="ing_semantic_threshold",
                help="코사인 유사도가 이 값 이하로 떨어지면 청크 경계로 판단 (낮을수록 덜 분할)",
            )
            sem2.number_input(
                "Window Size",
                min_value=1, max_value=10, value=3,
                key="ing_semantic_window",
                help="유사도 비교 시 앞뒤로 포함할 문장 수",
            )
            st.info(
                "🧠 Semantic Split은 임베딩 API를 추가로 호출합니다. "
                "처리 시간이 늘어나고 비용이 발생할 수 있습니다."
            )

        st.divider()

        # ── Post-processing ───────────────────────────────────────────
        st.markdown("**🔧 청크 후처리**")
        pp1, pp2 = st.columns(2)
        pp1.checkbox(
            "섹션 제목 prefix 추가",
            value=False,
            key="ing_prepend_title",
            help="각 청크 앞에 '[섹션명] 본문…' 형태로 섹션 제목을 추가합니다. 검색 정확도 향상에 도움.",
        )
        pp2.number_input(
            "최소 청크 길이 (chars)",
            min_value=0, max_value=500, value=50,
            key="ing_min_chunk_len",
            help="이 길이보다 짧은 청크는 제거합니다.",
        )

        st.divider()

        # ── Batch + Duplicate ─────────────────────────────────────────
        st.markdown("**⚡ Batch / 중복 처리**")
        bd1, bd2 = st.columns(2)
        bd1.number_input(
            "Batch Size",
            min_value=1, max_value=256, value=32,
            key="ing_batch_size",
        )
        bd2.selectbox(
            "Duplicate Policy",
            ["skip", "overwrite"],
            key="ing_dup_policy",
            help="skip: 이미 있으면 건너뜀 | overwrite: 기존 데이터 삭제 후 재저장",
        )

        st.divider()

        # ── Extra metadata ────────────────────────────────────────────
        st.markdown("**🏷 추가 메타데이터 (선택)**")
        metadata_json = st.text_area(
            "Metadata JSON",
            value="{}",
            height=80,
            key="ing_metadata_json",
            placeholder='{"project": "demo", "version": "1.0"}',
        )
        try:
            json.loads(metadata_json or "{}")
        except json.JSONDecodeError:
            st.error("올바른 JSON 형식이 아닙니다.")

        st.divider()

        # ── Debug options ─────────────────────────────────────────────
        st.markdown("**🐛 Debug 옵션**")
        d1, d2, d3 = st.columns(3)
        show_config = d1.checkbox("Config 출력", key="ing_debug_config")
        show_chunks = d2.checkbox("Raw Chunk 출력", key="ing_debug_chunks")
        show_vectors = d3.checkbox("Vector Shape 출력", key="ing_debug_vectors")

    # ── Config preview / validation ───────────────────────────────────
    config = _build_ingestion_config()
    errors = validate_ingestion_config(config)
    for err in errors:
        st.warning(err)

    if show_config:
        st.json(dict(config))

    # ── File uploader ─────────────────────────────────────────────────
    uploaded_files = st.file_uploader(
        "문서를 업로드하세요",
        type=supported,
        accept_multiple_files=True,
        key="doc_uploader",
    )

    if not uploaded_files:
        return

    st.markdown(f"**선택된 파일 {len(uploaded_files)}개**")
    for uf in uploaded_files:
        st.markdown(
            f"- {_file_icon(uf.name)} **{uf.name}** &nbsp;·&nbsp; {uf.size / 1024:.1f} KB"
        )

    # ── Debug: chunk + vector preview ────────────────────────────────
    if (show_chunks or show_vectors) and uploaded_files:
        if st.button("🔬 첫 번째 파일 청크/벡터 미리보기", key="debug_preview_btn"):
            uf = uploaded_files[0]
            try:
                from pathlib import Path

                from ingestion.chunker import get_chunker
                from ingestion.loaders import get_loader
                from ingestion.pdf_loader import _sections_to_raw_doc

                ft = Path(uf.name).suffix.lstrip(".").lower()
                loader_strat = st.session_state.get(
                    f"ing_loader_{ft}", config.get("loader_strategy")
                )
                loader = get_loader(ft, strategy=loader_strat)
                sections = loader.load_bytes(uf.getvalue(), uf.name)
                raw_doc = _sections_to_raw_doc(sections, uf.name, ft)

                chunker = get_chunker(
                    strategy=config.get("splitter_strategy", "sliding_window"),
                    chunk_size_tokens=config.get("chunk_size", 650),
                    chunk_overlap_tokens=config.get("chunk_overlap", 100),
                    sentence_count=config.get("sentence_count", 5),
                    sentence_overlap=config.get("sentence_overlap", 1),
                    semantic_threshold=config.get("semantic_threshold", 0.5),
                )
                chunks = chunker.chunk_document(raw_doc)

                st.markdown(
                    f"**🗂 Loader:** `{loader_strat}` &nbsp;|&nbsp; "
                    f"**✂️ Splitter:** `{config.get('splitter_strategy')}` &nbsp;|&nbsp; "
                    f"**총 {len(chunks)}개 청크**"
                )

                if show_chunks:
                    for i, c in enumerate(chunks[:5]):
                        with st.expander(
                            f"Chunk #{i} — {len(c.text)}chars  |  "
                            f"section: {c.metadata.get('section', '?')}"
                        ):
                            st.text(c.text[:600])
                            st.json(c.metadata)
                    if len(chunks) > 5:
                        st.caption(f"… 나머지 {len(chunks) - 5}개 청크는 생략됩니다.")

                if show_vectors and chunks:
                    from embedder.embedding_router import embed_passages
                    sample_vecs = embed_passages([chunks[0].text], config)
                    st.info(
                        f"Vector shape: ({len(sample_vecs)}, {len(sample_vecs[0])})  |  "
                        f"첫 5개 값: {[round(v, 4) for v in sample_vecs[0][:5]]}"
                    )
            except Exception as exc:
                st.error(f"미리보기 실패: {exc}")

    # ── Batch ingest button ───────────────────────────────────────────
    ingest_disabled = bool(errors) or not dbs
    if st.button(
        f"📥 선택한 모든 파일 인제스트 ({len(uploaded_files)}개)",
        type="primary",
        key="batch_ingest_btn",
        disabled=ingest_disabled,
    ):
        st.session_state.pop(_BATCH_RESULT_KEY, None)
        br = _run_batch_ingest(
            uploaded_files,
            active,
            config,
            debug=(show_config or show_chunks or show_vectors),
        )
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

    query = st.text_input(
        "질문 입력",
        placeholder="이 Knowledge Base에 대한 질문을 입력하세요",
        key="search_query_input",
    )

    c1, c2, c3 = st.columns([3, 1, 1])
    top_k = c1.slider("Top-K", 1, 20, 5, key="search_top_k")
    use_qdrant = c2.checkbox("🟦 Qdrant", value=True, key="search_use_qdrant")
    use_opensearch = c3.checkbox("🟧 OpenSearch", value=True, key="search_use_os")

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

    st.subheader("📊 Index 간 결과 중복도")
    cross_cols: list[tuple[str, float]] = []
    for key, label in [("qdrant", "Qdrant"), ("opensearch", "OpenSearch")]:
        oa, ob = out_a.get(key), out_b.get(key)
        if oa and ob and not oa.error and not ob.error:
            cross_cols.append(
                (label, _jaccard([r.id for r in oa.results], [r.id for r in ob.results]))
            )
    if cross_cols:
        cols = st.columns(len(cross_cols))
        for col, (label, j) in zip(cols, cross_cols):
            col.metric(f"{label}: A∩B Jaccard", f"{j:.3f}")
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

                cnt_cols = st.columns(2)
                with cnt_cols[0]:
                    try:
                        store = _make_qdrant(kb.index_name)
                        if store.exists():
                            cnt_cols[0].success(f"🟦 Qdrant: {store.count():,} vectors")
                        else:
                            cnt_cols[0].warning("🟦 Qdrant: 컬렉션 없음")
                    except Exception:
                        cnt_cols[0].error("🟦 Qdrant 연결 실패")

                with cnt_cols[1]:
                    try:
                        store = _make_opensearch(kb.index_name)
                        if store.exists():
                            cnt_cols[1].success(f"🟧 OS: {store.count():,} vectors")
                        else:
                            cnt_cols[1].warning("🟧 OpenSearch: 인덱스 없음")
                    except Exception:
                        cnt_cols[1].error("🟧 OpenSearch 연결 실패")

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
        "KB Index 기반 · 다중 DB / 다중 임베딩 모델 · "
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
