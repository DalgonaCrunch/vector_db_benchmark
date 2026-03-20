"""
AI-powered ingestion config advisor.

Analyzes a document description or sample text and recommends
the optimal IngestionConfig settings using an LLM (Gemini by default).

Usage::

    advisor = AIConfigAdvisor(api_key="YOUR_GEMINI_KEY")
    result = advisor.recommend("한국어 사규 문서, 조항 구조, 표 포함...")
    print(result.settings)
    print(result.reasoning)
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------

@dataclass
class ConfigRecommendation:
    """LLM-recommended ingestion settings with explanations."""

    # Loader strategies per file type
    loader_docx: str = "paragraph"
    loader_pdf: str = "page"
    loader_txt: str = "fulltext"
    loader_md: str = "heading"
    loader_html: str = "tag"

    # Chunking
    splitter_strategy: str = "recursive"
    chunk_size: int = 512
    chunk_overlap: int = 128
    sentence_count: int = 5
    sentence_overlap: int = 1
    semantic_threshold: float = 0.5

    # Embedding
    embedding_model: str = "upstage"

    # Post-processing
    prepend_section_title: bool = True
    min_chunk_length: int = 50

    # Explanations
    summary: str = ""
    reasoning: dict[str, str] = field(default_factory=dict)

    # Raw LLM output (for debugging)
    raw_response: str = ""


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
당신은 RAG 인제스트 전략 전문가입니다. 문서 특성을 분석해 최적 설정을 JSON으로만 반환합니다.

지원 옵션:
- loader_docx: heading|paragraph|fulltext
- loader_pdf: page|pdfplumber|pymupdf|fulltext|ocr
- loader_txt: fulltext|paragraph
- loader_md: heading|fulltext
- loader_html: tag|fulltext
- splitter_strategy: sliding_window|recursive|sentence|semantic
- chunk_size: 100~2000 (tokens, 한국어 1token≈2글자)
- chunk_overlap: 0~500 (chunk_size의 15~25%)
- sentence_count: 1~30 (sentence 전략용)
- sentence_overlap: 0~10 (sentence 전략용)
- semantic_threshold: 0.0~1.0 (semantic 전략용)
- embedding_model: upstage(한국어최적,4096d)|openai(1536d)|bge(한국어지원,1024d)|e5(1024d)
- prepend_section_title: true|false
- min_chunk_length: 0~500 (chars)

패턴: 법령/사규→paragraph+recursive+400~600+prepend=true / 표중심→paragraph+recursive+300~400 / 뉴스→sentence / 기술문서→heading+recursive

반드시 아래 JSON만 출력 (설명 없이):
{"loader_docx":"","loader_pdf":"","loader_txt":"","loader_md":"","loader_html":"","splitter_strategy":"","chunk_size":0,"chunk_overlap":0,"sentence_count":5,"sentence_overlap":1,"semantic_threshold":0.5,"embedding_model":"","prepend_section_title":true,"min_chunk_length":50,"summary":"한줄요약","reasoning":{"loader":"","splitter":"","chunk_size":"","embedding":"","postprocess":""}}
"""

_USER_TEMPLATE = """\
문서 특성:
{description}

위 문서에 최적화된 RAG 인제스트 설정 JSON을 반환하세요.
"""


# ---------------------------------------------------------------------------
# Advisor class
# ---------------------------------------------------------------------------

class AIConfigAdvisor:
    """
    LLM-based ingestion config advisor.

    Supports all project LLM providers (Company, Upstage, OpenAI, Claude, Gemini).
    Defaults to the Company (on-premise vLLM) provider.
    """

    def __init__(
        self,
        provider_name: str = "Company",
        model: str | None = None,
    ) -> None:
        """
        Parameters
        ----------
        provider_name:
            Provider name matching ``llm.PROVIDER_MODELS`` keys.
            Defaults to ``"Company"`` (on-premise vLLM).
        model:
            Model name. If *None*, the first model of the chosen provider is used.
        """
        from llm import get_provider, PROVIDER_MODELS

        self._provider = get_provider(provider_name)
        models = PROVIDER_MODELS.get(provider_name, [])
        self._model = model or (models[0] if models else "")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def recommend(self, description: str) -> ConfigRecommendation:
        """
        Analyze *description* and return a :class:`ConfigRecommendation`.

        Parameters
        ----------
        description:
            Free-form text describing the document — can be a sample of the
            document content, a structured description, or both.

        Returns
        -------
        ConfigRecommendation
            Populated with recommended settings and LLM reasoning.

        Raises
        ------
        ValueError
            If the LLM response cannot be parsed.
        RuntimeError
            If the LLM API call fails.
        """
        if not description.strip():
            raise ValueError("문서 설명이 비어 있습니다.")

        # Truncate very long descriptions to keep tokens reasonable
        truncated = description[:4000] if len(description) > 4000 else description

        user_prompt = _USER_TEMPLATE.format(description=truncated)

        try:
            raw = self._provider.generate(
                system=_SYSTEM_PROMPT,
                user=user_prompt,
                model=self._model,
                temperature=0.2,
                max_tokens=2048,
            )
        except Exception as exc:
            raise RuntimeError(f"LLM API 호출 실패: {exc}") from exc

        return self._parse_response(raw)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_response(raw: str) -> ConfigRecommendation:
        """Extract JSON from LLM output and build a ConfigRecommendation."""
        text = raw.strip()

        # Strategy 1: ```json ... ``` fenced block
        fence_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
        if fence_match:
            candidate = fence_match.group(1).strip()
            try:
                data: dict[str, Any] = json.loads(candidate)
                return AIConfigAdvisor._build_from_dict(data, raw)
            except json.JSONDecodeError:
                pass  # fall through to next strategy

        # Strategy 2: find the first { ... } block (greedy — handles pretty-printed JSON)
        brace_match = re.search(r"\{.*\}", text, re.DOTALL)
        if brace_match:
            candidate = brace_match.group(0).strip()
            try:
                data = json.loads(candidate)
                return AIConfigAdvisor._build_from_dict(data, raw)
            except json.JSONDecodeError:
                pass

        # Strategy 3: try the entire stripped text as JSON
        try:
            data = json.loads(text)
            return AIConfigAdvisor._build_from_dict(data, raw)
        except json.JSONDecodeError:
            pass

        # Strategy 4: partial JSON recovery — truncated responses (no closing })
        # Close any unclosed braces and try again
        partial = text
        # Count open vs close braces
        opens = partial.count("{")
        closes = partial.count("}")
        if opens > closes:
            partial = partial.rstrip().rstrip(",") + "}" * (opens - closes)
            try:
                data = json.loads(partial)
                return AIConfigAdvisor._build_from_dict(data, raw)
            except json.JSONDecodeError:
                pass

        # Strategy 5: regex key extraction as last resort
        data = AIConfigAdvisor._extract_kv_fallback(text)
        if data:
            return AIConfigAdvisor._build_from_dict(data, raw)

        raise ValueError(
            f"LLM 응답을 JSON으로 파싱할 수 없습니다.\n"
            f"원본 응답 (앞 600자):\n{raw[:600]}"
        )

    @staticmethod
    def _extract_kv_fallback(text: str) -> dict[str, Any]:
        """Last-resort: extract key:value pairs via regex from malformed JSON."""
        data: dict[str, Any] = {}
        # String values
        for m in re.finditer(r'"(\w+)"\s*:\s*"([^"]*)"', text):
            data[m.group(1)] = m.group(2)
        # Numeric values
        for m in re.finditer(r'"(\w+)"\s*:\s*(-?\d+(?:\.\d+)?)', text):
            key, val = m.group(1), m.group(2)
            data[key] = float(val) if "." in val else int(val)
        # Boolean values
        for m in re.finditer(r'"(\w+)"\s*:\s*(true|false)', text):
            data[m.group(1)] = m.group(2) == "true"
        return data

    @staticmethod
    def _build_from_dict(data: dict[str, Any], raw: str) -> "ConfigRecommendation":

        # ── Validate / clamp values ─────────────────────────────────────
        valid_splitters = {"sliding_window", "recursive", "sentence", "semantic"}
        valid_emb = {"upstage", "openai", "bge", "e5"}
        valid_docx = {"heading", "paragraph", "fulltext"}
        valid_pdf = {"page", "pdfplumber", "pymupdf", "fulltext", "ocr"}
        valid_txt = {"fulltext", "paragraph"}
        valid_md = {"heading", "fulltext"}
        valid_html = {"tag", "fulltext"}

        def _pick(key: str, valid: set[str], default: str) -> str:
            v = str(data.get(key, default))
            return v if v in valid else default

        def _clamp_int(key: str, lo: int, hi: int, default: int) -> int:
            try:
                return max(lo, min(hi, int(data.get(key, default))))
            except (TypeError, ValueError):
                return default

        def _clamp_float(key: str, lo: float, hi: float, default: float) -> float:
            try:
                return max(lo, min(hi, float(data.get(key, default))))
            except (TypeError, ValueError):
                return default

        chunk_size = _clamp_int("chunk_size", 100, 2000, 512)
        chunk_overlap = _clamp_int("chunk_overlap", 0, 500, 128)
        # Ensure overlap < size
        chunk_overlap = min(chunk_overlap, chunk_size - 1)

        reasoning_raw = data.get("reasoning", {})
        if not isinstance(reasoning_raw, dict):
            reasoning_raw = {}

        return ConfigRecommendation(
            loader_docx=_pick("loader_docx", valid_docx, "paragraph"),
            loader_pdf=_pick("loader_pdf", valid_pdf, "page"),
            loader_txt=_pick("loader_txt", valid_txt, "fulltext"),
            loader_md=_pick("loader_md", valid_md, "heading"),
            loader_html=_pick("loader_html", valid_html, "tag"),
            splitter_strategy=_pick("splitter_strategy", valid_splitters, "recursive"),
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            sentence_count=_clamp_int("sentence_count", 1, 30, 5),
            sentence_overlap=_clamp_int("sentence_overlap", 0, 10, 1),
            semantic_threshold=_clamp_float("semantic_threshold", 0.0, 1.0, 0.5),
            embedding_model=_pick("embedding_model", valid_emb, "upstage"),
            prepend_section_title=bool(data.get("prepend_section_title", True)),
            min_chunk_length=_clamp_int("min_chunk_length", 0, 500, 50),
            summary=str(data.get("summary", "")),
            reasoning={k: str(v) for k, v in reasoning_raw.items()},
            raw_response=raw,
        )


# ---------------------------------------------------------------------------
# Convenience: apply recommendation → Streamlit session_state
# ---------------------------------------------------------------------------

def apply_to_session_state(rec: ConfigRecommendation) -> None:
    """
    Write all recommended values into ``st.session_state`` so that
    the ingestion option widgets pick them up on the next render.

    Must be called *before* the corresponding widgets are rendered
    (i.e., before the expander that contains the sliders/selectboxes).
    """
    import streamlit as st
    from ingestion.ingestion_config import MODEL_DIMENSIONS

    st.session_state["ing_emb_model"] = rec.embedding_model
    st.session_state["ing_emb_dim"] = MODEL_DIMENSIONS.get(rec.embedding_model, 1024)
    st.session_state["ing_splitter_strategy"] = rec.splitter_strategy
    st.session_state["ing_chunk_size"] = rec.chunk_size
    st.session_state["ing_chunk_overlap"] = rec.chunk_overlap
    st.session_state["ing_sentence_count"] = rec.sentence_count
    st.session_state["ing_sentence_overlap"] = rec.sentence_overlap
    st.session_state["ing_semantic_threshold"] = rec.semantic_threshold
    st.session_state["ing_prepend_title"] = rec.prepend_section_title
    st.session_state["ing_min_chunk_len"] = rec.min_chunk_length

    # Loader strategies per file type
    st.session_state["ing_loader_pdf"] = rec.loader_pdf
    st.session_state["ing_loader_docx"] = rec.loader_docx
    st.session_state["ing_loader_txt"] = rec.loader_txt
    st.session_state["ing_loader_md"] = rec.loader_md
    st.session_state["ing_loader_html"] = rec.loader_html
