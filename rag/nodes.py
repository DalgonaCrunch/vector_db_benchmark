"""
RAG pipeline node functions — LangGraph logic reimplemented as plain Python.

Adapted from Webcash RuleAndGuide LangGraph project:
  node.generate_query                              → transform_query()
  node.documents_grade                             → grade_documents()
  node.grade_generation_v_documents_and_question   → check_hallucination()

LangGraph dependency is completely removed.
Each node is a pure function: (inputs) → output, no state graph required.

Prompts are written in Korean to match the existing chatbot system language.
"""
from __future__ import annotations

import json
import re

from llm.base_provider import BaseLLMProvider
from stores.base_store import SearchResult


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_TRANSFORM_QUERY_SYSTEM = """\
당신은 문서 검색 쿼리를 개선하는 전문가입니다.

주어진 원본 질문을 분석하여 관련 문서를 더 잘 검색할 수 있도록
더 구체적이고 명확한 검색 쿼리로 재작성하세요.

규칙:
- 핵심 개념과 키워드를 유지하세요
- 불필요한 조사·어미·대화체를 제거하세요
- 의미는 변경하지 마세요
- 개선된 쿼리 텍스트만 반환하고, 설명이나 추가 텍스트는 포함하지 마세요"""

_TRANSFORM_QUERY_USER = """\
원본 질문: {question}

개선된 검색 쿼리:"""

_GRADE_DOCUMENT_SYSTEM = """\
당신은 검색된 문서가 사용자 질문과 관련성이 있는지 평가하는 그레이더입니다.

평가 기준:
- 문서에 질문과 관련된 키워드, 개념, 또는 의미가 포함되어 있으면 관련 있음
- 판정이 엄격할 필요는 없습니다. 오류가 있는 검색 결과만 걸러내는 것이 목표입니다

반환 형식: JSON 한 줄
  관련 있음 → {"score": "yes"}
  관련 없음 → {"score": "no"}

설명 없이 JSON만 반환하세요."""

_GRADE_DOCUMENT_USER = """\
사용자 질문: {question}

검색된 문서:
{document}

이 문서가 질문과 관련이 있습니까? JSON으로 답하세요:"""

_HALLUCINATION_SYSTEM = """\
당신은 LLM이 생성한 답변이 주어진 참고 문서에 근거하는지 평가하는 그레이더입니다.

평가 기준:
1. 사실 근거성: 답변 내용이 참고 문서에 있는 정보만을 사용했는가?
2. 질문 응답성: 답변이 실제로 원본 질문에 답하고 있는가?

두 기준을 모두 충족해야 "yes"를 반환하세요.

반환 형식: JSON 한 줄
  {"score": "yes", "reason": "근거 설명"}
  {"score": "no",  "reason": "문제점 설명"}

설명 없이 JSON만 반환하세요."""

_HALLUCINATION_USER = """\
[참고 문서]
{documents}

[생성된 답변]
{generation}

[원본 질문]
{question}

답변이 참고 문서에 근거하며 질문에 정확히 답하고 있습니까? JSON으로 답하세요:"""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _parse_json_score(raw: str) -> str:
    """
    Extract the ``score`` field from an LLM JSON response.

    Attempts JSON parse first, then regex fallback, then keyword scan.
    Always returns a lowercase string ("yes" | "no").
    """
    raw = raw.strip()

    # 1) Direct JSON parse
    try:
        data = json.loads(raw)
        return str(data.get("score", "yes")).lower().strip()
    except json.JSONDecodeError:
        pass

    # 2) Regex: "score": "yes/no"
    m = re.search(r'"score"\s*:\s*"(\w+)"', raw, re.IGNORECASE)
    if m:
        return m.group(1).lower()

    # 3) Keyword scan (last resort)
    lower = raw.lower()
    if "\"no\"" in lower or ": no" in lower or "'no'" in lower:
        return "no"
    return "yes"


def _parse_json_reason(raw: str) -> str:
    """Extract the ``reason`` field from an LLM JSON response."""
    raw = raw.strip()
    try:
        data = json.loads(raw)
        return str(data.get("reason", "")).strip()
    except Exception:
        pass
    m = re.search(r'"reason"\s*:\s*"([^"]*)"', raw, re.IGNORECASE)
    if m:
        return m.group(1)
    return ""


def _docs_to_context_str(docs: list[SearchResult], max_chars: int = 4000) -> str:
    """Format docs into a numbered context string, capped at *max_chars*."""
    parts: list[str] = []
    total = 0
    for i, doc in enumerate(docs, 1):
        snippet = doc.text[:800].strip()
        entry = f"[문서 {i}] (score={doc.score:.3f})\n{snippet}"
        if total + len(entry) > max_chars:
            break
        parts.append(entry)
        total += len(entry)
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Node: transform_query
# ---------------------------------------------------------------------------


def transform_query(
    query: str,
    provider: BaseLLMProvider,
    model: str,
) -> str:
    """
    Rewrite *query* to improve retrieval recall.

    Analogous to ``node.generate_query`` in the LangGraph project.

    Parameters
    ----------
    query:
        Original user question.
    provider:
        Instantiated LLM provider.
    model:
        Model identifier string (e.g. ``"solar-pro"``, ``"gpt-4o-mini"``).

    Returns
    -------
    str
        Improved query string.  Falls back to the original on any error.
    """
    try:
        improved = provider.generate(
            system=_TRANSFORM_QUERY_SYSTEM,
            user=_TRANSFORM_QUERY_USER.format(question=query),
            model=model,
            temperature=0.0,
            max_tokens=256,
        )
        improved = improved.strip()
        return improved if improved else query
    except Exception as exc:
        print(f"[transform_query] Warning: {exc} — returning original query")
        return query


# ---------------------------------------------------------------------------
# Node: grade_documents
# ---------------------------------------------------------------------------


def grade_documents(
    query: str,
    docs: list[SearchResult],
    provider: BaseLLMProvider,
    model: str,
    relevance_threshold: float = 0.0,
) -> list[SearchResult]:
    """
    Filter *docs* to retain only those relevant to *query*.

    Analogous to ``node.documents_grade`` in the LangGraph project.

    Two-stage pipeline:
      1. Vector score pre-filter  (fast, no LLM call)
      2. LLM binary relevance grading per document  (slow, accurate)

    Parameters
    ----------
    query:
        User question used as grading reference.
    docs:
        Retrieved SearchResult objects (post-rerank).
    provider:
        Instantiated LLM provider.
    model:
        Model identifier.
    relevance_threshold:
        If > 0, documents with ``score < relevance_threshold`` are removed
        before the LLM grading step (saves API calls).

    Returns
    -------
    list[SearchResult]
        Filtered list of relevant documents (may be empty).
    """
    if not docs:
        return []

    # Stage 1: fast vector-score pre-filter
    if relevance_threshold > 0.0:
        docs = [d for d in docs if d.score >= relevance_threshold]

    if not docs:
        return []

    # Stage 2: LLM binary grading
    graded: list[SearchResult] = []
    for doc in docs:
        try:
            raw = provider.generate(
                system=_GRADE_DOCUMENT_SYSTEM,
                user=_GRADE_DOCUMENT_USER.format(
                    question=query,
                    document=doc.text[:1500],
                ),
                model=model,
                temperature=0.0,
                max_tokens=64,
            )
            score = _parse_json_score(raw)
            if score == "yes":
                graded.append(doc)
        except Exception as exc:
            print(f"[grade_documents] Warning (doc '{doc.id}'): {exc} — keeping doc")
            graded.append(doc)   # fail open: keep document on error

    return graded


# ---------------------------------------------------------------------------
# Node: check_hallucination
# ---------------------------------------------------------------------------


def check_hallucination(
    query: str,
    answer: str,
    docs: list[SearchResult],
    provider: BaseLLMProvider,
    model: str,
) -> tuple[bool, str]:
    """
    Verify *answer* is grounded in *docs* and actually answers *query*.

    Analogous to ``node.grade_generation_v_documents_and_question``
    in the LangGraph project.

    Parameters
    ----------
    query:
        Original user question.
    answer:
        LLM-generated answer to evaluate.
    docs:
        Retrieved supporting documents.
    provider:
        Instantiated LLM provider.
    model:
        Model identifier.

    Returns
    -------
    tuple[bool, str]
        ``(is_grounded, reason)``

        * ``is_grounded = True``  → answer is factually grounded, safe to return
        * ``is_grounded = False`` → answer contains hallucination or is off-topic
    """
    if not docs:
        return False, "참고 문서가 없어 근거 확인 불가"

    if not answer.strip():
        return False, "빈 답변"

    context = _docs_to_context_str(docs, max_chars=3000)
    try:
        raw = provider.generate(
            system=_HALLUCINATION_SYSTEM,
            user=_HALLUCINATION_USER.format(
                documents=context,
                generation=answer[:2000],
                question=query,
            ),
            model=model,
            temperature=0.0,
            max_tokens=256,
        )
        score = _parse_json_score(raw)
        reason = _parse_json_reason(raw) or raw.strip()[:200]
        return score == "yes", reason

    except Exception as exc:
        print(f"[check_hallucination] Warning: {exc} — fail open")
        return True, f"hallucination 체크 실패 (fail open): {exc}"
