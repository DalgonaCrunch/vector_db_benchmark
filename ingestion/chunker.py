"""
RAG text chunkers: multiple splitting strategies.

Strategies
----------
sliding_window  — fixed-size window with sentence-boundary snapping (default)
recursive       — hierarchical separator split (\\n\\n → \\n → . → space)
sentence        — sentence tokenisation then N-sentence batching
semantic        — embedding cosine-similarity breakpoint detection
"""
from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable

from ingestion.pdf_loader import RawDocument

# Approximate chars per token for Korean/English mixed content.
_CHARS_PER_TOKEN: float = 2.0


def _t2c(tokens: int) -> int:
    return int(tokens * _CHARS_PER_TOKEN)


# ---------------------------------------------------------------------------
# Chunk dataclass
# ---------------------------------------------------------------------------


@dataclass
class Chunk:
    chunk_id: str
    text: str
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Base chunker
# ---------------------------------------------------------------------------


class BaseChunker(ABC):
    @abstractmethod
    def chunk_document(self, doc: RawDocument) -> list[Chunk]:
        """Produce all Chunk objects for a single RawDocument."""

    @staticmethod
    def _section_range(
        start: int,
        end: int,
        section_offsets: list[tuple[int, str]],
    ) -> str:
        if not section_offsets:
            return "?"
        first_label = last_label = section_offsets[0][1]
        for char_pos, label in section_offsets:
            if char_pos <= start:
                first_label = label
            if char_pos < end:
                last_label = label
        if first_label == last_label:
            return first_label
        return f"{first_label}→{last_label}"

    @staticmethod
    def _build_full_text(doc: RawDocument) -> tuple[str, list[tuple[int, str]]]:
        """Join sections and record character offsets per section."""
        labels = doc.section_labels
        if len(labels) != len(doc.sections):
            labels = [str(i + 1) for i in range(len(doc.sections))]

        section_offsets: list[tuple[int, str]] = []
        parts: list[str] = []
        pos = 0
        for label, section_text in zip(labels, doc.sections):
            section_offsets.append((pos, label))
            parts.append(section_text)
            pos += len(section_text) + 2
        return "\n\n".join(parts), section_offsets

    @staticmethod
    def _make_chunks(
        doc: RawDocument,
        windows: list[tuple[int, int, str]],
        section_offsets: list[tuple[int, str]],
        section_range_fn: "Callable[[int, int, list], str]",
    ) -> list[Chunk]:
        chunks: list[Chunk] = []
        for idx, (start, end, text) in enumerate(windows):
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.filename}_{idx:04d}",
                    text=text,
                    metadata={
                        "doc_id": doc.filename,
                        "source_file": doc.source_file or doc.filename,
                        "file_type": doc.file_type,
                        "section": section_range_fn(start, end, section_offsets),
                        "chunk_index": idx,
                        "filepath": str(doc.filepath),
                    },
                )
            )
        return chunks


# ---------------------------------------------------------------------------
# 1. Sliding Window (original)
# ---------------------------------------------------------------------------


class TextChunker(BaseChunker):
    """
    Sliding-window chunker with sentence-boundary snapping.
    Original chunker — kept for backward compatibility.
    """

    def __init__(
        self,
        chunk_size_tokens: int = 650,
        chunk_overlap_tokens: int = 100,
    ) -> None:
        self._chunk_size = _t2c(chunk_size_tokens)
        self._chunk_overlap = _t2c(chunk_overlap_tokens)

    def chunk_document(self, doc: RawDocument) -> list[Chunk]:
        if not doc.sections:
            return []
        full_text, section_offsets = self._build_full_text(doc)
        windows = self._sliding_window(full_text)
        return self._make_chunks(doc, windows, section_offsets, self._section_range)

    def _sliding_window(self, text: str) -> list[tuple[int, int, str]]:
        results: list[tuple[int, int, str]] = []
        text_len = len(text)
        start = 0
        while start < text_len:
            end = min(start + self._chunk_size, text_len)
            if end < text_len:
                snapped = self._find_sentence_end(text, end)
                if snapped > start + self._chunk_size // 2:
                    end = snapped
            chunk_text = text[start:end].strip()
            if chunk_text:
                results.append((start, end, chunk_text))
            if end >= text_len:
                break
            start = max(start + 1, end - self._chunk_overlap)
        return results

    @staticmethod
    def _find_sentence_end(text: str, pos: int, window: int = 300) -> int:
        lo = max(0, pos - window)
        segment = text[lo:pos]
        for pattern in (r"\n\n", r"[.!?。]\s", r"[,，]\s"):
            matches = list(re.finditer(pattern, segment))
            if matches:
                return lo + matches[-1].end()
        return pos


# ---------------------------------------------------------------------------
# 2. Recursive Character Splitter
# ---------------------------------------------------------------------------


class RecursiveChunker(BaseChunker):
    """
    Hierarchical separator split: \\n\\n → \\n → . → space.
    Similar to LangChain's RecursiveCharacterTextSplitter.
    """

    _SEPARATORS = ["\n\n", "\n", ". ", "。", " ", ""]

    def __init__(
        self,
        chunk_size_tokens: int = 650,
        chunk_overlap_tokens: int = 100,
    ) -> None:
        self._chunk_size = _t2c(chunk_size_tokens)
        self._chunk_overlap = _t2c(chunk_overlap_tokens)

    def chunk_document(self, doc: RawDocument) -> list[Chunk]:
        if not doc.sections:
            return []
        full_text, section_offsets = self._build_full_text(doc)
        splits = self._recursive_split(full_text, self._SEPARATORS)
        windows = self._merge_splits(splits, full_text)
        return self._make_chunks(doc, windows, section_offsets, self._section_range)

    def _recursive_split(self, text: str, separators: list[str]) -> list[str]:
        if not separators or len(text) <= self._chunk_size:
            return [text] if text.strip() else []
        sep = separators[0]
        rest = separators[1:]
        if sep == "":
            # character-level split
            return [
                text[i : i + self._chunk_size]
                for i in range(0, len(text), max(1, self._chunk_size - self._chunk_overlap))
                if text[i : i + self._chunk_size].strip()
            ]
        parts = text.split(sep)
        result: list[str] = []
        for part in parts:
            if len(part) <= self._chunk_size:
                if part.strip():
                    result.append(part)
            else:
                result.extend(self._recursive_split(part, rest))
        return result

    def _merge_splits(
        self, splits: list[str], full_text: str
    ) -> list[tuple[int, int, str]]:
        """Merge small splits up to chunk_size; track approximate offsets."""
        windows: list[tuple[int, int, str]] = []
        buf: list[str] = []
        buf_len = 0
        pos = 0

        for split in splits:
            slen = len(split)
            if buf_len + slen > self._chunk_size and buf:
                chunk_text = " ".join(buf).strip()
                if chunk_text:
                    windows.append((pos, pos + len(chunk_text), chunk_text))
                    # overlap: keep last portion
                    overlap_text = chunk_text[-self._chunk_overlap :]
                    buf = [overlap_text]
                    buf_len = len(overlap_text)
                    pos = pos + len(chunk_text) - self._chunk_overlap
                else:
                    buf, buf_len = [], 0
            buf.append(split)
            buf_len += slen

        if buf:
            chunk_text = " ".join(buf).strip()
            if chunk_text:
                windows.append((pos, pos + len(chunk_text), chunk_text))

        return windows


# ---------------------------------------------------------------------------
# 3. Sentence-based Chunker
# ---------------------------------------------------------------------------

_SENT_SPLIT_RE = re.compile(r"(?<=[.!?。！？])\s+")


def _split_sentences(text: str) -> list[str]:
    """Simple rule-based sentence splitter supporting Korean and English."""
    raw = _SENT_SPLIT_RE.split(text)
    return [s.strip() for s in raw if s.strip()]


class SentenceChunker(BaseChunker):
    """
    Splits text into sentences then groups *sentence_count* sentences per chunk.
    """

    def __init__(
        self,
        sentence_count: int = 5,
        sentence_overlap: int = 1,
        max_chunk_tokens: int = 2000,
    ) -> None:
        self._sentence_count = max(1, sentence_count)
        self._sentence_overlap = max(0, sentence_overlap)
        self._max_chars = _t2c(max_chunk_tokens)

    def chunk_document(self, doc: RawDocument) -> list[Chunk]:
        if not doc.sections:
            return []
        full_text, section_offsets = self._build_full_text(doc)
        sentences = _split_sentences(full_text)
        if not sentences:
            return []

        windows: list[tuple[int, int, str]] = []
        i = 0
        char_pos = 0

        while i < len(sentences):
            batch = sentences[i : i + self._sentence_count]
            chunk_text = " ".join(batch).strip()

            # Enforce hard max-token limit by trimming
            if len(chunk_text) > self._max_chars:
                chunk_text = chunk_text[: self._max_chars].rsplit(" ", 1)[0]

            if chunk_text:
                windows.append((char_pos, char_pos + len(chunk_text), chunk_text))
            char_pos += len(chunk_text) + 1
            step = max(1, self._sentence_count - self._sentence_overlap)
            i += step

        return self._make_chunks(doc, windows, section_offsets, self._section_range)


# ---------------------------------------------------------------------------
# 4. Semantic Chunker
# ---------------------------------------------------------------------------


class SemanticChunker(BaseChunker):
    """
    Splits text at points where the cosine similarity between consecutive
    sentence-group embeddings drops below *threshold*.

    Parameters
    ----------
    embed_fn:
        Callable[list[str]] → list[list[float]].
        Must be provided externally (e.g. from the configured embedder).
    threshold:
        Cosine-similarity drop threshold. Range 0–1. Default 0.5.
    window_size:
        Number of sentences to embed together as one "window". Default 3.
    max_chunk_tokens:
        Hard cap on chunk length in tokens. Default 1200.
    """

    def __init__(
        self,
        embed_fn: Callable[[list[str]], list[list[float]]],
        threshold: float = 0.5,
        window_size: int = 3,
        max_chunk_tokens: int = 1200,
    ) -> None:
        self._embed_fn = embed_fn
        self._threshold = threshold
        self._window_size = window_size
        self._max_chars = _t2c(max_chunk_tokens)

    def chunk_document(self, doc: RawDocument) -> list[Chunk]:
        if not doc.sections:
            return []
        full_text, section_offsets = self._build_full_text(doc)
        sentences = _split_sentences(full_text)
        if len(sentences) < 2:
            return self._make_chunks(
                doc,
                [(0, len(full_text), full_text.strip())],
                section_offsets,
                self._section_range,
            )

        breakpoints = self._find_breakpoints(sentences)
        windows = self._build_windows(sentences, breakpoints)
        return self._make_chunks(doc, windows, section_offsets, self._section_range)

    def _find_breakpoints(self, sentences: list[str]) -> list[int]:
        """Return indices where similarity drops; these become chunk boundaries."""
        w = self._window_size
        groups = [
            " ".join(sentences[max(0, i - w) : i + w + 1])
            for i in range(len(sentences))
        ]
        try:
            embeddings = self._embed_fn(groups)
        except Exception:
            return []  # fall back to no split

        breakpoints: list[int] = []
        for i in range(1, len(embeddings)):
            sim = _cosine(embeddings[i - 1], embeddings[i])
            if sim < self._threshold:
                breakpoints.append(i)
        return breakpoints

    def _build_windows(
        self, sentences: list[str], breakpoints: list[int]
    ) -> list[tuple[int, int, str]]:
        windows: list[tuple[int, int, str]] = []
        boundaries = [0] + breakpoints + [len(sentences)]
        char_pos = 0

        for start_i, end_i in zip(boundaries, boundaries[1:]):
            batch = sentences[start_i:end_i]
            chunk_text = " ".join(batch).strip()
            # Hard cap
            if len(chunk_text) > self._max_chars:
                chunk_text = chunk_text[: self._max_chars].rsplit(" ", 1)[0]
            if chunk_text:
                windows.append((char_pos, char_pos + len(chunk_text), chunk_text))
            char_pos += len(chunk_text) + 1

        return windows


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def get_chunker(
    strategy: str,
    chunk_size_tokens: int = 650,
    chunk_overlap_tokens: int = 100,
    sentence_count: int = 5,
    sentence_overlap: int = 1,
    semantic_threshold: float = 0.5,
    semantic_window: int = 3,
    embed_fn: "Callable[[list[str]], list[list[float]]] | None" = None,
) -> BaseChunker:
    """
    Instantiate and return a chunker for the given *strategy*.

    Parameters
    ----------
    strategy:
        One of ``sliding_window`` | ``recursive`` | ``sentence`` | ``semantic``.
    """
    if strategy == "sliding_window":
        return TextChunker(chunk_size_tokens, chunk_overlap_tokens)
    if strategy == "recursive":
        return RecursiveChunker(chunk_size_tokens, chunk_overlap_tokens)
    if strategy == "sentence":
        return SentenceChunker(sentence_count, sentence_overlap)
    if strategy == "semantic":
        if embed_fn is None:
            raise ValueError(
                "semantic 전략에는 embed_fn이 필요합니다. "
                "SemanticChunker(embed_fn=...) 또는 get_chunker(..., embed_fn=fn)을 사용하세요."
            )
        return SemanticChunker(embed_fn, semantic_threshold, semantic_window)
    raise ValueError(
        f"Unknown splitter strategy '{strategy}'. "
        "Valid: sliding_window | recursive | sentence | semantic"
    )
