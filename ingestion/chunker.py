"""
RAG text chunker: sliding-window with sentence-boundary snapping.

Token count is approximated by character count using a chars-per-token
factor calibrated for Korean/English mixed text.

Usage::

    chunker = TextChunker(chunk_size_tokens=650, chunk_overlap_tokens=100)
    all_chunks: list[Chunk] = []
    for raw_doc in raw_docs:
        all_chunks.extend(chunker.chunk_document(raw_doc))
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from ingestion.pdf_loader import RawDocument

# Approximate chars per token for Korean/English mixed content.
# Korean: ~1.5–2 chars/token  |  English: ~4 chars/token
# 2.0 is a conservative midpoint that avoids over-splitting Korean text.
_CHARS_PER_TOKEN: float = 2.0


def _t2c(tokens: int) -> int:
    """Convert approximate token count → character count."""
    return int(tokens * _CHARS_PER_TOKEN)


@dataclass
class Chunk:
    """
    One text chunk ready for embedding and vector-store insertion.

    Fields
    ------
    chunk_id:
        Globally unique identifier: ``"{doc_filename}_{chunk_index:04d}"``.
    text:
        The chunk text to be embedded.
    metadata:
        Key-value pairs for payload filtering in vector stores::

            {
                "doc_id":       "report_2024_20240101_120000",
                "source_file":  "report_2024.pdf",
                "file_type":    "pdf",              # or "docx"
                "section":      "4-5",              # page range or heading name
                "chunk_index":  3,
                "filepath":     "./data/pdfs/report_2024.pdf",
            }

        ``upload_time`` is added by DocIngestor after chunking.
    """

    chunk_id: str
    text: str
    metadata: dict = field(default_factory=dict)


class TextChunker:
    """
    Splits a RawDocument into overlapping Chunk objects.

    Strategy
    --------
    * Sliding window over the concatenated full text.
    * Window size and overlap are specified in *approximate* tokens,
      converted to characters via ``_CHARS_PER_TOKEN``.
    * The right edge of each window is snapped to the nearest sentence
      boundary (paragraph break → sentence-ending punctuation → comma)
      within a look-back window of 300 chars.
    * Section-range metadata is tracked via character offset boundaries
      recorded at section-join points (pages for PDF, headings for DOCX).

    Parameters
    ----------
    chunk_size_tokens:
        Target chunk size in approximate tokens.
        Default 650 sits in the middle of the 500–800 range.
    chunk_overlap_tokens:
        Overlap in approximate tokens. Default 100.
    """

    def __init__(
        self,
        chunk_size_tokens: int = 650,
        chunk_overlap_tokens: int = 100,
    ) -> None:
        self._chunk_size = _t2c(chunk_size_tokens)       # chars
        self._chunk_overlap = _t2c(chunk_overlap_tokens)  # chars

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def chunk_document(self, doc: RawDocument) -> list[Chunk]:
        """
        Produce all Chunk objects for a single RawDocument.

        Works with any file type as long as ``doc.sections`` and
        ``doc.section_labels`` are set (they are, for all loaders).

        Returns an empty list only if the document has no text.
        """
        if not doc.sections:
            return []

        # Ensure section_labels is parallel to sections (fallback: numbers)
        labels = doc.section_labels
        if len(labels) != len(doc.sections):
            labels = [str(i + 1) for i in range(len(doc.sections))]

        # Build combined text; record character offset at which each section starts
        section_offsets: list[tuple[int, str]] = []   # (char_start, label)
        parts: list[str] = []
        pos = 0
        for label, section_text in zip(labels, doc.sections):
            section_offsets.append((pos, label))
            parts.append(section_text)
            pos += len(section_text) + 2   # +2 for the "\n\n" join separator

        full_text = "\n\n".join(parts)
        windows = self._sliding_window(full_text)

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
                        "section": self._section_range(start, end, section_offsets),
                        "chunk_index": idx,
                        "filepath": str(doc.filepath),
                    },
                )
            )
        return chunks

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _sliding_window(
        self, text: str
    ) -> list[tuple[int, int, str]]:
        """
        Generate ``(start, end, chunk_text)`` triples.

        The right boundary is snapped to the nearest sentence end found
        within the last 300 characters of the window.
        """
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
        """
        Return the character offset of the last sentence-ending boundary
        in ``text[pos - window : pos]``.

        Priority: paragraph break > sentence-ending punctuation > comma.
        Falls back to ``pos`` if no boundary is found.
        """
        lo = max(0, pos - window)
        segment = text[lo:pos]
        for pattern in (r"\n\n", r"[.!?。]\s", r"[,，]\s"):
            matches = list(re.finditer(pattern, segment))
            if matches:
                return lo + matches[-1].end()
        return pos

    @staticmethod
    def _section_range(
        start: int,
        end: int,
        section_offsets: list[tuple[int, str]],
    ) -> str:
        """
        Return the label (or label range) of the sections covering [start, end).

        For PDF: returns ``"4"`` or ``"4-5"`` (page numbers).
        For DOCX: returns ``"Introduction"`` or ``"Introduction→Methods"``.
        """
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
