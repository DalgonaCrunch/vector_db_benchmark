"""
DOCX loader: multiple extraction strategies.

Strategies
----------
heading    — heading-based sections (default)
paragraph  — each non-empty paragraph as a section
fulltext   — entire document as one section
"""
from __future__ import annotations

import io

from ingestion.loaders.base_loader import BaseLoader, LoadedSection

_VALID_STRATEGIES = frozenset(["heading", "paragraph", "fulltext"])

try:
    from docx import Document as _DocxDocument
    from docx.oxml.ns import qn
    from docx.table import Table
    from docx.text.paragraph import Paragraph
except ImportError as exc:
    raise ImportError("python-docx 설치 필요: uv add python-docx") from exc

_HEADING_STYLES = frozenset(f"heading {i}" for i in range(1, 10)) | {"title", "subtitle"}


def _is_heading(para: "Paragraph") -> bool:
    return (para.style.name or "").lower() in _HEADING_STYLES


def _table_to_text(table: "Table") -> str:
    rows: list[str] = []
    for row in table.rows:
        cells = [c.text.strip() for c in row.cells if c.text.strip()]
        if cells:
            rows.append(" | ".join(cells))
    return "\n".join(rows)


def _iter_body_items(doc: "_DocxDocument") -> list:
    items = []
    for child in doc.element.body.iterchildren():
        if child.tag == qn("w:p"):
            items.append(Paragraph(child, doc))
        elif child.tag == qn("w:tbl"):
            items.append(Table(child, doc))
    return items


class DocxLoader(BaseLoader):
    """
    Extracts text from .docx files using a selectable strategy.

    Parameters
    ----------
    strategy:
        One of ``heading`` | ``paragraph`` | ``fulltext``. Defaults to ``heading``.
    """

    def __init__(self, strategy: str = "heading") -> None:
        if strategy not in _VALID_STRATEGIES:
            raise ValueError(
                f"Unknown DOCX strategy '{strategy}'. "
                f"Valid: {sorted(_VALID_STRATEGIES)}"
            )
        self._strategy = strategy

    def load_bytes(self, file_bytes: bytes, filename: str) -> list[LoadedSection]:
        dispatch = {
            "heading": self._load_heading,
            "paragraph": self._load_paragraph,
            "fulltext": self._load_fulltext,
        }
        return dispatch[self._strategy](file_bytes, filename)

    # ------------------------------------------------------------------

    def _open(self, file_bytes: bytes, filename: str) -> "_DocxDocument":
        try:
            return _DocxDocument(io.BytesIO(file_bytes))
        except Exception as exc:
            raise RuntimeError(
                f"[DocxLoader] '{filename}' 파싱 실패: {exc}"
            ) from exc

    def _load_heading(self, file_bytes: bytes, filename: str) -> list[LoadedSection]:
        """Heading-based section split (original behaviour)."""
        doc = self._open(file_bytes, filename)
        sections: list[LoadedSection] = []
        current_heading = "Document"
        current_parts: list[str] = []
        table_counter = 0

        def _flush() -> None:
            if current_parts:
                combined = self.clean_text("\n\n".join(current_parts))
                if combined:
                    sections.append(
                        LoadedSection(text=combined, page_or_section=current_heading)
                    )
                current_parts.clear()

        for item in _iter_body_items(doc):
            if isinstance(item, Paragraph):
                text = item.text.strip()
                if not text:
                    continue
                if _is_heading(item):
                    _flush()
                    current_heading = text[:60]
                    current_parts.clear()
                else:
                    current_parts.append(text)
            elif isinstance(item, Table):
                table_counter += 1
                tbl_text = self.clean_text(_table_to_text(item))
                if tbl_text:
                    current_parts.append(f"[Table {table_counter}]\n{tbl_text}")

        _flush()
        if not sections:
            raise ValueError(
                f"[DocxLoader/heading] '{filename}'에서 추출 가능한 텍스트가 없습니다."
            )
        return sections

    def _load_paragraph(self, file_bytes: bytes, filename: str) -> list[LoadedSection]:
        """Each non-empty paragraph as its own section."""
        doc = self._open(file_bytes, filename)
        sections: list[LoadedSection] = []
        table_counter = 0
        para_counter = 0

        for item in _iter_body_items(doc):
            if isinstance(item, Paragraph):
                text = self.clean_text(item.text)
                if text:
                    para_counter += 1
                    sections.append(
                        LoadedSection(text=text, page_or_section=f"para_{para_counter}")
                    )
            elif isinstance(item, Table):
                table_counter += 1
                tbl_text = self.clean_text(_table_to_text(item))
                if tbl_text:
                    sections.append(
                        LoadedSection(
                            text=tbl_text,
                            page_or_section=f"table_{table_counter}",
                        )
                    )

        if not sections:
            raise ValueError(
                f"[DocxLoader/paragraph] '{filename}'에서 추출 가능한 텍스트가 없습니다."
            )
        return sections

    def _load_fulltext(self, file_bytes: bytes, filename: str) -> list[LoadedSection]:
        """Merge entire document into one section."""
        heading_sections = self._load_heading(file_bytes, filename)
        full = self.clean_text("\n\n".join(s.text for s in heading_sections))
        if not full:
            raise ValueError(
                f"[DocxLoader/fulltext] '{filename}'에서 추출 가능한 텍스트가 없습니다."
            )
        return [LoadedSection(text=full, page_or_section="fulltext")]
