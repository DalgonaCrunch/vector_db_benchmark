"""
DOCX loader: extracts text from Word documents using python-docx.

Document structure mapping:
  • Heading paragraphs (Heading 1–9, Title) → section boundaries
    The heading text becomes the ``page_or_section`` label.
  • Body paragraphs below each heading → concatenated as one section's text
  • Tables → extracted row-by-row as a separate section ("Table N")

If a DOCX has no headings, all paragraphs are grouped into a single
"Document" section.
"""
from __future__ import annotations

import io

try:
    from docx import Document as _DocxDocument
    from docx.oxml.ns import qn
    from docx.table import Table
    from docx.text.paragraph import Paragraph
except ImportError as exc:
    raise ImportError(
        "python-docx is required. Run: uv add python-docx"
    ) from exc

from ingestion.loaders.base_loader import BaseLoader, LoadedSection

_HEADING_STYLES = frozenset(
    f"heading {i}" for i in range(1, 10)
) | {"title", "subtitle"}


def _is_heading(para: Paragraph) -> bool:
    style_name = (para.style.name or "").lower()
    return style_name in _HEADING_STYLES


def _table_to_text(table: Table) -> str:
    """Render a table as pipe-separated rows."""
    rows: list[str] = []
    for row in table.rows:
        cells = [c.text.strip() for c in row.cells if c.text.strip()]
        if cells:
            rows.append(" | ".join(cells))
    return "\n".join(rows)


def _iter_body_items(doc: "_DocxDocument") -> list[Paragraph | Table]:
    """
    Yield paragraphs and tables in document order.

    python-docx exposes ``doc.paragraphs`` and ``doc.tables`` separately;
    to preserve order we iterate the XML body directly.
    """
    items: list[Paragraph | Table] = []
    for child in doc.element.body.iterchildren():
        if child.tag == qn("w:p"):
            items.append(Paragraph(child, doc))
        elif child.tag == qn("w:tbl"):
            items.append(Table(child, doc))
    return items


class DocxLoader(BaseLoader):
    """Extracts text section-by-section from a .docx file using python-docx."""

    def load_bytes(self, file_bytes: bytes, filename: str) -> list[LoadedSection]:
        """
        Load a DOCX from bytes; return one LoadedSection per heading block + tables.

        Parameters
        ----------
        file_bytes:
            Raw .docx bytes.
        filename:
            Original filename (used in error messages only).

        Raises
        ------
        ValueError
            If the document contains no extractable text.
        RuntimeError
            If python-docx cannot parse the file.
        """
        try:
            doc = _DocxDocument(io.BytesIO(file_bytes))
        except Exception as exc:
            raise RuntimeError(
                f"[DocxLoader] Failed to parse '{filename}': {exc}"
            ) from exc

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
                    # Truncate very long headings for label readability
                    current_heading = text[:60]
                    current_parts.clear()
                else:
                    current_parts.append(text)

            elif isinstance(item, Table):
                table_counter += 1
                table_text = self.clean_text(_table_to_text(item))
                if table_text:
                    # Inline table: attach to current section
                    current_parts.append(f"[Table {table_counter}]\n{table_text}")

        _flush()

        if not sections:
            raise ValueError(
                f"[DocxLoader] No extractable text found in '{filename}'."
            )

        return sections
