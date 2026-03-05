"""
PDF loader: extracts per-page text using pypdf.

Each page of the PDF becomes one LoadedSection; blank pages are skipped.
"""
from __future__ import annotations

import io

try:
    from pypdf import PdfReader
except ImportError as exc:
    raise ImportError("pypdf is required. Run: uv add pypdf") from exc

from ingestion.loaders.base_loader import BaseLoader, LoadedSection


class PDFLoader(BaseLoader):
    """Extracts text page-by-page from a PDF using pypdf."""

    def load_bytes(self, file_bytes: bytes, filename: str) -> list[LoadedSection]:
        """
        Load PDF from bytes; return one LoadedSection per non-blank page.

        Parameters
        ----------
        file_bytes:
            Raw PDF bytes.
        filename:
            Original filename (used in error messages only).

        Raises
        ------
        ValueError
            If no page contains extractable text.
        RuntimeError
            If pypdf cannot parse the file.
        """
        try:
            reader = PdfReader(io.BytesIO(file_bytes))
        except Exception as exc:
            raise RuntimeError(
                f"[PDFLoader] Failed to parse '{filename}': {exc}"
            ) from exc

        sections: list[LoadedSection] = []
        for page_num, page in enumerate(reader.pages, start=1):
            raw = page.extract_text() or ""
            cleaned = self.clean_text(raw)
            if cleaned:
                sections.append(
                    LoadedSection(text=cleaned, page_or_section=str(page_num))
                )

        if not sections:
            raise ValueError(
                f"[PDFLoader] No extractable text found in '{filename}'. "
                "The PDF may be image-based (scanned) and require OCR."
            )

        return sections
