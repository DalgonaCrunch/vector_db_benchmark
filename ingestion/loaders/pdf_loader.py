"""
PDF loader: multiple extraction strategies.

Strategies
----------
page        — page-by-page using pypdf (default)
pdfplumber  — layout-aware using pdfplumber
pymupdf     — high-speed using PyMuPDF/fitz
fulltext    — entire document merged into one section using pypdf
ocr         — scanned/image PDF via pytesseract + pdf2image
"""
from __future__ import annotations

import io

from ingestion.loaders.base_loader import BaseLoader, LoadedSection

_VALID_STRATEGIES = frozenset(["page", "pdfplumber", "pymupdf", "fulltext", "ocr"])


class PDFLoader(BaseLoader):
    """
    Extracts text from PDF files using a selectable strategy.

    Parameters
    ----------
    strategy:
        One of ``page`` | ``pdfplumber`` | ``pymupdf`` | ``fulltext`` | ``ocr``.
        Defaults to ``page``.
    ocr_lang:
        Tesseract language code, e.g. ``"kor+eng"``. Used only when
        ``strategy="ocr"``.
    """

    def __init__(self, strategy: str = "page", ocr_lang: str = "kor+eng") -> None:
        if strategy not in _VALID_STRATEGIES:
            raise ValueError(
                f"Unknown PDF strategy '{strategy}'. "
                f"Valid: {sorted(_VALID_STRATEGIES)}"
            )
        self._strategy = strategy
        self._ocr_lang = ocr_lang

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def load_bytes(self, file_bytes: bytes, filename: str) -> list[LoadedSection]:
        dispatch = {
            "page": self._load_page,
            "pdfplumber": self._load_pdfplumber,
            "pymupdf": self._load_pymupdf,
            "fulltext": self._load_fulltext,
            "ocr": self._load_ocr,
        }
        return dispatch[self._strategy](file_bytes, filename)

    # ------------------------------------------------------------------
    # Strategy implementations
    # ------------------------------------------------------------------

    def _load_page(self, file_bytes: bytes, filename: str) -> list[LoadedSection]:
        """Page-by-page extraction using pypdf."""
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise ImportError("pypdf 설치 필요: uv add pypdf") from exc

        try:
            reader = PdfReader(io.BytesIO(file_bytes))
        except Exception as exc:
            raise RuntimeError(f"[PDFLoader/page] '{filename}' 파싱 실패: {exc}") from exc

        sections: list[LoadedSection] = []
        for page_num, page in enumerate(reader.pages, start=1):
            raw = page.extract_text() or ""
            cleaned = self.clean_text(raw)
            if cleaned:
                sections.append(LoadedSection(text=cleaned, page_or_section=str(page_num)))

        if not sections:
            raise ValueError(
                f"[PDFLoader/page] '{filename}'에서 추출 가능한 텍스트가 없습니다. "
                "스캔 PDF라면 OCR 전략을 사용하세요."
            )
        return sections

    def _load_pdfplumber(self, file_bytes: bytes, filename: str) -> list[LoadedSection]:
        """Layout-aware extraction using pdfplumber."""
        try:
            import pdfplumber
        except ImportError as exc:
            raise ImportError("pdfplumber 설치 필요: uv add pdfplumber") from exc

        sections: list[LoadedSection] = []
        try:
            with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                for page_num, page in enumerate(pdf.pages, start=1):
                    raw = page.extract_text(x_tolerance=3, y_tolerance=3) or ""
                    # Extract tables if present
                    tables = page.extract_tables()
                    table_texts: list[str] = []
                    for tbl in tables:
                        rows = [
                            " | ".join(str(cell or "").strip() for cell in row if cell)
                            for row in tbl
                            if any(cell for cell in row)
                        ]
                        if rows:
                            table_texts.append("\n".join(rows))
                    combined = raw
                    if table_texts:
                        combined += "\n\n" + "\n\n".join(table_texts)
                    cleaned = self.clean_text(combined)
                    if cleaned:
                        sections.append(
                            LoadedSection(text=cleaned, page_or_section=str(page_num))
                        )
        except Exception as exc:
            raise RuntimeError(
                f"[PDFLoader/pdfplumber] '{filename}' 파싱 실패: {exc}"
            ) from exc

        if not sections:
            raise ValueError(
                f"[PDFLoader/pdfplumber] '{filename}'에서 추출 가능한 텍스트가 없습니다."
            )
        return sections

    def _load_pymupdf(self, file_bytes: bytes, filename: str) -> list[LoadedSection]:
        """High-speed extraction using PyMuPDF."""
        try:
            import fitz  # type: ignore[import]
        except ImportError as exc:
            raise ImportError("pymupdf 설치 필요: uv add pymupdf") from exc

        sections: list[LoadedSection] = []
        try:
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            for page_num in range(len(doc)):
                page = doc[page_num]
                raw = page.get_text("text") or ""
                cleaned = self.clean_text(raw)
                if cleaned:
                    sections.append(
                        LoadedSection(text=cleaned, page_or_section=str(page_num + 1))
                    )
            doc.close()
        except Exception as exc:
            raise RuntimeError(
                f"[PDFLoader/pymupdf] '{filename}' 파싱 실패: {exc}"
            ) from exc

        if not sections:
            raise ValueError(
                f"[PDFLoader/pymupdf] '{filename}'에서 추출 가능한 텍스트가 없습니다."
            )
        return sections

    def _load_fulltext(self, file_bytes: bytes, filename: str) -> list[LoadedSection]:
        """Merge all pages into a single section using pypdf."""
        page_sections = self._load_page(file_bytes, filename)
        full = self.clean_text("\n\n".join(s.text for s in page_sections))
        if not full:
            raise ValueError(
                f"[PDFLoader/fulltext] '{filename}'에서 추출 가능한 텍스트가 없습니다."
            )
        return [LoadedSection(text=full, page_or_section="fulltext")]

    def _load_ocr(self, file_bytes: bytes, filename: str) -> list[LoadedSection]:
        """OCR extraction for scanned/image PDFs."""
        try:
            from pdf2image import convert_from_bytes  # type: ignore[import]
            import pytesseract  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "OCR에는 pdf2image와 pytesseract가 필요합니다: "
                "uv add pdf2image pytesseract  (Tesseract 엔진도 별도 설치 필요)"
            ) from exc

        sections: list[LoadedSection] = []
        try:
            images = convert_from_bytes(file_bytes, dpi=200)
            for page_num, img in enumerate(images, start=1):
                raw = pytesseract.image_to_string(img, lang=self._ocr_lang)
                cleaned = self.clean_text(raw)
                if cleaned:
                    sections.append(
                        LoadedSection(text=cleaned, page_or_section=str(page_num))
                    )
        except Exception as exc:
            raise RuntimeError(
                f"[PDFLoader/ocr] '{filename}' OCR 실패: {exc}"
            ) from exc

        if not sections:
            raise ValueError(
                f"[PDFLoader/ocr] '{filename}'에서 OCR 텍스트를 추출하지 못했습니다."
            )
        return sections
