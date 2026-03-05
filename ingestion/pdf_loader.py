"""
PDF loading: discovers all .pdf files in a directory and extracts
per-page text with basic whitespace normalisation.

This module owns the ``RawDocument`` dataclass — the shared intermediate
representation consumed by ``TextChunker`` regardless of file format.

Usage::

    loader = PDFLoader("./data/pdfs")
    raw_docs = loader.load_all()          # list[RawDocument]

    # Or load a single PDF from bytes (e.g. Streamlit upload):
    raw_doc = PDFLoader.from_bytes(pdf_bytes, "report.pdf")
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

try:
    from pypdf import PdfReader
except ImportError as exc:
    raise ImportError("pypdf is required. Run: uv add pypdf") from exc

from ingestion.loaders.base_loader import LoadedSection
from ingestion.loaders.pdf_loader import PDFLoader as _NewPDFLoader


@dataclass
class RawDocument:
    """
    Format-agnostic intermediate representation of a loaded document.

    Fields
    ------
    filename:
        File stem without extension (e.g. ``"report_2024"``).
    filepath:
        Full path to the source file (or a ``Path(filename)`` when loaded
        from bytes).
    sections:
        List of text blocks — one per page (PDF) or heading section (DOCX).
    section_labels:
        Parallel list of human-readable labels for each section
        (``"1"``, ``"2"`` … for PDF pages; heading text for DOCX).
    file_type:
        Lowercase extension without dot: ``"pdf"``, ``"docx"``, etc.
    source_file:
        Original filename with extension (e.g. ``"report_2024.pdf"``).
    """

    filename: str              # stem, e.g. "report_2024"
    filepath: Path             # full path or Path(original_filename)
    sections: list[str]        # text per section
    section_labels: list[str]  # label per section
    file_type: str = "pdf"
    source_file: str = ""      # original filename with extension

    # ------------------------------------------------------------------
    # Backward-compatibility aliases
    # ------------------------------------------------------------------

    @property
    def pages(self) -> list[str]:
        """Alias for ``sections`` (backward compatibility)."""
        return self.sections

    @property
    def total_pages(self) -> int:
        return len(self.sections)

    @property
    def full_text(self) -> str:
        """All sections joined by a blank line."""
        return "\n\n".join(self.sections)


def _sections_to_raw_doc(
    sections: list[LoadedSection],
    filename: str,
    file_type: str,
) -> RawDocument:
    """Convert loader output to a ``RawDocument``."""
    stem = Path(filename).stem
    return RawDocument(
        filename=stem,
        filepath=Path(filename),
        sections=[s.text for s in sections],
        section_labels=[s.page_or_section for s in sections],
        file_type=file_type,
        source_file=filename,
    )


class PDFLoader:
    """
    Loads every .pdf from a given directory and returns RawDocument objects.

    Parameters
    ----------
    pdf_dir:
        Directory to scan for .pdf files (e.g. ``./data/pdfs``).
    """

    def __init__(self, pdf_dir: str | Path) -> None:
        self._pdf_dir = Path(pdf_dir)
        self._inner = _NewPDFLoader()

    def load_all(self) -> list[RawDocument]:
        """
        Return one RawDocument per successfully loaded PDF.

        Raises
        ------
        FileNotFoundError
            If the directory does not exist or contains no .pdf files.
        RuntimeError
            If every PDF fails to load or has no extractable text.
        """
        if not self._pdf_dir.exists():
            raise FileNotFoundError(
                f"PDF directory not found: {self._pdf_dir}\n"
                "Create it and place PDF files inside:\n"
                "  mkdir -p data/pdfs"
            )

        pdf_files = sorted(self._pdf_dir.glob("*.pdf"))
        if not pdf_files:
            raise FileNotFoundError(
                f"No .pdf files found in {self._pdf_dir}.\n"
                "Place PDF documents there before running."
            )

        print(f"[PDFLoader] Found {len(pdf_files)} PDF file(s) in {self._pdf_dir}")
        docs: list[RawDocument] = []
        for path in pdf_files:
            doc = self._load_one(path)
            if doc is not None:
                docs.append(doc)

        if not docs:
            raise RuntimeError(
                "All PDF files failed to load or contained no extractable text."
            )
        return docs

    @staticmethod
    def load_bytes(pdf_bytes: bytes, filename: str) -> RawDocument:
        """
        Load a PDF from raw bytes (e.g. from Streamlit file uploader).

        Raises
        ------
        RuntimeError
            If the PDF cannot be read or contains no extractable text.
        """
        loader = _NewPDFLoader()
        sections = loader.load_bytes(pdf_bytes, filename)
        doc = _sections_to_raw_doc(sections, filename, "pdf")
        print(
            f"[PDFLoader] Loaded '{filename}' from bytes "
            f"({len(sections)} pages with text)"
        )
        return doc

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _load_one(self, path: Path) -> RawDocument | None:
        """Extract text from a single PDF; return None on failure."""
        try:
            with open(path, "rb") as f:
                pdf_bytes = f.read()
            sections = self._inner.load_bytes(pdf_bytes, path.name)
            doc = _sections_to_raw_doc(sections, path.name, "pdf")
            # Override filepath with the real path
            doc = RawDocument(
                filename=doc.filename,
                filepath=path,
                sections=doc.sections,
                section_labels=doc.section_labels,
                file_type=doc.file_type,
                source_file=doc.source_file,
            )
            print(f"[PDFLoader] Loaded '{path.name}' ({len(sections)} pages with text)")
            return doc
        except Exception as exc:
            print(f"[PDFLoader] ERROR loading '{path.name}': {exc} — skipped")
            return None
