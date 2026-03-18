"""
Loader registry: maps file extensions → concrete BaseLoader classes.
Each loader now accepts an optional `strategy` parameter.

Usage::

    from ingestion.loaders import get_loader

    loader = get_loader("pdf", strategy="pdfplumber")
    sections = loader.load_bytes(data, "report.pdf")

    loader = get_loader("docx", strategy="paragraph")
"""
from __future__ import annotations

from ingestion.loaders.base_loader import BaseLoader, LoadedSection
from ingestion.loaders.csv_loader import CsvLoader
from ingestion.loaders.docx_loader import DocxLoader
from ingestion.loaders.html_loader import HtmlLoader
from ingestion.loaders.md_loader import MarkdownLoader
from ingestion.loaders.pdf_loader import PDFLoader
from ingestion.loaders.txt_loader import TxtLoader

# Registry: extension → (loader class, default strategy)
FILE_LOADER_REGISTRY: dict[str, type[BaseLoader]] = {
    "pdf": PDFLoader,
    "docx": DocxLoader,
    "txt": TxtLoader,
    "md": MarkdownLoader,
    "html": HtmlLoader,
    "htm": HtmlLoader,
    "csv": CsvLoader,
}

# Default strategy per file type
_DEFAULT_STRATEGY: dict[str, str] = {
    "pdf": "page",
    "docx": "heading",
    "txt": "fulltext",
    "md": "heading",
    "html": "tag",
    "htm": "tag",
    "csv": "row_batch",
}


class UnsupportedFileTypeError(ValueError):
    """Raised when get_loader() is called with an unregistered extension."""


def get_loader(file_type: str, strategy: str | None = None) -> BaseLoader:
    """
    Return an instantiated loader for *file_type* with the given *strategy*.

    Parameters
    ----------
    file_type:
        File extension without the dot (e.g. ``"pdf"``, ``"docx"``).
    strategy:
        Extraction strategy. If None, the default for the file type is used.

    Raises
    ------
    UnsupportedFileTypeError
        If *file_type* is not in ``FILE_LOADER_REGISTRY``.
    """
    key = file_type.lower().lstrip(".")
    cls = FILE_LOADER_REGISTRY.get(key)
    if cls is None:
        supported = ", ".join(sorted(FILE_LOADER_REGISTRY))
        raise UnsupportedFileTypeError(
            f"Unsupported file type: '{file_type}'. Supported: {supported}"
        )

    resolved_strategy = strategy or _DEFAULT_STRATEGY.get(key)

    # CsvLoader uses rows_per_section instead of strategy string
    if key == "csv":
        return CsvLoader()

    if resolved_strategy is not None:
        return cls(strategy=resolved_strategy)  # type: ignore[call-arg]
    return cls()


__all__ = [
    "BaseLoader",
    "LoadedSection",
    "PDFLoader",
    "DocxLoader",
    "TxtLoader",
    "MarkdownLoader",
    "HtmlLoader",
    "CsvLoader",
    "FILE_LOADER_REGISTRY",
    "UnsupportedFileTypeError",
    "get_loader",
]
