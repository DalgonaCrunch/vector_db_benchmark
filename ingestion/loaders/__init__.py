"""
Loader registry: maps file extensions → concrete BaseLoader classes.

Usage::

    from ingestion.loaders import get_loader, FILE_LOADER_REGISTRY

    loader = get_loader("pdf")          # PDFLoader instance
    sections = loader.load_bytes(data, "report.pdf")

To add a new format::

    1. Create ``ingestion/loaders/{fmt}_loader.py`` with a ``FmtLoader(BaseLoader)``
    2. Add  ``"{fmt}": FmtLoader``  to ``FILE_LOADER_REGISTRY`` below
"""
from __future__ import annotations

from ingestion.loaders.base_loader import BaseLoader, LoadedSection
from ingestion.loaders.docx_loader import DocxLoader
from ingestion.loaders.pdf_loader import PDFLoader

# Registry: extension (lowercase, no dot) → loader class
FILE_LOADER_REGISTRY: dict[str, type[BaseLoader]] = {
    "pdf": PDFLoader,
    "docx": DocxLoader,
    # Future extensions:
    # "txt":      TxtLoader,
    # "html":     HtmlLoader,
    # "md":       MarkdownLoader,
}


class UnsupportedFileTypeError(ValueError):
    """Raised when ``get_loader()`` is called with an unregistered extension."""


def get_loader(file_type: str) -> BaseLoader:
    """
    Return an instantiated loader for *file_type*.

    Parameters
    ----------
    file_type:
        File extension without the dot (e.g. ``"pdf"``, ``"docx"``).
        Case-insensitive.

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
            f"Unsupported file type: '{file_type}'. "
            f"Supported formats: {supported}"
        )
    return cls()


__all__ = [
    "BaseLoader",
    "LoadedSection",
    "PDFLoader",
    "DocxLoader",
    "FILE_LOADER_REGISTRY",
    "UnsupportedFileTypeError",
    "get_loader",
]
