"""
BaseLoader: abstract interface every document loader must implement.

All loaders:
  • Accept raw bytes + original filename
  • Return a list of LoadedSection objects (one per page / heading block / table)
  • Apply the shared ``clean_text()`` normalisation before returning

Adding a new format:
  1. Create ``ingestion/loaders/{fmt}_loader.py`` with a ``FmtLoader(BaseLoader)``
  2. Register it in ``ingestion/loaders/__init__.py :: FILE_LOADER_REGISTRY``
"""
from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class LoadedSection:
    """One logical section of a document."""

    text: str
    page_or_section: str   # "3" for page 3, "Introduction" for a DOCX heading


class BaseLoader(ABC):
    """Abstract document loader.  Subclasses implement ``load_bytes()``."""

    @abstractmethod
    def load_bytes(self, file_bytes: bytes, filename: str) -> list[LoadedSection]:
        """
        Load a document from raw bytes and return per-section text.

        Parameters
        ----------
        file_bytes:
            Raw bytes of the document (e.g. from Streamlit's ``UploadedFile.getvalue()``).
        filename:
            Original filename; used for error messages and stem extraction.

        Returns
        -------
        list[LoadedSection]
            Non-empty list; at least one section with non-blank text.

        Raises
        ------
        ValueError
            If the document contains no extractable text.
        RuntimeError
            If the file cannot be parsed at all.
        """

    # ------------------------------------------------------------------
    # Shared text cleaning (called by all concrete loaders)
    # ------------------------------------------------------------------

    @staticmethod
    def clean_text(text: str) -> str:
        """
        Normalise whitespace and remove noise common to all formats.

        Operations (in order):
          1. Normalise line endings → ``\\n``
          2. Collapse 3+ blank lines → single blank line
          3. Collapse runs of spaces/tabs → single space
          4. Replace non-breaking / Unicode whitespace with regular space
          5. Strip leading/trailing whitespace
        """
        # Line ending normalisation
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        # Collapse excess blank lines
        text = re.sub(r"\n{3,}", "\n\n", text)
        # Collapse inline spaces/tabs
        text = re.sub(r"[ \t]+", " ", text)
        # Replace Unicode whitespace variants with regular space
        text = re.sub(r"[\u00a0\u2000-\u200b\u202f\u3000]+", " ", text)
        return text.strip()
