"""TXT loader: plain text file extraction strategies."""
from __future__ import annotations

import re

from ingestion.loaders.base_loader import BaseLoader, LoadedSection

_VALID_STRATEGIES = frozenset(["fulltext", "paragraph"])


class TxtLoader(BaseLoader):
    """
    Loads plain-text (.txt) files.

    Parameters
    ----------
    strategy:
        ``fulltext`` — entire file as one section (default).
        ``paragraph`` — split on blank lines.
    encoding:
        File encoding. Defaults to ``utf-8``.
    """

    def __init__(self, strategy: str = "fulltext", encoding: str = "utf-8") -> None:
        if strategy not in _VALID_STRATEGIES:
            raise ValueError(
                f"Unknown TXT strategy '{strategy}'. "
                f"Valid: {sorted(_VALID_STRATEGIES)}"
            )
        self._strategy = strategy
        self._encoding = encoding

    def load_bytes(self, file_bytes: bytes, filename: str) -> list[LoadedSection]:
        try:
            text = file_bytes.decode(self._encoding, errors="replace")
        except Exception as exc:
            raise RuntimeError(
                f"[TxtLoader] '{filename}' 디코딩 실패: {exc}"
            ) from exc

        if self._strategy == "fulltext":
            return self._load_fulltext(text, filename)
        return self._load_paragraph(text, filename)

    def _load_fulltext(self, text: str, filename: str) -> list[LoadedSection]:
        cleaned = self.clean_text(text)
        if not cleaned:
            raise ValueError(f"[TxtLoader/fulltext] '{filename}'이 비어 있습니다.")
        return [LoadedSection(text=cleaned, page_or_section="fulltext")]

    def _load_paragraph(self, text: str, filename: str) -> list[LoadedSection]:
        raw_paras = re.split(r"\n{2,}", text)
        sections: list[LoadedSection] = []
        for i, para in enumerate(raw_paras, start=1):
            cleaned = self.clean_text(para)
            if cleaned:
                sections.append(LoadedSection(text=cleaned, page_or_section=f"para_{i}"))
        if not sections:
            raise ValueError(f"[TxtLoader/paragraph] '{filename}'에서 텍스트를 찾을 수 없습니다.")
        return sections
