"""Markdown loader: heading-based or full-text extraction."""
from __future__ import annotations

import re

from ingestion.loaders.base_loader import BaseLoader, LoadedSection

_VALID_STRATEGIES = frozenset(["heading", "fulltext"])
_HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+)", re.MULTILINE)


class MarkdownLoader(BaseLoader):
    """
    Loads Markdown (.md) files.

    Parameters
    ----------
    strategy:
        ``heading`` — split on # / ## / ### headings (default).
        ``fulltext`` — entire file as one section.
    """

    def __init__(self, strategy: str = "heading") -> None:
        if strategy not in _VALID_STRATEGIES:
            raise ValueError(
                f"Unknown Markdown strategy '{strategy}'. "
                f"Valid: {sorted(_VALID_STRATEGIES)}"
            )
        self._strategy = strategy

    def load_bytes(self, file_bytes: bytes, filename: str) -> list[LoadedSection]:
        try:
            text = file_bytes.decode("utf-8", errors="replace")
        except Exception as exc:
            raise RuntimeError(
                f"[MarkdownLoader] '{filename}' 디코딩 실패: {exc}"
            ) from exc

        if self._strategy == "fulltext":
            return self._load_fulltext(text, filename)
        return self._load_heading(text, filename)

    def _load_fulltext(self, text: str, filename: str) -> list[LoadedSection]:
        # Strip markdown syntax for cleaner text
        cleaned = self.clean_text(_strip_md(text))
        if not cleaned:
            raise ValueError(f"[MarkdownLoader/fulltext] '{filename}'이 비어 있습니다.")
        return [LoadedSection(text=cleaned, page_or_section="fulltext")]

    def _load_heading(self, text: str, filename: str) -> list[LoadedSection]:
        sections: list[LoadedSection] = []
        # Split by heading markers
        parts = _HEADING_PATTERN.split(text)
        # parts: [pre_content, level, heading_text, body, level, heading_text, body, ...]

        # Handle content before first heading
        if parts and parts[0].strip():
            cleaned = self.clean_text(_strip_md(parts[0]))
            if cleaned:
                sections.append(LoadedSection(text=cleaned, page_or_section="Preamble"))

        # Iterate heading groups (level, title, body)
        i = 1
        while i + 2 <= len(parts):
            _level = parts[i]
            title = parts[i + 1].strip()
            body = parts[i + 2] if i + 2 < len(parts) else ""
            cleaned_body = self.clean_text(_strip_md(body))
            if title or cleaned_body:
                combined = (title + "\n\n" + cleaned_body).strip() if cleaned_body else title
                sections.append(LoadedSection(text=combined, page_or_section=title[:60]))
            i += 3

        if not sections:
            # Fall back to fulltext if no headings found
            return self._load_fulltext(text, filename)
        return sections


def _strip_md(text: str) -> str:
    """Remove common Markdown syntax (code fences, inline code, links, bold, italic)."""
    # Code fences
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    text = re.sub(r"`[^`]+`", "", text)
    # Links and images
    text = re.sub(r"!\[.*?\]\(.*?\)", "", text)
    text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)
    # Bold / italic
    text = re.sub(r"\*{1,3}([^*]+)\*{1,3}", r"\1", text)
    text = re.sub(r"_{1,3}([^_]+)_{1,3}", r"\1", text)
    # Headings (remove # prefix)
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    # Horizontal rules
    text = re.sub(r"^[-*_]{3,}\s*$", "", text, flags=re.MULTILINE)
    return text
