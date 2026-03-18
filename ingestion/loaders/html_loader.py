"""HTML loader: tag-based or full-text extraction using BeautifulSoup."""
from __future__ import annotations

from ingestion.loaders.base_loader import BaseLoader, LoadedSection

_VALID_STRATEGIES = frozenset(["tag", "fulltext"])


class HtmlLoader(BaseLoader):
    """
    Loads HTML (.html / .htm) files using BeautifulSoup.

    Parameters
    ----------
    strategy:
        ``tag`` — h1~h6 tags as section boundaries (default).
        ``fulltext`` — entire HTML stripped to plain text.
    """

    def __init__(self, strategy: str = "tag") -> None:
        if strategy not in _VALID_STRATEGIES:
            raise ValueError(
                f"Unknown HTML strategy '{strategy}'. "
                f"Valid: {sorted(_VALID_STRATEGIES)}"
            )
        self._strategy = strategy

    def load_bytes(self, file_bytes: bytes, filename: str) -> list[LoadedSection]:
        try:
            from bs4 import BeautifulSoup  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "beautifulsoup4 설치 필요: uv add beautifulsoup4"
            ) from exc

        try:
            text = file_bytes.decode("utf-8", errors="replace")
            soup = BeautifulSoup(text, "html.parser")
        except Exception as exc:
            raise RuntimeError(
                f"[HtmlLoader] '{filename}' 파싱 실패: {exc}"
            ) from exc

        # Remove script and style elements
        for tag in soup(["script", "style", "noscript", "meta", "head"]):
            tag.decompose()

        if self._strategy == "fulltext":
            return self._load_fulltext(soup, filename)
        return self._load_tag(soup, filename)

    def _load_fulltext(self, soup, filename: str) -> list[LoadedSection]:
        cleaned = self.clean_text(soup.get_text(separator="\n"))
        if not cleaned:
            raise ValueError(f"[HtmlLoader/fulltext] '{filename}'에서 텍스트를 찾을 수 없습니다.")
        return [LoadedSection(text=cleaned, page_or_section="fulltext")]

    def _load_tag(self, soup, filename: str) -> list[LoadedSection]:
        """Split on h1-h6 boundaries."""
        sections: list[LoadedSection] = []
        current_heading = "Document"
        current_parts: list[str] = []

        def _flush() -> None:
            if current_parts:
                combined = self.clean_text("\n\n".join(current_parts))
                if combined:
                    sections.append(
                        LoadedSection(text=combined, page_or_section=current_heading)
                    )
            current_parts.clear()

        body = soup.find("body") or soup
        for element in body.descendants:
            if not hasattr(element, "name"):
                continue
            if element.name in {"h1", "h2", "h3", "h4", "h5", "h6"}:
                _flush()
                current_heading = self.clean_text(element.get_text())[:60] or current_heading
            elif element.name in {"p", "li", "td", "th", "blockquote"}:
                text = self.clean_text(element.get_text())
                if text:
                    current_parts.append(text)

        _flush()

        if not sections:
            # Fall back to fulltext
            cleaned = self.clean_text(soup.get_text(separator="\n"))
            if cleaned:
                return [LoadedSection(text=cleaned, page_or_section="fulltext")]
            raise ValueError(
                f"[HtmlLoader/tag] '{filename}'에서 텍스트를 찾을 수 없습니다."
            )
        return sections
