"""CSV loader: row-batch extraction using pandas."""
from __future__ import annotations

import io

from ingestion.loaders.base_loader import BaseLoader, LoadedSection


class CsvLoader(BaseLoader):
    """
    Loads CSV files, grouping *rows_per_section* rows per section.

    Parameters
    ----------
    rows_per_section:
        How many rows to include in one LoadedSection. Default 20.
    """

    def __init__(self, rows_per_section: int = 20) -> None:
        self._rows_per_section = max(1, rows_per_section)

    def load_bytes(self, file_bytes: bytes, filename: str) -> list[LoadedSection]:
        try:
            import pandas as pd  # type: ignore[import]
        except ImportError as exc:
            raise ImportError("pandas 설치 필요: uv add pandas") from exc

        try:
            df = pd.read_csv(io.BytesIO(file_bytes), dtype=str).fillna("")
        except Exception as exc:
            raise RuntimeError(
                f"[CsvLoader] '{filename}' 파싱 실패: {exc}"
            ) from exc

        if df.empty:
            raise ValueError(f"[CsvLoader] '{filename}'이 비어 있습니다.")

        sections: list[LoadedSection] = []
        cols = df.columns.tolist()

        for batch_start in range(0, len(df), self._rows_per_section):
            batch = df.iloc[batch_start : batch_start + self._rows_per_section]
            lines: list[str] = []
            for _, row in batch.iterrows():
                parts = [f"{col}: {row[col]}" for col in cols if row[col].strip()]
                if parts:
                    lines.append(" / ".join(parts))
            combined = self.clean_text("\n".join(lines))
            if combined:
                label = f"rows_{batch_start + 1}_{min(batch_start + self._rows_per_section, len(df))}"
                sections.append(LoadedSection(text=combined, page_or_section=label))

        if not sections:
            raise ValueError(f"[CsvLoader] '{filename}'에서 텍스트를 찾을 수 없습니다.")
        return sections
