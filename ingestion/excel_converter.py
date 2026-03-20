"""
Excel (xlsx / xls) workbook converter.

Usage flow (UI side)
--------------------
1. ``load_excel_sheets(file_bytes, filename)``  → list[ExcelSheet]
2. Show ``sheet.df`` preview to the user for verification.
3. ``sheet.to_csv_bytes()``  → download or feed into ingestion pipeline.

The converter is intentionally decoupled from the ingestion pipeline so it can
be used as a standalone preview-and-download utility as well as a direct ingest
helper.
"""
from __future__ import annotations

import io
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class ExcelSheet:
    """One sheet extracted from an Excel workbook."""

    name: str
    df: "pd.DataFrame"                          # noqa: F821  (type-check only)
    workbook_name: str = ""                     # original filename (no ext)
    row_count: int = 0
    col_count: int = 0
    columns: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.row_count = len(self.df)
        self.col_count = len(self.df.columns)
        self.columns = self.df.columns.tolist()

    # ------------------------------------------------------------------
    # Export helpers
    # ------------------------------------------------------------------

    @property
    def csv_filename(self) -> str:
        """Suggested CSV filename: ``{workbook}_{sheet}.csv``."""
        stem = self.workbook_name or "excel"
        safe_sheet = self.name.replace(" ", "_").replace("/", "-")
        return f"{stem}_{safe_sheet}.csv"

    def to_csv_bytes(self, encoding: str = "utf-8-sig") -> bytes:
        """
        Serialise the sheet as a CSV byte string.

        ``utf-8-sig`` is the default so that Excel on Windows opens the file
        without garbled Korean characters.
        """
        buf = io.BytesIO()
        self.df.to_csv(buf, index=False, encoding=encoding)
        return buf.getvalue()

    # ------------------------------------------------------------------
    # Preview helpers
    # ------------------------------------------------------------------

    def preview_df(self, max_rows: int = 50) -> "pd.DataFrame":
        """Return the first *max_rows* rows for display."""
        return self.df.head(max_rows)

    def grouped_preview(self, rows_per_group: int = 20) -> list[dict]:
        """
        Return a list of group summaries (for showing how rows will be
        batched into LoadedSections by CsvLoader).

        Each entry: ``{"label": str, "row_range": str, "sample": str}``
        """
        import math

        groups: list[dict] = []
        total = len(self.df)
        n_groups = math.ceil(total / rows_per_group) if total > 0 else 0

        for i in range(n_groups):
            start = i * rows_per_group
            end = min(start + rows_per_group, total)
            batch = self.df.iloc[start:end]
            # Build a short sample line from the first row
            first = batch.iloc[0]
            sample_parts = [
                f"{col}: {first[col]}"
                for col in self.columns[:4]          # show at most 4 cols
                if str(first[col]).strip()
            ]
            sample = " / ".join(sample_parts)
            if len(sample) > 120:
                sample = sample[:117] + "…"
            groups.append(
                {
                    "label": f"섹션 {i + 1}",
                    "row_range": f"행 {start + 1}–{end}",
                    "rows": end - start,
                    "sample": sample,
                }
            )
        return groups


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_excel_sheets(
    file_bytes: bytes,
    filename: str,
    *,
    skip_empty: bool = True,
) -> list[ExcelSheet]:
    """
    Parse an Excel workbook and return all sheets as :class:`ExcelSheet` objects.

    Parameters
    ----------
    file_bytes:
        Raw bytes of the uploaded ``.xlsx`` / ``.xls`` file.
    filename:
        Original filename (used for naming suggestions only).
    skip_empty:
        If *True* (default), sheets with no rows are omitted.

    Raises
    ------
    ImportError
        If ``pandas`` or ``openpyxl`` are not installed.
    RuntimeError
        If the file cannot be parsed.
    """
    try:
        import pandas as pd          # type: ignore[import]
    except ImportError as exc:
        raise ImportError("pandas 설치 필요: uv add pandas") from exc

    try:
        import openpyxl              # type: ignore[import]  # noqa: F401
    except ImportError as exc:
        raise ImportError("openpyxl 설치 필요: uv add openpyxl") from exc

    from pathlib import Path
    stem = Path(filename).stem

    try:
        xl = pd.ExcelFile(io.BytesIO(file_bytes), engine="openpyxl")
    except Exception as exc:
        raise RuntimeError(
            f"[ExcelConverter] '{filename}' 파싱 실패: {exc}"
        ) from exc

    sheets: list[ExcelSheet] = []
    for sheet_name in xl.sheet_names:
        try:
            df = xl.parse(sheet_name, dtype=str).fillna("")
        except Exception as exc:
            raise RuntimeError(
                f"[ExcelConverter] 시트 '{sheet_name}' 파싱 실패: {exc}"
            ) from exc

        if skip_empty and df.empty:
            continue

        # Strip leading/trailing whitespace from all string values
        df = df.applymap(lambda v: v.strip() if isinstance(v, str) else v)
        # Drop rows where every column is empty
        non_empty_mask = df.apply(lambda row: row.str.strip().any(), axis=1)
        df = df[non_empty_mask].reset_index(drop=True)

        if skip_empty and df.empty:
            continue

        sheets.append(
            ExcelSheet(name=sheet_name, df=df, workbook_name=stem)
        )

    if not sheets:
        raise ValueError(
            f"[ExcelConverter] '{filename}'에서 데이터가 있는 시트를 찾지 못했습니다."
        )

    return sheets
