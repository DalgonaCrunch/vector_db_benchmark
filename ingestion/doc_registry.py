"""
Document registry: tracks per-document Qdrant collection and OpenSearch index names.

Each uploaded PDF gets a unique doc_id = "{stem}_{YYYYMMDD_HHMMSS}", e.g.
"object_pascal_guide_20240101_120000".  The registry persists to
``./data/registry.json`` so uploads survive Streamlit restarts.
"""
from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path


def make_doc_id(filename: str) -> str:
    """Generate a stable, unique doc_id from an original filename."""
    stem = Path(filename).stem
    # Replace spaces and special chars with underscores
    safe_stem = "".join(c if c.isalnum() or c == "_" else "_" for c in stem)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{safe_stem}_{ts}"


@dataclass
class DocRecord:
    """Metadata for one ingested document."""

    doc_id: str               # unique ID, e.g. "report_2024_20240101_120000"
    source_file: str          # original filename, e.g. "report_2024.pdf"
    upload_time: str          # ISO format string
    qdrant_collection: str    # "{doc_id}_qdrant"
    os_index: str             # "{doc_id}_os"
    chunk_count: int
    vector_dim: int


class DocRegistry:
    """
    Thread-safe, JSON-persisted registry of uploaded documents.

    Parameters
    ----------
    registry_path:
        Path to the JSON file (created automatically if absent).
    """

    def __init__(self, registry_path: str | Path = "./data/registry.json") -> None:
        self._path = Path(registry_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._records: dict[str, DocRecord] = {}
        self._load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(self, record: DocRecord) -> None:
        with self._lock:
            self._records[record.doc_id] = record
            self._save()

    def get(self, doc_id: str) -> DocRecord | None:
        return self._records.get(doc_id)

    def remove(self, doc_id: str) -> None:
        with self._lock:
            self._records.pop(doc_id, None)
            self._save()

    def all_records(self) -> list[DocRecord]:
        return list(self._records.values())

    def doc_ids(self) -> list[str]:
        return list(self._records.keys())

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            for entry in raw:
                rec = DocRecord(**entry)
                self._records[rec.doc_id] = rec
        except Exception as exc:
            print(f"[DocRegistry] WARNING: could not load registry: {exc}")

    def _save(self) -> None:
        data = [asdict(r) for r in self._records.values()]
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
