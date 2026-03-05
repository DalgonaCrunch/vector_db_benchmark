"""
Index Registry: tracks KB (Knowledge Base) indices and their documents.

Architecture:
  KBIndex  (1 Qdrant collection + 1 OpenSearch index)
    └─ DocEntry  (one uploaded document, many chunks)

Naming:
  qdrant_collection = index_name
  os_index          = index_name

The registry persists to ``./data/index_registry.json``.

Usage::

    registry = IndexRegistry()
    registry.create("medical_kb")
    registry.add_document("medical_kb", DocEntry(...), vector_dim=4096)

    kb = registry.get("medical_kb")
    print(kb.total_chunks())
"""
from __future__ import annotations

import json
import re
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path


# ---------------------------------------------------------------------------
# Name sanitisation
# ---------------------------------------------------------------------------


def sanitize_index_name(name: str) -> str:
    """
    Normalise a user-supplied name to a valid index/collection identifier.

    Rules:
      • Lowercase
      • Only alphanumeric and underscore
      • No leading digits (prefix ``kb_`` if needed)
      • Empty → ``knowledge_base``
    """
    name = name.lower().strip()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    name = name.strip("_")
    if name and name[0].isdigit():
        name = "kb_" + name
    return name or "knowledge_base"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class DocEntry:
    """Metadata for one document ingested into a KB index."""

    doc_id: str          # unique timestamped ID, e.g. "report_20260304_133048"
    source_file: str     # original filename, e.g. "report.pdf"
    file_type: str       # "pdf" | "docx" | …
    upload_time: str     # ISO 8601
    chunk_count: int


@dataclass
class KBIndex:
    """
    A Knowledge Base Index.

    One Qdrant collection + one OpenSearch index, holding multiple documents.
    """

    index_name: str           # user-defined, sanitised
    qdrant_collection: str    # always == index_name
    os_index: str             # always == index_name
    created_at: str           # ISO 8601
    vector_dim: int           # 0 until first document is ingested
    documents: list[DocEntry] = field(default_factory=list)

    def total_chunks(self) -> int:
        return sum(d.chunk_count for d in self.documents)

    def doc_ids(self) -> list[str]:
        return [d.doc_id for d in self.documents]


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class IndexRegistry:
    """
    Thread-safe, JSON-persisted registry of KB indices.

    Parameters
    ----------
    registry_path:
        Path to the JSON file (created automatically if absent).
    """

    def __init__(
        self, registry_path: str | Path = "./data/index_registry.json"
    ) -> None:
        self._path = Path(registry_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._indices: dict[str, KBIndex] = {}
        self._load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def create(self, index_name: str) -> KBIndex:
        """
        Register a new KB index.  Does not touch any vector store.

        Raises
        ------
        ValueError
            If *index_name* already exists in the registry.
        """
        with self._lock:
            if index_name in self._indices:
                raise ValueError(f"Index '{index_name}' already exists.")
            kb = KBIndex(
                index_name=index_name,
                qdrant_collection=index_name,
                os_index=index_name,
                created_at=datetime.now().isoformat(),
                vector_dim=0,
            )
            self._indices[index_name] = kb
            self._save()
        return kb

    def get(self, index_name: str) -> KBIndex | None:
        return self._indices.get(index_name)

    def add_document(
        self, index_name: str, doc_entry: DocEntry, vector_dim: int
    ) -> None:
        """Append a document entry to an existing index and update vector_dim."""
        with self._lock:
            kb = self._indices.get(index_name)
            if kb is None:
                raise ValueError(f"Index '{index_name}' not found.")
            kb.documents.append(doc_entry)
            if vector_dim:
                kb.vector_dim = vector_dim
            self._save()

    def delete(self, index_name: str) -> None:
        with self._lock:
            self._indices.pop(index_name, None)
            self._save()

    def all_indices(self) -> list[KBIndex]:
        return list(self._indices.values())

    def index_names(self) -> list[str]:
        return list(self._indices.keys())

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                raw: list[dict] = json.load(f)
            for entry in raw:
                docs = [DocEntry(**d) for d in entry.get("documents", [])]
                kb = KBIndex(
                    index_name=entry["index_name"],
                    qdrant_collection=entry["qdrant_collection"],
                    os_index=entry["os_index"],
                    created_at=entry["created_at"],
                    vector_dim=entry.get("vector_dim", 0),
                    documents=docs,
                )
                self._indices[kb.index_name] = kb
        except Exception as exc:
            print(f"[IndexRegistry] WARNING: could not load registry: {exc}")

    def _save(self) -> None:
        data = [asdict(kb) for kb in self._indices.values()]
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
