"""Per-document ingestion registry (JSON, atomic writes).

The registry is the source of truth for *what is in the index and how it got
there*. It replaces the single-document manifest:

* collection level: the index fingerprint (embedding model + revision,
  chunking/parser and image settings) that every stored vector was built with;
* document level, keyed by ``doc_id`` (first 12 hex chars of the PDF SHA-256):
  source, local path, hash, status, record counts, timings and errors.

Document status lifecycle::

    ingesting --success--> indexed --(source changed)--> superseded / removed
        \\--failure--> failed

A document left in ``ingesting`` (crash, kill) is treated as dirty: its records
are deleted and it is re-ingested on the next run, never served as complete.
"""

from __future__ import annotations

import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..vector_store import read_manifest, write_manifest

INDEXED, INGESTING, FAILED, REMOVED = "indexed", "ingesting", "failed", "removed"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class DocumentEntry:
    doc_id: str
    source: str
    path: str
    filename: str
    sha256: str
    page_count: int
    title: str
    status: str
    fingerprint_hash: str
    text_chunks: int = 0
    section_records: int = 0
    visual_records: int = 0
    total_records: int = 0
    timings: dict[str, float] = field(default_factory=dict)
    error: str = ""
    updated_at: str = field(default_factory=now_iso)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class Registry:
    """Thread-safe registry persisted as one JSON file."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._lock = threading.RLock()
        data = read_manifest(self.path) or {}
        self.fingerprint_hash: str | None = data.get("fingerprint_hash")
        self.fingerprint: dict[str, Any] = data.get("fingerprint", {})
        self.documents: dict[str, DocumentEntry] = {
            k: DocumentEntry(**v) for k, v in data.get("documents", {}).items()}

    def save(self) -> None:
        with self._lock:
            write_manifest(self.path, {
                "fingerprint_hash": self.fingerprint_hash, "fingerprint": self.fingerprint,
                "documents": {k: v.to_dict() for k, v in self.documents.items()}})

    def set_fingerprint(self, fp: dict[str, Any], fp_hash: str) -> None:
        with self._lock:
            self.fingerprint, self.fingerprint_hash = fp, fp_hash
            self.save()

    def upsert(self, entry: DocumentEntry) -> None:
        with self._lock:
            entry.updated_at = now_iso()
            self.documents[entry.doc_id] = entry
            self.save()

    def get(self, doc_id: str) -> DocumentEntry | None:
        return self.documents.get(doc_id)

    def by_source(self, source: str) -> list[DocumentEntry]:
        return [d for d in self.documents.values() if d.source == source and d.status == INDEXED]

    def indexed(self) -> list[DocumentEntry]:
        return [d for d in self.documents.values() if d.status == INDEXED]

    def dirty(self) -> list[DocumentEntry]:
        return [d for d in self.documents.values() if d.status == INGESTING]

    def total_records(self) -> int:
        return sum(d.total_records for d in self.indexed())
