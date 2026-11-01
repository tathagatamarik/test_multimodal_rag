"""ChromaDB persistence layer plus atomic JSON helpers.

The collection uses cosine distance (``hnsw.space = cosine``) because Jina CLIP
embeddings are L2-normalised and trained with a cosine contrastive objective;
``similarity = 1 - distance``.

Chroma is used purely as a vector store: ``embedding_function=None`` so it never
downloads or applies its own default embedding model.

``read_manifest`` / ``write_manifest`` provide atomic JSON persistence (tmp file +
rename); the ingestion registry (``src/ingestion/registry.py``) is stored with them.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import chromadb
import numpy as np
from chromadb.errors import ChromaError, NotFoundError

from .exceptions import IndexCompatibilityError, VectorStoreError

logger = logging.getLogger(__name__)

MANIFEST_VERSION = 1
_SCALAR = (str, int, float, bool)


@dataclass(frozen=True)
class Hit:
    """One similarity-search result with its stored content and provenance."""

    id: str
    document: str
    metadata: dict[str, Any]
    distance: float

    @property
    def similarity(self) -> float:
        """Cosine similarity (1 - cosine distance) as computed by ChromaDB."""
        return 1.0 - self.distance

    @property
    def modality(self) -> str:
        return str(self.metadata.get("modality", "unknown"))


def fingerprint_hash(fingerprint: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode()).hexdigest()[:16]


# --------------------------------------------------------------------------- manifest
def read_manifest(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise VectorStoreError(f"Index manifest {path} is unreadable ({exc}); "
                               "rebuild the index with --rebuild") from exc


def write_manifest(path: Path, data: dict[str, Any]) -> None:
    """Atomic write (tmp file + rename) so a crash never leaves half a manifest."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    payload = {"manifest_version": MANIFEST_VERSION,
               "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds")} | data
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


# --------------------------------------------------------------------------- store
class VectorStore:
    """Thin, validated wrapper around one persistent Chroma collection."""

    def __init__(self, persist_dir: Path, collection_name: str,
                 client: chromadb.api.ClientAPI | None = None):
        self.persist_dir = Path(persist_dir)
        self.collection_name = collection_name
        try:
            if client is None:
                self.persist_dir.mkdir(parents=True, exist_ok=True)
                client = chromadb.PersistentClient(path=str(self.persist_dir))
        except (ChromaError, OSError, ValueError) as exc:
            raise VectorStoreError(f"Cannot open ChromaDB at {self.persist_dir}: {exc}") from exc
        self.client = client
        self._collection = None

    # ------------------------------------------------------------- collection
    def exists(self) -> bool:
        try:
            self.client.get_collection(self.collection_name)
            return True
        except (NotFoundError, ValueError):
            return False

    def open(self, expected: dict[str, Any] | None = None):
        """Open the existing collection, optionally checking its stored metadata."""
        try:
            col = self.client.get_collection(self.collection_name, embedding_function=None)
        except (NotFoundError, ValueError) as exc:
            raise VectorStoreError(f"Collection '{self.collection_name}' does not exist; "
                                   "run the indexing workflow first") from exc
        if expected:
            stored = col.metadata or {}
            diffs = {k: (stored.get(k), v) for k, v in expected.items() if stored.get(k) != v}
            if diffs:
                raise IndexCompatibilityError(
                    f"Collection '{self.collection_name}' was built with different settings "
                    f"(stored vs current): {diffs}. Rebuild with --rebuild.")
        self._collection = col
        return col

    def create(self, metadata: dict[str, Any]):
        """Create a fresh cosine-space collection (fails if it already exists)."""
        bad = {k: v for k, v in metadata.items() if not isinstance(v, _SCALAR)}
        if bad:
            raise VectorStoreError(f"Collection metadata must be scalar: {bad}")
        try:
            self._collection = self.client.create_collection(
                self.collection_name, embedding_function=None,
                configuration={"hnsw": {"space": "cosine"}}, metadata=metadata)
        except (ChromaError, ValueError) as exc:
            raise VectorStoreError(f"Cannot create collection '{self.collection_name}': {exc}") from exc
        logger.info("Created Chroma collection '%s' (cosine)", self.collection_name)
        return self._collection

    def delete(self) -> None:
        """Drop the collection if present (explicit, logged)."""
        if self.exists():
            self.client.delete_collection(self.collection_name)
            logger.warning("Deleted Chroma collection '%s'", self.collection_name)
        self._collection = None

    @property
    def collection(self):
        if self._collection is None:
            self.open()
        return self._collection

    # ------------------------------------------------------------------ writes
    def upsert(self, ids: list[str], embeddings: np.ndarray, documents: list[str],
               metadatas: list[dict[str, Any]]) -> None:
        """Validate and upsert records in client-sized batches.

        Upsert (not add) keyed by deterministic IDs makes re-runs idempotent:
        re-indexing the same content replaces records instead of duplicating them.
        """
        n = len(ids)
        if not (n == len(documents) == len(metadatas) == embeddings.shape[0]):
            raise VectorStoreError(f"Length mismatch: ids={n}, docs={len(documents)}, "
                                   f"metas={len(metadatas)}, embeddings={embeddings.shape[0]}")
        if len(set(ids)) != n:
            dupes = sorted({i for i in ids if ids.count(i) > 1})
            raise VectorStoreError(f"Duplicate IDs in one upsert: {dupes[:5]}")
        if embeddings.ndim != 2 or not np.isfinite(embeddings).all():
            raise VectorStoreError("Embeddings must be a finite 2-D array")
        expected_dim = (self.collection.metadata or {}).get("embedding_dim")
        if expected_dim is not None and embeddings.shape[1] != expected_dim:
            raise VectorStoreError(f"Embedding dimension {embeddings.shape[1]} does not match "
                                   f"collection dimension {expected_dim}")
        for i, md in enumerate(metadatas):
            bad = {k: type(v).__name__ for k, v in md.items() if not isinstance(v, _SCALAR)}
            if bad:
                raise VectorStoreError(f"Invalid metadata for id {ids[i]!r}: non-scalar values {bad}")

        step = max(1, self.client.get_max_batch_size())
        try:
            for s in range(0, n, step):
                self.collection.upsert(ids=ids[s:s + step], embeddings=embeddings[s:s + step].tolist(),
                                       documents=documents[s:s + step], metadatas=metadatas[s:s + step])
        except (ChromaError, ValueError) as exc:
            raise VectorStoreError(f"Chroma upsert failed: {exc}") from exc

    def delete_where(self, where: dict[str, Any]) -> int:
        """Delete all records matching a metadata filter; returns how many were removed."""
        try:
            ids = self.collection.get(where=where, include=[])["ids"]
            if ids:
                self.collection.delete(ids=ids)
        except (ChromaError, ValueError) as exc:
            raise VectorStoreError(f"Chroma delete failed for {where}: {exc}") from exc
        return len(ids)

    # ------------------------------------------------------------------- reads
    def count(self, where: dict[str, Any] | None = None) -> int:
        if where is None:
            return self.collection.count()
        return len(self.collection.get(where=where, include=[])["ids"])

    def get(self, ids: list[str]) -> list[Hit]:
        res = self.collection.get(ids=ids, include=["documents", "metadatas"])
        return [Hit(i, d or "", m or {}, 0.0)
                for i, d, m in zip(res["ids"], res["documents"], res["metadatas"])]

    def query(self, embedding: np.ndarray, n_results: int,
              where: dict[str, Any] | None = None) -> list[Hit]:
        """Nearest neighbours of *embedding*, optionally filtered by metadata."""
        if n_results <= 0:
            return []
        vec = np.asarray(embedding, dtype=np.float32).reshape(-1)
        if not np.isfinite(vec).all():
            raise VectorStoreError("Query embedding contains NaN/Inf")
        available = self.count(where)
        if available == 0:
            return []
        try:
            res = self.collection.query(query_embeddings=[vec.tolist()],
                                        n_results=min(n_results, available), where=where,
                                        include=["documents", "metadatas", "distances"])
        except (ChromaError, ValueError) as exc:
            raise VectorStoreError(f"Chroma query failed: {exc}") from exc
        return [Hit(i, d or "", m or {}, float(dist)) for i, d, m, dist in
                zip(res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0])]
