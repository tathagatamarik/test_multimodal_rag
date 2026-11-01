"""Ingestion service: single, batch and incremental ingestion into one collection.

Decoupled from querying: it only needs the embedding service, the vector store
and the registry, and can run in a separate process (CLI, watcher) or behind
the API's background queue. A lock file ensures only one writer at a time.

Incremental semantics (``ingest_file`` / ``sync``):

* same content hash already ``indexed`` with the current fingerprint -> skipped;
* new content for an existing source (file edited / URL updated) -> the new
  version is fully ingested first, then the superseded version's records are
  deleted, so queries never see a gap;
* a document left ``ingesting`` by a crash -> its records are purged and it is
  re-ingested;
* ``sync(directory, prune=True)`` also removes documents whose file disappeared.

A changed index fingerprint (embedding model, parser, chunking, image settings)
makes stored vectors incomparable with new ones; ingestion then refuses until
``rebuild()`` re-processes every registered document.
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np
from PIL import Image

from ..config import Config
from ..document_loader import DocumentInfo, validate_pdf
from ..embedding_service import EmbeddingService
from ..exceptions import IndexCompatibilityError, RAGError, VectorStoreError
from ..vector_store import VectorStore, fingerprint_hash
from .processor import (ProcessedDocument, process_document, section_metadata, text_metadata,
                        visual_metadata)
from .registry import FAILED, INDEXED, INGESTING, REMOVED, DocumentEntry, Registry

logger = logging.getLogger(__name__)


@dataclass
class DocumentResult:
    doc_id: str
    filename: str
    action: str                 # "indexed" | "unchanged" | "updated" | "failed" | "removed"
    records: int = 0
    timings: dict[str, float] = field(default_factory=dict)
    error: str = ""


@dataclass
class BatchReport:
    results: list[DocumentResult] = field(default_factory=list)
    seconds: float = 0.0

    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for r in self.results:
            out[r.action] = out.get(r.action, 0) + 1
        return out

    def summary(self) -> str:
        return f"Batch of {len(self.results)} documents in {self.seconds:.1f}s: {self.counts()}"


class WriterLock:
    """Cross-process exclusive lock via an O_EXCL lock file (stale after ``stale_s``)."""

    def __init__(self, path: Path, stale_s: float = 6 * 3600):
        self.path = Path(path)
        self.stale_s = stale_s
        self._fd: int | None = None

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists() and time.time() - self.path.stat().st_mtime > self.stale_s:
            logger.warning("Removing stale ingestion lock %s", self.path)
            self.path.unlink(missing_ok=True)
        try:
            self._fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as exc:
            raise VectorStoreError(f"Another ingestion process holds {self.path}; "
                                   "run only one writer (API worker or CLI) at a time") from exc
        os.write(self._fd, str(os.getpid()).encode())

    def release(self) -> None:
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None
            self.path.unlink(missing_ok=True)


class IngestionService:
    def __init__(self, cfg: Config, embedder: EmbeddingService, store: VectorStore,
                 registry: Registry | None = None):
        self.cfg = cfg
        self.embedder = embedder
        self.store = store
        self.registry = registry or Registry(cfg.path(cfg.vector_store.registry_path))
        self.fingerprint = cfg.index_fingerprint()
        self.fp_hash = fingerprint_hash(self.fingerprint)
        self._lock = WriterLock(cfg.path(cfg.vector_store.persist_dir).parent / "ingest.lock")
        self._held = 0

    # ------------------------------------------------------------------ locking
    @contextmanager
    def writer(self) -> Iterator[None]:
        """Re-entrant within this object; exclusive across processes."""
        if self._held == 0:
            self._lock.acquire()
        self._held += 1
        try:
            yield
        finally:
            self._held -= 1
            if self._held == 0:
                self._lock.release()

    # -------------------------------------------------------------- compatibility
    def check_compatible(self) -> None:
        """Raise if the stored index was built with a different fingerprint."""
        stored = self.registry.fingerprint_hash
        if self.store.exists():
            meta_hash = (self.store.open().metadata or {}).get("fingerprint_hash")
            if meta_hash != self.fp_hash:
                raise IndexCompatibilityError(
                    "The existing index was built with different embedding/parsing settings "
                    f"(stored {meta_hash}, current {self.fp_hash}). Re-run with --rebuild to "
                    "re-process all registered documents.")
            if stored not in (None, self.fp_hash):
                raise IndexCompatibilityError("Registry fingerprint differs from the collection; rebuild")
        elif self.registry.indexed():
            raise IndexCompatibilityError("Registry lists indexed documents but the collection is "
                                          "missing; re-run with --rebuild")

    def _ensure_collection(self) -> None:
        if not self.store.exists():
            self.store.create({"embedding_model_id": self.cfg.embedding.model_id,
                               "embedding_dim": int(self.embedder.dim or 0),
                               "fingerprint_hash": self.fp_hash})
            self.registry.set_fingerprint(self.fingerprint, self.fp_hash)
        else:
            self.store.open()
            if self.registry.fingerprint_hash is None:
                self.registry.set_fingerprint(self.fingerprint, self.fp_hash)

    # ------------------------------------------------------------------ single doc
    def ingest_file(self, path: str | Path, source: str | None = None) -> DocumentResult:
        """Ingest one PDF incrementally (skip if unchanged). Raises on failure."""
        with self.writer():
            self.check_compatible()
            self._recover_dirty()
            doc = validate_pdf(path, source=source or str(Path(path).resolve()))
            existing = self.registry.get(doc.doc_id)
            if existing and existing.status == INDEXED and existing.fingerprint_hash == self.fp_hash:
                logger.info("Unchanged: %s (doc_id %s) already indexed", doc.filename, doc.doc_id)
                return DocumentResult(doc.doc_id, doc.filename, "unchanged", existing.total_records)
            previous = [d for d in self.registry.by_source(doc.source) if d.doc_id != doc.doc_id]
            result = self._ingest(doc)
            for old in previous:  # new version is fully in place; now drop the superseded one
                self._delete_records(old.doc_id)
                old.status, old.error = REMOVED, f"superseded by {doc.doc_id}"
                self.registry.upsert(old)
                result.action = "updated"
                logger.info("Superseded %s with %s for source %s", old.doc_id, doc.doc_id, doc.source)
            return result

    def _ingest(self, doc: DocumentInfo) -> DocumentResult:
        entry = DocumentEntry(doc.doc_id, doc.source, doc.path, doc.filename, doc.sha256,
                              doc.page_count, doc.title, INGESTING, self.fp_hash)
        self.registry.upsert(entry)
        t0 = time.perf_counter()
        try:
            processed = process_document(doc, self.cfg)
            n = self._embed_and_store(processed)
        except RAGError as exc:
            self._delete_records(doc.doc_id)
            entry.status, entry.error = FAILED, f"{type(exc).__name__}: {exc}"
            self.registry.upsert(entry)
            logger.error("Ingestion of %s failed: %s", doc.filename, exc)
            raise
        entry.status = INDEXED
        entry.text_chunks, entry.section_records = len(processed.chunks), len(processed.sections)
        entry.visual_records, entry.total_records = len(processed.visuals), n
        entry.timings = {k: round(v, 3) for k, v in processed.timings.items()}
        entry.timings["total"] = round(time.perf_counter() - t0, 3)
        self.registry.upsert(entry)
        logger.info("Indexed %s (%s): %d records in %.1fs", doc.filename, doc.doc_id, n,
                    entry.timings["total"])
        return DocumentResult(doc.doc_id, doc.filename, "indexed", n, entry.timings)

    def _embed_and_store(self, p: ProcessedDocument) -> int:
        doc = p.doc
        t = time.perf_counter()
        texts = [c.text for c in p.chunks] + [s.text for s in p.sections]
        text_emb = self.embedder.embed_texts(texts) if texts else np.zeros((0, 0), np.float32)
        p.timings["embed_text"] = time.perf_counter() - t
        t = time.perf_counter()
        images = []
        for v in p.visuals:
            with Image.open(self.cfg.path(v.path)) as im:
                images.append(im.convert("RGB"))
        img_emb = self.embedder.embed_images(images)
        p.timings["embed_images"] = time.perf_counter() - t

        t = time.perf_counter()
        self._ensure_collection()
        self._delete_records(doc.doc_id)  # idempotent: clear any partial earlier attempt
        n_c = len(p.chunks)
        if n_c:
            self.store.upsert([c.chunk_id for c in p.chunks], text_emb[:n_c],
                              [c.text for c in p.chunks], [text_metadata(c, doc) for c in p.chunks])
        if p.sections:
            self.store.upsert([s.record_id for s in p.sections], text_emb[n_c:],
                              [s.text for s in p.sections], [section_metadata(s, doc) for s in p.sections])
        if p.visuals:
            # The stored "document" of a visual is its caption: shown to the LLM as the
            # textual companion of the image; the pixels themselves live at image_path.
            self.store.upsert([v.visual_id for v in p.visuals], img_emb,
                              [v.caption or v.label for v in p.visuals],
                              [visual_metadata(v, doc) for v in p.visuals])
        expected = n_c + len(p.sections) + len(p.visuals)
        stored = self.store.count({"doc_id": doc.doc_id})
        if stored != expected:
            raise VectorStoreError(f"Verification failed for {doc.doc_id}: expected {expected} "
                                   f"records, found {stored}")
        p.timings["chroma_index"] = time.perf_counter() - t
        return stored

    def _delete_records(self, doc_id: str) -> None:
        if self.store.exists():
            self.store.delete_where({"doc_id": doc_id})

    def _recover_dirty(self) -> None:
        for d in self.registry.dirty():
            logger.warning("Document %s was left mid-ingestion; purging its records", d.doc_id)
            self._delete_records(d.doc_id)
            d.status, d.error = FAILED, "interrupted during ingestion"
            self.registry.upsert(d)

    # ---------------------------------------------------------------- batch / sync
    def ingest_batch(self, paths: Iterable[str | Path]) -> BatchReport:
        """Ingest many PDFs; a failing document is recorded and the batch continues."""
        report = BatchReport()
        t0 = time.perf_counter()
        with self.writer():
            for p in paths:
                try:
                    report.results.append(self.ingest_file(p))
                except RAGError as exc:
                    report.results.append(DocumentResult("", Path(p).name, "failed", error=str(exc)))
        report.seconds = time.perf_counter() - t0
        logger.info(report.summary())
        return report

    def sync(self, directory: str | Path, prune: bool = False) -> BatchReport:
        """Incrementally mirror *directory* (``*.pdf``) into the index."""
        directory = Path(directory)
        files = sorted(directory.glob("*.pdf"))
        with self.writer():
            report = self.ingest_batch(files)
            if prune:
                present = {str(f.resolve()) for f in files}
                for d in self.registry.indexed():
                    if d.source.startswith(str(directory.resolve())) and d.source not in present:
                        report.results.append(self.remove(d.doc_id))
        return report

    def remove(self, doc_id: str) -> DocumentResult:
        with self.writer():
            d = self.registry.get(doc_id)
            if d is None:
                raise VectorStoreError(f"Unknown document {doc_id}")
            self._delete_records(doc_id)
            d.status, d.total_records = REMOVED, 0
            self.registry.upsert(d)
            logger.info("Removed document %s (%s)", doc_id, d.filename)
            return DocumentResult(doc_id, d.filename, "removed")

    def rebuild(self) -> BatchReport:
        """Drop the collection and re-process every registered document from its local copy."""
        with self.writer():
            docs = [d for d in self.registry.documents.values() if d.status in (INDEXED, INGESTING, FAILED)]
            self.store.delete()
            for d in docs:
                d.status = FAILED  # will be overwritten on success
            self.registry.set_fingerprint(self.fingerprint, self.fp_hash)
            report = BatchReport()
            t0 = time.perf_counter()
            for d in docs:
                try:
                    doc = validate_pdf(d.path, source=d.source)
                    report.results.append(self._ingest(doc))
                except RAGError as exc:
                    report.results.append(DocumentResult(d.doc_id, d.filename, "failed", error=str(exc)))
            report.seconds = time.perf_counter() - t0
            return report

    def status(self) -> dict[str, Any]:
        docs = list(self.registry.documents.values())
        return {"fingerprint_hash": self.fp_hash, "collection_exists": self.store.exists(),
                "collection_records": self.store.count() if self.store.exists() else 0,
                "documents": [d.to_dict() for d in docs],
                "indexed_documents": len(self.registry.indexed())}
