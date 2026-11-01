"""Indexing workflow for the configured paper (``python -m src.cli index``).

A thin wrapper over :class:`src.ingestion.service.IngestionService`: it obtains
the assignment's PDF (download or valid local copy) and ingests it
incrementally. Behaviour:

=======================================================  ==========================
State                                                    Action
=======================================================  ==========================
paper not in registry                                    build (ingest)
same content hash indexed with the current fingerprint   reuse (nothing recomputed)
index fingerprint changed (model/parser/chunking/images)  refuse, unless ``rebuild``
document left ``ingesting`` by a crash                   purged and re-ingested
=======================================================  ==========================

Further documents are added through the ingestion CLI/API (batch, sync,
streaming); see ``docs/ingestion.md``.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from .config import Config
from .document_loader import load_document
from .embedding_service import EmbeddingService
from .ingestion.service import IngestionService
from .vector_store import VectorStore

logger = logging.getLogger(__name__)


@dataclass
class IndexReport:
    """What the indexing run did, with per-stage timings in seconds."""

    action: str
    doc: dict[str, Any]
    text_chunks: int = 0
    section_records: int = 0
    visual_records: int = 0
    total_records: int = 0
    timings: dict[str, float] = field(default_factory=dict)

    def summary(self) -> str:
        t = ", ".join(f"{k}={v:.2f}s" for k, v in self.timings.items())
        return (f"Index {self.action}: {self.text_chunks} text chunks + {self.section_records} ToC "
                f"sections + {self.visual_records} visuals = {self.total_records} records | {t}")


class Indexer:
    def __init__(self, cfg: Config, embedder: EmbeddingService, store: VectorStore,
                 service: IngestionService | None = None):
        self.cfg = cfg
        self.service = service or IngestionService(cfg, embedder, store)

    def run(self, rebuild: bool = False, allow_download: bool = True) -> IndexReport:
        t0 = time.perf_counter()
        doc = load_document(self.cfg.document, self.cfg.path, allow_download=allow_download)
        timings = {"load_pdf": time.perf_counter() - t0}
        if rebuild:
            report = self.service.rebuild()
            if not any(r.doc_id == doc.doc_id for r in report.results):
                result = self.service.ingest_file(doc.path, source=self.cfg.document.source_url)
            action = "rebuilt"
        else:
            result = self.service.ingest_file(doc.path, source=self.cfg.document.source_url)
            action = {"unchanged": "reused", "indexed": "built", "updated": "updated"}[result.action]
        entry = self.service.registry.get(doc.doc_id)
        timings.update(entry.timings if action != "reused" else {})
        report = IndexReport(action, doc.to_dict(), entry.text_chunks, entry.section_records,
                             entry.visual_records, entry.total_records, timings)
        logger.info(report.summary())
        return report
