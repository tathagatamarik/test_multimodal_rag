"""End-to-end orchestration: indexing workflow and query workflow.

Models are loaded lazily and at most once per :class:`RAGPipeline` instance, so
many questions can be asked against a persisted index without re-reading the
PDF or reloading Jina CLIP / Phi-3-V.
"""

from __future__ import annotations

import logging
import sqlite3
import time
from dataclasses import asdict, dataclass, field, replace
from typing import Any

from .config import Config
from .context_builder import Context, build_context, default_image_loader
from .embedding_service import EmbeddingService
from .exceptions import RAGError
from .generator import PROMPT_VERSION, ContextTooLongError, GenerationResult, Phi3VisionGenerator
from .indexer import Indexer, IndexReport
from .ingestion.service import IngestionService
from .logging_config import get_run_id
from .retriever import RetrievalResult, Retriever
from .telemetry import TelemetryStore, config_hash
from .vector_store import VectorStore

logger = logging.getLogger(__name__)

_MAX_SHRINK_ATTEMPTS = 2


@dataclass
class QueryResponse:
    question: str
    answer: str
    insufficient_evidence: bool
    evidence: list[dict[str, Any]]
    citations: list[dict[str, Any]]
    invalid_citations: list[str]
    retrieval: dict[str, Any]
    timings: dict[str, float]
    tokens: dict[str, int] = field(default_factory=dict)
    cold_start: bool = False
    run_id: str = field(default_factory=get_run_id)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def render(self) -> str:
        lines = [f"Q: {self.question}", "", f"A: {self.answer}", "", "Evidence:"]
        for e in self.evidence:
            lines.append(f"  [{e['tag']}] {e['modality']:<5} {e['label'][:60]:<60} {e['page_ref']:<8} "
                         f"sim={e['similarity']:.3f}")
        t = ", ".join(f"{k}={v:.2f}s" for k, v in self.timings.items())
        lines += ["", f"Timings: {t}"]
        if self.tokens:
            lines.append(f"Tokens: {self.tokens}")
        return "\n".join(lines)


def evidence_summary(ctx: Context) -> list[dict[str, Any]]:
    return [{"tag": it.tag, "modality": it.modality, "label": it.label, "page_ref": it.page_ref,
             "source": it.source,
             "pages": list(range(it.page_start, it.page_end + 1)),
             "similarity": round(it.similarity, 4), "source_ids": it.source_ids,
             "image_path": it.image_path, "snippet": it.text[:300]} for it in ctx.items]


class RAGPipeline:
    """Facade over indexer, retriever, context builder and generator."""

    def __init__(self, cfg: Config, embedder: EmbeddingService | None = None,
                 store: VectorStore | None = None, generator: Phi3VisionGenerator | None = None,
                 telemetry: TelemetryStore | None = None, telemetry_source: str = "cli"):
        self.cfg = cfg
        self.embedder = embedder or EmbeddingService(cfg.embedding)
        self.store = store or VectorStore(cfg.path(cfg.vector_store.persist_dir),
                                          cfg.vector_store.collection_name)
        self.retriever = Retriever(self.store, self.embedder, cfg.retrieval)
        self._generator = generator
        self._last_qvec = None
        self._ingestion: IngestionService | None = None
        self._load_image = default_image_loader(cfg.project_root)
        if telemetry is None and cfg.monitoring.record_queries:
            telemetry = TelemetryStore(cfg.path(cfg.monitoring.telemetry_path))
        self.telemetry = telemetry
        self.telemetry_source = telemetry_source
        self.config_hash = config_hash(cfg.to_dict())

    @property
    def generator(self) -> Phi3VisionGenerator:
        if self._generator is None:
            self._generator = Phi3VisionGenerator(self.cfg.generation)
        return self._generator

    # ---------------------------------------------------------------- indexing
    @property
    def ingestion(self) -> IngestionService:
        """Shared ingestion service (same embedder/store; models loaded once)."""
        if self._ingestion is None:
            self._ingestion = IngestionService(self.cfg, self.embedder, self.store)
        return self._ingestion

    def index(self, rebuild: bool = False, allow_download: bool = True) -> IndexReport:
        return Indexer(self.cfg, self.embedder, self.store, self.ingestion).run(
            rebuild=rebuild, allow_download=allow_download)

    # ---------------------------------------------------------------- querying
    def retrieve(self, question: str, **kwargs) -> RetrievalResult:
        return self.retriever.retrieve(question, **kwargs)

    def ask(self, question: str, mode: str | None = None, source: str | None = None,
            **retrieval_kwargs) -> QueryResponse:
        """Retrieve evidence, build context and generate a grounded answer.

        Every call (including failures) is recorded in the telemetry store when enabled.
        """
        try:
            resp = self._ask(question, mode, **retrieval_kwargs)
        except RAGError as exc:
            self._record({"question": question, "run_id": get_run_id()}, None, source,
                         error=f"{type(exc).__name__}: {exc}")
            raise
        self._record(resp.to_dict(), self._last_qvec, source)
        return resp

    def _record(self, response: dict, qvec, source: str | None, error: str = "") -> None:
        if self.telemetry is None:
            return
        try:
            self.telemetry.record(response, source=source or self.telemetry_source, embedding=qvec,
                                  config_hash=self.config_hash, prompt_version=PROMPT_VERSION,
                                  error=error)
        except (OSError, sqlite3.Error) as exc:  # telemetry must never break answering
            logger.warning("Telemetry write failed: %s", exc)

    def _ask(self, question: str, mode: str | None = None, **retrieval_kwargs) -> QueryResponse:
        t_total = time.perf_counter()
        # Cold start = this call loads a model; reported separately from warm latency.
        cold_embed = self.embedder._model is None
        cold_gen = self.generator._model is None
        retrieval = self.retriever.retrieve(question, mode=mode, **retrieval_kwargs)
        self._last_qvec = retrieval.query_embedding
        timings = dict(retrieval.timings)

        t = time.perf_counter()
        ctx_cfg = self.cfg.context
        ctx = build_context(retrieval, ctx_cfg, self._load_image)
        timings["context"] = time.perf_counter() - t

        gen: GenerationResult | None = None
        for attempt in range(_MAX_SHRINK_ATTEMPTS + 1):
            try:
                gen = self.generator.generate(ctx)
                break
            except ContextTooLongError as exc:
                if attempt == _MAX_SHRINK_ATTEMPTS:
                    raise
                # Halve the text budget (and drop to one image) and rebuild - bounded.
                ctx_cfg = replace(ctx_cfg, max_context_chars=max(500, ctx_cfg.max_context_chars // 2),
                                  max_images=min(ctx_cfg.max_images, 1))
                logger.warning("%s; shrinking context to %d chars", exc, ctx_cfg.max_context_chars)
                ctx = build_context(retrieval, ctx_cfg, self._load_image)
        assert gen is not None
        timings.update({f"generation_{k}": v for k, v in gen.timings.items()})
        if cold_embed and self.embedder.load_seconds is not None:
            timings["embedding_model_load"] = self.embedder.load_seconds
        if cold_gen and self.generator.load_seconds is not None:
            timings["generation_model_load"] = self.generator.load_seconds
        timings["total"] = time.perf_counter() - t_total

        response = QueryResponse(
            question=retrieval.query, answer=gen.answer,
            insufficient_evidence=gen.insufficient_evidence, evidence=evidence_summary(ctx),
            citations=gen.citations, invalid_citations=gen.invalid_citations,
            retrieval={"mode": retrieval.mode, "text_hits": len(retrieval.text),
                       "image_hits": len(retrieval.images), "pages": retrieval.pages(),
                       "top_text_similarity": retrieval.top_text_similarity,
                       "routed_sections": [s.metadata.get("section_path") for s in retrieval.sections],
                       "dropped_from_context": ctx.dropped},
            timings={k: round(v, 3) for k, v in timings.items()},
            tokens={"input": gen.input_tokens, "output": gen.output_tokens},
            cold_start=cold_embed or cold_gen)
        logger.info("Query done in %.1fs: %d evidence items, %d citations, insufficient=%s",
                    timings["total"], len(response.evidence), len(response.citations),
                    response.insufficient_evidence)
        return response
