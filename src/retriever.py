"""Query-time retrieval over the shared Jina CLIP space.

Text: top-k text chunks by cosine similarity (Chroma, ``modality == text``).

Visuals: with Jina CLIP v1, text->image similarities live on a much lower scale
(about 0.05-0.35 on this paper) than text->text similarities (about 0.3-0.8),
so a single merged top-k would almost never surface an image. Visuals are
therefore retrieved with their own quota and scored in one of two ways:

* ``pixel``: cosine(query, image embedding) only - the plain CLIP baseline.
* ``fused`` (default): mean of cosine(query, image embedding) and
  cosine(query, caption-chunk embedding). Every figure/table caption is also
  indexed as a text chunk tagged ``figure_ref``, so both scores are real Jina
  CLIP similarities; no extra model is used. The mean is gated by
  ``min_visual_score`` so that an image (~750 Phi-3-V tokens, ~75 s on CPU) is
  attached only when it is likely to be relevant.

The evaluation report compares both scoring rules and the threshold.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .config import RetrievalConfig
from .embedding_service import EmbeddingService
from .exceptions import EmbeddingError, RetrievalError, VectorStoreError
from .vector_store import Hit, VectorStore

logger = logging.getLogger(__name__)

MAX_QUERY_CHARS = 2000
_TEXT = {"modality": "text"}
_IMAGE = {"modality": "image"}
_CAPTIONS = {"$and": [{"modality": "text"}, {"figure_ref": {"$ne": ""}}]}


@dataclass
class RetrievedItem:
    id: str
    modality: str              # "text" | "image"
    content: str               # chunk text, or caption for visuals
    metadata: dict[str, Any]
    similarity: float          # ranking score: cosine sim (text, pixel) or fused visual score
    rank: int                  # 1-based rank within its modality list
    scores: dict[str, float] = field(default_factory=dict)  # component similarities

    @property
    def page(self) -> int:
        return int(self.metadata.get("page", 0))


@dataclass
class RetrievalResult:
    query: str
    mode: str
    text: list[RetrievedItem] = field(default_factory=list)
    images: list[RetrievedItem] = field(default_factory=list)
    rejected_images: list[RetrievedItem] = field(default_factory=list)
    sections: list[RetrievedItem] = field(default_factory=list)   # routed ToC sections
    timings: dict[str, float] = field(default_factory=dict)
    query_embedding: np.ndarray | None = None

    @property
    def top_text_similarity(self) -> float | None:
        return max((t.scores.get("text", t.similarity) for t in self.text), default=None)

    @property
    def is_empty(self) -> bool:
        return not self.text and not self.images

    def pages(self) -> list[int]:
        return sorted({i.page for i in self.text + self.images})


def validate_query(query: Any) -> str:
    if not isinstance(query, str):
        raise RetrievalError(f"Query must be a string, got {type(query).__name__}")
    q = " ".join(query.split())
    if not q:
        raise RetrievalError("Query is empty")
    if len(q) > MAX_QUERY_CHARS:
        raise RetrievalError(f"Query is too long ({len(q)} > {MAX_QUERY_CHARS} characters)")
    return q


def _scoped(where: dict, scope: dict | None) -> dict:
    """AND a metadata filter with an optional document scope."""
    if not scope:
        return where
    parts = where["$and"] if "$and" in where else [where]
    return {"$and": [*parts, scope]}


class Retriever:
    def __init__(self, store: VectorStore, embedder: EmbeddingService, cfg: RetrievalConfig):
        self.store = store
        self.embedder = embedder
        self.cfg = cfg

    def retrieve(self, query: str, mode: str | None = None, top_k_text: int | None = None,
                 top_k_image: int | None = None, visual_scoring: str | None = None,
                 min_visual_score: float | None = None, structure_routing: bool | None = None,
                 doc_ids: list[str] | None = None) -> RetrievalResult:
        """Retrieve text chunks and/or visuals relevant to *query*.

        Keyword arguments override the configured values (used by the CLI and evaluation).
        """
        q = validate_query(query)
        mode = mode or self.cfg.mode
        if mode not in {"text", "image", "multimodal"}:
            raise RetrievalError(f"Unknown retrieval mode {mode!r}")
        k_text = self.cfg.top_k_text if top_k_text is None else top_k_text
        k_img = self.cfg.top_k_image if top_k_image is None else top_k_image
        scoring = visual_scoring or self.cfg.visual_scoring
        threshold = self.cfg.min_visual_score if min_visual_score is None else min_visual_score
        routing = self.cfg.structure_routing if structure_routing is None else structure_routing
        if scoring not in {"pixel", "fused"}:
            raise RetrievalError(f"Unknown visual scoring {scoring!r}")

        try:
            if self.store.count() == 0:
                raise RetrievalError("The vector index is empty; run the indexing workflow first")
        except VectorStoreError as exc:
            raise RetrievalError(f"Vector index unavailable: {exc}") from exc

        result = RetrievalResult(query=q, mode=mode)
        t = time.perf_counter()
        try:
            qvec = self.embedder.embed_query(q)
        except EmbeddingError as exc:
            raise RetrievalError(f"Could not embed query: {exc}") from exc
        result.timings["query_embedding"] = time.perf_counter() - t
        result.query_embedding = qvec
        scope = {"doc_id": {"$in": list(doc_ids)}} if doc_ids else None

        t = time.perf_counter()
        try:
            if mode in ("text", "multimodal"):
                hits = self.store.query(qvec, k_text, _scoped(_TEXT, scope))
                if routing:
                    hits = self._route(result, qvec, hits, k_text, scope)
                result.text = [RetrievedItem(h.id, "text", h.document, h.metadata, h.similarity, r,
                                             {"text": h.similarity})
                               for r, h in enumerate(hits, start=1)]
            if mode in ("image", "multimodal") and k_img > 0:
                self._retrieve_visuals(result, qvec, k_img, scoring, threshold, scope)
        except VectorStoreError as exc:
            raise RetrievalError(f"Similarity search failed: {exc}") from exc
        result.timings["vector_search"] = time.perf_counter() - t

        logger.info("Retrieved %d text + %d image records (mode=%s, visual=%s, %d below threshold) "
                    "in %.3fs", len(result.text), len(result.images), mode, scoring,
                    len(result.rejected_images), sum(result.timings.values()))
        return result

    def _route(self, result: RetrievalResult, qvec: np.ndarray, flat: list[Hit], k: int,
               scope: dict | None) -> list[Hit]:
        """STAIR-style structure routing (zero-shot): pick the best ToC sections first,
        fill up to ``routed_text_slots`` with chunks from those sections, then the rest
        from the flat ranking. Falls back to ``flat`` when no section records exist."""
        secs = self.store.query(qvec, self.cfg.top_k_sections, _scoped({"modality": "section"}, scope))
        if not secs:
            return flat
        result.sections = [RetrievedItem(h.id, "section", h.document, h.metadata, h.similarity, r,
                                         {"section": h.similarity})
                           for r, h in enumerate(secs, start=1)]
        keys = [str(h.metadata["section_key"]) for h in secs]
        routed = self.store.query(qvec, self.cfg.routed_text_slots,
                                  _scoped({"$and": [_TEXT, {"section_key": {"$in": keys}}]}, scope))
        out = list(routed)
        seen = {h.id for h in out}
        for h in flat:
            if len(out) >= k:
                break
            if h.id not in seen:
                out.append(h)
                seen.add(h.id)
        return out

    def _retrieve_visuals(self, result: RetrievalResult, qvec: np.ndarray, k: int,
                          scoring: str, threshold: float, scope: dict | None = None) -> None:
        # Candidates are keyed by (doc_id, label): "Figure 1" is only unique within a document.
        def key(md: dict) -> tuple[str, str]:
            return str(md.get("doc_id", "")), str(md.get("label") or md.get("figure_ref"))

        pixel_hits: dict[tuple[str, str], Hit] = {}
        for h in self.store.query(qvec, k, _scoped(_IMAGE, scope)):
            pixel_hits[key(h.metadata)] = h
        caption_sims: dict[tuple[str, str], float] = {}
        if scoring == "fused":
            for h in self.store.query(qvec, k, _scoped(_CAPTIONS, scope)):
                caption_sims.setdefault(key(h.metadata), h.similarity)
            # Candidates found only through their caption still need a real pixel score.
            for doc_id, label in [kk for kk in caption_sims if kk not in pixel_hits]:
                got = self.store.collection.get(
                    where={"$and": [_IMAGE, {"label": label}, {"doc_id": doc_id}]},
                    include=["documents", "metadatas", "embeddings"])
                if got["ids"]:
                    sim = float(np.dot(np.asarray(got["embeddings"][0], np.float32), qvec))  # unit vectors
                    pixel_hits[(doc_id, label)] = Hit(got["ids"][0], got["documents"][0] or "",
                                                      got["metadatas"][0], 1.0 - sim)
            # Visuals ranked by pixels but whose captions fell outside the caption top-k.
            for doc_id, label in [kk for kk in pixel_hits if kk not in caption_sims]:
                got = self.store.collection.get(
                    where={"$and": [_TEXT, {"figure_ref": label}, {"doc_id": doc_id}]},
                    include=["embeddings"])
                if got["ids"]:
                    caption_sims[(doc_id, label)] = float(
                        np.dot(np.asarray(got["embeddings"][0], np.float32), qvec))

        items: list[RetrievedItem] = []
        for lab, h in pixel_hits.items():
            scores = {"pixel": h.similarity}
            if scoring == "fused" and lab in caption_sims:
                scores["caption"] = caption_sims[lab]
                score = (scores["pixel"] + scores["caption"]) / 2
            else:
                score = h.similarity  # uncaptioned image: pixel similarity only
            items.append(RetrievedItem(h.id, "image", h.document, h.metadata, score, 0, scores))
        items.sort(key=lambda i: i.similarity, reverse=True)
        for it in items:
            (result.images if it.similarity >= threshold else result.rejected_images).append(it)
        result.images = result.images[:k]
        for r, it in enumerate(result.images, start=1):
            it.rank = r
