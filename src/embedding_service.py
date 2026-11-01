"""Jina CLIP v1 embedding service (text and images in one 768-d space).

API verification (jina-clip-implementation/modeling_clip.py, checked before use):
``JinaCLIPModel.encode_text(sentences, batch_size, normalize_embeddings=True,
**tokenizer_kwargs)`` and ``encode_image(images, batch_size,
normalize_embeddings=True)`` exist, run under ``torch.inference_mode`` and
return L2-normalised numpy arrays. Images may be PIL objects. Text and image
towers project into the same space, so cosine similarity between a text query
and an image embedding is meaningful (the model's contrastive objective).

We still drive batching ourselves so that an out-of-memory error on one batch
can be retried with a smaller batch instead of failing the whole index build.
"""

from __future__ import annotations

import logging
import time
from typing import Callable, Sequence, TypeVar

import numpy as np
import torch
from PIL import Image

from .config import EmbeddingConfig, resolve_device
from .exceptions import EmbeddingError

logger = logging.getLogger(__name__)

T = TypeVar("T")
_NORM_TOLERANCE = 1e-3


def _is_oom(exc: BaseException) -> bool:
    if isinstance(exc, torch.OutOfMemoryError):
        return True
    msg = str(exc).lower()
    return isinstance(exc, RuntimeError) and ("out of memory" in msg or "can't allocate memory" in msg)


def validate_embeddings(emb: np.ndarray, expected_rows: int, expected_dim: int | None) -> np.ndarray:
    """Check shape, finiteness and unit norm. Returns float32 array."""
    if not isinstance(emb, np.ndarray) or emb.ndim != 2:
        raise EmbeddingError(f"Expected a 2-D embedding matrix, got {type(emb).__name__} "
                             f"with shape {getattr(emb, 'shape', None)}")
    if emb.shape[0] != expected_rows:
        raise EmbeddingError(f"Expected {expected_rows} embeddings, got {emb.shape[0]}")
    if expected_dim is not None and emb.shape[1] != expected_dim:
        raise EmbeddingError(f"Embedding dimension {emb.shape[1]} != expected {expected_dim}")
    if not np.isfinite(emb).all():
        raise EmbeddingError("Embeddings contain NaN or Inf values")
    norms = np.linalg.norm(emb, axis=1)
    if np.any(np.abs(norms - 1.0) > _NORM_TOLERANCE):
        raise EmbeddingError(f"Embeddings are not L2-normalised (norm range "
                             f"{norms.min():.4f}..{norms.max():.4f})")
    return emb.astype(np.float32, copy=False)


class EmbeddingService:
    """Lazily loads Jina CLIP once and embeds texts, images and queries."""

    def __init__(self, cfg: EmbeddingConfig, model_loader: Callable[[], torch.nn.Module] | None = None):
        self.cfg = cfg
        self.device = resolve_device(cfg.device)
        self._model_loader = model_loader
        self._model: torch.nn.Module | None = None
        self.dim: int | None = None
        self.model_revision: str | None = None
        self.load_seconds: float | None = None

    # ------------------------------------------------------------------ loading
    def _default_loader(self) -> torch.nn.Module:
        from transformers import AutoModel

        return AutoModel.from_pretrained(self.cfg.model_id, revision=self.cfg.revision,
                                         trust_remote_code=True)

    @property
    def model(self) -> torch.nn.Module:
        if self._model is None:
            t0 = time.perf_counter()
            try:
                model = (self._model_loader or self._default_loader)()
            except (OSError, ValueError, ImportError, RuntimeError) as exc:
                raise EmbeddingError(f"Failed to load embedding model '{self.cfg.model_id}': {exc}") from exc
            for attr in ("encode_text", "encode_image"):
                if not callable(getattr(model, attr, None)):
                    raise EmbeddingError(f"Model '{self.cfg.model_id}' has no {attr}(); "
                                         "it is not a Jina CLIP compatible model")
            model.to(self.device).eval()
            self._model = model
            self.model_revision = getattr(getattr(model, "config", None), "_commit_hash", None)
            self.load_seconds = time.perf_counter() - t0
            logger.info("Loaded %s on %s in %.1fs (revision %s)", self.cfg.model_id,
                        self.device, self.load_seconds, self.model_revision)
        return self._model

    # ---------------------------------------------------------------- batching
    def _batched(self, items: Sequence[T], encode: Callable[[list[T], int], np.ndarray],
                 kind: str) -> np.ndarray:
        """Encode in batches, halving the batch size on OOM (bounded by size 1)."""
        out: list[np.ndarray] = []
        batch = self.cfg.batch_size
        i = 0
        while i < len(items):
            chunk = list(items[i:i + batch])
            try:
                out.append(np.asarray(encode(chunk, len(chunk)), dtype=np.float32).reshape(len(chunk), -1))
            except (RuntimeError, torch.OutOfMemoryError) as exc:
                if _is_oom(exc) and batch > 1:
                    batch = max(1, batch // 2)
                    logger.warning("OOM while embedding %s; retrying with batch_size=%d", kind, batch)
                    if self.device == "cuda":
                        torch.cuda.empty_cache()
                    continue
                raise EmbeddingError(f"Embedding {kind} batch starting at {i} failed: {exc}") from exc
            i += len(chunk)
            logger.debug("Embedded %s %d/%d", kind, i, len(items))
        return np.concatenate(out, axis=0) if out else np.zeros((0, self.dim or 0), np.float32)

    def _finish(self, emb: np.ndarray, n: int) -> np.ndarray:
        emb = validate_embeddings(emb, n, self.dim)
        if self.dim is None:
            self.dim = int(emb.shape[1])
        return emb

    # ---------------------------------------------------------------- public API
    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        """Embed passages (or queries) with the text tower. Returns (n, d) float32."""
        if not texts:
            return np.zeros((0, self.dim or 0), np.float32)
        if any(not isinstance(t, str) or not t.strip() for t in texts):
            raise EmbeddingError("All texts must be non-empty strings")
        model = self.model
        emb = self._batched(texts, lambda b, n: model.encode_text(
            b, batch_size=n, show_progress_bar=False, normalize_embeddings=True,
            max_length=self.cfg.max_text_tokens, truncation=True), "texts")
        return self._finish(emb, len(texts))

    def embed_images(self, images: Sequence[Image.Image]) -> np.ndarray:
        """Embed RGB PIL images with the vision tower. Returns (n, d) float32."""
        if not images:
            return np.zeros((0, self.dim or 0), np.float32)
        if any(not isinstance(im, Image.Image) for im in images):
            raise EmbeddingError("embed_images expects PIL.Image objects")
        model = self.model
        emb = self._batched(images, lambda b, n: model.encode_image(
            b, batch_size=n, show_progress_bar=False, normalize_embeddings=True), "images")
        return self._finish(emb, len(images))

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query with the text tower (same path as indexed passages)."""
        return self.embed_texts([query])[0]
