"""Turn retrieval results into model-ready, cited evidence.

* Text evidence is tagged ``[T1]..[Tn]`` and visual evidence ``[I1]..[In]``;
  every tag carries its page so the model can cite ``[T2, p.4]``.
* Adjacent chunks of the same section are merged and their overlap removed,
  and a figure caption chunk is dropped when its image is already attached
  (the image item carries the caption); table chunks are kept for their text.
* A character budget (``max_context_chars``) bounds the textual context; at most
  ``max_images`` images are attached because each image costs hundreds of
  Phi-3-V tokens and dominates CPU latency.
* Images are passed as real pixels (PIL) through Phi-3-V's ``<|image_k|>``
  pathway; only the caption is given as text, never a file path.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from PIL import Image

from .config import ContextConfig
from .exceptions import RetrievalError
from .retriever import RetrievalResult, RetrievedItem

logger = logging.getLogger(__name__)

_MIN_OVERLAP = 20


@dataclass
class EvidenceItem:
    tag: str                      # "T1" / "I1"
    modality: str
    text: str
    page_start: int
    page_end: int
    label: str                    # section title or "Figure 2"
    similarity: float
    source_ids: list[str]
    image_path: str | None = None
    source: str = ""              # document filename

    @property
    def page_ref(self) -> str:
        return (f"p.{self.page_start}" if self.page_start == self.page_end
                else f"pp.{self.page_start}-{self.page_end}")


@dataclass
class Context:
    question: str
    text_items: list[EvidenceItem] = field(default_factory=list)
    image_items: list[EvidenceItem] = field(default_factory=list)
    images: list[Image.Image] = field(default_factory=list)
    dropped: list[str] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not self.text_items and not self.image_items

    @property
    def items(self) -> list[EvidenceItem]:
        return self.image_items + self.text_items

    def evidence_block(self) -> str:
        """Plain-text evidence listing that goes into the prompt."""
        lines: list[str] = []
        for i, it in enumerate(self.image_items, start=1):
            # Refer to the picture in words only: every literal "<|image_k|>" in the prompt
            # is expanded into image tokens by the Phi-3-V processor.
            lines.append(f"[{it.tag}] ({it.label}, {it.page_ref}; shown as image {i} above) "
                         f"Caption: {it.text}")
        for it in self.text_items:
            lines.append(f"[{it.tag}] ({it.label}, {it.page_ref}) {it.text}")
        return "\n\n".join(lines)


def _strip_heading(text: str) -> str:
    return text.split("] ", 1)[1] if text.startswith("[") and "] " in text else text


def _merge_overlap(a: str, b: str) -> str:
    """Append *b* to *a*, removing the longest suffix-of-a / prefix-of-b overlap."""
    for k in range(min(len(a), len(b)), _MIN_OVERLAP - 1, -1):
        if a.endswith(b[:k]):
            return a + b[k:]
    return f"{a} {b}"


def _merge_adjacent(items: list[RetrievedItem]) -> list[tuple[list[RetrievedItem], str]]:
    """Group consecutive same-section chunks, keeping the best-ranked group first."""
    by_ord = sorted(items, key=lambda i: int(i.metadata.get("ordinal", 0)))
    groups: list[list[RetrievedItem]] = []
    for it in by_ord:
        prev = groups[-1][-1] if groups else None
        if (prev is not None and it.metadata.get("section") == prev.metadata.get("section")
                and int(it.metadata.get("ordinal", -9)) == int(prev.metadata.get("ordinal", -9)) + 1):
            groups[-1].append(it)
        else:
            groups.append([it])
    groups.sort(key=lambda g: min(i.rank for i in g))
    merged = []
    for g in groups:
        text = g[0].content
        for nxt in g[1:]:
            text = _merge_overlap(text, _strip_heading(nxt.content))
        merged.append((g, text))
    return merged


def default_image_loader(project_root: Path) -> Callable[[str], Image.Image]:
    def load(rel: str) -> Image.Image:
        path = Path(rel) if Path(rel).is_absolute() else project_root / rel
        try:
            with Image.open(path) as im:
                return im.convert("RGB")
        except (OSError, ValueError) as exc:
            raise RetrievalError(f"Retrieved image {path} cannot be loaded: {exc}") from exc
    return load


def build_context(result: RetrievalResult, cfg: ContextConfig,
                  load_image: Callable[[str], Image.Image]) -> Context:
    ctx = Context(question=result.query)

    attached_labels: set[str] = set()
    for item in result.images:
        if len(ctx.image_items) >= cfg.max_images:
            ctx.dropped.append(item.id)
            continue
        md = item.metadata
        path = md.get("image_path")
        if not path:
            logger.warning("Visual %s has no image_path; skipped", item.id)
            ctx.dropped.append(item.id)
            continue
        try:
            img = load_image(path)
        except RetrievalError as exc:
            logger.warning("%s", exc)
            ctx.dropped.append(item.id)
            continue
        ctx.images.append(img)
        attached_labels.add(str(md.get("label")))
        ctx.image_items.append(EvidenceItem(
            tag=f"I{len(ctx.image_items) + 1}", modality="image", text=item.content,
            page_start=item.page, page_end=item.page, label=str(md.get("label", "image")),
            similarity=item.similarity, source_ids=[item.id], image_path=path,
            source=str(md.get("source", ""))))

    budget = cfg.max_context_chars - sum(len(i.text) for i in ctx.image_items)
    for group, text in _merge_adjacent(result.text):
        ref = group[0].metadata.get("figure_ref")
        if ref and ref in attached_labels and ref.startswith("Figure"):
            # A figure caption chunk only repeats the caption carried by the image item.
            # Table chunks are kept: they hold the table body as text, which Phi-3-V reads
            # more reliably than small table pixels (evaluation run 1, q12).
            ctx.dropped.extend(i.id for i in group)
            continue
        # The section path becomes the evidence label (document structure shown to the
        # model), so the duplicated "[path] " prefix is removed from the body.
        text = _strip_heading(text)
        if len(text) > budget:
            ctx.dropped.extend(i.id for i in group)
            continue
        budget -= len(text)
        md0, mdl = group[0].metadata, group[-1].metadata
        ctx.text_items.append(EvidenceItem(
            tag=f"T{len(ctx.text_items) + 1}", modality="text", text=text,
            page_start=int(md0.get("page_start", md0.get("page", 0))),
            page_end=int(mdl.get("page_end", mdl.get("page", 0))),
            label=str(md0.get("section_path") or md0.get("section", "")),
            similarity=max(i.similarity for i in group),
            source_ids=[i.id for i in group], source=str(md0.get("source", ""))))

    logger.info("Context: %d text items, %d images, %d dropped, %d chars",
                len(ctx.text_items), len(ctx.image_items), len(ctx.dropped),
                sum(len(i.text) for i in ctx.items))
    return ctx
