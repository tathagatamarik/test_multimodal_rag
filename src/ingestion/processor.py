"""Pure document processing: validated PDF -> retrievable records (no store access).

Produces three record kinds, all keyed by deterministic IDs prefixed with the
document's content hash:

* ``text``    section-bounded chunks (layout-aware or basic parser);
* ``section`` one record per ToC entry (path + opening text). These are the
  STAIR-style structural retrieval units used for section routing;
* ``image``   one rendered region per figure/table caption.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any

from ..config import Config
from ..document_loader import DocumentInfo, open_pdf
from ..exceptions import DocumentExtractionError
from ..image_extractor import VisualRecord, extract_visuals
from ..layout_parser import SectionNode, layout_paragraphs, parse_layout
from ..preprocessing import TextChunk, caption_label, chunk_blocks, pack_paragraphs
from ..text_extractor import extract_text

logger = logging.getLogger(__name__)

SECTION_PREVIEW_CHARS = 600


@dataclass
class SectionRecord:
    record_id: str
    text: str
    path: str
    level: int
    page_start: int
    page_end: int


@dataclass
class ProcessedDocument:
    doc: DocumentInfo
    chunks: list[TextChunk]
    sections: list[SectionRecord]
    visuals: list[VisualRecord]
    stats: dict[str, Any] = field(default_factory=dict)
    timings: dict[str, float] = field(default_factory=dict)


_FLOAT_SUFFIX = re.compile(r" > (Figure \d+|Table \d+|Table|Footnote)$")


def section_key(path: str) -> str:
    """Enclosing ToC section of a chunk path (strips float/footnote suffixes)."""
    return _FLOAT_SUFFIX.sub("", path)


def text_metadata(c: TextChunk, doc: DocumentInfo) -> dict[str, Any]:
    return {"modality": "text", "doc_id": doc.doc_id, "source": doc.filename,
            "section_key": section_key(c.section_path or c.section),
            "page": c.page_start, "page_start": c.page_start, "page_end": c.page_end,
            "section": c.section, "section_path": c.section_path or c.section,
            "chunk_id": c.chunk_id, "ordinal": c.ordinal, "char_count": c.char_count,
            # "Figure 2" / "Table 3" when this chunk is a caption; links it to the visual record.
            "figure_ref": caption_label(c.section) or ""}


def section_metadata(s: SectionRecord, doc: DocumentInfo) -> dict[str, Any]:
    return {"modality": "section", "doc_id": doc.doc_id, "source": doc.filename,
            "page": s.page_start, "page_start": s.page_start, "page_end": s.page_end,
            "section_path": s.path, "section_key": s.path, "level": s.level}


def visual_metadata(v: VisualRecord, doc: DocumentInfo) -> dict[str, Any]:
    return {"modality": "image", "doc_id": doc.doc_id, "source": doc.filename,
            "page": v.page, "page_start": v.page, "page_end": v.page,
            "visual_id": v.visual_id, "kind": v.kind, "label": v.label, "image_path": v.path,
            "width": v.width, "height": v.height, "extraction": v.source,
            "bbox": ",".join(str(x) for x in v.bbox)}


def _section_records(sections: list[SectionNode], doc_id: str, exclude_references: bool) -> list[SectionRecord]:
    out = []
    for i, s in enumerate(sections):
        if exclude_references and s.path[0] == "References":
            continue
        body = " ".join(s.text)[:SECTION_PREVIEW_CHARS]
        if not body:
            continue  # pure container heading (e.g. "3.2 Attention" with no own text) is skipped
        out.append(SectionRecord(f"{doc_id}-section-{i:03d}", f"[{s.path_str}] {body}", s.path_str,
                                 s.level, s.page_start, s.page_end))
    return out


def process_document(doc: DocumentInfo, cfg: Config) -> ProcessedDocument:
    """Extract chunks, ToC section records and visuals from a validated PDF."""
    timings: dict[str, float] = {}
    sections: list[SectionRecord] = []
    stats: dict[str, Any] = {"parser": cfg.chunking.parser}
    with open_pdf(doc.path) as pdf:
        t = time.perf_counter()
        if cfg.chunking.parser == "layout":
            layout = parse_layout(pdf)
            chunks = pack_paragraphs(layout_paragraphs(layout, cfg.chunking.exclude_references),
                                     cfg.chunking, doc.doc_id)
            sections = _section_records(layout.sections, doc.doc_id, cfg.chunking.exclude_references)
            stats.update(roles=layout.role_counts, toc=layout.toc(), outline_used=layout.outline_used)
        else:
            text_res = extract_text(pdf)
            chunks = chunk_blocks(text_res.blocks, cfg.chunking, doc.doc_id)
            stats["flagged_pages"] = text_res.flagged_pages
        timings["extract_text"] = time.perf_counter() - t

        t = time.perf_counter()
        img_res = extract_visuals(pdf, cfg.images, doc.doc_id,
                                  cfg.path(cfg.images.output_dir), cfg.project_root)
        timings["extract_images"] = time.perf_counter() - t

    if not chunks and not img_res.visuals:
        raise DocumentExtractionError(f"Nothing to index in {doc.filename}: no text and no visuals")
    stats.update(embedded_images=len(img_res.embedded), rejected_images=len(img_res.rejected))
    logger.info("Processed %s: %d chunks, %d sections, %d visuals", doc.filename, len(chunks),
                len(sections), len(img_res.visuals))
    return ProcessedDocument(doc, chunks, sections, img_res.visuals, stats, timings)
