"""Page-level text extraction with PyMuPDF.

Returns raw text *blocks* (PyMuPDF's paragraph-like units) with page numbers
and bounding boxes; cleaning and chunking happen in :mod:`preprocessing`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import fitz

from .exceptions import DocumentExtractionError

logger = logging.getLogger(__name__)

# Expand ligatures (ﬁ -> fi) and join words hyphenated across line breaks, but
# keep everything else (math symbols, Greek letters, superscripts) as-is.
TEXT_FLAGS = (fitz.TEXTFLAGS_TEXT & ~fitz.TEXT_PRESERVE_LIGATURES) | fitz.TEXT_DEHYPHENATE

# Pages with fewer extractable characters than this are flagged for inspection.
LOW_TEXT_CHARS = 50


@dataclass(frozen=True)
class TextBlock:
    page: int  # 1-based
    bbox: tuple[float, float, float, float]
    text: str


@dataclass
class PageText:
    page: int
    blocks: list[TextBlock]
    char_count: int
    image_count: int
    status: str  # "ok" | "empty" | "needs_visual_inspection"


@dataclass
class TextExtractionResult:
    pages: list[PageText] = field(default_factory=list)

    @property
    def blocks(self) -> list[TextBlock]:
        return [b for p in self.pages for b in p.blocks]

    @property
    def flagged_pages(self) -> list[int]:
        return [p.page for p in self.pages if p.status != "ok"]


def extract_text(doc: fitz.Document) -> TextExtractionResult:
    """Extract text blocks from every page of an open PDF.

    Pages with almost no text are classified as ``empty`` (nothing on them) or
    ``needs_visual_inspection`` (images/drawings but no text, i.e. probably a
    scanned page that would require OCR, which is outside the allowed libraries).
    """
    result = TextExtractionResult()
    for page in doc:
        try:
            raw = page.get_text("blocks", flags=TEXT_FLAGS)
            n_images = len(page.get_images(full=False))
            has_drawings = bool(page.get_drawings()) if n_images == 0 else True
        except (RuntimeError, ValueError) as exc:
            raise DocumentExtractionError(f"Text extraction failed on page {page.number + 1}: {exc}") from exc

        # block tuple: (x0, y0, x1, y1, text, block_no, block_type); type 0 = text
        blocks = [
            TextBlock(page=page.number + 1, bbox=tuple(b[:4]), text=b[4])
            for b in raw if b[6] == 0 and b[4].strip()
        ]
        chars = sum(len(b.text.strip()) for b in blocks)
        if chars >= LOW_TEXT_CHARS:
            status = "ok"
        elif n_images or has_drawings:
            status = "needs_visual_inspection"
            logger.warning("Page %d has %d chars of text but visual content; OCR is not "
                           "available, only its images will be indexed", page.number + 1, chars)
        else:
            status = "empty"
            logger.info("Page %d is empty", page.number + 1)
        result.pages.append(PageText(page.number + 1, blocks, chars, n_images, status))

    logger.info("Extracted %d text blocks from %d pages (%d flagged)",
                len(result.blocks), len(result.pages), len(result.flagged_pages))
    return result
