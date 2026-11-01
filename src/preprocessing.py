"""Text normalisation, section tracking and structure-aware chunking.

Chunking strategy (see README "Design decisions"):

* Units are PyMuPDF text blocks, i.e. the paper's own paragraphs, so chunk
  boundaries fall between paragraphs whenever possible.
* Paragraphs are packed into chunks of at most ``chunk_size_chars`` *within one
  section*. A paragraph that is longer than the budget is split at sentence
  boundaries.
* Consecutive chunks of the same section share up to ``chunk_overlap_chars`` of
  trailing whole sentences, so a statement cut at a boundary remains retrievable.
* Each chunk is prefixed with its section heading (e.g. ``[3.2.1 Scaled
  Dot-Product Attention]``) so it is understandable on its own and the heading
  words contribute to its embedding.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from .config import ChunkingConfig
from .text_extractor import TextBlock

logger = logging.getLogger(__name__)

_LIGATURES = {"ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl", "ﬅ": "st", "ﬆ": "st"}
_NUMBERED_HEADING = re.compile(r"^(\d+(?:\.\d+){0,3})\s*\n\s*([A-Z][^\n]{1,80})\s*$")
_NAMED_HEADINGS = {"abstract", "references", "acknowledgements", "acknowledgments"}
_PAGE_NUMBER = re.compile(r"^\s*\d{1,3}\s*$")
_CAPTION = re.compile(r"^(Figure|Table)\s+(\d+)\s*:", re.IGNORECASE)
# Sentence end: . ! ? followed by whitespace and an uppercase letter/digit/bracket.
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\[])")


@dataclass(frozen=True)
class TextChunk:
    """A retrievable unit of text with full provenance."""

    chunk_id: str
    text: str           # heading-prefixed text that is embedded and shown to the LLM
    section: str
    page_start: int
    page_end: int
    ordinal: int
    char_count: int
    section_path: str = ""   # "3 Model Architecture > 3.2 Attention > ..." (layout parser)

    @property
    def pages(self) -> list[int]:
        return list(range(self.page_start, self.page_end + 1))


def normalize_text(text: str) -> str:
    """Conservative cleanup: ligatures, line-wrap joins, whitespace runs.

    Mathematical symbols, Greek letters, sub/superscript digits and punctuation
    are left untouched because they carry technical meaning.
    """
    for lig, rep in _LIGATURES.items():
        text = text.replace(lig, rep)
    text = text.replace(" ", " ").replace("­", "")
    text = re.sub(r"\s*\n\s*", " ", text)
    return re.sub(r"[ \t]{2,}", " ", text).strip()


def detect_heading(block_text: str) -> str | None:
    """Return a normalised section title if *block_text* is a heading block."""
    stripped = block_text.strip()
    m = _NUMBERED_HEADING.match(stripped)
    if m:
        return f"{m.group(1)} {m.group(2).strip()}"
    if stripped.lower() in _NAMED_HEADINGS:
        return stripped.title()
    return None


def caption_label(text: str) -> str | None:
    """``'Figure 1'`` / ``'Table 3'`` if the text is a caption, else None."""
    m = _CAPTION.match(text.strip())
    return f"{m.group(1).title()} {m.group(2)}" if m else None


def split_sentences(text: str) -> list[str]:
    return [s for s in _SENTENCE_SPLIT.split(text) if s]


def body_column(blocks: list[TextBlock]) -> tuple[float, float]:
    """Estimate the main text column (left x, width) from wide blocks.

    The prose column is the left-most wide block; wide table rows are indented, so
    taking the minimum (not the median) stays correct on table-heavy pages.
    """
    wide = [b for b in blocks if b.bbox[2] - b.bbox[0] > 300]
    if not wide:
        return (min((b.bbox[0] for b in blocks), default=0.0), 0.0)
    return float(min(b.bbox[0] for b in wide)), max(b.bbox[2] - b.bbox[0] for b in wide)


def is_table_block(block: TextBlock, column: tuple[float, float]) -> bool:
    """Table cells are indented from, or much narrower than, the prose column."""
    left, width = column
    x0, _, x1, _ = block.bbox
    return x0 > left + 3 or (width > 0 and (x1 - x0) < 0.8 * width)


def table_body(page_blocks: list[TextBlock], caption_idx: int) -> list[TextBlock]:
    """Contiguous table-like blocks immediately following a table caption."""
    column = body_column(page_blocks)
    body: list[TextBlock] = []
    for b in page_blocks[caption_idx + 1:]:
        if _PAGE_NUMBER.match(b.text) or not is_table_block(b, column):
            break
        body.append(b)
    return body


@dataclass
class Para:
    """A paragraph-level unit with its section (leaf title) and full section path."""

    text: str
    page: int
    section: str
    path: str = ""


def blocks_to_paragraphs(blocks: list[TextBlock], exclude_references: bool) -> list[Para]:
    """Normalise blocks, drop page numbers, and attach the current section title.

    Figure captions and table captions+bodies are assigned to their own pseudo
    section (``"Table 2: The Transformer achieves..."``) so floats do not pollute
    the chunk of whatever prose section happens to surround them.
    """
    section = "Front matter"
    paras: list[Para] = []
    by_page: dict[int, list[TextBlock]] = {}
    for b in blocks:
        by_page.setdefault(b.page, []).append(b)

    for page, page_blocks in by_page.items():
        skip: set[int] = set()
        for i, b in enumerate(page_blocks):
            if i in skip or _PAGE_NUMBER.match(b.text):
                continue
            heading = detect_heading(b.text)
            if heading:
                section = heading
                continue
            if exclude_references and section == "References":
                continue
            text = normalize_text(b.text)
            label = caption_label(text)
            if label:
                float_section = f"{label}: {text.split(':', 1)[1].strip()[:80]}"
                parts = [text]
                if label.startswith("Table"):
                    body = table_body(page_blocks, i)
                    skip.update(range(i + 1, i + 1 + len(body)))
                    parts += [normalize_text(t.text) for t in body]
                paras.append(Para(" ".join(parts), page, float_section))
                continue
            if text:
                paras.append(Para(text, page, section))
    return paras


def _split_long(para: Para, limit: int) -> list[Para]:
    """Split an over-long paragraph at sentence boundaries (hard-split as last resort)."""
    pieces: list[str] = []
    current = ""
    for sent in split_sentences(para.text):
        while len(sent) > limit:  # a single "sentence" longer than the budget
            pieces.append(sent[:limit])
            sent = sent[limit:]
        if current and len(current) + 1 + len(sent) > limit:
            pieces.append(current)
            current = sent
        else:
            current = f"{current} {sent}".strip()
    if current:
        pieces.append(current)
    return [Para(p, para.page, para.section, para.path) for p in pieces]


def _overlap_tail(text: str, max_chars: int) -> str:
    """Trailing whole sentences of *text* totalling at most *max_chars*."""
    if max_chars <= 0:
        return ""
    tail: list[str] = []
    total = 0
    for sent in reversed(split_sentences(text)):
        if total + len(sent) > max_chars:
            break
        tail.insert(0, sent)
        total += len(sent) + 1
    return " ".join(tail)


def chunk_blocks(blocks: list[TextBlock], cfg: ChunkingConfig, doc_id: str) -> list[TextChunk]:
    """Basic parser path: regex headings over raw blocks, then :func:`pack_paragraphs`."""
    return pack_paragraphs(blocks_to_paragraphs(blocks, cfg.exclude_references), cfg, doc_id)


def pack_paragraphs(paras: list[Para], cfg: ChunkingConfig, doc_id: str) -> list[TextChunk]:
    """Pack paragraphs into section-bounded, overlapping chunks with stable IDs.

    The chunk prefix is the full section path when available (layout parser),
    otherwise the leaf section title.
    """
    body_budget = cfg.chunk_size_chars

    units: list[Para] = []
    for p in paras:
        units.extend(_split_long(p, body_budget) if len(p.text) > body_budget else [p])

    chunks: list[TextChunk] = []
    buf: list[Para] = []
    carry = ""  # overlap text from the previous chunk of the same section

    def flush() -> None:
        nonlocal buf, carry
        if not buf:
            return
        body = " ".join(u.text for u in buf)
        full_body = f"{carry} {body}".strip() if carry else body
        section = buf[0].section
        path = buf[0].path
        if len(body) >= cfg.min_chunk_chars or not chunks or chunks[-1].section != section:
            text = f"[{path or section}] {full_body}"
            chunks.append(TextChunk(
                chunk_id=f"{doc_id}-text-{len(chunks):04d}", text=text, section=section,
                page_start=min(u.page for u in buf), page_end=max(u.page for u in buf),
                ordinal=len(chunks), char_count=len(text), section_path=path,
            ))
        else:
            # Tiny remainder: merge into the previous chunk of the same section
            # instead of emitting a fragment that would embed poorly.
            prev = chunks[-1]
            text = f"{prev.text} {body}"
            chunks[-1] = TextChunk(prev.chunk_id, text, prev.section, prev.page_start,
                                   max(prev.page_end, buf[-1].page), prev.ordinal, len(text),
                                   prev.section_path)
        carry = _overlap_tail(body, cfg.chunk_overlap_chars)
        buf = []

    for u in units:
        if buf and (u.section, u.path) != (buf[0].section, buf[0].path):
            flush()
            carry = ""  # never overlap across sections
        size = sum(len(x.text) + 1 for x in buf)
        if buf and size + len(u.text) > body_budget:
            flush()
        buf.append(u)
    flush()

    logger.info("Chunked %d paragraphs into %d chunks (size=%d, overlap=%d)",
                len(paras), len(chunks), cfg.chunk_size_chars, cfg.chunk_overlap_chars)
    return chunks
