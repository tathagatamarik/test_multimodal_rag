"""Layout-aware PDF parsing: block roles and a table-of-contents (ToC) section tree.

Motivation: plain text extraction (``text_extractor`` + regex headings) loses the
document's global structure and mixes non-content into chunks. On this paper it
merged the conference footer into the Introduction and attached footnote 4
(about dot-product variance, which belongs to Sec. 3.2.1) to Sec. 3.2.2.
STAIR (Kumar et al., 2026, arXiv:2609.03874) argues that a document's ToC is a
better retrieval unit than length-based chunks, because ToC entries are semantically
coherent and clearly bounded. Our layout parser recovers that structure from
typography when the PDF has no outline (this one has none):

* body font size = the most common span size (by characters);
* headings: short blocks whose first span is bold and either numbered
  ("3.2.1") or in a small set of named headings; level = numbering depth, or
  1 for larger-than-body bold text; the PDF outline is used instead when present;
* title: the largest text on page 1;
* page numbers, running headers/footers: short blocks in the top/bottom margin,
  or small-font blocks in the bottom margin (e.g. the venue footer);
* footnotes: smaller-than-body blocks in the lower part of the page that start
  with a marker (digit, *, dagger);
* equations: blocks dominated by math fonts (Computer Modern / symbol fonts),
  kept attached to the preceding paragraph;
* captions / tables: ``Figure N:`` / ``Table N:`` captions and the indented
  table body that follows them.

Every content unit carries its full section path, e.g.
``3 Model Architecture > 3.2 Attention > 3.2.1 Scaled Dot-Product Attention``.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass, field

import fitz

from .exceptions import DocumentExtractionError
from .preprocessing import Para, caption_label, normalize_text
from .text_extractor import TEXT_FLAGS

logger = logging.getLogger(__name__)

ROLES = ("title", "front_matter", "heading", "paragraph", "equation", "caption", "table",
         "footnote", "page_number", "header_footer")
NON_CONTENT = {"page_number", "header_footer"}

_NUMBERED = re.compile(r"^(\d+(?:\.\d+){0,3})\s+(\S.{0,90})$")
_NAMED = {"abstract", "references", "acknowledgements", "acknowledgments", "appendix",
          "conclusion", "introduction", "background", "related work"}
_FOOTNOTE_MARK = re.compile(r"^\s*(\d{1,2}|[∗*†‡§¶])\s*\S")
_MATH_FONT = re.compile(r"^(CM(MI|SY|EX|R)|MSBM|MSAM|Symbol|Math)", re.IGNORECASE)
_MARGIN_FRAC = 0.07          # top/bottom margin band for headers/footers
_FOOTNOTE_ZONE_FRAC = 0.70   # footnotes live below this fraction of page height
_SMALL_FONT_RATIO = 0.95     # "smaller than body" threshold


@dataclass
class LayoutBlock:
    page: int
    bbox: tuple[float, float, float, float]
    text: str                      # normalised text
    role: str
    size: float                    # dominant font size
    bold: bool                     # first span bold
    math_ratio: float              # share of characters in math fonts
    section_path: tuple[str, ...] = ()
    heading_level: int = 0
    label: str = ""                # "Figure 2" / "Table 1" for captions and tables

    @property
    def section(self) -> str:
        return self.section_path[-1] if self.section_path else "Front matter"

    @property
    def path_str(self) -> str:
        return " > ".join(self.section_path) if self.section_path else "Front matter"


@dataclass
class SectionNode:
    """One ToC entry, with the text it covers."""

    number: str
    title: str
    level: int
    path: tuple[str, ...]
    page_start: int
    page_end: int
    text: list[str] = field(default_factory=list)

    @property
    def name(self) -> str:
        return f"{self.number} {self.title}".strip()

    @property
    def path_str(self) -> str:
        return " > ".join(self.path)


@dataclass
class LayoutDocument:
    blocks: list[LayoutBlock]
    sections: list[SectionNode]
    body_size: float
    title: str
    outline_used: bool
    role_counts: dict[str, int]

    def content_blocks(self) -> list[LayoutBlock]:
        return [b for b in self.blocks if b.role not in NON_CONTENT]

    def toc(self) -> list[str]:
        return [("  " * (s.level - 1)) + s.name for s in self.sections]


def _spans(block: dict) -> list[dict]:
    return [s for line in block.get("lines", []) for s in line.get("spans", []) if s["text"].strip()]


def _block_features(block: dict) -> tuple[str, float, bool, float]:
    spans = _spans(block)
    raw = "\n".join("".join(s["text"] for s in line["spans"]) for line in block["lines"])
    sizes = Counter()
    math_chars = 0
    total = 0
    for s in spans:
        n = len(s["text"].strip())
        sizes[round(s["size"], 1)] += n
        total += n
        if _MATH_FONT.match(s["font"].split("+")[-1]):
            math_chars += n
    size = sizes.most_common(1)[0][0] if sizes else 0.0
    bold = bool(spans) and (bool(spans[0]["flags"] & 16) or "Bold" in spans[0]["font"]
                            or "Medi" in spans[0]["font"])
    return raw, size, bold, (math_chars / total if total else 0.0)


def _heading(text: str, size: float, bold: bool, body: float) -> tuple[str, str, int] | None:
    """(number, title, level) if the block is a heading."""
    one_line = " ".join(text.split())
    if len(one_line) > 100 or not bold and size <= body * 1.05:
        return None
    m = _NUMBERED.match(one_line)
    if m and (bold or size > body * 1.05) and not one_line.endswith("."):
        number = m.group(1)
        return number, m.group(2).strip(), number.count(".") + 1
    if one_line.lower().rstrip(":") in _NAMED and (bold or size > body * 1.05):
        return "", one_line.rstrip(":"), 1
    return None


def parse_layout(doc: fitz.Document) -> LayoutDocument:
    """Classify every text block of *doc* and build the section tree."""
    try:
        pages = [(p.number + 1, p.rect, p.get_text("dict", flags=TEXT_FLAGS)["blocks"]) for p in doc]
    except (RuntimeError, ValueError) as exc:
        raise DocumentExtractionError(f"Layout extraction failed: {exc}") from exc

    feats = []
    size_chars: Counter = Counter()
    for pno, rect, blocks in pages:
        for b in blocks:
            if b.get("type") != 0:
                continue
            raw, size, bold, math = _block_features(b)
            if not raw.strip():
                continue
            feats.append((pno, rect, tuple(b["bbox"]), raw, size, bold, math))
            size_chars[size] += len(raw)
    if not feats:
        raise DocumentExtractionError("No text blocks found; the PDF may be scanned (OCR unavailable)")
    body = size_chars.most_common(1)[0][0]

    # Repeated short margin text across pages = running header/footer.
    margin_texts = Counter(" ".join(raw.split()) for pno, rect, bb, raw, *_ in feats
                           if bb[1] < rect.height * _MARGIN_FRAC or bb[3] > rect.height * (1 - _MARGIN_FRAC))

    outline = doc.get_toc(simple=True)
    title_block = max((f for f in feats if f[0] == 1), key=lambda f: f[4], default=None)

    blocks: list[LayoutBlock] = []
    for pno, rect, bb, raw, size, bold, math in feats:
        text = normalize_text(raw)
        one = " ".join(raw.split())
        in_top = bb[1] < rect.height * _MARGIN_FRAC
        in_bottom = bb[3] > rect.height * (1 - _MARGIN_FRAC)
        role, level, label = "paragraph", 0, ""
        if re.fullmatch(r"\d{1,3}", one) and (in_top or in_bottom):
            role = "page_number"
        elif (in_top or in_bottom) and (margin_texts[one] > 1 or size < body * _SMALL_FONT_RATIO):
            role = "header_footer"
        elif title_block is not None and (pno, bb) == (title_block[0], title_block[2]) and size > body * 1.3:
            role = "title"
        elif (h := _heading(raw, size, bold, body)) is not None:
            role, level = "heading", h[2]
        elif caption_label(text):
            role, label = "caption", caption_label(text)
        elif (size < body * _SMALL_FONT_RATIO and bb[1] > rect.height * _FOOTNOTE_ZONE_FRAC
              and _FOOTNOTE_MARK.match(one)):
            role = "footnote"
        elif math >= 0.5 or (math >= 0.2 and len(one) < 80):
            role = "equation"
        blocks.append(LayoutBlock(pno, bb, text, role, size, bold, round(math, 3), (), level, label))

    _attach_footnote_continuations(blocks, body)
    _mark_tables(blocks)
    _mark_front_matter(blocks)
    sections = _assign_sections(blocks, outline)
    counts = Counter(b.role for b in blocks)
    logger.info("Layout: body=%.1fpt, %d blocks %s, %d sections (outline=%s)",
                body, len(blocks), dict(counts), len(sections), bool(outline))
    return LayoutDocument(blocks, sections, body, title_block[3].strip() if title_block else "",
                          bool(outline), dict(counts))


def _attach_footnote_continuations(blocks: list[LayoutBlock], body: float) -> None:
    """Small-font blocks right after a footnote on the same page continue that footnote
    (e.g. the inline formula that ends footnote 4)."""
    for prev, b in zip(blocks, blocks[1:]):
        if (prev.role == "footnote" and b.page == prev.page and b.role in ("equation", "paragraph")
                and b.size < body * _SMALL_FONT_RATIO and b.bbox[1] >= prev.bbox[1] - 2):
            b.role = "footnote_cont"
    for i, b in enumerate(blocks):
        if b.role == "footnote_cont":
            b.role = "footnote"
            b.label = "continuation"


def _column(page_blocks: list[LayoutBlock]) -> tuple[float, float]:
    wide = [b for b in page_blocks if b.bbox[2] - b.bbox[0] > 300]
    if not wide:
        return (min((b.bbox[0] for b in page_blocks), default=0.0), 0.0)
    return min(b.bbox[0] for b in wide), max(b.bbox[2] - b.bbox[0] for b in wide)


def _mark_tables(blocks: list[LayoutBlock]) -> None:
    """Indented/narrow blocks directly after a table caption form the table body."""
    by_page: dict[int, list[int]] = {}
    for i, b in enumerate(blocks):
        by_page.setdefault(b.page, []).append(i)
    for idxs in by_page.values():
        page_blocks = [blocks[i] for i in idxs]
        left, width = _column(page_blocks)
        for pos, i in enumerate(idxs):
            cap = blocks[i]
            if cap.role != "caption" or not cap.label.startswith("Table"):
                continue
            for j in idxs[pos + 1:]:
                b = blocks[j]
                x0, _, x1, _ = b.bbox
                if b.role in NON_CONTENT or b.role in ("heading", "caption"):
                    break
                if not (x0 > left + 3 or (width > 0 and (x1 - x0) < 0.8 * width)):
                    break
                b.role, b.label = "table", cap.label


def _mark_front_matter(blocks: list[LayoutBlock]) -> None:
    """Page-1 blocks between the title and the first heading (authors, affiliations)."""
    for b in blocks:
        if b.page != 1 or b.role == "heading":
            if b.role == "heading" or b.page != 1:
                break
        if b.role == "paragraph":
            b.role = "front_matter"


def _assign_sections(blocks: list[LayoutBlock], outline: list) -> list[SectionNode]:
    sections: list[SectionNode] = []
    stack: list[SectionNode] = []
    current: SectionNode | None = None
    outline_titles = {" ".join(str(t).split()).lower(): lvl for lvl, t, _ in outline} if outline else {}

    for b in blocks:
        if b.role == "heading":
            one = " ".join(b.text.split())
            m = _NUMBERED.match(one)
            number, title = (m.group(1), m.group(2).strip()) if m else ("", one)
            level = outline_titles.get(one.lower(), b.heading_level or 1)
            b.heading_level = level
            while stack and stack[-1].level >= level:
                stack.pop()
            name = f"{number} {title}".strip()
            path = tuple(s.name for s in stack) + (name,)
            current = SectionNode(number, title, level, path, b.page, b.page)
            sections.append(current)
            stack.append(current)
            b.section_path = path
            continue
        if b.role in NON_CONTENT:
            continue
        b.section_path = current.path if current else ()
        if current:
            current.page_end = max(current.page_end, b.page)
            if b.role in ("paragraph", "equation", "caption", "table", "footnote"):
                current.text.append(b.text)
    return sections


def layout_paragraphs(layout: LayoutDocument, exclude_references: bool) -> list[Para]:
    """Convert classified blocks into paragraph units for :func:`pack_paragraphs`.

    * title + front matter -> one "Front matter" unit;
    * equations are appended to the preceding paragraph of the same section, so an
      equation stays with the sentence that introduces it;
    * a caption (plus its table body) becomes its own float unit, with section
      ``"Table 2: ..."`` and path ``<enclosing section path> > Table 2``;
    * footnotes become their own unit, tagged ``Footnote`` under the enclosing path;
    * page numbers and running headers/footers are dropped.
    """
    paras: list[Para] = []
    floats: dict[tuple[int, str], Para] = {}
    for b in layout.blocks:
        if b.role in NON_CONTENT or b.role == "heading":
            continue
        top = b.section_path[0] if b.section_path else ""
        if exclude_references and top == "References":
            continue
        path = b.path_str
        if b.role in ("title", "front_matter"):
            if paras and paras[-1].section == "Front matter":
                paras[-1].text += " " + b.text
            else:
                paras.append(Para(b.text, b.page, "Front matter", "Front matter"))
        elif b.role == "caption":
            cap = b.text.split(":", 1)[1].strip()[:80] if ":" in b.text else ""
            unit = Para(b.text, b.page, f"{b.label}: {cap}", f"{path} > {b.label}")
            floats[(b.page, b.label)] = unit
            paras.append(unit)
        elif b.role == "table":
            unit = floats.get((b.page, b.label))
            if unit is not None:
                unit.text += " " + b.text
            else:  # table body without a detected caption
                paras.append(Para(b.text, b.page, b.label or "Table", f"{path} > {b.label or 'Table'}"))
        elif b.role == "footnote":
            if b.label == "continuation" and paras and paras[-1].section == "Footnote":
                paras[-1].text += " " + b.text
            else:
                paras.append(Para(b.text, b.page, "Footnote", f"{path} > Footnote"))
        elif (b.role == "equation" and paras and paras[-1].path == path
              and paras[-1].section not in ("Footnote", "Front matter")):
            paras[-1].text += " " + b.text
        else:
            paras.append(Para(b.text, b.page, b.section, path))
    return paras
