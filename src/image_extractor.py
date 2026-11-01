"""Figure and table extraction.

Two complementary PyMuPDF mechanisms are used:

1. **Embedded raster extraction** (``Document.extract_image``): returns the
   original image bytes stored in the PDF. Every asset is decoded with Pillow,
   size-checked and de-duplicated by content hash. On their own, embedded
   images can be *fragments*: Figure 2 of this paper is stored as two separate
   rasters, and its panel titles are PDF text, not pixels.
2. **Region rendering** (``Page.get_pixmap(clip=...)``): rasterises a
   rectangle of the page exactly as a reader sees it, including vector graphics
   and text labels. Vector-drawn tables (Tables 1-3 here) exist *only* this way.

Strategy: each ``Figure N`` / ``Table N`` caption anchors one *visual record*.
For figures the region is the union of the embedded images above the caption,
any label text between them and the caption itself; for tables it is the
caption plus the table body. That gives one complete, captioned image per
figure/table instead of fragments or full-page renders (which would dilute the
CLIP embedding with unrelated prose). Embedded images with no caption are kept
as standalone records so no visual content is silently dropped.
"""

from __future__ import annotations

import hashlib
import io
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

import fitz
from PIL import Image, UnidentifiedImageError

from .config import ImageConfig
from .exceptions import DocumentExtractionError
from .preprocessing import caption_label, normalize_text, table_body
from .text_extractor import TEXT_FLAGS, TextBlock

logger = logging.getLogger(__name__)

_REGION_PAD_PT = 6.0          # breathing room around rendered regions
_LABEL_BAND_PT = 20.0         # how far above the images figure labels may sit
_CAPTION_GAP_PT = 120.0       # max vertical gap between a figure's images and its caption


@dataclass
class EmbeddedImage:
    page: int
    xref: int
    width: int
    height: int
    ext: str
    sha256: str
    rects: list[tuple[float, float, float, float]]


@dataclass
class VisualRecord:
    """One indexed visual: a figure, a table, or an uncaptioned image."""

    visual_id: str
    kind: str                # "figure" | "table" | "image"
    label: str               # "Figure 1", "Table 2", "Image p4-1"
    caption: str
    page: int
    bbox: tuple[float, float, float, float]
    width: int
    height: int
    path: str                # rendered PNG on disk (relative to project root when possible)
    source: str              # "embedded_raster_region" | "vector_region" | "embedded_raster"
    embedded_xrefs: list[int] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ImageExtractionResult:
    embedded: list[EmbeddedImage] = field(default_factory=list)
    visuals: list[VisualRecord] = field(default_factory=list)
    rejected: list[dict] = field(default_factory=list)


def to_model_rgb(img: Image.Image, max_side: int) -> Image.Image:
    """Convert any PIL mode to RGB (alpha composited on white) and cap the size."""
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        bg = Image.new("RGB", rgba.size, (255, 255, 255))
        bg.paste(rgba, mask=rgba.getchannel("A"))
        img = bg
    elif img.mode != "RGB":
        img = img.convert("RGB")
    if max(img.size) > max_side:
        img = img.copy()
        img.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return img


def decode_image(data: bytes) -> Image.Image:
    """Decode and fully verify image bytes; raises DocumentExtractionError."""
    try:
        with Image.open(io.BytesIO(data)) as probe:
            probe.verify()  # structural check; invalidates the object
        img = Image.open(io.BytesIO(data))
        img.load()  # force full decode so truncated data fails here
        return img
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as exc:
        raise DocumentExtractionError(f"Image bytes could not be decoded: {exc}") from exc


def _extract_embedded(doc: fitz.Document, cfg: ImageConfig,
                      result: ImageExtractionResult) -> None:
    seen: set[str] = set()
    for page in doc:
        for info in page.get_images(full=True):
            xref = info[0]
            try:
                raw = doc.extract_image(xref)
                if not raw or not raw.get("image"):
                    raise DocumentExtractionError("empty image stream")
                data: bytes = raw["image"]
                img = decode_image(data)
            except (DocumentExtractionError, RuntimeError, ValueError) as exc:
                # One bad asset must not abort the document: record and continue.
                logger.warning("Skipping undecodable image xref=%d on page %d: %s",
                               xref, page.number + 1, exc)
                result.rejected.append({"page": page.number + 1, "xref": xref, "reason": str(exc)})
                continue
            w, h = img.size
            digest = hashlib.sha256(data).hexdigest()
            reason = None
            if min(w, h) < cfg.min_image_side_px:
                reason = f"too small ({w}x{h} < {cfg.min_image_side_px}px)"
            elif digest in seen:
                reason = "duplicate of an earlier image"
            if reason:
                logger.info("Rejecting image xref=%d on page %d: %s", xref, page.number + 1, reason)
                result.rejected.append({"page": page.number + 1, "xref": xref, "reason": reason})
                continue
            seen.add(digest)
            rects = [tuple(r) for r in page.get_image_rects(xref)]
            result.embedded.append(EmbeddedImage(page.number + 1, xref, w, h,
                                                 raw.get("ext", "?"), digest, rects))


def _render(page: fitz.Page, rect: fitz.Rect, cfg: ImageConfig) -> Image.Image:
    pix = page.get_pixmap(dpi=cfg.render_dpi, clip=rect, alpha=False)
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    return to_model_rgb(img, cfg.max_image_side_px)


def _save(img: Image.Image, out_dir: Path, visual_id: str, project_root: Path) -> str:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{visual_id}.png"
    img.save(path, format="PNG", optimize=True)
    try:
        return str(path.relative_to(project_root))
    except ValueError:
        return str(path)


def extract_visuals(doc: fitz.Document, cfg: ImageConfig, doc_id: str,
                    out_dir: Path, project_root: Path) -> ImageExtractionResult:
    """Extract embedded images and render one captioned region per figure/table."""
    result = ImageExtractionResult()
    try:
        _extract_embedded(doc, cfg, result)
    except RuntimeError as exc:
        raise DocumentExtractionError(f"Embedded image enumeration failed: {exc}") from exc

    by_page: dict[int, list[EmbeddedImage]] = {}
    for e in result.embedded:
        by_page.setdefault(e.page, []).append(e)

    for page in doc:
        pno = page.number + 1
        blocks = [TextBlock(pno, tuple(b[:4]), b[4])
                  for b in page.get_text("blocks", flags=TEXT_FLAGS) if b[6] == 0 and b[4].strip()]
        unclaimed = list(by_page.get(pno, []))

        for idx, blk in enumerate(blocks):
            text = normalize_text(blk.text)
            label = caption_label(text)
            if not label:
                continue
            cap_rect = fitz.Rect(blk.bbox)
            region = fitz.Rect(cap_rect)
            xrefs: list[int] = []

            if label.startswith("Figure"):
                above = [e for e in unclaimed
                         if any(r[3] <= cap_rect.y0 + 1 and cap_rect.y0 - r[3] <= _CAPTION_GAP_PT
                                for r in e.rects)]
                if not above:
                    logger.warning("%s on page %d has no embedded raster above it; rendering caption "
                                   "band only", label, pno)
                for e in above:
                    for r in e.rects:
                        region.include_rect(fitz.Rect(r))
                    xrefs.append(e.xref)
                    unclaimed.remove(e)
                # Panel titles / labels drawn as text between the images and the caption.
                for b in blocks:
                    br = fitz.Rect(b.bbox)
                    if br.y0 >= region.y0 - _LABEL_BAND_PT and br.y1 <= cap_rect.y0 and b is not blk:
                        region.include_rect(br)
                kind, source = "figure", "embedded_raster_region"
            elif cfg.extract_tables:
                body = table_body(blocks, idx)
                if not body:
                    logger.warning("%s on page %d: no table body detected; skipped", label, pno)
                    continue
                for b in body:
                    region.include_rect(fitz.Rect(b.bbox))
                kind, source = "table", "vector_region"
            else:
                continue

            region = (region + (-_REGION_PAD_PT, -_REGION_PAD_PT, _REGION_PAD_PT, _REGION_PAD_PT)) & page.rect
            num = label.split()[1]
            visual_id = f"{doc_id}-{kind}-{int(num):02d}"
            img = _render(page, region, cfg)
            path = _save(img, out_dir, visual_id, project_root)
            result.visuals.append(VisualRecord(
                visual_id=visual_id, kind=kind, label=label, caption=text, page=pno,
                bbox=tuple(round(v, 1) for v in region), width=img.width, height=img.height,
                path=path, source=source, embedded_xrefs=xrefs,
            ))

        for n, e in enumerate(unclaimed, start=1):
            rect = fitz.Rect(e.rects[0]) if e.rects else page.rect
            visual_id = f"{doc_id}-image-p{pno:02d}-{n}"
            img = _render(page, rect, cfg)
            path = _save(img, out_dir, visual_id, project_root)
            result.visuals.append(VisualRecord(
                visual_id=visual_id, kind="image", label=f"Image p{pno}-{n}", caption="",
                page=pno, bbox=tuple(round(v, 1) for v in rect), width=img.width,
                height=img.height, path=path, source="embedded_raster", embedded_xrefs=[e.xref],
            ))

    logger.info("Images: %d embedded accepted, %d rejected; %d visual records (%s)",
                len(result.embedded), len(result.rejected), len(result.visuals),
                ", ".join(v.label for v in result.visuals))
    return result
