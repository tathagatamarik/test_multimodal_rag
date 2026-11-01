"""PDF acquisition and validation.

* Reuses a valid local copy instead of re-downloading.
* Downloads to a ``.part`` file and atomically renames it only after the bytes
  are verified to be a readable PDF, so an interrupted or bogus download can
  never replace a good file.
* Retries only transient failures (timeouts, connection errors, HTTP 429/5xx)
  with bounded exponential backoff.
"""

from __future__ import annotations

import hashlib
import logging
import os
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import fitz
import requests

from .config import DocumentConfig
from .exceptions import DocumentLoadError

logger = logging.getLogger(__name__)

PDF_MAGIC = b"%PDF-"
_CHUNK_BYTES = 64 * 1024
_TRANSIENT_STATUS = {429, 500, 502, 503, 504}
_BACKOFF_BASE_S = 1.0


@dataclass(frozen=True)
class DocumentInfo:
    """Provenance record for the processed PDF."""

    path: str
    filename: str
    source: str
    sha256: str
    page_count: int
    title: str
    size_bytes: int
    loaded_at: str

    @property
    def doc_id(self) -> str:
        """Short, content-derived identifier used as an ID prefix."""
        return self.sha256[:12]

    def to_dict(self) -> dict:
        return asdict(self) | {"doc_id": self.doc_id}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(_CHUNK_BYTES), b""):
            h.update(block)
    return h.hexdigest()


@contextmanager
def open_pdf(path: str | Path) -> Iterator[fitz.Document]:
    """Open a PDF with PyMuPDF and always close it."""
    try:
        doc = fitz.open(str(path))
    except (fitz.FileDataError, RuntimeError, ValueError) as exc:
        raise DocumentLoadError(f"Cannot open '{path}' as a PDF: {exc}") from exc
    try:
        yield doc
    finally:
        doc.close()


def validate_pdf(path: str | Path, source: str | None = None) -> DocumentInfo:
    """Check that *path* is a non-empty, unencrypted, readable PDF with pages.

    Raises:
        DocumentLoadError: with an actionable message describing what is wrong.
    """
    p = Path(path)
    if not p.is_file():
        raise DocumentLoadError(f"PDF not found: {p}. Check document.local_path or enable download.")
    size = p.stat().st_size
    if size == 0:
        raise DocumentLoadError(f"PDF is empty (0 bytes): {p}. Delete it and re-run to download again.")
    with p.open("rb") as fh:
        head = fh.read(1024)
    if PDF_MAGIC not in head:
        raise DocumentLoadError(
            f"{p} does not start with a PDF header; it may be an HTML error page or a corrupt file."
        )
    with open_pdf(p) as doc:
        if doc.needs_pass:
            raise DocumentLoadError(f"{p} is encrypted/password-protected; supply a decrypted copy.")
        if doc.page_count == 0:
            raise DocumentLoadError(f"{p} contains no pages.")
        # Touch every page so truncated xref tables or broken page trees fail here
        # rather than halfway through extraction.
        try:
            for page in doc:
                page.get_text("text")
        except (RuntimeError, ValueError) as exc:
            raise DocumentLoadError(f"{p} is corrupt (page parsing failed): {exc}") from exc
        title = (doc.metadata or {}).get("title") or p.stem
        page_count = doc.page_count

    return DocumentInfo(
        path=str(p), filename=p.name, source=source or str(p), sha256=_sha256(p),
        page_count=page_count, title=title, size_bytes=size,
        loaded_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )


def download_pdf(url: str, dest: str | Path, *, timeout_s: float = 60.0, max_retries: int = 3,
                 max_bytes: int = 50 * 1024 * 1024,
                 session: requests.Session | None = None) -> Path:
    """Download *url* to *dest* atomically, validating it is a PDF.

    Raises:
        DocumentLoadError: on invalid URL, non-transient HTTP errors, oversize or
            non-PDF payloads, or when all retries for transient errors are exhausted.
    """
    if not url.startswith(("http://", "https://")):
        raise DocumentLoadError(f"Refusing to download non-http(s) URL: {url!r}")
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    http = session or requests.Session()
    last_error: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            logger.info("Downloading PDF (attempt %d/%d): %s", attempt, max_retries, url)
            with http.get(url, stream=True, timeout=timeout_s) as resp:
                if resp.status_code in _TRANSIENT_STATUS:
                    raise requests.HTTPError(f"transient HTTP {resp.status_code}", response=resp)
                if resp.status_code != 200:
                    raise DocumentLoadError(
                        f"Download failed with HTTP {resp.status_code} for {url} (not retried)."
                    )
                ctype = resp.headers.get("Content-Type", "")
                if "pdf" not in ctype.lower() and "octet-stream" not in ctype.lower():
                    logger.warning("Unexpected Content-Type %r; validating bytes anyway", ctype)
                expected = int(resp.headers.get("Content-Length") or 0)
                written = 0
                with tmp.open("wb") as fh:
                    for block in resp.iter_content(_CHUNK_BYTES):
                        written += len(block)
                        if written > max_bytes:
                            raise DocumentLoadError(
                                f"Download exceeds max_download_bytes={max_bytes}; aborting."
                            )
                        fh.write(block)
            if expected and written != expected:
                raise requests.ConnectionError(
                    f"incomplete download: got {written} of {expected} bytes"
                )
            validate_pdf(tmp, source=url)  # raises DocumentLoadError if not a usable PDF
            os.replace(tmp, dest)
            logger.info("Saved PDF to %s (%d bytes)", dest, written)
            return dest
        except (requests.ConnectionError, requests.Timeout, requests.HTTPError) as exc:
            last_error = exc
            tmp.unlink(missing_ok=True)
            if attempt < max_retries:
                delay = _BACKOFF_BASE_S * 2 ** (attempt - 1)
                logger.warning("Transient download error (%s); retrying in %.1fs", exc, delay)
                time.sleep(delay)
        except DocumentLoadError:
            tmp.unlink(missing_ok=True)
            raise
    raise DocumentLoadError(f"Download failed after {max_retries} attempts: {last_error}") from last_error


def load_document(cfg: DocumentConfig, resolve_path, *, allow_download: bool = True) -> DocumentInfo:
    """Return a validated local PDF, downloading it only if no valid copy exists.

    Args:
        cfg: document section of the configuration.
        resolve_path: callable mapping a configured relative path to an absolute one.
        allow_download: if False, a missing/invalid local file is an error.
    """
    local = resolve_path(cfg.local_path)
    if local.exists():
        try:
            info = validate_pdf(local, source=cfg.source_url)
            logger.info("Reusing local PDF %s (%d pages, sha256=%s)",
                        local, info.page_count, info.sha256[:12])
            return info
        except DocumentLoadError as exc:
            if not allow_download:
                raise
            # Keep the bad file for inspection instead of silently deleting it.
            quarantine = local.with_name(local.name + ".invalid")
            os.replace(local, quarantine)
            logger.warning("Local PDF invalid (%s); moved to %s and re-downloading", exc, quarantine)
    elif not allow_download:
        raise DocumentLoadError(f"PDF not found at {local} and downloading is disabled.")

    t0 = time.perf_counter()
    download_pdf(cfg.source_url, local, timeout_s=cfg.download_timeout_s,
                 max_retries=cfg.download_max_retries, max_bytes=cfg.max_download_bytes)
    info = validate_pdf(local, source=cfg.source_url)
    logger.info("Downloaded and validated PDF in %.2fs (%d pages)",
                time.perf_counter() - t0, info.page_count)
    return info
