"""Config validation, PDF validation, downloading and text/image extraction."""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import requests
from PIL import Image

from tests.helpers import make_config, make_sample_pdf, png_bytes

import fitz

from src.config import config_from_dict, load_config, resolve_device
from src.document_loader import download_pdf, load_document, validate_pdf
from src.exceptions import ConfigurationError, DocumentExtractionError, DocumentLoadError
from src.image_extractor import decode_image, extract_visuals, to_model_rgb
from src.text_extractor import extract_text


class FakeResponse:
    def __init__(self, status=200, body=b"", headers=None, exc=None):
        self.status_code = status
        self._body = body
        self.headers = headers if headers is not None else {"Content-Type": "application/pdf",
                                                            "Content-Length": str(len(body))}
        self._exc = exc

    def iter_content(self, n):
        if self._exc:
            raise self._exc
        for i in range(0, len(self._body), n):
            yield self._body[i:i + n]

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def get(self, url, stream=True, timeout=None):
        self.calls += 1
        r = self.responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


class ConfigTests(unittest.TestCase):
    def test_project_config_is_valid(self):
        cfg = load_config()
        self.assertEqual(cfg.embedding.model_id, "jinaai/jina-clip-v1")
        self.assertEqual(cfg.generation.model_id, "microsoft/Phi-3-vision-128k-instruct")

    def test_invalid_values_are_all_reported(self):
        with self.assertRaises(ConfigurationError) as cm:
            config_from_dict({"chunking": {"chunk_size_chars": 100, "chunk_overlap_chars": 90},
                              "retrieval": {"mode": "fancy"}})
        msg = str(cm.exception)
        self.assertIn("chunk_size_chars", msg)
        self.assertIn("retrieval.mode", msg)

    def test_unknown_keys_rejected(self):
        with self.assertRaises(ConfigurationError):
            config_from_dict({"embedding": {"modle_id": "typo"}})
        with self.assertRaises(ConfigurationError):
            config_from_dict({"nonsense": {}})

    def test_missing_and_malformed_file(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ConfigurationError):
                load_config(Path(d) / "missing.json")
            bad = Path(d) / "bad.json"
            bad.write_text("{not json")
            with self.assertRaises(ConfigurationError):
                load_config(bad)

    def test_explicit_cuda_without_gpu_is_not_silently_overridden(self):
        with mock.patch("torch.cuda.is_available", return_value=False):
            self.assertEqual(resolve_device("auto"), "cpu")
            with self.assertRaises(ConfigurationError):
                resolve_device("cuda")

    def test_fingerprint_changes_with_chunking(self):
        with tempfile.TemporaryDirectory() as d:
            a = make_config(Path(d)).index_fingerprint()
            b = make_config(Path(d), chunking={"chunk_size_chars": 400}).index_fingerprint()
            self.assertNotEqual(a, b)


class ValidatePdfTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_valid_pdf_metadata(self):
        info = validate_pdf(make_sample_pdf(self.dir / "ok.pdf"), source="unit")
        self.assertEqual(info.page_count, 3)
        self.assertEqual(len(info.sha256), 64)
        self.assertEqual(info.doc_id, info.sha256[:12])
        self.assertEqual(info.source, "unit")

    def test_missing_file(self):
        with self.assertRaisesRegex(DocumentLoadError, "not found"):
            validate_pdf(self.dir / "nope.pdf")

    def test_empty_file(self):
        p = self.dir / "empty.pdf"
        p.write_bytes(b"")
        with self.assertRaisesRegex(DocumentLoadError, "empty"):
            validate_pdf(p)

    def test_html_error_page(self):
        p = self.dir / "fake.pdf"
        p.write_bytes(b"<html><body>404 Not Found</body></html>")
        with self.assertRaisesRegex(DocumentLoadError, "PDF header"):
            validate_pdf(p)

    def test_corrupt_pdf(self):
        good = make_sample_pdf(self.dir / "good.pdf").read_bytes()
        p = self.dir / "corrupt.pdf"
        p.write_bytes(good[:200])  # header intact, body truncated
        with self.assertRaises(DocumentLoadError):
            validate_pdf(p)

    def test_encrypted_pdf(self):
        src = make_sample_pdf(self.dir / "plain.pdf")
        p = self.dir / "enc.pdf"
        with fitz.open(str(src)) as doc:
            doc.save(str(p), encryption=fitz.PDF_ENCRYPT_AES_256, owner_pw="o", user_pw="u")
        with self.assertRaisesRegex(DocumentLoadError, "encrypted"):
            validate_pdf(p)

    def test_zero_page_pdf_bytes(self):
        p = self.dir / "nopages.pdf"
        p.write_bytes(b"%PDF-1.4\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
                      b"2 0 obj<</Type/Pages/Kids[]/Count 0>>endobj\ntrailer<</Root 1 0 R>>\n%%EOF")
        with self.assertRaises(DocumentLoadError):
            validate_pdf(p)


class DownloadTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.pdf_bytes = make_sample_pdf(self.dir / "src.pdf").read_bytes()
        patcher = mock.patch("src.document_loader.time.sleep")  # no real backoff waits
        self.sleep = patcher.start()
        self.addCleanup(patcher.stop)

    def tearDown(self):
        self.tmp.cleanup()

    def test_invalid_url(self):
        with self.assertRaisesRegex(DocumentLoadError, "non-http"):
            download_pdf("ftp://x/y.pdf", self.dir / "o.pdf")

    def test_success(self):
        s = FakeSession([FakeResponse(body=self.pdf_bytes)])
        out = download_pdf("https://x/p.pdf", self.dir / "o.pdf", session=s)
        self.assertEqual(out.read_bytes(), self.pdf_bytes)
        self.assertFalse((self.dir / "o.pdf.part").exists())

    def test_404_not_retried(self):
        s = FakeSession([FakeResponse(status=404), FakeResponse(body=self.pdf_bytes)])
        with self.assertRaisesRegex(DocumentLoadError, "HTTP 404"):
            download_pdf("https://x/p.pdf", self.dir / "o.pdf", session=s, max_retries=3)
        self.assertEqual(s.calls, 1)

    def test_transient_errors_retried_then_succeed(self):
        s = FakeSession([requests.ConnectionError("reset"), FakeResponse(status=503),
                         FakeResponse(body=self.pdf_bytes)])
        download_pdf("https://x/p.pdf", self.dir / "o.pdf", session=s, max_retries=3)
        self.assertEqual(s.calls, 3)
        self.assertEqual(self.sleep.call_count, 2)

    def test_retries_are_bounded(self):
        s = FakeSession([requests.Timeout("t")] * 5)
        with self.assertRaisesRegex(DocumentLoadError, "after 2 attempts"):
            download_pdf("https://x/p.pdf", self.dir / "o.pdf", session=s, max_retries=2)
        self.assertEqual(s.calls, 2)

    def test_incomplete_download_detected(self):
        r = FakeResponse(body=self.pdf_bytes[:100],
                         headers={"Content-Type": "application/pdf", "Content-Length": "999999"})
        with self.assertRaises(DocumentLoadError):
            download_pdf("https://x/p.pdf", self.dir / "o.pdf", session=FakeSession([r]), max_retries=1)

    def test_html_payload_never_overwrites_existing_file(self):
        dest = self.dir / "o.pdf"
        dest.write_bytes(self.pdf_bytes)
        r = FakeResponse(body=b"<html>blocked</html>", headers={"Content-Type": "text/html"})
        with self.assertRaises(DocumentLoadError):
            download_pdf("https://x/p.pdf", dest, session=FakeSession([r]))
        self.assertEqual(dest.read_bytes(), self.pdf_bytes)
        self.assertFalse((self.dir / "o.pdf.part").exists())

    def test_size_cap(self):
        r = FakeResponse(body=self.pdf_bytes)
        with self.assertRaisesRegex(DocumentLoadError, "max_download_bytes"):
            download_pdf("https://x/p.pdf", self.dir / "o.pdf", session=FakeSession([r]), max_bytes=100)

    def test_load_document_reuses_valid_local_copy(self):
        cfg = make_config(self.dir)
        local = cfg.path(cfg.document.local_path)
        local.parent.mkdir(parents=True)
        local.write_bytes(self.pdf_bytes)
        with mock.patch("src.document_loader.download_pdf") as dl:
            info = load_document(cfg.document, cfg.path)
        dl.assert_not_called()
        self.assertEqual(info.page_count, 3)

    def test_load_document_without_download(self):
        cfg = make_config(self.dir)
        with self.assertRaisesRegex(DocumentLoadError, "downloading is disabled"):
            load_document(cfg.document, cfg.path, allow_download=False)


class ExtractionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.cfg = make_config(self.dir)

    def tearDown(self):
        self.tmp.cleanup()

    def test_text_extraction_preserves_pages(self):
        with fitz.open(str(make_sample_pdf(self.dir / "a.pdf"))) as doc:
            res = extract_text(doc)
        self.assertEqual([p.page for p in res.pages], [1, 2, 3])
        self.assertTrue(all(p.status == "ok" for p in res.pages))
        self.assertIn("Scaled dot-product", " ".join(b.text for b in res.blocks if b.page == 1))

    def test_empty_and_image_only_pages_flagged(self):
        doc = fitz.open()
        doc.new_page()
        img_page = doc.new_page()
        img_page.insert_image(fitz.Rect(50, 50, 250, 200), stream=png_bytes())
        res = extract_text(doc)
        doc.close()
        self.assertEqual([p.status for p in res.pages], ["empty", "needs_visual_inspection"])

    def test_visuals_figure_and_table(self):
        pdf = make_sample_pdf(self.dir / "v.pdf", tiny_image=True, duplicate_image=True)
        with fitz.open(str(pdf)) as doc:
            res = extract_visuals(doc, self.cfg.images, "doc", self.dir / "vis", self.dir)
        labels = {v.label: v for v in res.visuals}
        self.assertEqual(set(labels), {"Figure 1", "Table 1"})
        self.assertTrue(labels["Figure 1"].embedded_xrefs)
        self.assertEqual(labels["Table 1"].source, "vector_region")
        self.assertEqual(labels["Figure 1"].visual_id, "doc-figure-01")  # deterministic id
        reasons = " ".join(r["reason"] for r in res.rejected)
        self.assertIn("too small", reasons)
        self.assertIn("duplicate", reasons)
        for v in res.visuals:
            with Image.open(self.dir / v.path) as im:
                self.assertEqual(im.mode, "RGB")
                self.assertEqual(im.size, (v.width, v.height))

    def test_uncaptioned_image_kept_as_standalone_record(self):
        doc = fitz.open()
        page = doc.new_page()
        page.insert_image(fitz.Rect(50, 50, 250, 200), stream=png_bytes())
        res = extract_visuals(doc, self.cfg.images, "doc", self.dir / "vis", self.dir)
        doc.close()
        self.assertEqual([v.kind for v in res.visuals], ["image"])

    def test_decode_failure(self):
        with self.assertRaises(DocumentExtractionError):
            decode_image(b"\x89PNG\r\n\x1a\nnot really a png")

    def test_to_model_rgb_modes_and_size(self):
        rgba = Image.new("RGBA", (100, 50), (255, 0, 0, 0))  # fully transparent red
        out = to_model_rgb(rgba, 1000)
        self.assertEqual(out.mode, "RGB")
        self.assertEqual(out.getpixel((0, 0)), (255, 255, 255))  # composited on white
        for mode in ("L", "CMYK", "P", "1"):
            self.assertEqual(to_model_rgb(Image.new(mode, (10, 10)), 1000).mode, "RGB")
        big = to_model_rgb(Image.new("RGB", (4000, 1000)), 2048)
        self.assertEqual(max(big.size), 2048)


if __name__ == "__main__":
    unittest.main()
