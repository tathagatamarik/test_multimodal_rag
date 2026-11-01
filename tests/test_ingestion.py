"""Layout parsing, decoupled ingestion (batch / incremental / streaming) and its safety rules."""

import shutil
import tempfile
import time
import unittest
from pathlib import Path

from tests.helpers import FakeJinaModel, make_config, make_sample_pdf

import fitz

from src.document_loader import open_pdf
from src.embedding_service import EmbeddingService
from src.exceptions import VectorStoreError
from src.ingestion.registry import Registry
from src.ingestion.service import IngestionService
from src.ingestion.streaming import IngestionQueue, InboxWatcher
from src.layout_parser import layout_paragraphs, parse_layout
from src.vector_store import VectorStore


def variant_pdf(path: Path, marker: str) -> Path:
    """Same structure as the sample PDF but different content (=> different hash)."""
    make_sample_pdf(path)
    with fitz.open(str(path)) as doc:
        doc[2].insert_text((72, 700), f"Revision marker {marker}.", fontsize=10)
        tmp = path.with_suffix(".tmp.pdf")
        doc.save(str(tmp))
    tmp.replace(path)
    return path


class LayoutParserTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.pdf = make_sample_pdf(Path(self.tmp.name) / "s.pdf")
        with open_pdf(self.pdf) as doc:
            self.layout = parse_layout(doc)

    def tearDown(self):
        self.tmp.cleanup()

    def test_toc_and_roles(self):
        self.assertEqual(self.layout.toc(), ["1 Introduction", "2 Attention", "3 Training", "References"])
        c = self.layout.role_counts
        self.assertEqual((c["page_number"], c["header_footer"], c["footnote"], c["equation"]), (3, 3, 1, 1))
        self.assertEqual(c["table"], 3)

    def test_paragraphs_exclude_furniture_and_keep_structure(self):
        paras = layout_paragraphs(self.layout, exclude_references=True)
        text = " ".join(p.text for p in paras)
        self.assertNotIn("Workshop on Test Documents", text)        # running footer dropped
        self.assertFalse(any(p.section == "References" for p in paras))
        foot = [p for p in paras if p.section == "Footnote"]
        self.assertEqual(foot[0].path, "2 Attention > Footnote")
        table = [p for p in paras if p.section.startswith("Table 1")]
        self.assertIn("Model B 28.4 41.0", table[0].text)
        attn = [p for p in paras if p.section == "2 Attention"][0]
        self.assertIn("E = m c 2", attn.text)                       # equation attached to its paragraph


class IngestionServiceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.cfg = make_config(self.root)
        self.model = FakeJinaModel()
        self.svc = self._service()
        self.docs = self.root / "docs"
        self.docs.mkdir()

    def _service(self):
        emb = EmbeddingService(self.cfg.embedding, model_loader=lambda: self.model)
        store = VectorStore(self.cfg.path(self.cfg.vector_store.persist_dir), "ingest_test")
        return IngestionService(self.cfg, emb, store)

    def tearDown(self):
        self.tmp.cleanup()

    def test_batch_unchanged_and_failure_isolation(self):
        a = make_sample_pdf(self.docs / "a.pdf")
        b = variant_pdf(self.docs / "b.pdf", "B")
        bad = self.docs / "bad.pdf"
        bad.write_bytes(b"%PDF-1.4 garbage")
        report = self.svc.ingest_batch([a, bad, b])
        self.assertEqual(report.counts(), {"indexed": 2, "failed": 1})
        n = self.svc.store.count()
        calls = len(self.model.calls)
        again = self.svc.ingest_batch([a, b])
        self.assertEqual(again.counts(), {"unchanged": 2})
        self.assertEqual(len(self.model.calls), calls)               # nothing re-embedded
        self.assertEqual(self.svc.store.count(), n)

    def test_updated_source_supersedes_old_version(self):
        p = make_sample_pdf(self.docs / "paper.pdf")
        first = self.svc.ingest_file(p)
        variant_pdf(p, "v2")
        second = self.svc.ingest_file(p)
        self.assertEqual(second.action, "updated")
        self.assertNotEqual(first.doc_id, second.doc_id)
        self.assertEqual(self.svc.store.count({"doc_id": first.doc_id}), 0)
        self.assertEqual(self.svc.registry.get(first.doc_id).status, "removed")
        self.assertEqual(self.svc.store.count(), second.records)

    def test_sync_with_prune_and_remove(self):
        make_sample_pdf(self.docs / "a.pdf")
        variant_pdf(self.docs / "b.pdf", "B")
        self.assertEqual(self.svc.sync(self.docs).counts(), {"indexed": 2})
        (self.docs / "b.pdf").unlink()
        report = self.svc.sync(self.docs, prune=True)
        self.assertEqual(report.counts(), {"unchanged": 1, "removed": 1})
        self.assertEqual(len(self.svc.registry.indexed()), 1)
        doc_id = self.svc.registry.indexed()[0].doc_id
        self.svc.remove(doc_id)
        self.assertEqual(self.svc.store.count(), 0)

    def test_registry_persists_across_instances(self):
        r = self.svc.ingest_file(make_sample_pdf(self.docs / "a.pdf"))
        reg = Registry(self.cfg.path(self.cfg.vector_store.registry_path))
        self.assertEqual(reg.get(r.doc_id).status, "indexed")
        self.assertEqual(reg.get(r.doc_id).total_records, r.records)

    def test_writer_lock_is_exclusive_across_instances(self):
        other = self._service()
        with self.svc.writer():
            with self.assertRaisesRegex(VectorStoreError, "Another ingestion process"):
                other.ingest_file(make_sample_pdf(self.docs / "a.pdf"))
        other.ingest_file(self.docs / "a.pdf")                       # free again afterwards

    def test_queue_processes_jobs_in_background(self):
        q = IngestionQueue(self.svc)
        try:
            job = q.submit("file", path=str(make_sample_pdf(self.docs / "a.pdf")))
            bad = q.submit("file", path=str(self.docs / "missing.pdf"))
            for _ in range(200):
                if job.status in ("done", "failed") and bad.status in ("done", "failed"):
                    break
                time.sleep(0.05)
            self.assertEqual(job.status, "done")
            self.assertEqual(job.to_dict()["result"]["action"], "indexed")
            self.assertEqual(bad.status, "failed")
            self.assertIn("DocumentLoadError", bad.error)
        finally:
            q.stop()

    def test_inbox_watcher_waits_for_stable_file_then_moves_it(self):
        inbox = self.root / "inbox"
        w = InboxWatcher(self.svc, inbox, poll_s=0)
        shutil.copy(make_sample_pdf(self.docs / "a.pdf"), inbox / "a.pdf")
        (inbox / "broken.pdf").write_bytes(b"not a pdf")
        self.assertEqual(w.poll_once(), [])                            # first sighting: wait
        handled = dict(w.poll_once())
        self.assertEqual(handled["a.pdf"], "indexed")
        self.assertTrue(handled["broken.pdf"].startswith("failed"))
        self.assertTrue((inbox / "processed" / "a.pdf").exists())
        self.assertTrue((inbox / "failed" / "broken.pdf").exists())
        entry = self.svc.registry.indexed()[0]
        self.assertTrue(entry.path.endswith("processed/a.pdf"))      # rebuild can find it


if __name__ == "__main__":
    unittest.main()
