"""HTTP API tests with fake models (skipped when the optional service packages are absent)."""

import time
import unittest
from unittest import mock

try:
    from fastapi.testclient import TestClient
    import services.api as api
except ImportError:  # core-only environment
    api = None

from tests.helpers import FakeJinaModel, FakePhi, FakeProcessor, make_project

from src.embedding_service import EmbeddingService
from src.generator import Phi3VisionGenerator
from src.pipeline import RAGPipeline


@unittest.skipIf(api is None, "fastapi not installed (pip install -r requirements-services.txt)")
class ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp, cls.cfg = make_project()
        emb = EmbeddingService(cls.cfg.embedding, model_loader=FakeJinaModel)
        gen = Phi3VisionGenerator(cls.cfg.generation,
                                  loader=lambda: (FakePhi(), FakeProcessor("Adam [T1, p.2].")))
        cls.pipe = RAGPipeline(cls.cfg, embedder=emb, generator=gen, telemetry_source="api")
        cls.pipe.index()
        cls.patch = mock.patch.object(api, "create_pipeline", return_value=cls.pipe)
        cls.patch.start()
        cls.client_cm = TestClient(api.app)
        cls.client = cls.client_cm.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.client_cm.__exit__(None, None, None)
        cls.patch.stop()
        cls.tmp.cleanup()

    def test_health_and_status(self):
        h = self.client.get("/health").json()
        self.assertEqual(h["status"], "ok")
        s = self.client.get("/index/status").json()
        self.assertEqual(s["indexed_documents"], 1)
        self.assertEqual(len(self.client.get("/documents").json()), 1)

    def test_query_retrieve_and_errors(self):
        r = self.client.post("/query", json={"question": "Which optimizer was used?"})
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(body["citations"][0]["tag"], "T1")
        self.assertIn("input", body["tokens"])
        r = self.client.post("/retrieve", json={"question": "attention diagram", "structure_routing": True})
        self.assertTrue(r.json()["sections"])
        vid = r.json()["images"][0]["visual_id"] if r.json()["images"] else None
        if vid:
            self.assertEqual(self.client.get(f"/visuals/{vid}").headers["content-type"], "image/png")
        self.assertEqual(self.client.post("/query", json={"question": ""}).status_code, 422)
        self.assertEqual(self.client.post("/query", json={"question": "   "}).status_code, 422)
        self.assertEqual(self.client.get("/visuals/..%2Fsecret").status_code, 404)

    def test_metrics_and_drift(self):
        self.client.post("/query", json={"question": "What is attention?"})
        s = self.client.get("/metrics/summary").json()
        self.assertGreaterEqual(s["live"]["n"], 1)
        self.assertIn("latency_total_p95", s["live"])
        rows = self.client.get("/metrics/queries").json()
        self.assertNotIn("embedding", rows[0])
        self.assertEqual(self.client.get("/monitoring/drift").json()["status"], "insufficient_data")

    def test_upload_ingestion_job(self):
        r = self.client.post("/ingest/upload", files={"file": ("notes.txt", b"hello", "text/plain")})
        self.assertEqual(r.status_code, 415)
        pdf = self.cfg.path(self.cfg.document.local_path).read_bytes()
        job = self.client.post("/ingest/upload", files={"file": ("copy.pdf", pdf, "application/pdf")}).json()
        for _ in range(100):
            j = self.client.get(f"/ingest/jobs/{job['job_id']}").json()
            if j["status"] in ("done", "failed"):
                break
            time.sleep(0.05)
        # Same bytes as the indexed paper -> content hash matches -> skipped, not duplicated.
        self.assertEqual(j["status"], "done")
        self.assertEqual(j["result"]["action"], "unchanged")
        self.assertEqual(self.client.get("/ingest/jobs/nope").status_code, 404)
        self.assertEqual(self.client.post("/ingest/batch", json={}).status_code, 422)


if __name__ == "__main__":
    unittest.main()
