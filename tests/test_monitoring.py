"""Telemetry store, summaries and drift statistics."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.drift import detect_drift, embedding_drift, ks_2samp, psi
from src.telemetry import TelemetryStore, summarize


def response(lat=100.0, sim=0.7, tokens=900, insufficient=False, images=0):
    ev = [{"modality": "text", "similarity": sim}] + [{"modality": "image", "similarity": 0.45}] * images
    return {"question": "q", "run_id": "r", "timings": {"total": lat, "query_embedding": 0.05,
                                                        "vector_search": 0.01, "generation_generate": lat - 1},
            "tokens": {"input": tokens, "output": 60}, "evidence": ev,
            "retrieval": {"mode": "multimodal", "top_text_similarity": sim},
            "citations": [{}] if not insufficient else [], "invalid_citations": [],
            "insufficient_evidence": insufficient, "cold_start": False}


def unit(v):
    v = np.asarray(v, np.float32)
    return v / np.linalg.norm(v)


class StatsTests(unittest.TestCase):
    def test_psi_and_ks(self):
        rng = np.random.default_rng(0)
        a, b = rng.normal(0, 1, 2000), rng.normal(0, 1, 2000)
        self.assertLess(psi(a, b), 0.05)
        self.assertGreater(psi(a, rng.normal(1.5, 1, 2000)), 0.25)
        d, p = ks_2samp(a, b)
        self.assertGreater(p, 0.05)
        d, p = ks_2samp(a, a + 1.0)
        self.assertLess(p, 1e-6)
        self.assertTrue(np.isnan(psi([1.0], [1.0])))

    def test_embedding_drift_detects_out_of_domain(self):
        rng = np.random.default_rng(1)
        base = unit(rng.normal(size=16))
        ref = np.stack([unit(base + 0.1 * rng.normal(size=16)) for _ in range(30)])
        same = np.stack([unit(base + 0.1 * rng.normal(size=16)) for _ in range(30)])
        other = np.stack([unit(rng.normal(size=16)) for _ in range(30)])
        self.assertLess(embedding_drift(ref, same)["out_of_domain_share"], 0.2)
        self.assertGreater(embedding_drift(ref, other)["out_of_domain_share"], 0.8)


class TelemetryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = TelemetryStore(Path(self.tmp.name) / "t.db")
        self.rng = np.random.default_rng(2)
        self.base = unit(self.rng.normal(size=16))

    def tearDown(self):
        self.tmp.cleanup()

    def _log(self, n, source, shifted=False):
        for _ in range(n):
            if shifted:
                emb = unit(self.rng.normal(size=16))
                r = response(lat=self.rng.normal(300, 10), sim=self.rng.normal(0.35, 0.03), insufficient=True)
            else:
                emb = unit(self.base + 0.1 * self.rng.normal(size=16))
                r = response(lat=self.rng.normal(100, 10), sim=self.rng.normal(0.7, 0.03))
            self.store.record(r, source=source, embedding=emb, config_hash="c", prompt_version="v")

    def test_record_and_summarize(self):
        self._log(5, "api")
        rows = self.store.rows(source="api")
        self.assertEqual(len(rows), 5)
        s = summarize(rows)
        self.assertEqual(s["n"], 5)
        self.assertEqual(s["input_tokens_total"], 4500)
        self.assertEqual(s["abstention_rate"], 0.0)
        self.assertIsNotNone(s["latency_total_p95"])
        self.assertEqual(TelemetryStore.embeddings(rows).shape, (5, 16))

    def test_drift_stable_vs_shifted(self):
        self._log(20, "evaluation")
        self.assertEqual(detect_drift(self.store)["status"], "insufficient_data")
        self._log(20, "api")
        stable = detect_drift(self.store)
        self.assertEqual(stable["signals"]["query_embedding"]["level"], "stable")
        self.assertNotEqual(stable["status"], "significant")
        self._log(40, "api", shifted=True)
        shifted = detect_drift(self.store, current_limit=40)
        self.assertEqual(shifted["status"], "significant")
        self.assertEqual(shifted["signals"]["top_text_similarity"]["level"], "significant")
        self.assertEqual(shifted["signals"]["insufficient_rate"]["level"], "significant")


if __name__ == "__main__":
    unittest.main()
