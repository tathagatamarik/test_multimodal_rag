"""ChromaDB wrapper, persistence, manifest and index reuse policy."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.exceptions import IndexCompatibilityError, VectorStoreError
from src.vector_store import VectorStore, read_manifest, write_manifest


def unit(rows, dim=4, seed=0):
    v = np.random.default_rng(seed).normal(size=(rows, dim)).astype(np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


class VectorStoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.store = VectorStore(self.dir / "chroma", "unit_test_col")
        self.store.create({"embedding_dim": 4, "fingerprint_hash": "abc"})

    def tearDown(self):
        self.tmp.cleanup()

    def _add(self, ids, emb, modality="text"):
        self.store.upsert(ids, emb, [f"doc {i}" for i in ids],
                          [{"modality": modality, "page": 1} for _ in ids])

    def test_insert_query_and_similarity(self):
        emb = unit(3)
        self._add(["a", "b", "c"], emb)
        hits = self.store.query(emb[1], 2)
        self.assertEqual(hits[0].id, "b")
        self.assertAlmostEqual(hits[0].similarity, 1.0, places=4)
        self.assertEqual(hits[0].document, "doc b")

    def test_metadata_filter(self):
        self._add(["t1"], unit(1, seed=1))
        self._add(["i1"], unit(1, seed=2), modality="image")
        hits = self.store.query(unit(1, seed=2)[0], 5, {"modality": "text"})
        self.assertEqual([h.id for h in hits], ["t1"])
        self.assertEqual(self.store.count({"modality": "image"}), 1)

    def test_upsert_is_idempotent(self):
        emb = unit(3)
        self._add(["a", "b", "c"], emb)
        self._add(["a", "b", "c"], emb)
        self.assertEqual(self.store.count(), 3)

    def test_persistence_and_reload(self):
        self._add(["a", "b"], unit(2))
        reopened = VectorStore(self.dir / "chroma", "unit_test_col")
        reopened.open(expected={"fingerprint_hash": "abc"})
        self.assertEqual(reopened.count(), 2)
        with self.assertRaises(IndexCompatibilityError):
            VectorStore(self.dir / "chroma", "unit_test_col").open(expected={"fingerprint_hash": "zzz"})

    def test_rejects_bad_input(self):
        with self.assertRaisesRegex(VectorStoreError, "Duplicate"):
            self._add(["a", "a"], unit(2))
        with self.assertRaisesRegex(VectorStoreError, "dimension"):
            self._add(["x"], unit(1, dim=8))
        with self.assertRaisesRegex(VectorStoreError, "Invalid metadata"):
            self.store.upsert(["m"], unit(1), ["d"], [{"pages": [1, 2], "nested": {"a": 1}}])
        bad = unit(1)
        bad[0, 0] = np.nan
        with self.assertRaises(VectorStoreError):
            self._add(["n"], bad)
        with self.assertRaisesRegex(VectorStoreError, "Length mismatch"):
            self.store.upsert(["a", "b"], unit(1), ["d"], [{}])

    def test_delete_where(self):
        self._add(["a", "b"], unit(2))
        self._add(["i"], unit(1, seed=3), modality="image")
        self.assertEqual(self.store.delete_where({"modality": "text"}), 2)
        self.assertEqual(self.store.count(), 1)
        self.assertEqual(self.store.delete_where({"modality": "text"}), 0)

    def test_empty_collection_query_returns_nothing(self):
        self.assertEqual(self.store.query(unit(1)[0], 3), [])

    def test_open_missing_collection(self):
        with self.assertRaisesRegex(VectorStoreError, "does not exist"):
            VectorStore(self.dir / "chroma", "other_collection").open()

    def test_create_twice_fails_and_delete(self):
        with self.assertRaises(VectorStoreError):
            VectorStore(self.dir / "chroma", "unit_test_col").create({"embedding_dim": 4})
        self.store.delete()
        self.assertFalse(self.store.exists())


class ManifestTests(unittest.TestCase):
    def test_roundtrip_and_corruption(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "m.json"
            self.assertIsNone(read_manifest(p))
            write_manifest(p, {"status": "complete", "x": 1})
            data = read_manifest(p)
            self.assertEqual(data["status"], "complete")
            self.assertIn("updated_at", data)
            p.write_text("{broken")
            with self.assertRaises(VectorStoreError):
                read_manifest(p)


if __name__ == "__main__":
    unittest.main()
