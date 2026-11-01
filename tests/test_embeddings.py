"""Embedding validation, batching, OOM back-off and failure handling (no real model)."""

import unittest

import numpy as np
from PIL import Image

from tests.helpers import DIM, FakeJinaModel

from src.config import EmbeddingConfig
from src.embedding_service import EmbeddingService, validate_embeddings
from src.exceptions import EmbeddingError


def service(model=None, batch_size=4):
    m = model or FakeJinaModel()
    return EmbeddingService(EmbeddingConfig(model_id="fake", batch_size=batch_size, device="cpu"),
                            model_loader=lambda: m), m


class ValidateTests(unittest.TestCase):
    def test_accepts_unit_vectors(self):
        v = np.eye(3, 5, dtype=np.float32)
        self.assertEqual(validate_embeddings(v, 3, 5).shape, (3, 5))

    def test_rejects_bad_shapes_values_and_norms(self):
        with self.assertRaises(EmbeddingError):
            validate_embeddings(np.ones(5), 1, 5)                      # 1-D
        with self.assertRaises(EmbeddingError):
            validate_embeddings(np.eye(2, 5), 3, 5)                    # row count
        with self.assertRaises(EmbeddingError):
            validate_embeddings(np.eye(2, 5), 2, 7)                    # dimension mismatch
        nan = np.eye(2, 5)
        nan[0, 0] = np.nan
        with self.assertRaises(EmbeddingError):
            validate_embeddings(nan, 2, 5)                             # NaN
        with self.assertRaises(EmbeddingError):
            validate_embeddings(np.ones((2, 5)), 2, 5)                 # not normalised


class ServiceTests(unittest.TestCase):
    def test_text_and_image_shapes_and_order(self):
        svc, model = service(batch_size=2)
        texts = ["attention is all you need", "a", "positional encoding with sine"]
        emb = svc.embed_texts(texts)
        self.assertEqual(emb.shape, (3, DIM))
        self.assertEqual(model.calls, [2, 1])  # our own batching
        # Order preserved: re-embedding one text yields the same row.
        np.testing.assert_allclose(svc.embed_query(texts[2]), emb[2], rtol=1e-6)
        imgs = [Image.new("RGB", (32, 32), c) for c in [(255, 0, 0), (0, 255, 0)]]
        self.assertEqual(svc.embed_images(imgs).shape, (2, DIM))
        self.assertEqual(svc.dim, DIM)

    def test_model_loaded_once(self):
        loads = []
        m = FakeJinaModel()
        svc = EmbeddingService(EmbeddingConfig(model_id="fake", device="cpu"),
                               model_loader=lambda: loads.append(1) or m)
        svc.embed_texts(["a b"])
        svc.embed_texts(["c d"])
        self.assertEqual(len(loads), 1)

    def test_oom_halves_batch_and_recovers(self):
        svc, model = service(FakeJinaModel(oom_until_batch=2), batch_size=8)
        emb = svc.embed_texts([f"text {i}" for i in range(5)])
        self.assertEqual(emb.shape[0], 5)
        self.assertEqual(model.calls[:3], [5, 4, 2])  # 8->5 items fail, 4 fails, 2 succeeds

    def test_non_oom_error_is_not_retried(self):
        svc, model = service(FakeJinaModel(fail_with=RuntimeError("shape bug")))
        with self.assertRaisesRegex(EmbeddingError, "shape bug"):
            svc.embed_texts(["x"])
        self.assertEqual(len(model.calls), 1)

    def test_invalid_inputs(self):
        svc, _ = service()
        with self.assertRaises(EmbeddingError):
            svc.embed_texts(["ok", "   "])
        with self.assertRaises(EmbeddingError):
            svc.embed_images(["not an image"])
        self.assertEqual(svc.embed_texts([]).shape[0], 0)

    def test_load_failure_and_incompatible_model(self):
        def boom():
            raise OSError("repo not found")
        svc = EmbeddingService(EmbeddingConfig(model_id="missing", device="cpu"), model_loader=boom)
        with self.assertRaisesRegex(EmbeddingError, "Failed to load"):
            svc.embed_texts(["x"])
        import torch
        svc = EmbeddingService(EmbeddingConfig(model_id="plain", device="cpu"),
                               model_loader=lambda: torch.nn.Linear(2, 2))
        with self.assertRaisesRegex(EmbeddingError, "encode_text"):
            svc.embed_texts(["x"])


if __name__ == "__main__":
    unittest.main()
