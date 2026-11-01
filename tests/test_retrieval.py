"""Query validation, retrieval modes, visual scoring and context construction."""

import unittest

from PIL import Image

from tests.helpers import FakeJinaModel, make_project

from src.config import ContextConfig
from src.context_builder import _merge_overlap, build_context
from src.embedding_service import EmbeddingService
from src.exceptions import RetrievalError
from src.pipeline import RAGPipeline
from src.retriever import RetrievalResult, RetrievedItem, validate_query
from src.vector_store import VectorStore


class QueryValidationTests(unittest.TestCase):
    def test_validation(self):
        self.assertEqual(validate_query("  what   is attention? "), "what is attention?")
        for bad in ["", "   ", None, 42, "x" * 5000]:
            with self.assertRaises(RetrievalError):
                validate_query(bad)


class RetrieverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp, cls.cfg = make_project()
        emb = EmbeddingService(cls.cfg.embedding, model_loader=FakeJinaModel)
        cls.pipe = RAGPipeline(cls.cfg, embedder=emb)
        cls.pipe.index()

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_text_mode_returns_relevant_chunk_with_provenance(self):
        res = self.pipe.retrieve("Adam optimizer warmup learning rate", mode="text")
        self.assertEqual(res.images, [])
        self.assertLessEqual(len(res.text), self.cfg.retrieval.top_k_text)
        self.assertEqual(res.text[0].metadata["section"], "3 Training")
        self.assertEqual(res.text[0].metadata["page"], 2)
        sims = [t.similarity for t in res.text]
        self.assertEqual(sims, sorted(sims, reverse=True))

    def test_image_and_multimodal_modes(self):
        img = self.pipe.retrieve("attention diagram", mode="image")
        self.assertFalse(img.text)
        self.assertTrue(img.images)
        self.assertTrue(all(i.modality == "image" for i in img.images))
        mm = self.pipe.retrieve("attention diagram query key value", mode="multimodal",
                                visual_scoring="fused")
        self.assertTrue(mm.text and mm.images)
        top = mm.images[0]
        self.assertEqual(set(top.scores), {"pixel", "caption"})
        self.assertAlmostEqual(top.similarity, (top.scores["pixel"] + top.scores["caption"]) / 2)

    def test_visual_threshold_rejects(self):
        res = self.pipe.retrieve("attention diagram", mode="multimodal", min_visual_score=0.99)
        self.assertEqual(res.images, [])
        self.assertTrue(res.rejected_images)

    def test_unknown_mode(self):
        with self.assertRaises(RetrievalError):
            self.pipe.retrieve("x", mode="audio")

    def test_empty_index(self):
        store = VectorStore(self.cfg.path("storage/empty"), "empty_col")
        store.create({"embedding_dim": 4})
        pipe = RAGPipeline(self.cfg, embedder=self.pipe.embedder, store=store)
        with self.assertRaisesRegex(RetrievalError, "empty"):
            pipe.retrieve("anything")


def item(i, section, ordinal, text, rank, page=1, modality="text", **md):
    meta = {"section": section, "ordinal": ordinal, "page": page, "page_start": page, "page_end": page,
            "figure_ref": md.pop("figure_ref", "")} | md
    return RetrievedItem(f"id{i}", modality, text, meta, 1.0 - rank / 10, rank)


class ContextBuilderTests(unittest.TestCase):
    def loader(self, path):
        if path == "missing.png":
            raise RetrievalError("cannot load")
        return Image.new("RGB", (10, 10))

    def test_merge_overlap(self):
        # Overlaps shorter than 20 chars are ignored on purpose (could be coincidental).
        a = "Intro text. The decoder is also composed of six layers."
        b = "The decoder is also composed of six layers. It masks future positions."
        self.assertEqual(_merge_overlap(a, b),
                         "Intro text. The decoder is also composed of six layers. It masks future positions.")
        self.assertEqual(_merge_overlap("abc", "xyz"), "abc xyz")

    def test_adjacent_chunks_merged_and_ranked(self):
        res = RetrievalResult("q", "text", text=[
            item(1, "S", 5, "[S] Alpha beta gamma delta epsilon zeta eta theta.", 2),
            item(2, "S", 6, "[S] epsilon zeta eta theta. Iota kappa.", 3),
            item(3, "T", 9, "[T] Unrelated best hit.", 1)])
        ctx = build_context(res, ContextConfig(max_context_chars=5000, max_images=1), self.loader)
        self.assertEqual([i.tag for i in ctx.text_items], ["T1", "T2"])
        self.assertEqual(ctx.text_items[0].source_ids, ["id3"])  # best rank first
        merged = ctx.text_items[1]
        self.assertEqual(merged.source_ids, ["id1", "id2"])
        self.assertEqual(merged.text.count("epsilon zeta eta theta."), 1)

    def test_budget_and_image_limits(self):
        res = RetrievalResult("q", "multimodal",
                              text=[item(1, "A", 1, "x" * 900, 1), item(2, "B", 5, "y" * 900, 2)],
                              images=[item(3, "", 0, "Figure 1: cap", 1, modality="image",
                                           label="Figure 1", image_path="ok.png"),
                                      item(4, "", 0, "Table 1: cap", 2, modality="image",
                                           label="Table 1", image_path="ok.png")])
        ctx = build_context(res, ContextConfig(max_context_chars=1000, max_images=1), self.loader)
        self.assertEqual(len(ctx.images), 1)
        self.assertEqual(len(ctx.text_items), 1)
        self.assertIn("id2", ctx.dropped)
        self.assertIn("id4", ctx.dropped)

    def test_caption_chunk_deduplicated_and_bad_image_skipped(self):
        res = RetrievalResult("q", "multimodal",
                              text=[item(1, "Figure 1: cap", 3, "[Figure 1] Figure 1: cap", 1,
                                         figure_ref="Figure 1")],
                              images=[item(2, "", 0, "Figure 1: cap", 1, modality="image",
                                           label="Figure 1", image_path="ok.png")])
        ctx = build_context(res, ContextConfig(max_context_chars=1000, max_images=2), self.loader)
        self.assertEqual(ctx.text_items, [])
        self.assertEqual(ctx.image_items[0].tag, "I1")
        res.images[0].metadata["image_path"] = "missing.png"
        ctx = build_context(res, ContextConfig(max_context_chars=1000, max_images=2), self.loader)
        self.assertEqual(ctx.images, [])
        self.assertEqual(len(ctx.text_items), 1)  # caption text kept when image failed

    def test_table_chunk_kept_when_table_image_attached(self):
        res = RetrievalResult("q", "multimodal",
                              text=[item(1, "Table 1: cap", 3, "[Table 1] Table 1: cap Layer O(1)", 1,
                                         figure_ref="Table 1")],
                              images=[item(2, "", 0, "Table 1: cap", 1, modality="image",
                                           label="Table 1", image_path="ok.png")])
        ctx = build_context(res, ContextConfig(max_context_chars=1000, max_images=1), self.loader)
        self.assertEqual(len(ctx.image_items), 1)
        self.assertIn("O(1)", ctx.text_items[0].text)  # run-1 q12 regression

    def test_empty_result(self):
        ctx = build_context(RetrievalResult("q", "text"), ContextConfig(), self.loader)
        self.assertTrue(ctx.is_empty)


if __name__ == "__main__":
    unittest.main()
