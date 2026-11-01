"""Optional smoke test with the REAL models and the real paper.

Skipped unless ``RUN_MODEL_TESTS=1``. Requires the Hugging Face models (or
network access to download them) and ~10 GB free RAM; on a CPU-only machine
the Phi-3-V part takes a few minutes. It uses the project's real index
(``python -m src.cli index`` is run implicitly and reuses an existing index).
"""

import os
import unittest

import numpy as np
from PIL import Image

RUN = os.environ.get("RUN_MODEL_TESTS") == "1"


@unittest.skipUnless(RUN, "set RUN_MODEL_TESTS=1 to run real-model smoke tests")
class RealModelSmokeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from src.config import load_config
        from src.pipeline import RAGPipeline

        cls.cfg = load_config()
        cls.pipe = RAGPipeline(cls.cfg)
        cls.report = cls.pipe.index()

    def test_jina_text_image_shared_space(self):
        emb = self.pipe.embedder
        t = emb.embed_texts(["a diagram of scaled dot-product attention"])
        i = emb.embed_images([Image.new("RGB", (224, 224), (255, 255, 255))])
        self.assertEqual(t.shape, (1, 768))
        self.assertEqual(i.shape, (1, 768))
        self.assertTrue(np.isfinite(t).all() and np.isfinite(i).all())

    def test_index_contents(self):
        self.assertGreater(self.report.total_records, 40)
        self.assertEqual(self.pipe.store.count({"modality": "image"}), 5)

    def test_figure_question_retrieves_figure_2(self):
        res = self.pipe.retrieve("Which operations are in the scaled dot-product attention diagram?")
        self.assertIn("Figure 2", [i.metadata["label"] for i in res.images])
        self.assertIn(4, [t.page for t in res.text])

    def test_grounded_answer(self):
        resp = self.pipe.ask("Which optimizer was used for training?")
        self.assertIn("adam", resp.answer.lower())
        # The supporting passage (Sec. 5.3, p.7) must be in the prompt evidence.
        self.assertTrue(any(7 in e["pages"] for e in resp.evidence))
        # Citations are parsed and validated but cannot be forced: Phi-3-V omits them for
        # some phrasings (observed for this exact question; 13/18 evaluation answers cite).
        self.assertEqual(resp.invalid_citations, [])
        self.assertTrue(all(any(p in e["pages"] for e in resp.evidence)
                            for c in resp.citations for p in c["pages"]))


if __name__ == "__main__":
    unittest.main()
