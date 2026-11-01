"""Prompt construction, citation parsing and generator failure handling (fake Phi-3-V)."""

import unittest

from PIL import Image

from tests.helpers import FakePhi, FakeProcessor

from src.config import GenerationConfig
from src.context_builder import Context, EvidenceItem
from src.exceptions import GenerationError
from src.generator import (INSUFFICIENT, ContextTooLongError, Phi3VisionGenerator, build_prompt_text,
                           parse_citations)


def ctx_with(n_images=1, question="What is in Figure 2?", text="Softmax of QK^T over sqrt(dk)."):
    ctx = Context(question=question)
    for i in range(n_images):
        ctx.image_items.append(EvidenceItem(f"I{i + 1}", "image", "Figure 2: caption", 4, 4,
                                            "Figure 2", 0.4, ["img"], "x.png"))
        ctx.images.append(Image.new("RGB", (20, 20)))
    ctx.text_items.append(EvidenceItem("T1", "text", text, 3, 4, "3.2.1 Attention", 0.8, ["t1", "t2"]))
    return ctx


def generator(reply="It uses softmax [T1, pp.3-4] and the figure [I1].", n_new=5, fail=None, **cfg):
    proc, model = FakeProcessor(reply), FakePhi(n_new=n_new, fail_with=fail)
    gen = Phi3VisionGenerator(GenerationConfig(model_id="fake", device="cpu", **cfg),
                              loader=lambda: (model, proc))
    return gen, proc, model


class PromptTests(unittest.TestCase):
    def test_one_placeholder_per_image(self):
        # Regression: evidence text once contained "<|image_1|>", doubling the image tokens.
        for n in (0, 1, 2):
            prompt = build_prompt_text(ctx_with(n))
            self.assertEqual(prompt.count("<|image_"), n)

    def test_placeholders_in_retrieved_text_are_neutralised(self):
        prompt = build_prompt_text(ctx_with(1, question="see <|image_7|>", text="bad <|image_3|> text"))
        self.assertEqual(prompt.count("<|image_"), 1)

    def test_prompt_contains_tags_pages_question_and_rules_last(self):
        prompt = build_prompt_text(ctx_with(1))
        self.assertIn("[T1] (3.2.1 Attention, pp.3-4)", prompt)
        self.assertIn("[I1] (Figure 2, p.4", prompt)
        self.assertLess(prompt.index("QUESTION:"), prompt.index(INSUFFICIENT))

    def test_parse_citations_variants(self):
        ctx = ctx_with(1)
        cites, invalid = parse_citations("A [T1, p.3]. B [I1]. C [T1][T9]. D [T1, I1] E [figure].", ctx)
        self.assertEqual([c["tag"] for c in cites], ["T1", "I1"])
        self.assertEqual(cites[0]["pages"], [3, 4])
        self.assertEqual(invalid, ["T9"])

    def test_label_style_citations_mapped(self):
        ctx = ctx_with(1)
        cites, invalid = parse_citations("Shown in [Figure 2, p.4] and [3.2.1 Attention, p.3-4]. [p.4]", ctx)
        self.assertEqual([c["tag"] for c in cites], ["I1", "T1"])
        self.assertEqual(invalid, [])

    def test_prompt_example_is_not_a_factual_sentence(self):
        # Run-1 regression: a concrete example sentence was copied into an answer.
        self.assertNotIn("encoder has six layers", build_prompt_text(ctx_with(0)))


class GeneratorTests(unittest.TestCase):
    def test_successful_generation(self):
        gen, proc, model = generator()
        res = gen.generate(ctx_with(1))
        self.assertEqual(model.calls, 1)
        self.assertEqual(len(proc.last_images), 1)
        self.assertEqual([c["tag"] for c in res.citations], ["T1", "I1"])
        self.assertFalse(res.insufficient_evidence)
        self.assertEqual(res.output_tokens, 5)
        self.assertGreater(res.input_tokens, 0)

    def test_empty_context_does_not_call_model(self):
        gen, _, model = generator()
        res = gen.generate(Context(question="anything"))
        self.assertEqual(res.answer, INSUFFICIENT)
        self.assertTrue(res.insufficient_evidence)
        self.assertEqual(model.calls, 0)

    def test_model_abstention_detected(self):
        gen, _, _ = generator(reply=INSUFFICIENT)
        self.assertTrue(gen.generate(ctx_with(0)).insufficient_evidence)

    def test_context_too_long(self):
        gen, _, model = generator(max_input_tokens=256)
        with self.assertRaises(ContextTooLongError):
            gen.generate(ctx_with(1, text="word " * 400))
        self.assertEqual(model.calls, 0)

    def test_empty_output_and_runtime_failure(self):
        gen, _, _ = generator(n_new=0)
        with self.assertRaisesRegex(GenerationError, "empty"):
            gen.generate(ctx_with(0))
        gen, _, _ = generator(fail=RuntimeError("kernel crashed"))
        with self.assertRaisesRegex(GenerationError, "kernel crashed"):
            gen.generate(ctx_with(0))

    def test_load_failure_and_cpu_fp16_rejected(self):
        def boom():
            raise OSError("no such repo")
        gen = Phi3VisionGenerator(GenerationConfig(model_id="missing", device="cpu"), loader=boom)
        with self.assertRaisesRegex(GenerationError, "Failed to load"):
            gen.generate(ctx_with(0))
        gen = Phi3VisionGenerator(GenerationConfig(model_id="x", device="cpu", dtype="float16"))
        with self.assertRaisesRegex(GenerationError, "float16"):
            gen.load()


if __name__ == "__main__":
    unittest.main()
