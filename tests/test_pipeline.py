"""End-to-end indexing/query workflows with fake models, plus regression tests for
index reuse, incompatibility detection, interrupted/failed indexing and telemetry."""

import json
import logging
import unittest
from unittest import mock

from tests.helpers import FakeJinaModel, FakePhi, FakeProcessor, make_config, make_project

from src.embedding_service import EmbeddingService
from src.exceptions import EmbeddingError, IndexCompatibilityError, RetrievalError
from src.generator import INSUFFICIENT, ContextTooLongError, Phi3VisionGenerator
from src.ingestion.registry import Registry
from src.logging_config import setup_logging
from src.pipeline import RAGPipeline


def build(cfg, embed_model=None, reply="Adam was used [T1, p.2]."):
    emb = EmbeddingService(cfg.embedding, model_loader=lambda: embed_model or FakeJinaModel())
    proc, phi = FakeProcessor(reply), FakePhi()
    gen = Phi3VisionGenerator(cfg.generation, loader=lambda: (phi, proc))
    return RAGPipeline(cfg, embedder=emb, generator=gen), phi, proc


def registry(cfg):
    return Registry(cfg.path(cfg.vector_store.registry_path))


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmp, self.cfg = make_project()

    def tearDown(self):
        self.tmp.cleanup()

    def test_index_then_query(self):
        pipe, phi, proc = build(self.cfg)
        report = pipe.index()
        self.assertEqual(report.action, "built")
        self.assertEqual(report.visual_records, 2)          # Figure 1 + Table 1
        self.assertGreater(report.section_records, 0)       # ToC records (layout parser)
        self.assertEqual(report.total_records,
                         report.text_chunks + report.section_records + report.visual_records)
        for stage in ("load_pdf", "extract_text", "extract_images", "embed_text", "embed_images",
                      "chroma_index"):
            self.assertIn(stage, report.timings)
        reg = registry(self.cfg)
        [entry] = reg.indexed()
        self.assertEqual(entry.total_records, pipe.store.count())

        resp = pipe.ask("Which optimizer and warmup learning rate schedule were used?")
        self.assertEqual(resp.citations[0]["tag"], "T1")
        self.assertTrue(resp.evidence)
        self.assertIn(" > ", resp.evidence[0]["label"] + " > ")   # section-path labels
        self.assertIn("total", resp.timings)
        self.assertTrue(resp.cold_start)
        resp2 = pipe.ask("What is the attention diagram?")
        self.assertFalse(resp2.cold_start)                  # models reused, not reloaded
        json.dumps(resp2.to_dict())                          # response is serialisable

    def test_repeated_indexing_reuses_without_duplicates(self):
        pipe, _, _ = build(self.cfg)
        first = pipe.index()
        model = FakeJinaModel()
        pipe2, _, _ = build(self.cfg, embed_model=model)
        second = pipe2.index()
        self.assertEqual(second.action, "reused")
        self.assertEqual(model.calls, [])                    # nothing re-embedded
        self.assertEqual(pipe2.store.count(), first.total_records)

    def test_incompatible_config_detected_and_rebuild(self):
        build(self.cfg)[0].index()
        changed = make_config(self.cfg.project_root, chunking={"chunk_size_chars": 500})
        pipe, _, _ = build(changed)
        with self.assertRaisesRegex(IndexCompatibilityError, "--rebuild"):
            pipe.index()
        report = pipe.index(rebuild=True)
        self.assertEqual(report.action, "rebuilt")
        self.assertEqual(pipe.store.count(), report.total_records)

    def test_parser_switch_requires_rebuild(self):
        build(self.cfg)[0].index()
        basic = make_config(self.cfg.project_root, chunking={"parser": "basic"})
        with self.assertRaises(IndexCompatibilityError):
            build(basic)[0].index()

    def test_failed_indexing_is_not_reported_as_success(self):
        pipe, _, _ = build(self.cfg, embed_model=FakeJinaModel(fail_with=RuntimeError("bad weights")))
        with self.assertRaises(EmbeddingError):
            pipe.index()
        [entry] = registry(self.cfg).documents.values()
        self.assertEqual(entry.status, "failed")
        self.assertIn("bad weights", entry.error)
        self.assertEqual(registry(self.cfg).indexed(), [])
        # Next run retries the document instead of treating it as indexed.
        self.assertEqual(build(self.cfg)[0].index().action, "built")

    def test_interrupted_indexing_recovers(self):
        pipe, _, _ = build(self.cfg)
        with mock.patch.object(pipe.store, "upsert", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                pipe.index()
        [entry] = registry(self.cfg).documents.values()
        self.assertEqual(entry.status, "ingesting")          # crash left it dirty
        pipe.ingestion._lock.release()                        # simulate the crashed process exiting
        report = build(self.cfg)[0].index()
        self.assertEqual(report.action, "built")
        self.assertEqual(registry(self.cfg).indexed()[0].status, "indexed")

    def test_context_shrinks_when_prompt_too_long(self):
        pipe, _, _ = build(self.cfg)
        pipe.index()
        real = pipe.generator.generate
        calls = []

        def flaky(ctx):
            calls.append(sum(len(i.text) for i in ctx.items))
            if len(calls) == 1:
                raise ContextTooLongError(9999, 100)
            return real(ctx)
        with mock.patch.object(pipe.generator, "generate", side_effect=flaky):
            resp = pipe.ask("attention softmax dot products")
        self.assertEqual(len(calls), 2)
        self.assertLessEqual(calls[1], calls[0])
        self.assertTrue(resp.answer)

    def test_insufficient_evidence_passthrough(self):
        pipe, _, _ = build(self.cfg, reply=INSUFFICIENT)
        pipe.index()
        resp = pipe.ask("How many parameters does GPT-4 have?")
        self.assertTrue(resp.insufficient_evidence)
        self.assertEqual(resp.citations, [])

    def test_structure_routing_returns_routed_sections(self):
        pipe, _, _ = build(self.cfg)
        pipe.index()
        res = pipe.retrieve("Adam optimizer warmup", mode="text", structure_routing=True)
        self.assertTrue(res.sections)
        self.assertTrue(all(s.modality == "section" for s in res.sections))
        self.assertLessEqual(len(res.text), self.cfg.retrieval.top_k_text)
        self.assertEqual(len({t.id for t in res.text}), len(res.text))   # no duplicates

    def test_telemetry_records_success_and_failure(self):
        pipe, _, _ = build(self.cfg)
        pipe.index()
        pipe.ask("Which optimizer was used?", source="evaluation")
        with self.assertRaises(RetrievalError):
            pipe.ask("   ")
        rows = pipe.telemetry.rows()
        self.assertEqual([r["source"] for r in rows], ["evaluation", "cli"])
        self.assertGreater(rows[0]["latency_total"], 0)
        self.assertGreater(rows[0]["input_tokens"], 0)
        self.assertIsNotNone(rows[0]["embedding"])
        self.assertIn("RetrievalError", rows[1]["error"])

    def test_logging_configured_once(self):
        cfg = make_config(self.cfg.project_root, logging={"log_file": "logs/test.log", "level": "INFO"})
        root = setup_logging(cfg)
        setup_logging(cfg)
        ours = [h for h in root.handlers if getattr(h, "_mmrag_handler", False)]
        self.assertEqual(len(ours), 2)                       # console + file, not duplicated
        logging.getLogger("src.test").info("hello log")
        for h in ours:
            h.flush()
        text = cfg.path("logs/test.log").read_text()
        self.assertIn("hello log", text)
        self.assertIn("run=", text)
        for h in ours:                                       # restore quiet test logging
            root.removeHandler(h)
            h.close()


if __name__ == "__main__":
    unittest.main()
