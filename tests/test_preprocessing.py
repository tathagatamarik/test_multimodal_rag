"""Normalisation, heading detection and chunking."""

import unittest

from src.config import ChunkingConfig
from src.preprocessing import (caption_label, chunk_blocks, detect_heading, normalize_text,
                               split_sentences)
from src.text_extractor import TextBlock


def blk(text, page=1, x0=108.0, width=396.0, y=100.0):
    return TextBlock(page=page, bbox=(x0, y, x0 + width, y + 20), text=text)


class NormalizeTests(unittest.TestCase):
    def test_ligatures_and_whitespace(self):
        self.assertEqual(normalize_text("signiﬁcantly  more\n  eﬃcient models"),
                         "significantly more efficient models")

    def test_math_symbols_preserved(self):
        s = "softmax(QK^T/√dk) with ϵls = 0.1 and β2 = 0.98 · 10−9"
        self.assertEqual(normalize_text(s), s)

    def test_headings(self):
        self.assertEqual(detect_heading("3.2.1\nScaled Dot-Product Attention\n"),
                         "3.2.1 Scaled Dot-Product Attention")
        self.assertEqual(detect_heading("References\n"), "References")
        self.assertIsNone(detect_heading("We trained the model for 3 days.\n"))
        self.assertIsNone(detect_heading("2\n"))

    def test_caption_label(self):
        self.assertEqual(caption_label("Figure 2: (left) Scaled"), "Figure 2")
        self.assertEqual(caption_label("Table 3: Variations"), "Table 3")
        self.assertIsNone(caption_label("As shown in Figure 2, the model"))

    def test_sentence_split(self):
        self.assertEqual(split_sentences("A b. C d? E f."), ["A b.", "C d?", "E f."])


class ChunkingTests(unittest.TestCase):
    cfg = ChunkingConfig(chunk_size_chars=200, chunk_overlap_chars=60, min_chunk_chars=20)

    def _blocks(self):
        para = " ".join(f"Sentence number {i} talks about attention." for i in range(12))
        return [blk("1\nIntroduction\n", y=60), blk(para, y=100),
                blk("2\nMethod\n", page=2, y=60), blk("The method section is short.", page=2, y=100),
                blk("References\n", page=3, y=60), blk("[1] Someone. A paper. 2017.", page=3, y=100)]

    def test_size_bound_and_overlap(self):
        chunks = chunk_blocks(self._blocks(), self.cfg, "doc")
        intro = [c for c in chunks if c.section == "1 Introduction"]
        self.assertGreater(len(intro), 1)
        for c in intro:
            body = c.text.split("] ", 1)[1]
            self.assertLessEqual(len(body), self.cfg.chunk_size_chars + self.cfg.chunk_overlap_chars + 1)
        # Overlap: the tail sentence of chunk i starts chunk i+1.
        first_body = intro[0].text.split("] ", 1)[1]
        last_sentence = split_sentences(first_body)[-1]
        self.assertIn(last_sentence, intro[1].text)

    def test_no_overlap_across_sections_and_heading_prefix(self):
        chunks = chunk_blocks(self._blocks(), self.cfg, "doc")
        method = [c for c in chunks if c.section == "2 Method"]
        self.assertEqual(len(method), 1)
        self.assertEqual(method[0].text, "[2 Method] The method section is short.")
        self.assertEqual((method[0].page_start, method[0].page_end), (2, 2))

    def test_references_excluded_by_default_and_optional(self):
        self.assertFalse(any(c.section == "References" for c in chunk_blocks(self._blocks(), self.cfg, "d")))
        cfg = ChunkingConfig(200, 60, 20, exclude_references=False)
        self.assertTrue(any(c.section == "References" for c in chunk_blocks(self._blocks(), cfg, "d")))

    def test_deterministic_ids(self):
        a = chunk_blocks(self._blocks(), self.cfg, "doc")
        b = chunk_blocks(self._blocks(), self.cfg, "doc")
        self.assertEqual([c.chunk_id for c in a], [c.chunk_id for c in b])
        self.assertEqual(len({c.chunk_id for c in a}), len(a))
        self.assertTrue(a[0].chunk_id.startswith("doc-text-"))

    def test_table_caption_and_body_form_own_chunk(self):
        blocks = [blk("4\nResults\n", y=50),
                  blk("Table 1: Scores of the models.", y=72),
                  blk("Model A 27.3", x0=140, width=200, y=90),
                  blk("Model B 28.4", x0=140, width=200, y=110),
                  blk("The prose resumes after the table with enough text to be wide.", y=150)]
        chunks = chunk_blocks(blocks, self.cfg, "doc")
        table = [c for c in chunks if c.section.startswith("Table 1")]
        self.assertEqual(len(table), 1)
        self.assertIn("Model B 28.4", table[0].text)
        prose = [c for c in chunks if c.section == "4 Results"]
        self.assertNotIn("Model A", prose[0].text)

    def test_page_numbers_dropped(self):
        chunks = chunk_blocks([blk("1\nIntro\n"), blk("Some real text here for the chunk."), blk("7")],
                              self.cfg, "doc")
        self.assertEqual(chunks[0].text, "[1 Intro] Some real text here for the chunk.")


if __name__ == "__main__":
    unittest.main()
