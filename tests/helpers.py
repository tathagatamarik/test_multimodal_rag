"""Test fixtures: a synthetic research-paper PDF, a deterministic fake embedder
and a fake Phi-3-V, so unit/integration tests run without network or models."""

from __future__ import annotations

import io
import re
import tempfile
import zlib
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import src  # noqa: F401  (installs the fitz deprecation-warning filter)
import fitz

from src.config import Config, config_from_dict

DIM = 64

PARAGRAPHS = {
    "intro": ("Recurrent networks process tokens sequentially. This sequential nature precludes "
              "parallelization within training examples. Attention mechanisms allow modeling of "
              "dependencies without regard to their distance in the sequence. ") * 2,
    "attention": ("Scaled dot-product attention computes the dot products of the query with all keys, "
                  "divides each by the square root of dk, and applies a softmax function to obtain "
                  "the weights on the values. Multi-head attention runs several attention layers in "
                  "parallel. ") * 2,
    "training": ("We trained the models with the Adam optimizer and a warmup learning rate schedule. "
                 "Dropout and label smoothing were used for regularization. ") * 2,
}


def png_bytes(size=(200, 150), color=(200, 30, 30), mode="RGB") -> bytes:
    img = Image.new(mode, size, color if mode != "L" else 128)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def make_sample_pdf(path: Path, *, tiny_image: bool = False, duplicate_image: bool = False) -> Path:
    """Three pages: numbered headings, prose, a captioned figure, a captioned table,
    and a References section."""
    doc = fitz.open()
    p1 = doc.new_page()
    p1.insert_text((72, 72), "1\nIntroduction", fontsize=11)
    p1.insert_textbox(fitz.Rect(72, 100, 523, 300), PARAGRAPHS["intro"], fontsize=10)
    p1.insert_text((72, 330), "2\nAttention", fontsize=11)
    p1.insert_textbox(fitz.Rect(72, 360, 523, 560), PARAGRAPHS["attention"], fontsize=10)

    p2 = doc.new_page()
    p2.insert_image(fitz.Rect(200, 72, 400, 222), stream=png_bytes())
    if duplicate_image:
        p2.insert_image(fitz.Rect(420, 72, 520, 147), stream=png_bytes())
    if tiny_image:
        p2.insert_image(fitz.Rect(72, 72, 82, 82), stream=png_bytes(size=(8, 8), color=(0, 0, 255)))
    p2.insert_textbox(fitz.Rect(72, 235, 523, 260),
                      "Figure 1: The attention diagram with query, key and value inputs.", fontsize=10)
    p2.insert_text((72, 300), "3\nTraining", fontsize=11)
    p2.insert_textbox(fitz.Rect(72, 330, 523, 500), PARAGRAPHS["training"], fontsize=10)

    p3 = doc.new_page()
    p3.insert_textbox(fitz.Rect(72, 72, 523, 100),
                      "Table 1: BLEU scores of the test models on the translation task.", fontsize=10)
    for i, row in enumerate(["Model A 27.3 38.1", "Model B 28.4 41.0", "Model C 25.1 39.9"]):
        p3.insert_textbox(fitz.Rect(140, 110 + i * 22, 400, 132 + i * 22), row, fontsize=9)
    p3.insert_textbox(fitz.Rect(72, 200, 523, 300),
                      "The table above summarises the results of all models on the task.", fontsize=10)
    p3.insert_text((72, 340), "References", fontsize=11)
    p3.insert_textbox(fitz.Rect(72, 360, 523, 420), "[1] A. Author. A cited paper about attention. 2017.",
                      fontsize=9)
    # Layout furniture the layout parser must classify (not index): page numbers, a
    # running footer, a footnote and a math-font equation.
    doc[0].insert_text((72, 600), "E = m c 2", fontsize=10, fontname="symb")
    doc[0].insert_text((72, 735), "1 This footnote explains the parallelization claim.", fontsize=8)
    for i, page in enumerate(doc, start=1):
        page.insert_text((290, 810), str(i), fontsize=10)
        page.insert_text((72, 825), "Workshop on Test Documents 2026", fontsize=8)
    doc.save(str(path))
    doc.close()
    return path


def make_config(root: Path, **section_overrides) -> Config:
    raw = {
        "document": {"source_url": "https://example.org/paper.pdf", "local_path": "data/raw/paper.pdf"},
        "chunking": {"chunk_size_chars": 300, "chunk_overlap_chars": 60, "min_chunk_chars": 20},
        "images": {"output_dir": "data/processed/visuals", "render_dpi": 72},
        "embedding": {"model_id": "fake/embedder", "revision": None, "batch_size": 4},
        "vector_store": {"persist_dir": "storage/chroma", "collection_name": "test_collection",
                         "registry_path": "storage/registry.json", "inbox_dir": "data/inbox"},
        "monitoring": {"telemetry_path": "storage/telemetry.db"},
        "retrieval": {"top_k_text": 3, "top_k_image": 2, "min_visual_score": -1.0},
        "context": {"max_context_chars": 2000, "max_images": 1},
        "generation": {"model_id": "fake/generator", "revision": None, "max_input_tokens": 4096},
        "logging": {"level": "WARNING", "log_file": None},
    }
    for section, values in section_overrides.items():
        raw.setdefault(section, {}).update(values)
    return config_from_dict(raw, project_root=root)


def make_project(**section_overrides) -> tuple[tempfile.TemporaryDirectory, Config]:
    tmp = tempfile.TemporaryDirectory()
    root = Path(tmp.name)
    cfg = make_config(root, **section_overrides)
    pdf = cfg.path(cfg.document.local_path)
    pdf.parent.mkdir(parents=True, exist_ok=True)
    make_sample_pdf(pdf)
    return tmp, cfg


def _bow(text: str) -> np.ndarray:
    v = np.zeros(DIM, np.float32)
    for tok in re.findall(r"[a-z0-9]+", text.lower()):
        v[zlib.crc32(tok.encode()) % DIM] += 1.0
    if not v.any():
        v[0] = 1.0
    return v / np.linalg.norm(v)


class FakeJinaModel(torch.nn.Module):
    """Mimics the verified JinaCLIPModel encode_text / encode_image signatures."""

    def __init__(self, fail_with: Exception | None = None, oom_until_batch: int | None = None):
        super().__init__()
        self.fail_with = fail_with
        self.oom_until_batch = oom_until_batch
        self.calls: list[int] = []

    def _maybe_fail(self, n: int) -> None:
        self.calls.append(n)
        if self.fail_with is not None:
            raise self.fail_with
        if self.oom_until_batch is not None and n > self.oom_until_batch:
            raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")

    def encode_text(self, sentences, batch_size=32, show_progress_bar=None, normalize_embeddings=True,
                    **_):
        self._maybe_fail(len(sentences))
        return np.stack([_bow(s) for s in sentences])

    def encode_image(self, images, batch_size=32, show_progress_bar=None, normalize_embeddings=True):
        self._maybe_fail(len(images))
        out = []
        for im in images:
            arr = np.asarray(im.convert("RGB").resize((8, 8)), np.float32).reshape(-1)[:DIM]
            arr = arr + 1e-3
            out.append(arr / np.linalg.norm(arr))
        return np.stack(out)


class _Tok:
    eos_token_id = 2

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return "".join(f"<|{m['role']}|>\n{m['content']}<|end|>\n" for m in messages) + "<|assistant|>\n"


class _Batch(dict):
    def to(self, device):
        return self


class FakeProcessor:
    """Counts one token per whitespace word plus 10 per image placeholder; validates
    that placeholders match the number of images exactly like Phi-3-V does."""

    tokenizer = _Tok()

    def __init__(self, reply: str = "The answer is supported [T1, p.1]."):
        self.reply = reply
        self.last_prompt = None
        self.last_images = None

    def __call__(self, prompt, images=None, return_tensors="pt"):
        self.last_prompt, self.last_images = prompt, images
        n_tags = len(re.findall(r"<\|image_\d+\|>", prompt))
        if n_tags != len(images or []):
            raise ValueError(f"{n_tags} image tags but {len(images or [])} images")
        n = len(prompt.split()) + 10 * n_tags
        return _Batch(input_ids=torch.ones((1, n), dtype=torch.long))

    def batch_decode(self, ids, skip_special_tokens=True, clean_up_tokenization_spaces=False):
        return [self.reply if ids.shape[1] else ""]


class FakePhi(torch.nn.Module):
    def __init__(self, n_new: int = 5, fail_with: Exception | None = None):
        super().__init__()
        self.n_new = n_new
        self.fail_with = fail_with
        self.calls = 0

    def generate(self, input_ids, **kwargs):
        self.calls += 1
        if self.fail_with is not None:
            raise self.fail_with
        return torch.cat([input_ids, torch.ones((1, self.n_new), dtype=torch.long)], dim=1)
