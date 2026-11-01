"""Generate notebooks/multimodal_rag_solution.ipynb (cells defined here, executed with nbconvert).

Keeping the notebook source in a script makes it diffable and guarantees a
top-to-bottom execution order:

    python notebooks/build_notebook.py
    jupyter nbconvert --to notebook --execute --inplace \
        --ExecutePreprocessor.timeout=3600 notebooks/multimodal_rag_solution.ipynb
"""

import json
from pathlib import Path

CELLS: list[tuple[str, str]] = []


def md(text: str) -> None:
    CELLS.append(("markdown", text.strip("\n")))


def code(text: str) -> None:
    CELLS.append(("code", text.strip("\n")))


md("""
# Multimodal RAG over "Attention Is All You Need"

## 1. Assignment overview

The task is a Retrieval-Augmented Generation system, built from scratch, that answers
questions about the paper *Attention Is All You Need*
([NeurIPS PDF](https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)).

| Component | Implementation |
|---|---|
| Document ingestion | PyMuPDF text blocks, embedded images and caption-anchored figure/table renders |
| Embeddings | `jinaai/jina-clip-v1`, one 768-d space for text and images |
| Vector database | ChromaDB (persistent, cosine) with an index manifest |
| Retrieval | text top-k plus visuals scored by fused pixel and caption similarity |
| Generation | `microsoft/Phi-3-vision-128k-instruct`, given the retrieved text and the retrieved figure pixels |

This notebook is a thin interface over the tested modules in `src/`, which keep the logic
in one place. Every stage has an explanation, the call, a **verification** cell and its
output. The notebook runs top to bottom. The PDF is processed once, the index is built
once (and reused when it already exists), and each model is loaded once.
""")

md("""
## 2. Requirements and constraints

* **Allowed libraries:** `torch, chromadb, numpy, io, fitz, requests, PIL, transformers`.
  `src/` imports nothing else from outside the standard library; this is checked in the
  next cell. `torchvision`, `timm`, `einops` and `accelerate` are installed only because the
  mandated models' remote code needs them.
* **Hardware used for this run:** CPU only (Intel i7-1185G7, 4 cores) with about 10 GB of
  free RAM. Phi-3-V therefore runs in bfloat16 with at most one image per prompt, and each
  answer takes 1–3 minutes.

**Traceability matrix:** where each requirement is implemented and how it is verified.

| Requirement | Implementation | Verification (this notebook / tests) |
|---|---|---|
| PDF ingestion | `src/document_loader.py` | §5 checks; `tests/test_document_loader.py` (download, retry, corrupt, encrypted, empty) |
| Text extraction | `src/text_extractor.py`, `src/preprocessing.py` | §6 checks; `tests/test_preprocessing.py` |
| Image extraction | `src/image_extractor.py` | §7 renders and checks; `ExtractionTests` |
| Text and image embeddings | `src/embedding_service.py` | §8 shape, finiteness and norm checks; `tests/test_embeddings.py` |
| Vector persistence | `src/vector_store.py`, `src/indexer.py` | §9 build or reuse and manifest; `tests/test_vector_store.py`, `tests/test_pipeline.py` |
| Multimodal retrieval | `src/retriever.py` | §10 and §15 metrics; `tests/test_retrieval.py` |
| Context construction | `src/context_builder.py` | §11; `ContextBuilderTests` |
| Answer generation | `src/generator.py` | §12–14 live answers; `tests/test_generation.py` |
| Error handling | every module | negative tests in each test file |
| Logging | `src/logging_config.py` | log output below; `test_logging_configured_once` |
| Documentation | `README.md`, `docs/` | – |
""")

code(r"""
# Dependency audit: every non-standard-library import used by src/
import ast, sys
from pathlib import Path

ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "src" / "pipeline.py").exists())
sys.path.insert(0, str(ROOT))

third_party = set()
for f in (ROOT / "src").glob("*.py"):
    for node in ast.walk(ast.parse(f.read_text())):
        names = ([a.name for a in node.names] if isinstance(node, ast.Import)
                 else [node.module] if isinstance(node, ast.ImportFrom) and node.level == 0 else [])
        for n in names:
            top = n.split(".")[0]
            if top not in sys.stdlib_module_names:
                third_party.add(top)
ALLOWED = {"torch", "chromadb", "numpy", "io", "fitz", "requests", "PIL", "transformers"}
print("third-party imports in src/:", sorted(third_party))
assert third_party <= ALLOWED, third_party - ALLOWED
print("OK - all within the allow-list")
""")

md("""
## 3. Configuration

All settings live in `config/config.json`. They are validated before any expensive work
starts, and the README documents every key with its range and default.
""")

code(r"""
import json
from src.config import load_config

cfg = load_config()
print(json.dumps({k: v for k, v in cfg.to_dict().items()
                  if k in ("chunking", "images", "retrieval", "context", "generation")}, indent=1))
""")

md("""
## 4. Imports and logging

Logging is configured once, to the console and to `logs/rag.log`, and every line carries a
run id.
""")

code(r"""
import time
import numpy as np
import torch
from PIL import Image

from src.logging_config import setup_logging, get_run_id
from src.document_loader import load_document, open_pdf
from src.text_extractor import extract_text
from src.preprocessing import chunk_blocks
from src.image_extractor import extract_visuals
from src.pipeline import RAGPipeline
from src.context_builder import build_context, default_image_loader
from src.generator import build_prompt_text

log = setup_logging(cfg)
print("run id:", get_run_id(), "| torch", torch.__version__, "| threads", torch.get_num_threads(),
      "| CUDA available:", torch.cuda.is_available())
""")

md("""
## 5. PDF loading

The loader reuses a valid local copy. Otherwise it downloads to a `.part` file with
bounded retries (transient errors only), checks that the bytes are a readable,
unencrypted PDF with pages, and only then renames the file into place.
""")

code(r"""
t = time.perf_counter()
doc_info = load_document(cfg.document, cfg.path)
print(f"loaded in {time.perf_counter() - t:.2f}s")
doc_info.to_dict()
""")

code(r"""
# Verification
assert doc_info.page_count == 11
assert doc_info.sha256.startswith("d87d482d")
print("PDF valid:", doc_info.filename, doc_info.page_count, "pages, doc_id", doc_info.doc_id)
""")

md("""
## 6. Layout-aware text extraction and chunking

The paper has no PDF outline, so its structure is **recovered from typography**: font
sizes and weights, position on the page, and math fonts. Each block gets a role:
heading at level *n*, paragraph, equation, caption, table, footnote, page number or
running footer. The headings form a table of contents (ToC). This follows STAIR
(arXiv:2609.03874), which argues that ToC entries make better retrieval units than
length-based chunks because they are semantically coherent and clearly bounded.

Chunking then:
* packs whole paragraphs into chunks of at most **900 characters** inside one ToC section,
  splitting only oversized paragraphs, at sentence boundaries, with **≤ 150 characters**
  of trailing-sentence overlap;
* prefixes each chunk with its **full section path**, for example
  `[3 Model Architecture > 3.2 Attention > 3.2.1 Scaled Dot-Product Attention]`;
* keeps equations with the paragraph that introduces them, and turns captions, tables and
  footnotes into their own units;
* drops page numbers and the venue footer, and excludes References.
""")

code(r"""
from src.layout_parser import parse_layout, layout_paragraphs
from src.preprocessing import pack_paragraphs

with open_pdf(doc_info.path) as pdf:
    layout = parse_layout(pdf)
print("body font:", layout.body_size, "pt | roles:", layout.role_counts)
print("\nRecovered table of contents:")
print("\n".join("  " + t for t in layout.toc()))
chunks = pack_paragraphs(layout_paragraphs(layout, cfg.chunking.exclude_references), cfg.chunking, doc_info.doc_id)
sizes = [c.char_count for c in chunks]
print(f"\n{len(chunks)} chunks; size min/mean/max = {min(sizes)}/{np.mean(sizes):.0f}/{max(sizes)} chars")
for c in chunks[19:22]:
    print(f"\n{c.chunk_id}  pages {c.page_start}-{c.page_end}\n  {c.text[:300]}...")
""")

code(r"""
# Verification: structure, provenance, furniture removed, deterministic ids
assert layout.toc()[3] == "3 Model Architecture" and "    3.2.1 Scaled Dot-Product Attention" in layout.toc()
assert len({c.chunk_id for c in chunks}) == len(chunks)
assert all(1 <= c.page_start <= c.page_end <= 11 for c in chunks)
assert not any("31st Conference on Neural Information" in c.text for c in chunks)   # venue footer dropped
assert not any(c.section_path.startswith("References") for c in chunks)
footnotes = [c for c in chunks if c.section == "Footnote"]
print("footnote chunks:", [c.section_path for c in footnotes])
""")

md("""
## 7. Image and figure extraction

* **Embedded raster extraction** (`extract_image`) returns the image bytes stored in the
  PDF. They are decoded with Pillow, size-checked and de-duplicated. On their own they are
  fragments: Figure 2 is two separate rasters, and its panel titles are PDF text.
* **Region rendering** (`get_pixmap(clip=…)`) rasterises part of the page as a reader sees
  it. Tables 1–3 are vector drawings and exist only this way.

Each `Figure N:` / `Table N:` caption anchors **one rendered region**, giving complete,
captioned visuals without full-page renders.
""")

code(r"""
with open_pdf(doc_info.path) as pdf:
    img_res = extract_visuals(pdf, cfg.images, doc_info.doc_id,
                              cfg.path(cfg.images.output_dir), cfg.project_root)
print("embedded rasters:", [(e.page, e.xref, f"{e.width}x{e.height}") for e in img_res.embedded])
print("rejected:", img_res.rejected)
for v in img_res.visuals:
    print(f"{v.visual_id:<24} {v.label:<9} page {v.page}  {v.width}x{v.height}  source={v.source}")
""")

code(r"""
# Verification: every visual decodes, is RGB and matches its recorded size
for v in img_res.visuals:
    with Image.open(cfg.path(v.path)) as im:
        assert im.mode == "RGB" and im.size == (v.width, v.height)
assert [v.label for v in img_res.visuals] == ["Figure 1", "Figure 2", "Table 1", "Table 2", "Table 3"]
fig2 = Image.open(cfg.path(img_res.visuals[1].path))
fig2.thumbnail((700, 700)); fig2
""")

code(r"""
tab2 = Image.open(cfg.path(img_res.visuals[3].path)); tab2.thumbnail((700, 700)); tab2
""")

md("""
## 8. Text and image embeddings (Jina CLIP v1)

I checked the model API in its source code before using it: `encode_text` and
`encode_image` exist, run in inference mode, and return L2-normalised 768-d vectors in a
**shared** text/image space. The service adds its own batching with OOM back-off and
validates the shape, finiteness and unit norm of every vector.
""")

code(r"""
pipeline = RAGPipeline(cfg)           # owns the single instances of both models
emb = pipeline.embedder
probe_q = ["scaled dot-product attention diagram", "BLEU scores compared with other models",
           "complexity per layer of self-attention, recurrent and convolutional layers"]
q_vecs = emb.embed_texts(probe_q)
visuals = [Image.open(cfg.path(v.path)).convert("RGB") for v in img_res.visuals]
v_vecs = emb.embed_images(visuals)
print("text", q_vecs.shape, "image", v_vecs.shape, "| model load", round(emb.load_seconds, 2), "s")
print("cross-modal cosine similarity (rows = queries, cols =", [v.label for v in img_res.visuals], ")")
print(np.round(q_vecs @ v_vecs.T, 3))
""")

code(r"""
# Verification: shared space works - each probe's best visual is the expected one
assert q_vecs.shape[1] == v_vecs.shape[1] == 768
assert np.isfinite(q_vecs).all() and np.allclose(np.linalg.norm(v_vecs, axis=1), 1, atol=1e-3)
labels = [v.label for v in img_res.visuals]
print([labels[i] for i in (q_vecs @ v_vecs.T).argmax(axis=1)])
assert [labels[i] for i in (q_vecs @ v_vecs.T).argmax(axis=1)] == ["Figure 2", "Table 2", "Table 1"]
""")

md("""
## 9. ChromaDB initialisation and indexing

Indexing runs through the decoupled ingestion service (`src/ingestion/`; batch, streaming
and incremental modes are described in `docs/ingestion.md`). `pipeline.index()` does one
of two things:
* it **ingests** the paper: process, embed text chunks, ToC section records and visuals,
  upsert into a cosine collection with document-prefixed deterministic IDs, verify the
  count and record the document in `storage/registry.json`; or
* it **reuses** the existing index when the content hash and index fingerprint are
  unchanged, recomputing nothing.

A changed model, parser, chunking or image setting makes it refuse with instructions to
`--rebuild`.
""")

code(r"""
report = pipeline.index()
print(report.summary())
again = pipeline.index()                     # second call must reuse
print(again.summary())
""")

code(r"""
# Verification: persisted counts, modality split, registry status, no duplicates on re-run
store = pipeline.store
counts = {m: store.count({"modality": m}) for m in ("text", "section", "image")}
entry = pipeline.ingestion.registry.get(doc_info.doc_id)
print(f"collection '{store.collection_name}': {store.count()} records {counts}")
print("registry entry:", {k: getattr(entry, k) for k in ("status", "total_records", "fingerprint_hash")})
assert again.action == "reused" and store.count() == entry.total_records == sum(counts.values())
assert counts["image"] == 5 and counts["section"] > 15
""")

md("""
## 10. Retrieval

* Text: Chroma top-5 among `modality == text`.
* Visuals: text-to-image cosine similarities from Jina CLIP (about 0.05–0.35) are much
  lower than text-to-text ones (about 0.3–0.8), so visuals get their own quota. The
  **fused score** is the mean of the query-to-pixels and query-to-caption-chunk cosines.
  Both are real Jina CLIP similarities, and an image is attached only if the score is
  ≥ 0.40 (§15 measures this choice).
""")

code(r"""
def show(res):
    print(f"Q: {res.query}  [mode={res.mode}]")
    for t in res.text:
        print(f"  T{t.rank} {t.similarity:.3f} p.{t.metadata['page_start']}-{t.metadata['page_end']} "
              f"[{t.metadata['section'][:45]}]")
    for i in res.images:
        print(f"  I{i.rank} {i.similarity:.3f} {i.scores} {i.metadata['label']} (p.{i.page})")
    for i in res.rejected_images:
        print(f"  (rejected visual {i.metadata['label']} score {i.similarity:.3f})")

show(pipeline.retrieve("What BLEU score did the big Transformer achieve on English-to-German?"))
show(pipeline.retrieve("How does the decoder prevent attending to subsequent positions?"))
show(pipeline.retrieve("scaled dot-product attention diagram", mode="image"))
print()
# STAIR-style structure routing: pick ToC sections first, then chunks inside them
r = pipeline.retrieve("How does the decoder prevent attending to subsequent positions?", structure_routing=True)
print("routed sections:", [(s.metadata["section_path"], round(s.similarity, 3)) for s in r.sections])
show(r)
""")

code(r"""
# Verification: modality filters and visual gating behave as specified
r = pipeline.retrieve("What BLEU score did the big Transformer achieve on English-to-German?")
assert r.text[0].page == 8 and r.images and r.images[0].metadata["label"] == "Table 2"
assert not pipeline.retrieve("Which optimizer was used?").images       # gate: no irrelevant image
assert not pipeline.retrieve("x", mode="text").images
print("retrieval checks passed")
""")

md("""
## 11. Context construction

The context builder turns retrieval results into tagged evidence: `[I1]` for a figure
passed **as pixels** with its caption, and `[T1]…` for text, each with pages. It merges
adjacent chunks and removes their overlap, drops a caption chunk when its image is already
attached, and enforces the 3,600-character and 1-image budgets.
""")

code(r"""
res = pipeline.retrieve("In the scaled dot-product attention diagram, which operations are applied to Q, K and V?")
ctx = build_context(res, cfg.context, default_image_loader(cfg.project_root))
print("images passed as pixels:", [(i.tag, i.label, ctx.images[k].size) for k, i in enumerate(ctx.image_items)])
print("dropped:", ctx.dropped)
print(build_prompt_text(ctx)[:1500], "...")
""")

code(r"""
# Verification: one image placeholder per real image, budget respected
prompt = build_prompt_text(ctx)
assert prompt.count("<|image_") == len(ctx.images) == 1
assert sum(len(i.text) for i in ctx.text_items) <= cfg.context.max_context_chars
print("context checks passed")
""")

md("""
## 12. Phi-3 Vision integration

I checked the remote code of the pinned revision before integrating it:

* `AutoModelForCausalLM` and `AutoProcessor` with `trust_remote_code=True`;
  `_attn_implementation="eager"`, because the shipped config asks for flash-attention-2,
  which is CUDA-only;
* Transformers 4.45.2, because the remote code uses `DynamicCache.seen_tokens` and
  `get_max_length`, which later versions removed;
* chat template `<|user|>\n…<|end|>\n<|assistant|>\n`, with images referenced as
  `<|image_1|>` and passed as a PIL list. With `num_crops=4`, one image costs 757 tokens;
* bfloat16 on CPU (fp32 would need about 16.5 GB of RAM).

The exact prompt length, image tokens included, is checked against `max_input_tokens`
before generation.
""")

code(r"""
gen = pipeline.generator
gen.load()
n_tokens = gen.count_tokens(ctx)
print(f"Phi-3-V loaded in {gen.load_seconds:.1f}s on {gen.device} ({cfg.generation.dtype}); "
      f"prompt above = {n_tokens} tokens (limit {cfg.generation.max_input_tokens})")
assert n_tokens <= cfg.generation.max_input_tokens
""")

md("""
## 13. End-to-end RAG pipeline

`pipeline.ask(question)` runs retrieval, context building, generation and citation
checking, and returns a structured response with timings. Models stay loaded between
questions.
""")

code(r"""
resp = pipeline.ask("In the scaled dot-product attention diagram, which operations are applied to Q, K and V?")
print(resp.render())
""")

code(r"""
# Verification: structured response with evidence, a visual item and per-stage timings
d = resp.to_dict()
assert d["evidence"][0]["modality"] == "image" and d["evidence"][0]["label"] == "Figure 2"
assert {"query_embedding", "vector_search", "generation_generate", "total"} <= set(d["timings"])
print("citations:", d["citations"], "| invalid:", d["invalid_citations"])
""")

md("""
## 14. Sample queries and outputs

Five more questions, all executed live: a table question (text + Table 2 image), a
text-only question, a masked-attention question, a model-variation question, and an
out-of-scope question that must be refused.
""")

code(r"""
demo_questions = [
    "What BLEU score did the big Transformer achieve on the WMT 2014 English-to-German task?",
    "Which optimizer and learning rate schedule were used for training?",
    "How does the decoder prevent positions from attending to subsequent positions?",
    "What does the Transformer model architecture diagram show?",
    "How many parameters does GPT-4 have?",
]
demo_responses = []
for q in demo_questions:
    r = pipeline.ask(q)
    demo_responses.append(r)
    print(r.render()); print("=" * 100)
""")

code(r"""
# Verification: the out-of-scope question is refused; in-scope answers cite evidence
assert demo_responses[-1].insufficient_evidence
for r in demo_responses[:-1]:
    print(f"{r.question[:70]:<72} citations={[c['tag'] for c in r.citations]} "
          f"pages={sorted({p for c in r.citations for p in c['pages']})}")
""")

md("""
## 15. Evaluation

`python -m src.cli evaluate` runs the 18-question set in `data/sample_queries.json` (page
and visual labels were checked against the PDF text) and writes
`artifacts/evaluation/<run>/`. Four runs were recorded; run 4 is the final configuration.
A full generation run takes about 45 minutes on this CPU, so the cells below **load the
recorded results** of run 4 instead of repeating it. The retrieval part is fast and is
recomputed live. All four runs are compared in `docs/evaluation_report.md` and in MLflow.
""")

code(r"""
from src.evaluation import load_eval_set, evaluate_retrieval
items = load_eval_set(cfg.path("data/sample_queries.json"))
retr = evaluate_retrieval(pipeline, items, cfg, sweep_thresholds=(0.30, 0.35, 0.45, 0.50))
cols = ["hit@5", "precision@5", "recall@5", "mrr", "visual_hit", "visual_in_context",
        "questions_with_irrelevant_image_in_context", "images_in_context_total"]
print(f"{'variant':<34}" + "".join(f"{c[:14]:>16}" for c in cols))
for name, v in retr["variants"].items():
    print(f"{name:<34}" + "".join(f"{str(v[c]):>16}" for c in cols))
""")

code(r"""
gen_summary = json.loads((ROOT / "artifacts/evaluation/run4_layout_parser_final/generation_summary.json").read_text())
records = [json.loads(l) for l in (ROOT / "artifacts/evaluation/run4_layout_parser_final/generation_results.jsonl").read_text().splitlines()]
print(json.dumps({k: v for k, v in gen_summary.items() if k != "cold_query_latency_s"}, indent=1))
print(f"\n{'id':<4} {'kw':>5} {'cites rel. page':>16} {'abstained':>10} {'total s':>8}  answer")
for r in records:
    c = r["checks"]
    print(f"{r['id']:<4} {str(c['keyword_coverage']):>5} {str(c['cites_relevant_page']):>16} "
          f"{str(c['abstained']):>10} {r['response']['timings']['total']:>8.0f}  {r['response']['answer'][:90]!r}")
""")

md("""
## 16. Performance measurements

The measurements come from this run (index build or reuse, live queries) and from the
recorded evaluation. Cold-start queries (including model loading) are reported separately
from warm ones.
""")

code(r"""
import resource
print("Index timings (from the registry entry of the build run):")
for k, v in entry.timings.items():
    print(f"  {k:<22} {v:7.2f}s")
print("\nLive queries in this notebook:")
for r in [resp, *demo_responses]:
    t = r.timings
    print(f"  cold={str(r.cold_start):<5} retrieval={t['query_embedding'] + t['vector_search']:.2f}s "
          f"generate={t.get('generation_generate', 0):6.1f}s total={t['total']:6.1f}s tokens={r.tokens}")
print(f"\nPeak RSS of this kernel: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6:.2f} GB")

# Every query above was also written to the telemetry store (latency, tokens, similarity, embedding)
from src.telemetry import summarize
from src.drift import detect_drift
print("\nTelemetry, notebook/CLI traffic:", json.dumps(summarize(pipeline.telemetry.rows(exclude_source="evaluation")), indent=1))
print("Drift vs evaluation reference:", detect_drift(pipeline.telemetry, min_samples=5)["status"])
""")

md("""
## 17. Limitations and conclusions

**What works (measured, run 4):**
* Text retrieval on the verified labels: Hit@5 = 1.0 and MRR = 0.93 over 17 answerable
  questions.
* Cross-modal retrieval puts a correct figure or table in the prompt for 7 of 8 visual
  questions, with zero irrelevant images.
* Answers: mean keyword coverage 0.89, 12 of 18 cite a relevant page, no invalid
  citations, and the out-of-scope question is refused.
* Layout-aware parsing recovers the paper's full ToC from typography and removes page
  furniture from chunks.
* Zero-shot ToC routing lowered MRR (0.93 to 0.86), so it is left off.

**Limitations:**
* Only a CPU was available: answers take 1–3 minutes, and GPU execution is untested.
* The 0.40 visual threshold and the relevance labels come from the same small set (18
  questions), with no held-out set.
* Phi-3-V (4.2B) sometimes omits citations, copies evidence verbatim, or misreads a
  diagram (run 4, q14). Citations are parsed and checked, not enforced.
* q05 (positional encoding) is refused because the sine/cosine chunk is not retrieved in
  the top 5.
* Equations are extracted as linear text, and OCR is unavailable under the dependency
  rules.
* The Jina CLIP modelling code repo cannot be revision-pinned through `from_pretrained`.
  Use `HF_HUB_OFFLINE=1` after the first download.

See `docs/evaluation_report.md` for the full measured report and `README.md` for setup.
""")


def build(path: Path) -> None:
    nb = {"cells": [], "metadata": {"kernelspec": {"display_name": "Python 3 (.venv)", "language": "python",
                                                   "name": "python3"},
                                    "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    for i, (kind, src) in enumerate(CELLS):
        cell = {"cell_type": kind, "id": f"cell-{i:02d}", "metadata": {}, "source": src}
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        nb["cells"].append(cell)
    path.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print(f"wrote {path} ({len(CELLS)} cells)")


if __name__ == "__main__":
    build(Path(__file__).with_name("multimodal_rag_solution.ipynb"))
