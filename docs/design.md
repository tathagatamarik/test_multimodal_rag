# Technical design

## 1. Scope and constraints

* Primary document: the NeurIPS 2017 PDF (11 pages, SHA-256 `d87d482d…`), questions in
  English. Ingestion supports more documents (`docs/ingestion.md`).
* Mandated models: Jina CLIP v1 for embeddings and Phi-3-Vision-128k-instruct for generation; ChromaDB as the store.
* `src/` imports only `torch, chromadb, numpy, fitz, requests, PIL, transformers` from
  outside the standard library (README §8). The optional `services/` layer (FastAPI,
  Streamlit, MLflow) sits on top and is never imported by `src/`.
* The verification hardware has no GPU (4-core laptop CPU, about 10 GB free RAM). That shaped the context-size and image-budget decisions.

## 2. Verified model interfaces (Phase 1 audit)

| Item | Finding (from the model repos' code, checked before implementation) | Consequence |
|---|---|---|
| Jina CLIP API | `JinaCLIPModel.encode_text(sentences, batch_size, normalize_embeddings, **tokenizer_kwargs)` and `encode_image(images, …)` exist in `jina-clip-implementation/modeling_clip.py`, run under `torch.inference_mode`, accept PIL images, and L2-normalise by default | used directly; our own batching wraps them for OOM back-off |
| Jina CLIP space | text and vision towers project to the same 768-d space (`projection_dim: 768`) | cross-modal cosine similarity is meaningful; verified empirically (the correct visual ranked first for 4 of 5 probe queries) |
| Jina runtime deps | `timm` (EVA vision tower), `einops` (rotary embeddings) | installed as model runtime requirements, never imported by `src/` |
| Phi-3-V loading | `AutoModelForCausalLM` + `AutoProcessor`, `trust_remote_code=True` | revision pinned |
| Phi-3-V attention | the config requests `flash_attention_2` (CUDA-only) | `_attn_implementation="eager"` |
| Phi-3-V cache API | remote code calls `DynamicCache.seen_tokens`, `get_max_length()`, `get_usable_length()` | Transformers pinned to 4.45.2 (they were removed in later versions) |
| Phi-3-V prompt | chat template `<|user|>\n…<|end|>\n<|assistant|>\n`; images referenced as `<|image_1|>`… and passed as a PIL list to the processor | implemented in `generator.py`; every placeholder in the text is expanded into image tokens, so retrieved text is sanitised |
| Phi-3-V image cost | `num_crops` HD transform; 4 crops → 757 tokens for one image (measured) | `num_crops=4`, at most 1 image on CPU |
| Phi-3-V processor deps | `torchvision` | installed as a model runtime requirement |
| ChromaDB 1.5.9 | `configuration={"hnsw": {"space": "cosine"}}`, `embedding_function=None`, `where` filters with `$and/$ne/$in`, clear errors on a dimension mismatch or non-scalar metadata | used as is; our own validation runs first for clearer messages |

## 3. Data flow

**Ingestion** (`IngestionService.ingest_file`; `Indexer.run` wraps it for the configured paper):

1. `load_document` / `validate_pdf`: reuse a valid local PDF; otherwise download to
   `*.part`, verify the header, page parse and encryption state, then rename atomically.
   SHA-256 gives `doc_id`.
2. Check compatibility: the collection's fingerprint (embedding model and revision,
   parser, chunking, image settings) must match the current config; otherwise refuse
   until `--rebuild`. Purge any document left `ingesting` by a crash.
3. If the same `doc_id` is already `indexed`, stop (unchanged). Otherwise mark it
   `ingesting` in the registry.
4. `parse_layout` assigns block roles and the ToC tree (23 headings for this paper).
   `layout_paragraphs` and `pack_paragraphs` produce 53 chunks with section paths;
   21 ToC section records.
5. `extract_visuals`: 3 embedded rasters (all valid) produce 5 visual records (Figures 1–2,
   Tables 1–3), saved as PNG.
6. Embed chunks and section records (text tower) and visuals (vision tower); validate
   shape, finiteness and norms.
7. Delete any earlier records of this `doc_id`, upsert 79 records with deterministic,
   document-prefixed IDs, and verify the per-document count.
8. Mark the document `indexed` with counts and timings. A superseded version of the same
   source is deleted only now.

**Query** (`RAGPipeline.ask`):

1. Validate the question (non-empty, ≤ 2000 characters) and embed it with the text tower,
   the same path as the indexed passages.
2. Text: Chroma top-5, filtered `modality == text` (optionally scoped to `doc_ids`;
   optional ToC routing, off by default because it lowered MRR).
3. Visuals: pixel top-2 plus caption-chunk top-2 form the candidates. Score each one as
   `(cos_pixel + cos_caption) / 2` and keep it if the score is ≥ 0.40.
4. Context: attach at most 1 image as pixels and drop a duplicate *figure* caption chunk
   (table chunks are kept for their text). Merge adjacent chunks of the same section
   (removing the overlap), label each item with its section path, keep rank order, and
   stop at 3,600 characters.
5. Generate: chat template, exact token count checked against `max_input_tokens` (on
   overflow the pipeline shrinks the context, at most twice), greedy decoding, citation
   parsing and checking.
6. Return a structured response with evidence, citations, timings, tokens and the
   cold-start flag, and write one telemetry row (latency, tokens, similarities, outcome,
   query embedding). Failed queries are recorded too.

## 4. Error-handling strategy

* **Exception hierarchy** (`exceptions.py`): `ConfigurationError`, `DocumentLoadError`,
  `DocumentExtractionError`, `EmbeddingError`, `VectorStoreError` → `IndexCompatibilityError`,
  `RetrievalError`, `GenerationError` → `ContextTooLongError`, `PipelineError`. All re-raises
  use `raise … from exc`.
* **Where recovery happens:**
  * Download: retry only timeouts, connection errors and HTTP 429/5xx, at most 3 attempts
    with exponential backoff; 4xx and invalid payloads fail immediately.
  * Embedding: an OOM halves the batch (down to 1); any other error fails immediately.
  * Single bad image: skipped and reported in `rejected`; the document continues.
  * Prompt too long: the context shrinks, at most twice.
  * CLI `chat`: one failing question does not end the session.
* **No silent success:** a document is `indexed` only after its per-document record
  count is verified. A domain failure purges its records and marks it `failed`; a crash
  leaves it `ingesting`, and it is purged and redone on the next run. Batch ingestion
  records per-document failures and continues. The CLI exits non-zero on any `RAGError`;
  the API maps error classes to HTTP status codes (422, 409, 503).
* **No fabricated output:** with no evidence, the generator returns a fixed
  insufficient-evidence sentence without calling the model, and empty model output raises
  `GenerationError`.
* **Resources:** PDFs are opened only through a context manager; temporary files are
  removed on failure; the registry is written atomically (tmp file + rename); a writer
  lock file prevents concurrent writers.
* **Telemetry never breaks answering:** write errors are logged as warnings.
* There are no bare `except:` blocks and none that swallow errors. Catches at the CLI and
  API boundaries are limited to `RAGError`. The only broad catch is in the monitoring
  dashboard's optional MLflow section, which displays the error instead of crashing the
  page.

## 5. Performance decisions (all measured on the test machine)

| Decision | Measurement behind it |
|---|---|
| bf16 Phi-3-V on CPU | fp32 weights alone would need about 16.5 GB, against about 10 GB free; bf16 peak RSS is 8.8 GB for Phi-3-V alone and 14.1 GB for the full evaluation process |
| ≤ 1 image, `num_crops=4` | prefill ≈ 10 tokens/s and decode ≈ 3.3 tokens/s; one image = 757 tokens ≈ 75 s |
| Visual gate 0.40 | images attached on 7 of 18 questions instead of 18 of 18, none irrelevant, with no loss of correct visuals (evaluation report) |
| 3,600-character text budget | ≈ 900 tokens ≈ 90 s of prefill; covers the merged top-5 chunks in most queries |
| Models loaded once per process | Jina load 1.5 s and Phi-3-V load 0.8 s (memory-mapped safetensors); `chat` and the evaluation reuse them |
| Chroma for search | about 10 ms per filtered query; no Python-side scan of embeddings |
| Index reuse | a second `index` run takes about 0.02 s (hash match), against 21 s to build 79 records |
| Generation serialised in the API | one CPU-bound `generate` at a time; health and status endpoints stay responsive |

## 6. Trade-offs not taken

* **Merged cross-modal ranking or reciprocal-rank fusion**: rejected because the score
  scales differ by modality, and a merged list would hide images. Per-modality quotas are
  simpler and measured.
* **Full-page renders**: they would dilute CLIP image embeddings with prose and cost
  Phi-3-V tokens.
* **Partial (chunk-level) updates within a document**: a changed document is re-processed
  as a whole (about 21 s per paper); document-level incremental ingestion is enough for
  this scale.
* **ToC routing on by default**: measured to lower MRR (0.927 → 0.858), so it is available
  but off.
* **External reranker or OCR**: excluded by the dependency rules.

## 7. Layout-aware parsing

See README §3 and `src/layout_parser.py`. Typography features per block: dominant font
size, bold first span, share of math-font characters, and position relative to the
page's margin bands and footnote zone. From these come the roles and the ToC (PDF outline
used when present). Measured on this paper:
- the parser recovers all 23 headings;
- it removes the venue footer and 10 page numbers from the content;
- it separates 4 footnotes;
- it attaches 17 equation blocks to their sentences.

Retrieval precision@5 rose from 0.577 to 0.600, and answer keyword coverage from 0.86 to
0.89 (evaluation report).

## 8. Services, telemetry and drift

The services and the drift method are described in `docs/monitoring.md` and README §6–7.
The API holds one pipeline per process, so models load once. Ingestion requests go to a
single background worker, and every query is written to `storage/telemetry.db`. Drift is
computed against the evaluation run as the reference window, and evaluations and
telemetry windows are logged to MLflow.

