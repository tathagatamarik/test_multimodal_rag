# Evaluation report

Every number here was measured by running the system. The raw data is in
`artifacts/evaluation/<run>/` (`retrieval_report.json`, `generation_results.jsonl` with
every answer, `generation_summary.json`, `run_info.json`), and all runs are logged in
MLflow (`storage/mlflow.db`, experiment `multimodal-rag`).

## 1. Setup

* **Evaluation set:** 18 questions (`data/sample_queries.json`): 17 answerable and 1 out of
  scope ("How many parameters does GPT-4 have?"). Eight questions have an expected figure or
  table. Expected pages were verified by searching the PDF's extracted text for the
  supporting statements; nothing was taken from memory.
* **Hardware:** Intel i7-1185G7 (4 cores), CPU only, Phi-3-V in bfloat16, greedy decoding,
  `max_new_tokens = 200`.
* **Retrieval metrics** (k = 5, page-level relevance): Hit@5, Precision@5, Recall@5 (share
  of relevant pages covered) and MRR. Visual metrics: *visual hit* (expected visual among
  retrieved images), *visual in context* (expected visual actually attached to the prompt)
  and the number of questions with an irrelevant image attached.
* **Generation checks (automatic):**
  - keyword coverage: groups of required terms, any alternative counts;
  - whether a citation points to a relevant page;
  - invalid citations;
  - whether the refuse/answer decision is correct.

  These checks are deliberately simple and transparent. Section 5 adds a manual review.

**Caveat:** 18 questions with no held-out split. The visual threshold and the prompt
wording were selected on this same set, so the numbers are optimistic estimates for this
paper, not general performance claims. Page-level labels are also lenient: a chunk from
the right page but the wrong subsection counts as relevant (see q05 in section 5).

## 2. Runs

| Run | Change | Prompt version | Parser |
|---|---|---|---|
| 1 | Baseline | v1: rules after evidence, concrete citation example | basic (regex headings) |
| 2 | Keep table text chunks next to table images; abstract citation example; label-style citation parsing | v2: strict "if the evidence does not answer" rule | basic |
| 3 | Prompt chosen by A/B test on run-2 failures | v3: "only if none of the evidence is related", cite only listed tags | basic |
| 4 (final) | Layout-aware parser (ToC paths, footer, footnote and equation handling), section-path evidence labels, ToC section records | v3 | layout |

## 3. Retrieval

### 3.1 Variants (17 answerable questions; 8 with an expected visual)

| Parser | Variant | Hit@5 | P@5 | R@5 | MRR | Visual in context | Irrelevant images attached | Images attached (18 q) |
|---|---|---|---|---|---|---|---|---|
| basic | text only | 1.00 | 0.577 | 0.971 | 0.931 | 0/8 | 0 | 0 |
| basic | multimodal, pixel score | 1.00 | 0.577 | 0.971 | 0.931 | 6/8 | 11 | 18 |
| basic | multimodal, fused score, no gate | 1.00 | 0.577 | 0.971 | 0.931 | 6/8 | 11 | 18 |
| basic | multimodal, fused score, gate 0.40 | 1.00 | 0.577 | 0.971 | 0.931 | 6/8 | **0** | 6 |
| layout | text only | 1.00 | 0.600 | 0.971 | 0.927 | 0/8 | 0 | 0 |
| layout | multimodal, pixel score | 1.00 | 0.600 | 0.971 | 0.927 | 6/8 | 11 | 18 |
| layout | multimodal, fused score, no gate | 1.00 | 0.600 | 0.971 | 0.927 | 7/8 | 10 | 18 |
| **layout** | **multimodal, fused score, gate 0.40 (final)** | **1.00** | **0.600** | **0.971** | **0.927** | **7/8** | **0** | **7** |
| layout | final + ToC structure routing | 1.00 | 0.612 | 0.971 | **0.858** | 7/8 | 0 | 7 |

Findings:
* **Text retrieval is strong for both parsers.** A relevant chunk is in the top 5 for every
  answerable question, and it is first for most of them (MRR ≈ 0.93). The layout parser
  raises precision@5 from 0.577 to 0.600 at equal MRR.
* **Multimodal retrieval adds visual evidence that text-only retrieval cannot provide.**
  Gating is essential: without it, every question gets an image and 10–11 of them are
  irrelevant. Each image costs about 757 tokens, or about 75 s of CPU prefill.
* **Fused scoring plus layout parsing** puts the correct visual in the prompt for 7 of 8
  visual questions (6 of 8 with pixels only or with the basic parser). It attaches only 7
  images in total, none of them irrelevant.
* **Zero-shot ToC routing (STAIR-inspired) hurt ranking:** MRR fell from 0.927 to 0.858,
  because q01 and q02 lost their best chunk to section-local chunks. Routing is therefore
  off by default. STAIR fine-tunes an LLM on the ToC; embedding-based routing without
  training does not reproduce its gains here.

### 3.2 Visual gate sensitivity (fused score; 8 visual questions)

| Gate | 0.30 | 0.35 | **0.40** | 0.45 | 0.50 |
|---|---|---|---|---|---|
| basic: correct visual in context / irrelevant attached | 6 / 5 | 6 / 1 | **6 / 0** | 5 / 0 | 1 / 0 |
| layout: correct visual in context / irrelevant attached | 7 / 4 | 7 / 1 | **7 / 0** | 2 / 0 | 1 / 0 |

0.40 is the only setting with no irrelevant images and no loss of correct ones for both
parsers. The safe band is narrow, especially for the layout parser: its longer
section-path prefixes lower caption similarities, so 0.45 already drops most correct
visuals. The threshold should be re-checked whenever the parser or the embedding model
changes.

### 3.3 Latency

Query embedding plus vector search takes about 0.04–0.13 s per query (mean over the
evaluation, warm models; ToC routing adds about 0.01 s). Retrieval is negligible next to
generation.

## 4. Generation

| Metric | Run 1 | Run 2 | Run 3 | **Run 4 (final)** |
|---|---|---|---|---|
| Mean keyword coverage | 0.77 | 0.64 | 0.86 | **0.89** |
| Answers with full keyword coverage (of 17) | 12 | 10 | 13 | **14** |
| Answers citing a relevant page (of 17) | 11 | 8 | 12 | **12** |
| Answers with any citation | 11 | 10 | 13 | 13 |
| Answers with invalid citation tags | 0 | 0 | 0 | 0 |
| Correct refuse/answer decision (of 18) | **17** | 13 | 16 | 16 |
| Mean input / output tokens | 1177 / 66 | 1191 / 67 | 1193 / 112 | 1256 / 114 |
| Cold query latency (s) | 121 | 129 | 117 | 113 |
| Warm query latency, mean / max (s) | 139 / 235 | 139 / 249 | 160 / 261 | 164 / 308 |
| Peak RSS of the evaluation process (GB) | 14.1 | 14.8 | 14.9 | 15.7 |

Latencies are measured with `time.perf_counter()` (`CLOCK_MONOTONIC` on Linux), so a
machine suspend during run 3, visible as a gap in the log timestamps, does not inflate
them. Latency rises from run 2 to run 4 mainly because answers became longer (output
tokens 66 → 114, about 0.3 s per token) and run 4 carries slightly larger contexts.

### 4.1 What changed between runs, and why

* **Run 1 → 2.** Two run-1 failures were addressed:
  1. q10 copied the prompt's concrete citation example ("The encoder has six layers
     [T1, p.3]") and added a **fabricated claim** ("the decoder has eight layers").
  2. q12 refused because the Table 1 text chunk had been dropped when its image was
     attached.

  Fix: an abstract citation example, and table chunks kept with their images. The new
  strict refusal wording caused **5 false refusals**, and the abstract example
  `[T2, p.5]` was itself copied into answers with wrong pages. **Net: worse.**
* **Run 2 → 3.** Two prompt variants were A/B-tested on the five falsely refused questions
  plus the out-of-scope control, with 30-token generations. Variant A (strict rule, cite
  listed tags) fixed 2 of 5. Variant B ("only if none of the evidence is related") fixed
  4 of 5 and still refused q18. B was adopted.
* **Run 3 → 4.** The layout-aware parser and section-path evidence labels lifted keyword
  coverage from 0.86 to 0.89 and full coverage from 13 to 14 at the same citation and
  refusal accuracy.

The 4.2B model under greedy decoding is very sensitive to prompt wording. Both of the
concrete examples tried were copied verbatim into answers.

## 5. Manual review of the final run (run 4)

Each answer was read against the paper.

| ID | Question (short) | Verdict | Notes |
|---|---|---|---|
| q01 | Why the Transformer | correct | parallelisation and training speed; cites p.2 |
| q02 | What is self-attention | correct, padded | correct definition [T1, p.2], then copies unrelated passages from Sec. 4 |
| q03 | Scaled dot-product attention | correct | dot products, ÷√dk, softmax, and the small-gradient reason; refers to Figure 2 and "page 4" in prose, without tags |
| q04 | Multi-head attention | correct | h = 8; no citation |
| q05 | Positional encoding | **false refusal** | the Sec. 3.5 chunks were not in the top 5 (retrieval miss); page-level labels still counted a p.5 chunk from Sec. 3.2.3 as relevant |
| q06 | Encoder/decoder layers | correct | N = 6, two sub-layers |
| q07 | Decoder masking | correct | masking plus output offset |
| q08 | Training data | correct | WMT 2014 EN-DE 4.5M pairs, EN-FR 36M sentences |
| q09 | Optimizer and schedule | correct | Adam (β values), warmup, then inverse square root |
| q10 | Regularization | correct, padded | residual dropout and label smoothing; copies text and adds an unrelated optimizer sentence; hits the 200-token cap |
| q11 | BLEU EN-DE big model | correct | 28.4 (Table 2 attached); also adds the EN-FR 41.0 figure, which the evidence supports |
| q12 | Complexity vs recurrent | correct, verbatim | constant vs O(n) sequential operations; largely copied text |
| q13 | Architecture diagram | **partially correct** | encoder-decoder, N = 6, embeddings, positional encoding; but "the decoder follows the same architecture as the encoder" is **unsupported**, since the decoder has a third sub-layer |
| q14 | Operations in Fig. 2 (left) | **incorrect** | invents "K is multiplied by V" and "V is multiplied by K"; the actual order is MatMul(Q,K), Scale, Mask (opt.), SoftMax, MatMul with V |
| q15 | Number of heads | correct, plus a stray refusal line | single head 0.9 BLEU worse, too many heads also hurt; then appends the refusal sentence, which the automatic check counts as a refusal |
| q16 | Hardware and time | correct | 8 P100 GPUs, 3.5 days |
| q17 | FFN inner dimension | correct | dff = 2048 |
| q18 | GPT-4 parameters | correct refusal | |

**Summary:**
- 14 of 17 answerable questions are correct and 1 is partially correct; 1 has a factual
  error (q14) and 1 was falsely refused (q05).
- The out-of-scope question was refused.
- 2 of 16 produced answers contain an unsupported claim (q13 minor, q14 major).
- The hardest case is the only question that requires reading a diagram (q14). It failed
  in all four runs, so Phi-3-V does not reliably read the operation order of a small block
  diagram at `num_crops = 4`.

## 6. Resource usage

| Item | Measured |
|---|---|
| Index build (layout parser) | 21.3 s, of which text embedding 19.2 s, image embedding 1.2 s, extraction 0.7 s, Chroma 0.1 s; 79 records |
| Index reuse | about 0.02 s (hash and fingerprint match, nothing recomputed) |
| Jina CLIP / Phi-3-V load | about 1.5 s / about 0.8 s (memory-mapped safetensors, warm page cache) |
| Prefill / decode throughput (Phi-3-V bf16, CPU) | about 10 tokens/s / about 3.3 tokens/s |
| One image at `num_crops = 4` | 757 prompt tokens |
| Peak RSS | Phi-3-V alone about 8.8 GB; full evaluation process 14–16 GB |

## 7. Known weaknesses and next steps

1. **Diagram reading (q14).** Try `num_crops = 8–16` (about 1,400–2,500 more tokens),
   crop each panel separately, or add a short text description of each figure at
   ingestion time.
2. **Retrieval misses for paraphrased questions (q05).** Add query rewriting, or
   section-title matching (ToC records are already indexed) as an additive score rather
   than as routing. Use passage-level relevance labels.
3. **Verbatim copying and padding.** Lower `max_new_tokens` and add a sentence limit in
   post-processing, or use a larger instruction-tuned model.
4. **Evaluation:** build a larger, held-out question set and add an LLM-as-judge or human
   rating for faithfulness. Keep logging every run to MLflow so regressions stay visible
   (run 2 would have been caught).
