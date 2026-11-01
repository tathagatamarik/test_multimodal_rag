# Example outputs

These are **real outputs** of the final configuration (evaluation run 4, layout parser,
prompt v3), copied programmatically from
`artifacts/evaluation/run4_layout_parser_final/generation_results.jsonl`. They ran on CPU
with Phi-3-Vision in bf16 and greedy decoding. Successes and failures are both shown;
the "Grounding" lines are a manual review against the paper.

Evidence tags: `T` = text chunk and `I` = image passed to Phi-3-V as pixels. The
similarity of an image is the fused score (mean of pixel and caption cosine similarity).

## q11: Table question: text evidence plus the Table 2 image

**Question:** What BLEU score did the big Transformer achieve on the WMT 2014 English-to-German task?

**Retrieved evidence passed to the model:**

| Tag | Type | Section / figure | Page | Similarity |
|---|---|---|---|---|
| I1 | visual | Table 2 | p.8 | 0.456 |
| T1 | text | 6 Results > 6.1 Machine Translation | p.8 | 0.809 |
| T2 | text | Abstract | p.1 | 0.690 |
| T3 | text | 5 Training > 5.4 Regularization > Table 2 | p.8 | 0.672 |
| T4 | text | 6 Results > 6.2 Model Variations > Table 3 | p.9 | 0.591 |
| T5 | text | 6 Results > 6.2 Model Variations | p.8 | 0.549 |

Image passed to the model: `data/processed/visuals/d87d482d5ae7-table-02.png`

**Generated answer:**

> The big Transformer achieved a BLEU score of 28.4 on the WMT 2014 English-to-German task. [T2] (Abstract, p.1) On the WMT 2014 English-to-French translation task, our model establishes a new single-model state-of-the-art BLEU score of 41.0 after training for 3.5 days on eight GPUs, a small fraction of the training costs of the best models from the literature. [T3] (5 Training > 5.4 Regularization > Table 2, p.8) Table 2: The Transformer achieves better BLEU scores than previous state-of-the-art models on the English-to-German and English-to-French newstest2014 tests at a fraction of the training cost. Model BLEU Training

**Parsed citations:** T2 (p.1), T3 (p.8)
 | **Tokens:** 2196 in / 200 out | **Latency:** 308 s (retrieval 0.07 s)

**Grounding:** Correct. 28.4 BLEU appears in the cited abstract chunk [T2, p.1] and in the attached Table 2 (Transformer (big), EN-DE column). The extra EN-FR sentence is also supported by T2.

## q13: Figure question: Figure 1 passed as pixels

**Question:** What does the Transformer model architecture diagram show?

**Retrieved evidence passed to the model:**

| Tag | Type | Section / figure | Page | Similarity |
|---|---|---|---|---|
| I1 | visual | Figure 1 | p.3 | 0.423 |
| T1 | text | 2 Background | p.2 | 0.537 |
| T2 | text | 3 Model Architecture | p.2 | 0.515 |
| T3 | text | 6 Results > 6.2 Model Variations > Table 3 | p.9 | 0.502 |
| T4 | text | 3 Model Architecture > 3.1 Encoder and Decoder Stacks | p.2 | 0.460 |

Image passed to the model: `data/processed/visuals/d87d482d5ae7-figure-01.png`

**Generated answer:**

> The Transformer model architecture diagram shows the encoder-decoder structure with stacked self-attention and point-wise, fully connected layers for both the encoder and decoder. The encoder is composed of a stack of N = 6 identical layers, each with two sub-layers: a multi-head self-attention mechanism and a simple, position-wise, fully connected layer. The decoder follows the same architecture as the encoder. The diagram also shows the input and output embeddings, as well as the positional encoding. The diagram is labeled as Figure 1: The Transformer - model architecture. [T2, T3, T4]

**Parsed citations:** T2 (p.2), T3 (p.9), T4 (p.2)
 | **Tokens:** 1842 in / 143 out | **Latency:** 248 s (retrieval 0.06 s)

**Grounding:** Partially correct. Encoder-decoder stacks, N = 6, input/output embeddings and positional encoding are visible in Figure 1 and stated in T2/T4. "The decoder follows the same architecture as the encoder" is **not supported**: the figure and Sec. 3.1 show a third (encoder-decoder attention) sub-layer in the decoder. The cited tags are pages 2 and 9; the figure itself (I1, p.3) was not cited.

## q14: Figure question requiring diagram reading (Figure 2, left panel)

**Question:** In the scaled dot-product attention diagram, which operations are applied to Q, K and V?

**Retrieved evidence passed to the model:**

| Tag | Type | Section / figure | Page | Similarity |
|---|---|---|---|---|
| I1 | visual | Figure 2 | p.4 | 0.413 |
| T1 | text | 3 Model Architecture > 3.2 Attention > 3.2.1 Scaled Dot-Product Attention | p.4 | 0.724 |
| T2 | text | 3 Model Architecture > 3.2 Attention > 3.2.1 Scaled Dot-Product Attention | pp.3-4 | 0.708 |
| T3 | text | 3 Model Architecture > 3.2 Attention > 3.2.2 Multi-Head Attention > Footnote | p.4 | 0.660 |

Image passed to the model: `data/processed/visuals/d87d482d5ae7-figure-02.png`

**Generated answer:**

> In the scaled dot-product attention diagram, the operations applied to Q, K, and V are as follows: Q is multiplied by K, then the result is divided by √dk, and a softmax function is applied to obtain the weights on the values. K is multiplied by V, then the result is divided by √dk, and a softmax function is applied to obtain the weights on the values. V is multiplied by K, then the result is divided by √dk, and a softmax function is applied to obtain the weights on the values. These operations are depicted in the diagram as Scaled Dot-Product Attention, where Q, K, and V are represented by the matrices Q, K, and V respectively. The diagram also shows the output of the scaled dot-product attention as the matrix Attention(Q, K, V ), which is the result of applying the softmax function to the matrix obtained by multip

**Parsed citations:** none
 | **Tokens:** 1484 in / 200 out | **Latency:** 218 s (retrieval 0.06 s)

**Grounding:** **Incorrect (hallucination).** The figure shows MatMul(Q, K), Scale, Mask (opt.), SoftMax, then MatMul with V. The answer invents "K is multiplied by V" and "V is multiplied by K". It is kept here deliberately: diagram reading at `num_crops = 4` is the system's weakest point (see the evaluation report, section 5).

## q09: Text-only question

**Question:** Which optimizer and learning rate schedule were used for training?

**Retrieved evidence passed to the model:**

| Tag | Type | Section / figure | Page | Similarity |
|---|---|---|---|---|
| T1 | text | 5 Training > 5.3 Optimizer | p.7 | 0.714 |
| T2 | text | 5 Training > 5.4 Regularization | p.8 | 0.466 |
| T3 | text | 5 Training > 5.2 Hardware and Schedule | p.7 | 0.466 |
| T4 | text | 5 Training > 5.4 Regularization | p.7 | 0.462 |
| T5 | text | 5 Training | p.7 | 0.447 |

**Generated answer:**

> The optimizer used for training was Adam with β1 = 0.9, β2 = 0.98 and ϵ = 10−9. The learning rate schedule used was to increase the learning rate linearly for the first warmup_steps training steps, and decrease it thereafter proportionally to the inverse square root of the step number. [T1] [T2] [T3] [T4] [T5]

**Parsed citations:** T1 (p.7), T2 (p.8), T3 (p.7), T4 (p.7), T5 (p.7)
 | **Tokens:** 658 in / 99 out | **Latency:** 87 s (retrieval 0.09 s)

**Grounding:** Correct and fully grounded in [T1] (Sec. 5.3, p.7): Adam with β1 = 0.9, β2 = 0.98, ϵ = 10⁻⁹; linear warmup, then decay proportional to the inverse square root of the step number. No image was attached, because the fused visual score was below 0.40.

## q07: Masked attention

**Question:** How does the decoder prevent positions from attending to subsequent positions?

**Retrieved evidence passed to the model:**

| Tag | Type | Section / figure | Page | Similarity |
|---|---|---|---|---|
| T1 | text | 3 Model Architecture > 3.1 Encoder and Decoder Stacks | p.3 | 0.477 |
| T2 | text | 3 Model Architecture > 3.2 Attention > 3.2.3 Applications of Attention in our Model | p.5 | 0.448 |
| T3 | text | 3 Model Architecture > 3.5 Positional Encoding | p.5 | 0.429 |
| T4 | text | 3 Model Architecture > 3.5 Positional Encoding | p.6 | 0.402 |

**Generated answer:**

> The decoder prevents positions from attending to subsequent positions by modifying the self-attention sub-layer in the decoder stack to prevent positions from attending to subsequent positions. This is achieved by masking, combined with the fact that the output embeddings are offset by one position. [T2, p.5]

**Parsed citations:** T2 (p.5)
 | **Tokens:** 991 in / 69 out | **Latency:** 110 s (retrieval 0.06 s)

**Grounding:** Correct: masking plus the one-position offset of the output embeddings (Sec. 3.1, p.3, and Sec. 3.2.3, p.5).

## q18: Out-of-scope question (must acknowledge insufficient evidence)

**Question:** How many parameters does GPT-4 have?

**Retrieved evidence passed to the model:**

| Tag | Type | Section / figure | Page | Similarity |
|---|---|---|---|---|
| T1 | text | 6 Results > 6.2 Model Variations > Footnote | p.8 | 0.362 |
| T2 | text | 6 Results > 6.2 Model Variations > Table 3 | p.9 | 0.345 |
| T3 | text | 6 Results > 6.1 Machine Translation | p.8 | 0.338 |
| T4 | text | 5 Training > 5.2 Hardware and Schedule | p.7 | 0.309 |
| T5 | text | 5 Training > 5.1 Training Data and Batching | p.7 | 0.281 |

**Generated answer:**

> The retrieved evidence does not contain enough information to answer this question.

**Parsed citations:** none
 | **Tokens:** 1322 in / 15 out | **Latency:** 129 s (retrieval 0.06 s)

**Grounding:** Correct refusal. The retrieved chunks (results and hardware sections) do not mention GPT-4, and the model returned the fixed insufficient-evidence sentence instead of using outside knowledge.
