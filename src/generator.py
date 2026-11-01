"""Phi-3-Vision answer generation.

Verified against the model repo (revision pinned in config):

* Loaded with ``AutoModelForCausalLM`` / ``AutoProcessor`` and
  ``trust_remote_code=True``; the remote code needs transformers < 4.47
  (it uses ``DynamicCache.seen_tokens``/``get_max_length``) - pinned to 4.45.2.
* ``_attn_implementation="eager"``: the shipped config requests
  flash-attention-2, which is CUDA-only.
* Prompt format is the tokenizer's chat template
  (``<|user|>\\n...<|end|>\\n<|assistant|>\\n``); images are referenced in the
  text as ``<|image_1|>``, ``<|image_2|>``... and passed as PIL images to
  ``processor(prompt, images)``.
* ``num_crops`` controls the HD transform: each image costs
  ``(num_crops + 1) * 144`` visual tokens plus separators (~757 at 4 crops).
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import torch

from .config import GenerationConfig, resolve_device
from .context_builder import Context
from .exceptions import GenerationError

logger = logging.getLogger(__name__)

# Bumped whenever the prompt wording changes; logged with every query and evaluation.
PROMPT_VERSION = "v3-soft-abstain-listed-tags"

INSUFFICIENT = "The retrieved evidence does not contain enough information to answer this question."

# Rules come *after* the evidence and question: Phi-3-V (4.2B) follows instructions
# placed closest to the answer more reliably. Wording was chosen by measurement
# (docs/evaluation_report.md): a concrete citation example was copied verbatim into
# answers (runs 1-2), and a strict "if the evidence does not answer" rule caused false
# abstentions (run 2); this variant fixed 4 of 5 of those without losing the
# out-of-scope refusal.
_INSTRUCTIONS = (
    "Answer the QUESTION about the paper \"Attention Is All You Need\" using ONLY the EVIDENCE above.\n"
    "- Do not use outside knowledge.\n"
    "- After each sentence, cite the supporting evidence tag and its page in square brackets, "
    "using only the tags listed in the EVIDENCE.\n"
    f"- Only if none of the evidence is related to the question, reply exactly: \"{INSUFFICIENT}\"\n"
    "- Use at most 4 sentences."
)

_IMAGE_TOKEN = re.compile(r"<\|image_\d+\|>")
_BRACKET = re.compile(r"\[([^\[\]]{1,60})\]")
_TAG = re.compile(r"\b([TI]\d{1,2})\b")


class ContextTooLongError(GenerationError):
    """The tokenised prompt exceeds ``max_input_tokens``."""

    def __init__(self, n_tokens: int, limit: int):
        super().__init__(f"Prompt has {n_tokens} tokens, exceeding max_input_tokens={limit}")
        self.n_tokens = n_tokens
        self.limit = limit


@dataclass
class GenerationResult:
    answer: str
    citations: list[dict[str, Any]] = field(default_factory=list)
    invalid_citations: list[str] = field(default_factory=list)
    insufficient_evidence: bool = False
    input_tokens: int = 0
    output_tokens: int = 0
    timings: dict[str, float] = field(default_factory=dict)


def build_prompt_text(ctx: Context) -> str:
    """User-turn content: image placeholders, evidence, question, then the rules."""
    image_tags = "".join(f"<|image_{i}|>\n" for i in range(1, len(ctx.images) + 1))
    evidence = ctx.evidence_block() or "(no evidence was retrieved)"
    # Retrieved text must never smuggle in extra image placeholders.
    evidence = _IMAGE_TOKEN.sub("[image]", evidence)
    question = _IMAGE_TOKEN.sub("[image]", ctx.question)
    return (f"{image_tags}EVIDENCE:\n{evidence}\n\nQUESTION: {question}\n\n{_INSTRUCTIONS}")


def _cites_label(group: str, item) -> bool:
    """True if a bracket like "[3.2.1 Scaled Dot-Product Attention, p.4]" names *item*.

    Labels may be full section paths ("3 Model ... > 3.2.1 Scaled ..."), so the cited
    text only has to match one path component (prefix match)."""
    label = group.split(",")[0].strip().lower()
    if len(label) < 5:
        return False
    parts = [p.strip() for p in item.label.lower().split(" > ")] + [item.label.lower()]
    return any(p.startswith(label) for p in parts)


def parse_citations(answer: str, ctx: Context) -> tuple[list[dict[str, Any]], list[str]]:
    """Map ``[T1]``-style tags in the answer to their evidence; collect unknown tags."""
    by_tag = {it.tag: it for it in ctx.items}
    valid: dict[str, dict[str, Any]] = {}
    invalid: list[str] = []
    tags: list[str] = []
    for group in _BRACKET.findall(answer):
        found = _TAG.findall(group)
        # Phi-3-V sometimes cites by label ("[Figure 1, p.3]", "[3.3 Position-wise ..., p.5]")
        # instead of by tag; map such citations to every item with that label.
        tags.extend(found or [it.tag for it in ctx.items if _cites_label(group, it)])
    for tag in tags:
        it = by_tag.get(tag)
        if it is None:
            if tag not in invalid:
                invalid.append(tag)
        elif tag not in valid:
            valid[tag] = {"tag": tag, "modality": it.modality, "label": it.label,
                          "pages": list(range(it.page_start, it.page_end + 1)),
                          "source_ids": it.source_ids}
    return list(valid.values()), invalid


class Phi3VisionGenerator:
    """Loads Phi-3-Vision once and answers questions from a :class:`Context`."""

    def __init__(self, cfg: GenerationConfig,
                 loader: Callable[[], tuple[Any, Any]] | None = None):
        self.cfg = cfg
        self.device = resolve_device(cfg.device)
        self._loader = loader
        self._model = None
        self._processor = None
        self.load_seconds: float | None = None

    def _default_loader(self) -> tuple[Any, Any]:
        from transformers import AutoModelForCausalLM, AutoProcessor

        dtype = getattr(torch, self.cfg.dtype)
        if self.device == "cpu" and dtype == torch.float16:
            raise GenerationError("float16 is not supported for CPU inference; use bfloat16 or float32")
        processor = AutoProcessor.from_pretrained(self.cfg.model_id, revision=self.cfg.revision,
                                                  trust_remote_code=True, num_crops=self.cfg.num_crops)
        model = AutoModelForCausalLM.from_pretrained(
            self.cfg.model_id, revision=self.cfg.revision, trust_remote_code=True,
            torch_dtype=dtype, _attn_implementation=self.cfg.attn_implementation,
            low_cpu_mem_usage=True)
        return model, processor

    def load(self) -> None:
        if self._model is not None:
            return
        t0 = time.perf_counter()
        try:
            model, processor = (self._loader or self._default_loader)()
            model.to(self.device).eval()
        except GenerationError:
            raise
        except (OSError, ValueError, ImportError, RuntimeError) as exc:
            raise GenerationError(f"Failed to load generation model '{self.cfg.model_id}': {exc}") from exc
        self._model, self._processor = model, processor
        self.load_seconds = time.perf_counter() - t0
        logger.info("Loaded %s (%s, %s) in %.1fs", self.cfg.model_id, self.device,
                    self.cfg.dtype, self.load_seconds)

    def count_tokens(self, ctx: Context) -> int:
        """Exact prompt length (text + image tokens) as the model will see it."""
        self.load()
        return int(self._encode(ctx)["input_ids"].shape[1])

    def _encode(self, ctx: Context):
        messages = [{"role": "user", "content": build_prompt_text(ctx)}]
        prompt = self._processor.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        return self._processor(prompt, ctx.images or None, return_tensors="pt")

    def generate(self, ctx: Context) -> GenerationResult:
        """Answer ``ctx.question`` from ``ctx`` evidence.

        With no evidence at all the model is not called: the fixed
        insufficient-evidence answer is returned (no fabricated content).

        Raises:
            ContextTooLongError: prompt exceeds ``max_input_tokens`` (caller may shrink context).
            GenerationError: model failure or empty output.
        """
        if ctx.is_empty:
            logger.warning("No evidence retrieved; returning insufficient-evidence answer")
            return GenerationResult(answer=INSUFFICIENT, insufficient_evidence=True)

        self.load()
        timings: dict[str, float] = {}
        t = time.perf_counter()
        try:
            inputs = self._encode(ctx)
        except (ValueError, RuntimeError, KeyError) as exc:
            raise GenerationError(f"Phi-3-V processor rejected the prompt/images: {exc}") from exc
        n_in = int(inputs["input_ids"].shape[1])
        if n_in > self.cfg.max_input_tokens:
            raise ContextTooLongError(n_in, self.cfg.max_input_tokens)
        inputs = inputs.to(self.device)
        timings["preprocess"] = time.perf_counter() - t

        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": self.cfg.max_new_tokens, "do_sample": self.cfg.do_sample,
            "repetition_penalty": self.cfg.repetition_penalty,
            "eos_token_id": self._processor.tokenizer.eos_token_id,
        }
        if self.cfg.do_sample:
            gen_kwargs["temperature"] = self.cfg.temperature

        t = time.perf_counter()
        try:
            with torch.inference_mode():
                out = self._model.generate(**inputs, **gen_kwargs)
        except (RuntimeError, ValueError, torch.OutOfMemoryError) as exc:
            raise GenerationError(f"Phi-3-V generation failed: {exc}") from exc
        finally:
            if self.device == "cuda":
                torch.cuda.empty_cache()
        timings["generate"] = time.perf_counter() - t

        new_tokens = out[:, n_in:]
        answer = self._processor.batch_decode(new_tokens, skip_special_tokens=True,
                                              clean_up_tokenization_spaces=False)[0].strip()
        if not answer:
            raise GenerationError("Phi-3-V returned an empty answer")
        citations, invalid = parse_citations(answer, ctx)
        if invalid:
            logger.warning("Answer cites unknown evidence tags: %s", invalid)
        n_out = int(new_tokens.shape[1])
        logger.info("Generated %d tokens from %d input tokens in %.1fs", n_out, n_in, timings["generate"])
        return GenerationResult(answer=answer, citations=citations, invalid_citations=invalid,
                                insufficient_evidence=INSUFFICIENT.lower().rstrip(".") in answer.lower(),
                                input_tokens=n_in, output_tokens=n_out, timings=timings)
