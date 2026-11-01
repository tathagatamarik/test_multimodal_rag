"""Retrieval and generation evaluation against ``data/sample_queries.json``.

Relevance labels are page-level (a text chunk is relevant if its page range
intersects ``relevant_pages``) plus visual-level (``relevant_visuals`` such as
"Table 2"). All metrics are computed from actual runs; nothing is estimated.

Retrieval metrics (text list, cut-off k):
  * Hit@k       - at least one relevant chunk in the top k.
  * Precision@k - fraction of the top-k chunks that are relevant.
  * Recall@k    - fraction of relevant pages covered by the top-k chunks.
  * MRR         - reciprocal rank of the first relevant chunk (0 if none).
Visual metrics (questions with ``relevant_visuals``):
  * VisualHit   - an expected visual is among the retrieved images.
  * VisualInCtx - an expected visual is actually attached to the prompt (respects max_images).
  * Irrelevant images - images attached for questions that have no expected visual.

Generation checks (automatic, deliberately simple and transparent):
  * keyword coverage - share of expected keyword groups present in the answer.
  * citation validity - cited tags exist and at least one cited item is on a relevant page.
  * abstention - unanswerable questions must produce the insufficient-evidence reply,
    answerable ones must not.
"""

from __future__ import annotations

import json
import logging
import resource
import statistics
import time
from pathlib import Path
from typing import Any

from .config import Config
from .context_builder import build_context, default_image_loader
from .retriever import RetrievalResult

logger = logging.getLogger(__name__)


def load_eval_set(path: Path) -> list[dict[str, Any]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return data["queries"]


def _relevant(meta: dict[str, Any], pages: set[int]) -> bool:
    start = int(meta.get("page_start", meta.get("page", 0)))
    end = int(meta.get("page_end", start))
    return any(p in pages for p in range(start, end + 1))


def retrieval_metrics(result: RetrievalResult, item: dict[str, Any], k: int) -> dict[str, Any]:
    pages = set(item["relevant_pages"])
    top = result.text[:k]
    rel_flags = [_relevant(t.metadata, pages) for t in top]
    covered = {p for t in top for p in range(int(t.metadata["page_start"]), int(t.metadata["page_end"]) + 1)}
    first = next((i + 1 for i, f in enumerate(rel_flags) if f), None)
    m: dict[str, Any] = {
        "hit": bool(first), "precision": sum(rel_flags) / k if top else 0.0,
        "recall": len(covered & pages) / len(pages) if pages else None,
        "mrr": 1.0 / first if first else 0.0,
        "retrieved_pages": [[int(t.metadata["page_start"]), int(t.metadata["page_end"])] for t in top],
        "top_text_similarity": round(top[0].similarity, 4) if top else None,
        "images": [{"label": i.metadata.get("label"), "score": round(i.similarity, 4),
                    **{k2: round(v, 4) for k2, v in i.scores.items()}} for i in result.images],
    }
    expected = set(item.get("relevant_visuals", []))
    if expected:
        m["visual_hit"] = any(i.metadata.get("label") in expected for i in result.images)
    return m


def _mean(vals: list[float]) -> float | None:
    vals = [v for v in vals if v is not None]
    return round(statistics.mean(vals), 4) if vals else None


def evaluate_retrieval(pipeline, items: list[dict[str, Any]], cfg: Config,
                       sweep_thresholds: tuple[float, ...] = ()) -> dict[str, Any]:
    """Compare retrieval variants (and optional visual-score thresholds)."""
    k = cfg.retrieval.top_k_text
    thr = cfg.retrieval.min_visual_score
    variants = {
        "text_only": {"mode": "text"},
        "multimodal_pixel": {"mode": "multimodal", "visual_scoring": "pixel", "min_visual_score": -1.0},
        "multimodal_fused_ungated": {"mode": "multimodal", "visual_scoring": "fused",
                                     "min_visual_score": -1.0},
        f"multimodal_fused_gated@{thr}": {"mode": "multimodal", "visual_scoring": "fused",
                                          "min_visual_score": thr},
    }
    variants[f"multimodal_fused_gated@{thr}+structure_routing"] = {
        "mode": "multimodal", "visual_scoring": "fused", "min_visual_score": thr,
        "structure_routing": True}
    for sweep in sweep_thresholds:
        variants[f"multimodal_fused_gated@{sweep}"] = {"mode": "multimodal", "visual_scoring": "fused",
                                                       "min_visual_score": sweep}
    loader = default_image_loader(cfg.project_root)
    report: dict[str, Any] = {"k": k, "top_k_image": cfg.retrieval.top_k_image,
                              "max_images_in_context": cfg.context.max_images,
                              "configured_min_visual_score": thr,
                              "variants": {}, "per_query": {}}
    for name, kw in variants.items():
        rows = []
        for item in items:
            t = time.perf_counter()
            res = pipeline.retrieve(item["question"], **kw)
            latency = time.perf_counter() - t
            m = retrieval_metrics(res, item, k)
            ctx = build_context(res, cfg.context, loader)
            in_ctx = {it.label for it in ctx.image_items}
            expected = set(item.get("relevant_visuals", []))
            m["visual_in_context"] = bool(in_ctx & expected) if expected else None
            m["irrelevant_images_in_context"] = len(in_ctx - expected)
            m["latency_s"] = round(latency, 4)
            rows.append((item, m))
            report["per_query"].setdefault(item["id"], {})[name] = m
        answerable = [m for it, m in rows if it["answerable"]]
        visual_q = [m for it, m in rows if it.get("relevant_visuals")]
        report["variants"][name] = {
            f"hit@{k}": _mean([float(m["hit"]) for m in answerable]),
            f"precision@{k}": _mean([m["precision"] for m in answerable]),
            f"recall@{k}": _mean([m["recall"] for m in answerable]),
            "mrr": _mean([m["mrr"] for m in answerable]),
            "visual_hit": _mean([float(m.get("visual_hit", False)) for m in visual_q]),
            "visual_in_context": _mean([float(bool(m["visual_in_context"])) for m in visual_q]),
            "questions_with_irrelevant_image_in_context": sum(
                1 for m in answerable if m["irrelevant_images_in_context"]),
            "images_in_context_total": sum(
                1 for it, m in rows if m["visual_in_context"] or m["irrelevant_images_in_context"]),
            "mean_retrieval_latency_s": _mean([m["latency_s"] for m in answerable]),
            "n_answerable": len(answerable), "n_visual_questions": len(visual_q),
        }
    return report


def check_answer(response: dict[str, Any], item: dict[str, Any]) -> dict[str, Any]:
    answer = response["answer"].lower()
    groups = item.get("expected_keywords", [])
    found = [any(alt.lower() in answer for alt in g) for g in groups]
    pages = set(item["relevant_pages"])
    cited_pages = {p for c in response["citations"] for p in c["pages"]}
    out = {
        "keyword_coverage": round(sum(found) / len(groups), 3) if groups else None,
        "missing_keyword_groups": [g for g, f in zip(groups, found) if not f],
        "n_citations": len(response["citations"]),
        "invalid_citations": response["invalid_citations"],
        "cites_relevant_page": bool(cited_pages & pages) if pages else None,
        "abstained": response["insufficient_evidence"],
    }
    out["abstention_correct"] = out["abstained"] == (not item["answerable"])
    return out


def evaluate_generation(pipeline, items: list[dict[str, Any]], out_dir: Path,
                        ids: list[str] | None = None) -> dict[str, Any]:
    """Run the full pipeline per question, writing results incrementally."""
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "generation_results.jsonl"
    done: dict[str, dict] = {}
    if results_path.exists():  # resume: a CPU run takes minutes per question
        for line in results_path.read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            done[rec["id"]] = rec
    selected = [it for it in items if ids is None or it["id"] in ids]
    for item in selected:
        if item["id"] in done:
            logger.info("Skipping %s (already evaluated)", item["id"])
            continue
        # Logged as source="evaluation": these rows are the reference window for drift detection.
        resp = pipeline.ask(item["question"], source="evaluation").to_dict()
        rec = {"id": item["id"], "category": item["category"], "response": resp,
               "checks": check_answer(resp, item),
               "peak_rss_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 2)}
        with results_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")
        done[item["id"]] = rec
        logger.info("[%s] keywords=%s cites_relevant=%s abstained=%s total=%.0fs", item["id"],
                    rec["checks"]["keyword_coverage"], rec["checks"]["cites_relevant_page"],
                    rec["checks"]["abstained"], resp["timings"]["total"])
    return summarize_generation([done[it["id"]] for it in selected if it["id"] in done])


def summarize_generation(records: list[dict[str, Any]]) -> dict[str, Any]:
    checks = [r["checks"] for r in records]
    warm = [r["response"]["timings"]["total"] for r in records if not r["response"]["cold_start"]]
    cold = [r["response"]["timings"]["total"] for r in records if r["response"]["cold_start"]]
    gen = [r["response"]["timings"].get("generation_generate") for r in records]
    return {
        "n": len(records),
        "mean_keyword_coverage": _mean([c["keyword_coverage"] for c in checks]),
        "full_keyword_coverage": sum(1 for c in checks if c["keyword_coverage"] == 1.0),
        "answers_citing_relevant_page": sum(1 for c in checks if c["cites_relevant_page"]),
        "answers_with_any_citation": sum(1 for c in checks if c["n_citations"]),
        "answers_with_invalid_citations": sum(1 for c in checks if c["invalid_citations"]),
        "abstention_correct": sum(1 for c in checks if c["abstention_correct"]),
        "mean_input_tokens": _mean([r["response"]["tokens"].get("input", 0) for r in records]),
        "mean_output_tokens": _mean([r["response"]["tokens"].get("output", 0) for r in records]),
        "cold_query_latency_s": cold,
        "warm_query_latency_mean_s": _mean(warm),
        "warm_query_latency_max_s": max(warm) if warm else None,
        "mean_generate_s": _mean(gen),
        "peak_rss_gb": max((r.get("peak_rss_gb", 0) for r in records), default=None),
    }
