"""Drift detection between a reference window and a current window of queries.

Reference = queries logged during an evaluation run (known-good behaviour);
current = recent live queries. Three families of signals, all computed with
NumPy only:

* **Input (query) drift** - are users asking different things?
  - centroid cosine distance between the two embedding sets;
  - out-of-domain share: fraction of current queries whose nearest reference
    query is less similar than the reference set's own 5th percentile of
    nearest-neighbour similarity.
* **Retrieval drift** - does the index still match the questions? PSI and a
  two-sample Kolmogorov-Smirnov test on the top text similarity.
* **Operational drift** - latency, token usage and outcome rates (abstention,
  citations) compared with the reference.

Thresholds follow common practice: PSI < 0.1 stable, 0.1-0.25 moderate,
> 0.25 significant; KS p < 0.05 significant. With small windows these tests
have little power, so ``min_samples`` guards against reporting noise as drift.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .telemetry import TelemetryStore

PSI_MODERATE, PSI_SIGNIFICANT = 0.1, 0.25
KS_ALPHA = 0.05
_EPS = 1e-6


def psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index using reference-quantile bins."""
    reference, current = np.asarray(reference, float), np.asarray(current, float)
    if reference.size < 2 or current.size < 1:
        return float("nan")
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if edges.size < 2:
        return 0.0 if np.allclose(current, reference[0]) else float("inf")
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(reference, edges)[0] / reference.size + _EPS
    c = np.histogram(current, edges)[0] / current.size + _EPS
    return float(np.sum((c - r) * np.log(c / r)))


def ks_2samp(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Two-sample KS statistic and asymptotic p-value (Kolmogorov distribution)."""
    a, b = np.sort(np.asarray(a, float)), np.sort(np.asarray(b, float))
    if a.size == 0 or b.size == 0:
        return float("nan"), float("nan")
    grid = np.concatenate([a, b])
    d = float(np.max(np.abs(np.searchsorted(a, grid, "right") / a.size
                            - np.searchsorted(b, grid, "right") / b.size)))
    n = a.size * b.size / (a.size + b.size)
    lam = (math.sqrt(n) + 0.12 + 0.11 / math.sqrt(n)) * d
    p = 2 * sum((-1) ** (k - 1) * math.exp(-2 * k * k * lam * lam) for k in range(1, 101))
    return d, float(min(1.0, max(0.0, p)))


def embedding_drift(ref: np.ndarray, cur: np.ndarray) -> dict[str, float]:
    if ref.size == 0 or cur.size == 0:
        return {}
    rc, cc = ref.mean(0), cur.mean(0)
    centroid_dist = 1.0 - float(rc @ cc / (np.linalg.norm(rc) * np.linalg.norm(cc) + _EPS))
    sim_rr = ref @ ref.T
    np.fill_diagonal(sim_rr, -np.inf)
    ref_nn = sim_rr.max(1) if ref.shape[0] > 1 else np.array([1.0])
    floor = float(np.percentile(ref_nn, 5))
    cur_nn = (cur @ ref.T).max(1)
    return {"centroid_cosine_distance": round(centroid_dist, 4),
            "reference_nn_floor": round(floor, 4),
            "current_nn_similarity_mean": round(float(cur_nn.mean()), 4),
            "out_of_domain_share": round(float((cur_nn < floor).mean()), 4)}


def _level(value: float, moderate: float, significant: float) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "insufficient_data"
    return "significant" if value > significant else "moderate" if value > moderate else "stable"


def detect_drift(store: TelemetryStore, *, reference_source: str = "evaluation",
                 current_since: float | None = None, current_limit: int = 200,
                 min_samples: int = 10) -> dict[str, Any]:
    ref = store.rows(source=reference_source)
    cur = store.rows(exclude_source=reference_source, since=current_since, limit=current_limit)
    report: dict[str, Any] = {"reference_n": len(ref), "current_n": len(cur), "signals": {},
                              "status": "insufficient_data"}
    if len(ref) < min_samples or len(cur) < min_samples:
        report["message"] = (f"Need at least {min_samples} reference and current queries "
                             f"(have {len(ref)} / {len(cur)}).")
        return report

    sig: dict[str, Any] = {}
    emb = embedding_drift(TelemetryStore.embeddings(ref), TelemetryStore.embeddings(cur))
    if emb:
        sig["query_embedding"] = emb | {"level": _level(emb["out_of_domain_share"], 0.2, 0.4)}

    def series(rows, name):
        return np.array([r[name] for r in rows if r.get(name) is not None], float)

    for name in ("top_text_similarity", "latency_total", "input_tokens", "output_tokens"):
        a, b = series(ref, name), series(cur, name)
        if a.size >= 2 and b.size >= 2:
            p = psi(a, b)
            d, pv = ks_2samp(a, b)
            level = _level(p, PSI_MODERATE, PSI_SIGNIFICANT)
            if level != "stable" and pv >= KS_ALPHA:
                level = "moderate" if level == "significant" else level
            sig[name] = {"psi": round(p, 4), "ks_stat": round(d, 4), "ks_p": round(pv, 4),
                         "ref_mean": round(float(a.mean()), 4), "cur_mean": round(float(b.mean()), 4),
                         "level": level}
    for name in ("insufficient", "n_images"):
        a, b = (series(ref, name) > 0).mean(), (series(cur, name) > 0).mean()
        delta = float(b - a)
        sig[f"{name}_rate"] = {"ref": round(float(a), 4), "cur": round(float(b), 4),
                               "delta": round(delta, 4), "level": _level(abs(delta), 0.1, 0.25)}
    report["signals"] = sig
    levels = [s["level"] for s in sig.values()]
    report["status"] = ("significant" if "significant" in levels else
                        "moderate" if "moderate" in levels else "stable")
    return report
