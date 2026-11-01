"""Per-query telemetry in SQLite (standard library) for latency, tokens and drift.

Every answered query becomes one row: stage latencies, token counts, retrieval
similarity, visual usage, abstention and citation outcomes, the configuration
that served it, and the query embedding (float32 blob) used for drift
detection. ``source`` separates live traffic (``api``/``cli``) from evaluation
runs (``evaluation``), which serve as the drift reference.

The core never depends on MLflow; ``services/tracking.py`` reads this store and
logs aggregates to MLflow.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

_SCHEMA = """
CREATE TABLE IF NOT EXISTS queries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    run_id TEXT, source TEXT, question TEXT, question_sha TEXT, mode TEXT,
    latency_total REAL, latency_retrieval REAL, latency_generation REAL,
    input_tokens INTEGER, output_tokens INTEGER,
    top_text_similarity REAL, top_visual_score REAL, n_text INTEGER, n_images INTEGER,
    insufficient INTEGER, n_citations INTEGER, n_invalid_citations INTEGER,
    cold_start INTEGER, config_hash TEXT, prompt_version TEXT, error TEXT,
    embedding BLOB
);
CREATE INDEX IF NOT EXISTS idx_queries_ts ON queries(ts);
CREATE INDEX IF NOT EXISTS idx_queries_source ON queries(source);
"""

NUMERIC_FIELDS = ("latency_total", "latency_retrieval", "latency_generation", "input_tokens",
                  "output_tokens", "top_text_similarity", "top_visual_score", "n_images",
                  "insufficient", "n_citations", "n_invalid_citations")


class TelemetryStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        with self._connect() as con:
            con.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.path, timeout=10)
        con.row_factory = sqlite3.Row
        return con

    def record(self, response: dict[str, Any], *, source: str, embedding: np.ndarray | None,
               config_hash: str, prompt_version: str, error: str = "") -> None:
        """Store one query outcome (``response`` = ``QueryResponse.to_dict()``)."""
        t = response.get("timings", {})
        ev = response.get("evidence", [])
        text_sims = [e["similarity"] for e in ev if e["modality"] == "text"]
        vis = [e["similarity"] for e in ev if e["modality"] == "image"]
        row = {
            "ts": time.time(), "run_id": response.get("run_id"), "source": source,
            "question": response.get("question", ""),
            "question_sha": hashlib.sha256(response.get("question", "").encode()).hexdigest()[:16],
            "mode": response.get("retrieval", {}).get("mode"),
            "latency_total": t.get("total"),
            "latency_retrieval": (t.get("query_embedding", 0) + t.get("vector_search", 0)),
            "latency_generation": t.get("generation_generate"),
            "input_tokens": response.get("tokens", {}).get("input"),
            "output_tokens": response.get("tokens", {}).get("output"),
            "top_text_similarity": response.get("retrieval", {}).get("top_text_similarity",
                                                                     max(text_sims, default=None)),
            "top_visual_score": max(vis, default=None), "n_text": len(text_sims), "n_images": len(vis),
            "insufficient": int(bool(response.get("insufficient_evidence"))),
            "n_citations": len(response.get("citations", [])),
            "n_invalid_citations": len(response.get("invalid_citations", [])),
            "cold_start": int(bool(response.get("cold_start"))), "config_hash": config_hash,
            "prompt_version": prompt_version, "error": error,
            "embedding": (np.asarray(embedding, np.float32).tobytes() if embedding is not None else None),
        }
        cols = ", ".join(row)
        with self._lock, self._connect() as con:
            con.execute(f"INSERT INTO queries ({cols}) VALUES ({', '.join('?' * len(row))})",
                        list(row.values()))

    def rows(self, source: str | None = None, since: float | None = None,
             exclude_source: str | None = None, limit: int | None = None) -> list[dict[str, Any]]:
        sql, args = "SELECT * FROM queries WHERE 1=1", []
        if source:
            sql += " AND source = ?"
            args.append(source)
        if exclude_source:
            sql += " AND source != ?"
            args.append(exclude_source)
        if since:
            sql += " AND ts >= ?"
            args.append(since)
        sql += " ORDER BY ts DESC"
        if limit:
            sql += f" LIMIT {int(limit)}"
        with self._connect() as con:
            out = [dict(r) for r in con.execute(sql, args)]
        return out[::-1]  # chronological

    @staticmethod
    def embeddings(rows: list[dict[str, Any]]) -> np.ndarray:
        vecs = [np.frombuffer(r["embedding"], np.float32) for r in rows if r.get("embedding")]
        return np.stack(vecs) if vecs else np.zeros((0, 0), np.float32)


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Latency percentiles, token usage and outcome rates for a set of rows."""
    if not rows:
        return {"n": 0}

    def col(name):
        return np.array([r[name] for r in rows if r.get(name) is not None], dtype=float)

    def pct(name, q):
        v = col(name)
        return round(float(np.percentile(v, q)), 3) if v.size else None

    out: dict[str, Any] = {"n": len(rows)}
    for name in ("latency_total", "latency_retrieval", "latency_generation"):
        out[f"{name}_p50"], out[f"{name}_p95"] = pct(name, 50), pct(name, 95)
    for name in ("input_tokens", "output_tokens"):
        v = col(name)
        out[f"{name}_mean"] = round(float(v.mean()), 1) if v.size else None
        out[f"{name}_total"] = int(v.sum()) if v.size else 0
    for name in ("top_text_similarity", "top_visual_score"):
        v = col(name)
        out[f"{name}_mean"] = round(float(v.mean()), 4) if v.size else None
    out["abstention_rate"] = round(float(col("insufficient").mean()), 4)
    out["citation_rate"] = round(float((col("n_citations") > 0).mean()), 4)
    out["image_rate"] = round(float((col("n_images") > 0).mean()), 4)
    out["error_rate"] = round(sum(1 for r in rows if r.get("error")) / len(rows), 4)
    return out


def config_hash(cfg_dict: dict[str, Any]) -> str:
    keep = {k: cfg_dict[k] for k in ("embedding", "chunking", "retrieval", "context", "generation")}
    return hashlib.sha256(json.dumps(keep, sort_keys=True, default=str).encode()).hexdigest()[:12]
