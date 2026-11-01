"""MLflow experiment tracking for evaluations and production telemetry.

    python -m services.tracking log-eval [artifacts/evaluation]   # one MLflow run per evaluation
    python -m services.tracking log-telemetry [--hours 24]         # window aggregates + drift
    mlflow ui --backend-store-uri sqlite:///storage/mlflow.db --port 5000

* ``log-eval``: params = retrieval/generation config, parser, prompt version;
  metrics = retrieval (Hit@k, MRR, visual metrics per variant) and generation
  (keyword coverage, citations, abstention, latency, tokens); artifacts = the
  raw reports. Comparing runs in the MLflow UI shows the effect of each change.
* ``log-telemetry``: latency percentiles, token totals, outcome rates of live
  traffic plus every drift signal, logged as a run tagged ``kind=monitoring``.
  Schedule it (cron/systemd timer) for a time series of production health.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import mlflow

from src.config import load_config
from src.drift import detect_drift
from src.generator import PROMPT_VERSION
from src.telemetry import TelemetryStore, summarize


def _setup(cfg) -> None:
    uri = cfg.monitoring.mlflow_tracking_uri
    if uri.startswith("sqlite:///") and not uri.startswith("sqlite:////"):
        uri = "sqlite:///" + str(cfg.path(uri[len("sqlite:///"):]))
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(cfg.monitoring.mlflow_experiment)


def _metric_name(s: str) -> str:
    return "".join(c if c.isalnum() or c in "_-./ " else "_" for c in s)


def _log_numbers(prefix: str, d: dict[str, Any]) -> None:
    for k, v in d.items():
        if isinstance(v, bool):
            v = float(v)
        if isinstance(v, (int, float)) and not (isinstance(v, float) and math.isnan(v)):
            mlflow.log_metric(_metric_name(f"{prefix}{k}"), float(v))


def log_evaluation(eval_dir: Path, run_name: str | None = None) -> str:
    cfg = load_config()
    _setup(cfg)
    retrieval = json.loads((eval_dir / "retrieval_report.json").read_text())
    gen_path = eval_dir / "generation_summary.json"
    generation = json.loads(gen_path.read_text()) if gen_path.exists() else None
    # run_info.json (written by `evaluate`) records the prompt/parser the run actually used;
    # the current config is only a fallback for runs that predate it.
    info_path = eval_dir / "run_info.json"
    info = json.loads(info_path.read_text()) if info_path.exists() else {}
    with mlflow.start_run(run_name=run_name or f"eval-{eval_dir.name}") as run:
        mlflow.set_tags({"kind": "evaluation", "eval_dir": str(eval_dir),
                         "prompt_version": info.get("prompt_version", PROMPT_VERSION),
                         "description": info.get("description", "")})
        mlflow.log_params({
            "parser": info.get("parser", retrieval.get("parser", "basic")), "top_k_text": retrieval["k"],
            "top_k_image": retrieval["top_k_image"], "max_images": retrieval["max_images_in_context"],
            "min_visual_score": retrieval.get("configured_min_visual_score"),
            "embedding_model": cfg.embedding.model_id, "generation_model": cfg.generation.model_id,
            "num_crops": cfg.generation.num_crops, "max_context_chars": cfg.context.max_context_chars,
            "structure_routing": cfg.retrieval.structure_routing})
        for variant, metrics in retrieval["variants"].items():
            _log_numbers(f"retrieval.{variant}.", metrics)
        if generation:
            _log_numbers("generation.", {k: v for k, v in generation.items() if not isinstance(v, list)})
        for f in eval_dir.glob("*.json*"):
            mlflow.log_artifact(str(f))
        return run.info.run_id


def log_telemetry(hours: float | None = None) -> str:
    cfg = load_config()
    _setup(cfg)
    store = TelemetryStore(cfg.path(cfg.monitoring.telemetry_path))
    since = time.time() - hours * 3600 if hours else None
    live = summarize(store.rows(exclude_source="evaluation", since=since))
    drift = detect_drift(store, current_since=since)
    with mlflow.start_run(run_name=f"monitoring-{time.strftime('%Y%m%d-%H%M')}") as run:
        mlflow.set_tags({"kind": "monitoring", "drift_status": drift["status"],
                         "window_hours": str(hours or "last-200")})
        _log_numbers("live.", live)
        for name, sig in drift.get("signals", {}).items():
            _log_numbers(f"drift.{name}.", {k: v for k, v in sig.items() if k != "level"})
        mlflow.log_dict(drift, "drift_report.json")
        return run.info.run_id


def main() -> None:
    p = argparse.ArgumentParser(prog="python -m services.tracking")
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("log-eval")
    e.add_argument("eval_dir", nargs="?", default="artifacts/evaluation")
    e.add_argument("--name")
    t = sub.add_parser("log-telemetry")
    t.add_argument("--hours", type=float)
    a = p.parse_args()
    if a.cmd == "log-eval":
        print("MLflow run:", log_evaluation(Path(a.eval_dir), a.name))
    else:
        print("MLflow run:", log_telemetry(a.hours))


if __name__ == "__main__":
    main()
