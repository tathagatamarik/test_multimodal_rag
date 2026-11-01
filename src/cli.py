"""Command-line interface.

    python -m src.cli index [--rebuild] [--no-download]
    python -m src.cli retrieve "question" [--mode text|image|multimodal]
    python -m src.cli ask "question" [--mode ...] [--json]
    python -m src.cli chat                       # many questions, models loaded once
    python -m src.cli evaluate [--retrieval-only] [--ids q01,q14]
    python -m src.cli ingest file|batch|sync|watch|remove|rebuild|status ...
    python -m src.cli monitor summary|drift [--hours N]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone

from .config import load_config
from .evaluation import evaluate_generation, evaluate_retrieval, load_eval_set
from .exceptions import ConfigurationError, RAGError
from .logging_config import get_run_id, setup_logging
from .pipeline import RAGPipeline

logger = logging.getLogger(__name__)


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m src.cli", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", help="path to config JSON (default: config/config.json)")
    p.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("index", help="build or reuse the multimodal index")
    s.add_argument("--rebuild", action="store_true", help="drop and rebuild the index")
    s.add_argument("--no-download", action="store_true", help="fail instead of downloading the PDF")

    for name in ("retrieve", "ask"):
        s = sub.add_parser(name, help=f"{name} for one question")
        s.add_argument("question")
        s.add_argument("--mode", choices=["text", "image", "multimodal"])
        s.add_argument("--json", action="store_true", help="print the structured response as JSON")

    s = sub.add_parser("chat", help="interactive loop; models stay loaded between questions")
    s.add_argument("--mode", choices=["text", "image", "multimodal"])

    s = sub.add_parser("ingest", help="decoupled ingestion: file, batch, sync, watch, remove, status")
    isub = s.add_subparsers(dest="ingest_command", required=True)
    x = isub.add_parser("file", help="ingest one PDF (skipped if unchanged)")
    x.add_argument("path")
    x.add_argument("--source", help="stable source id (default: absolute path)")
    x = isub.add_parser("batch", help="ingest several PDFs; failures do not stop the batch")
    x.add_argument("paths", nargs="+")
    x = isub.add_parser("sync", help="incrementally mirror a directory of PDFs")
    x.add_argument("directory")
    x.add_argument("--prune", action="store_true", help="remove documents whose file disappeared")
    x = isub.add_parser("watch", help="streaming: ingest PDFs dropped into the inbox directory")
    x.add_argument("--inbox", help="default: vector_store.inbox_dir")
    x.add_argument("--poll", type=float, default=5.0, help="seconds between scans")
    x.add_argument("--max-polls", type=int, help="stop after N scans (default: run forever)")
    x = isub.add_parser("remove", help="remove a document's records")
    x.add_argument("doc_id")
    isub.add_parser("rebuild", help="re-process every registered document (after config changes)")
    isub.add_parser("status", help="registry and collection status")

    s = sub.add_parser("monitor", help="telemetry summary and drift report")
    s.add_argument("what", choices=["summary", "drift"])
    s.add_argument("--hours", type=float, help="current window: last N hours (default: last 200 queries)")

    s = sub.add_parser("evaluate", help="run the evaluation set")
    s.add_argument("--retrieval-only", action="store_true")
    s.add_argument("--ids", help="comma-separated query ids for generation evaluation")
    s.add_argument("--queries", default="data/sample_queries.json")
    s.add_argument("--out", default="artifacts/evaluation")
    return p


def _print_retrieval(res) -> None:
    print(f"Query: {res.query}  (mode={res.mode})")
    for it in res.text:
        md = it.metadata
        print(f"  T{it.rank} sim={it.similarity:.3f} p.{md['page_start']}-{md['page_end']} "
              f"[{md['section'][:50]}] {it.content[:110]!r}")
    for it in res.images:
        comp = ", ".join(f"{k}={v:.3f}" for k, v in it.scores.items())
        print(f"  I{it.rank} score={it.similarity:.3f} ({comp}) p.{it.page} {it.metadata['label']} "
              f"-> {it.metadata['image_path']}")
    for it in res.rejected_images:
        print(f"  (below visual threshold: {it.metadata['label']} score={it.similarity:.3f})")


def _run_ingest(args, cfg, pipeline: RAGPipeline) -> None:
    svc = pipeline.ingestion
    cmd = args.ingest_command
    if cmd == "file":
        r = svc.ingest_file(args.path, source=args.source)
        print(f"{r.action}: {r.filename} (doc_id {r.doc_id}, {r.records} records)")
    elif cmd == "batch":
        print(svc.ingest_batch(args.paths).summary())
    elif cmd == "sync":
        print(svc.sync(args.directory, prune=args.prune).summary())
    elif cmd == "watch":
        from .ingestion.streaming import InboxWatcher
        inbox = cfg.path(args.inbox or cfg.vector_store.inbox_dir)
        with svc.writer():  # hold the writer lock for the whole watch session
            InboxWatcher(svc, inbox, poll_s=args.poll).run(max_polls=args.max_polls)
    elif cmd == "remove":
        r = svc.remove(args.doc_id)
        print(f"removed {r.doc_id} ({r.filename})")
    elif cmd == "rebuild":
        print(svc.rebuild().summary())
    else:
        print(json.dumps(svc.status(), indent=2))


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    overrides = {"logging": {"level": args.log_level}} if args.log_level else None
    try:
        cfg = load_config(args.config, overrides)
    except ConfigurationError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2
    setup_logging(cfg)
    logger.info("Starting '%s' (run %s)", args.command, get_run_id())
    pipeline = RAGPipeline(cfg)

    try:
        if args.command == "index":
            report = pipeline.index(rebuild=args.rebuild, allow_download=not args.no_download)
            print(report.summary())
        elif args.command == "retrieve":
            res = pipeline.retrieve(args.question, mode=args.mode)
            if args.json:
                print(json.dumps({"query": res.query, "text": [vars(i) for i in res.text],
                                  "images": [vars(i) for i in res.images]}, indent=2, default=str))
            else:
                _print_retrieval(res)
        elif args.command == "ask":
            resp = pipeline.ask(args.question, mode=args.mode)
            print(json.dumps(resp.to_dict(), indent=2) if args.json else resp.render())
        elif args.command == "chat":
            print("Ask questions about 'Attention Is All You Need' (empty line or Ctrl-D to quit).")
            while True:
                try:
                    q = input("\n> ").strip()
                except EOFError:
                    break
                if not q:
                    break
                try:
                    print(pipeline.ask(q, mode=args.mode).render())
                except RAGError as exc:  # keep the session alive on a bad question
                    logger.error("Query failed: %s", exc)
        elif args.command == "ingest":
            _run_ingest(args, cfg, pipeline)
        elif args.command == "monitor":
            from .drift import detect_drift
            from .telemetry import summarize
            store = pipeline.telemetry
            since = time.time() - args.hours * 3600 if args.hours else None
            if args.what == "summary":
                print(json.dumps({"live": summarize(store.rows(exclude_source="evaluation", since=since)),
                                  "evaluation": summarize(store.rows(source="evaluation"))}, indent=2))
            else:
                print(json.dumps(detect_drift(store, current_since=since), indent=2))
        elif args.command == "evaluate":
            items = load_eval_set(cfg.path(args.queries))
            out = cfg.path(args.out)
            out.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
            t = time.perf_counter()
            retrieval = evaluate_retrieval(pipeline, items, cfg, sweep_thresholds=(0.30, 0.35, 0.45, 0.50))
            from .generator import PROMPT_VERSION
            (out / "run_info.json").write_text(json.dumps(
                {"prompt_version": PROMPT_VERSION, "parser": cfg.chunking.parser,
                 "structure_routing": cfg.retrieval.structure_routing, "evaluated_at": stamp,
                 "config": cfg.to_dict()}, indent=2, default=str), encoding="utf-8")
            retrieval["evaluated_at"] = stamp
            retrieval["parser"] = cfg.chunking.parser
            (out / "retrieval_report.json").write_text(json.dumps(retrieval, indent=2), encoding="utf-8")
            print(json.dumps(retrieval["variants"], indent=2))
            if not args.retrieval_only:
                ids = args.ids.split(",") if args.ids else None
                summary = evaluate_generation(pipeline, items, out, ids)
                summary["evaluated_at"] = stamp
                (out / "generation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
                print(json.dumps(summary, indent=2))
            logger.info("Evaluation finished in %.1fs; reports in %s", time.perf_counter() - t, out)
    except RAGError as exc:
        logger.critical("%s failed: %s: %s", args.command, type(exc).__name__, exc)
        logger.debug("Traceback", exc_info=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
