"""FastAPI service over the RAG pipeline.

    uvicorn services.api:app --host 0.0.0.0 --port 8010

Design:
* one :class:`RAGPipeline` per process; Jina CLIP and Phi-3-V load lazily on first
  use and are reused for every request;
* generation is serialised with a lock (one CPU/GPU-bound generate at a time);
  FastAPI runs these sync endpoints in its threadpool, so health/status calls stay
  responsive while an answer is being generated;
* ingestion goes through a single background worker (:class:`IngestionQueue`):
  uploads and batch requests return a job id immediately;
* every query is written to the telemetry store with ``source="api"``.

Endpoints are documented at ``/docs`` (OpenAPI).
"""

from __future__ import annotations

import os
import re
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from src.config import load_config
from src.drift import detect_drift
from src.exceptions import (ConfigurationError, DocumentLoadError, GenerationError,
                            IndexCompatibilityError, RAGError, RetrievalError)
from src.generator import PROMPT_VERSION
from src.ingestion.streaming import IngestionQueue
from src.logging_config import get_run_id, setup_logging
from src.pipeline import RAGPipeline
from src.telemetry import summarize

MAX_UPLOAD_BYTES = 50 * 1024 * 1024


class State:
    pipeline: RAGPipeline
    queue: IngestionQueue
    generate_lock = threading.Lock()
    started_at = time.time()


state = State()


def create_pipeline() -> RAGPipeline:
    """Build the process-wide pipeline (overridable in tests)."""
    cfg = load_config(os.environ.get("RAG_CONFIG"))
    setup_logging(cfg)
    return RAGPipeline(cfg, telemetry_source="api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    state.pipeline = create_pipeline()
    state.queue = IngestionQueue(state.pipeline.ingestion)
    yield
    state.queue.stop()


app = FastAPI(title="Multimodal RAG API",
              description="Question answering over indexed research papers (Jina CLIP + ChromaDB + Phi-3-Vision).",
              version="1.0.0", lifespan=lifespan)


# ------------------------------------------------------------------- schemas
class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    mode: Literal["text", "image", "multimodal"] | None = None
    generate: bool = Field(True, description="false = retrieval only (fast)")
    structure_routing: bool | None = None
    doc_ids: list[str] | None = Field(None, description="restrict retrieval to these documents")


class BatchRequest(BaseModel):
    paths: list[str] | None = None
    directory: str | None = None
    prune: bool = False


def _error(exc: RAGError) -> HTTPException:
    status = {RetrievalError: 422, DocumentLoadError: 422, ConfigurationError: 500,
              IndexCompatibilityError: 409, GenerationError: 503}
    code = next((v for k, v in status.items() if isinstance(exc, k)), 500)
    return HTTPException(code, detail={"error": type(exc).__name__, "message": str(exc)})


def _retrieval_payload(res) -> dict[str, Any]:
    def item(i):
        return {"id": i.id, "modality": i.modality, "rank": i.rank, "similarity": round(i.similarity, 4),
                "scores": {k: round(v, 4) for k, v in i.scores.items()},
                "page": i.page, "label": i.metadata.get("label") or i.metadata.get("section_path"),
                "source": i.metadata.get("source"), "doc_id": i.metadata.get("doc_id"),
                "visual_id": i.metadata.get("visual_id"), "content": i.content[:600]}
    return {"query": res.query, "mode": res.mode, "text": [item(i) for i in res.text],
            "images": [item(i) for i in res.images],
            "rejected_images": [item(i) for i in res.rejected_images],
            "sections": [item(i) for i in res.sections],
            "timings": {k: round(v, 4) for k, v in res.timings.items()}}


# ------------------------------------------------------------------- endpoints
@app.get("/health")
def health() -> dict[str, Any]:
    p = state.pipeline
    return {"status": "ok", "run_id": get_run_id(), "uptime_s": round(time.time() - state.started_at),
            "embedding_model_loaded": p.embedder._model is not None,
            "generation_model_loaded": p._generator is not None and p.generator._model is not None,
            "generation_busy": state.generate_lock.locked(), "ingestion_queue": state.queue.pending(),
            "prompt_version": PROMPT_VERSION}


@app.get("/index/status")
def index_status() -> dict[str, Any]:
    return state.pipeline.ingestion.status()


@app.get("/documents")
def documents() -> list[dict[str, Any]]:
    return [d.to_dict() for d in state.pipeline.ingestion.registry.documents.values()]


@app.post("/retrieve")
def retrieve(req: QueryRequest) -> dict[str, Any]:
    try:
        res = state.pipeline.retrieve(req.question, mode=req.mode, structure_routing=req.structure_routing,
                                      doc_ids=req.doc_ids)
    except RAGError as exc:
        raise _error(exc) from exc
    return _retrieval_payload(res)


@app.post("/query")
def query(req: QueryRequest) -> dict[str, Any]:
    if not req.generate:
        return {"retrieval": retrieve(req)}
    with state.generate_lock:
        try:
            resp = state.pipeline.ask(req.question, mode=req.mode, source="api",
                                      structure_routing=req.structure_routing, doc_ids=req.doc_ids)
        except RAGError as exc:
            raise _error(exc) from exc
    return resp.to_dict()


@app.get("/visuals/{visual_id}")
def visual(visual_id: str) -> FileResponse:
    if not re.fullmatch(r"[A-Za-z0-9_.-]{1,120}", visual_id):
        raise HTTPException(400, "invalid visual id")
    hits = state.pipeline.store.get([visual_id])
    if not hits or not hits[0].metadata.get("image_path"):
        raise HTTPException(404, "visual not found")
    path = state.pipeline.cfg.path(hits[0].metadata["image_path"])
    if not path.is_file():
        raise HTTPException(404, "image file missing")
    return FileResponse(path, media_type="image/png")


@app.post("/ingest/upload", status_code=202)
async def ingest_upload(file: UploadFile = File(...)) -> dict[str, Any]:
    name = Path(file.filename or "upload.pdf").name
    if not name.lower().endswith(".pdf"):
        raise HTTPException(415, "only PDF files are accepted")
    data = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "file too large")
    if not data.startswith(b"%PDF-"):
        raise HTTPException(415, "file is not a PDF")
    cfg = state.pipeline.cfg
    dest_dir = cfg.path("data/uploads")
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / re.sub(r"[^A-Za-z0-9_.-]", "_", name)
    dest.write_bytes(data)
    job = state.queue.submit("file", path=str(dest), source=f"upload:{dest.name}")
    return job.to_dict()


@app.post("/ingest/batch", status_code=202)
def ingest_batch(req: BatchRequest) -> dict[str, Any]:
    if bool(req.paths) == bool(req.directory):
        raise HTTPException(422, "give exactly one of 'paths' or 'directory'")
    job = (state.queue.submit("batch", paths=req.paths) if req.paths
           else state.queue.submit("sync", directory=req.directory, prune=req.prune))
    return job.to_dict()


@app.get("/ingest/jobs")
def ingest_jobs() -> list[dict[str, Any]]:
    return [j.to_dict() for j in reversed(state.queue.jobs())]


@app.get("/ingest/jobs/{job_id}")
def ingest_job(job_id: str) -> dict[str, Any]:
    job = state.queue.get(job_id)
    if job is None:
        raise HTTPException(404, "unknown job")
    return job.to_dict()


@app.delete("/documents/{doc_id}", status_code=202)
def delete_document(doc_id: str) -> dict[str, Any]:
    if state.pipeline.ingestion.registry.get(doc_id) is None:
        raise HTTPException(404, "unknown document")
    return state.queue.submit("remove", doc_id=doc_id).to_dict()


@app.get("/metrics/summary")
def metrics_summary(hours: float | None = None) -> dict[str, Any]:
    t = state.pipeline.telemetry
    since = time.time() - hours * 3600 if hours else None
    return {"live": summarize(t.rows(exclude_source="evaluation", since=since)),
            "evaluation": summarize(t.rows(source="evaluation"))}


@app.get("/metrics/queries")
def metrics_queries(limit: int = 100) -> list[dict[str, Any]]:
    rows = state.pipeline.telemetry.rows(limit=min(limit, 1000))
    return [{k: v for k, v in r.items() if k != "embedding"} for r in rows]


@app.get("/monitoring/drift")
def drift(hours: float | None = None) -> dict[str, Any]:
    since = time.time() - hours * 3600 if hours else None
    return detect_drift(state.pipeline.telemetry, current_since=since)
