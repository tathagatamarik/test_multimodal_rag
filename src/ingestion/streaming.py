"""Streaming ingestion: an in-process job queue and a polling inbox watcher.

* :class:`IngestionQueue` - thread-safe FIFO with one background worker. The
  API enqueues uploads/batches and returns a job id immediately; clients poll
  the job status. One worker = one writer, which is what ChromaDB's embedded
  persistent client supports safely.
* :class:`InboxWatcher` - polls a directory (standard library only, no inotify
  dependency). A file is ingested once its size is stable across two polls (so a
  half-copied file is never read), then moved to ``processed/`` or ``failed/``.

Both reuse :class:`IngestionService`, so streaming and batch ingestion share the
same incremental semantics, registry and verification.
"""

from __future__ import annotations

import logging
import queue
import shutil
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from ..exceptions import RAGError
from .registry import now_iso
from .service import IngestionService

logger = logging.getLogger(__name__)


@dataclass
class Job:
    job_id: str
    kind: str                      # "file" | "batch" | "sync" | "remove"
    payload: dict[str, Any]
    status: str = "queued"         # queued | running | done | failed
    result: Any = None
    error: str = ""
    created_at: str = field(default_factory=now_iso)
    finished_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["result"] = _jsonable(self.result)
        return d


def _jsonable(obj: Any) -> Any:
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, (list, tuple)):
        return [_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if hasattr(obj, "__dataclass_fields__"):
        return _jsonable(asdict(obj))
    return str(obj)


class IngestionQueue:
    """Single-worker background queue around an :class:`IngestionService`."""

    MAX_JOBS_KEPT = 500

    def __init__(self, service: IngestionService):
        self.service = service
        self._q: queue.Queue[Job | None] = queue.Queue()
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._run, name="ingestion-worker", daemon=True)
        self._thread.start()

    def submit(self, kind: str, **payload: Any) -> Job:
        if kind not in {"file", "batch", "sync", "remove"}:
            raise ValueError(f"Unknown job kind {kind!r}")
        job = Job(uuid.uuid4().hex[:12], kind, payload)
        with self._lock:
            self._jobs[job.job_id] = job
            if len(self._jobs) > self.MAX_JOBS_KEPT:
                for old in list(self._jobs)[: len(self._jobs) - self.MAX_JOBS_KEPT]:
                    self._jobs.pop(old, None)
        self._q.put(job)
        logger.info("Queued ingestion job %s (%s)", job.job_id, kind)
        return job

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def jobs(self) -> list[Job]:
        return list(self._jobs.values())

    def pending(self) -> int:
        return self._q.qsize()

    def stop(self, timeout: float = 5.0) -> None:
        self._q.put(None)
        self._thread.join(timeout)

    def _execute(self, job: Job) -> Any:
        p = job.payload
        if job.kind == "file":
            return self.service.ingest_file(p["path"], source=p.get("source"))
        if job.kind == "batch":
            return self.service.ingest_batch(p["paths"])
        if job.kind == "sync":
            return self.service.sync(p["directory"], prune=p.get("prune", False))
        return self.service.remove(p["doc_id"])

    def _run(self) -> None:
        while True:
            job = self._q.get()
            if job is None:
                break
            job.status = "running"
            try:
                job.result = self._execute(job)
                job.status = "done"
            except RAGError as exc:  # domain failure: recorded on the job, worker keeps running
                job.status, job.error = "failed", f"{type(exc).__name__}: {exc}"
                logger.error("Ingestion job %s failed: %s", job.job_id, exc)
            finally:
                job.finished_at = now_iso()
                self._q.task_done()


class InboxWatcher:
    """Poll ``inbox`` for PDFs and ingest each one exactly once."""

    def __init__(self, service: IngestionService, inbox: Path, poll_s: float = 5.0,
                 on_result: Callable[[str, Any], None] | None = None):
        self.service = service
        self.inbox = Path(inbox)
        self.processed = self.inbox / "processed"
        self.failed = self.inbox / "failed"
        for d in (self.inbox, self.processed, self.failed):
            d.mkdir(parents=True, exist_ok=True)
        self.poll_s = poll_s
        self.on_result = on_result
        self._sizes: dict[Path, int] = {}

    def poll_once(self) -> list[tuple[str, str]]:
        """One scan; returns ``(filename, outcome)`` for files handled in this scan."""
        handled = []
        for f in sorted(self.inbox.glob("*.pdf")):
            size = f.stat().st_size
            if self._sizes.get(f) != size:      # new or still being written
                self._sizes[f] = size
                continue
            self._sizes.pop(f, None)
            try:
                result = self.service.ingest_file(f)
                dest, outcome = self.processed / f.name, result.action
            except RAGError as exc:
                dest, outcome = self.failed / f.name, f"failed: {exc}"
                result = exc
            shutil.move(str(f), dest)
            # Keep the registry pointing at the file's new location for future rebuilds.
            entry = next((d for d in self.service.registry.documents.values()
                          if d.path == str(f)), None)
            if entry is not None:
                entry.path = str(dest)
                self.service.registry.upsert(entry)
            handled.append((f.name, outcome))
            logger.info("Inbox: %s -> %s", f.name, outcome)
            if self.on_result:
                self.on_result(f.name, result)
        return handled

    def run(self, stop: threading.Event | None = None, max_polls: int | None = None) -> None:
        polls = 0
        logger.info("Watching %s every %.1fs", self.inbox, self.poll_s)
        while not (stop and stop.is_set()) and (max_polls is None or polls < max_polls):
            self.poll_once()
            polls += 1
            time.sleep(self.poll_s)
