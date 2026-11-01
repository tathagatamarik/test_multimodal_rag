"""One-time logging setup (console + optional rotating file).

Modules obtain loggers with ``logging.getLogger(__name__)`` and never configure
handlers themselves, so messages are never duplicated.
"""

from __future__ import annotations

import logging
import logging.handlers
import uuid
from pathlib import Path

from .config import Config

_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | run=%(run_id)s | %(message)s"
_HANDLER_TAG = "_mmrag_handler"
_RUN_ID = uuid.uuid4().hex[:8]

# Third-party libraries that are chatty at INFO level.
_NOISY = ("chromadb", "httpx", "urllib3", "transformers", "huggingface_hub", "PIL", "timm")


class _RunIdFilter(logging.Filter):
    """Stamp every record with the process-wide run identifier."""

    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "run_id"):
            record.run_id = _RUN_ID
        return True


def get_run_id() -> str:
    return _RUN_ID


def setup_logging(config: Config) -> logging.Logger:
    """Configure the ``src`` logger tree. Idempotent: safe to call repeatedly."""
    root = logging.getLogger("src")
    root.setLevel(config.logging.level.upper())
    root.propagate = False

    for h in list(root.handlers):
        if getattr(h, _HANDLER_TAG, False):
            root.removeHandler(h)
            h.close()

    formatter = logging.Formatter(_FORMAT)
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if config.logging.log_file:
        log_path: Path = config.path(config.logging.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.handlers.RotatingFileHandler(
            log_path, maxBytes=config.logging.max_bytes,
            backupCount=config.logging.backup_count, encoding="utf-8",
        ))
    for h in handlers:
        h.setFormatter(formatter)
        h.addFilter(_RunIdFilter())
        setattr(h, _HANDLER_TAG, True)
        root.addHandler(h)

    for name in _NOISY:
        logging.getLogger(name).setLevel(logging.WARNING)
    return root
