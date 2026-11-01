"""Centralised, validated configuration.

All tunable values live in ``config/config.json``; this module maps them onto
typed dataclasses, resolves relative paths against the project root and
validates ranges *before* any expensive work (downloads, model loads) starts.

Parameter reference (type, accepted range, default) is documented in README.md
section "Configuration reference".
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

from .exceptions import ConfigurationError

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.json"

_DEVICES = {"auto", "cpu", "cuda"}
_DTYPES = {"float32", "bfloat16", "float16"}
_RETRIEVAL_MODES = {"text", "image", "multimodal"}
_LOG_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}


@dataclass
class DocumentConfig:
    source_url: str = (
        "https://proceedings.neurips.cc/paper_files/paper/2017/file/"
        "3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf"
    )
    local_path: str = "data/raw/attention_is_all_you_need.pdf"
    download_timeout_s: float = 60.0
    download_max_retries: int = 3
    max_download_bytes: int = 50 * 1024 * 1024


@dataclass
class ChunkingConfig:
    chunk_size_chars: int = 900
    chunk_overlap_chars: int = 150
    min_chunk_chars: int = 80
    exclude_references: bool = True
    parser: str = "layout"          # "layout" (font/position-aware, ToC paths) | "basic" (regex)


@dataclass
class ImageConfig:
    render_dpi: int = 200
    min_image_side_px: int = 64
    max_image_side_px: int = 2048
    extract_tables: bool = True
    output_dir: str = "data/processed/visuals"


@dataclass
class EmbeddingConfig:
    model_id: str = "jinaai/jina-clip-v1"
    revision: str | None = "ceb3e44ca4d6eceaa4f3fb58b1c1a5748b3f29b6"
    batch_size: int = 16
    max_text_tokens: int = 512
    device: str = "auto"


@dataclass
class VectorStoreConfig:
    persist_dir: str = "storage/chroma"
    collection_name: str = "attention_paper_mm"
    registry_path: str = "storage/registry.json"
    inbox_dir: str = "data/inbox"


@dataclass
class RetrievalConfig:
    mode: str = "multimodal"
    top_k_text: int = 5
    top_k_image: int = 2
    visual_scoring: str = "fused"
    min_visual_score: float = 0.40
    structure_routing: bool = False    # STAIR-style: route via ToC section records first
    top_k_sections: int = 2
    routed_text_slots: int = 3


@dataclass
class ContextConfig:
    max_context_chars: int = 3600
    max_images: int = 1


@dataclass
class GenerationConfig:
    model_id: str = "microsoft/Phi-3-vision-128k-instruct"
    revision: str | None = "ed2772fabe9dc9acd0caad54b62761d92520cc44"
    device: str = "auto"
    dtype: str = "bfloat16"
    num_crops: int = 4
    max_new_tokens: int = 200
    max_input_tokens: int = 4096
    do_sample: bool = False
    temperature: float = 0.0
    repetition_penalty: float = 1.0
    attn_implementation: str = "eager"


@dataclass
class MonitoringConfig:
    telemetry_path: str = "storage/telemetry.db"
    record_queries: bool = True
    mlflow_tracking_uri: str = "sqlite:///storage/mlflow.db"
    mlflow_experiment: str = "multimodal-rag"


@dataclass
class LoggingConfig:
    level: str = "INFO"
    log_file: str | None = "logs/rag.log"
    max_bytes: int = 5 * 1024 * 1024
    backup_count: int = 3


@dataclass
class Config:
    """Root configuration object. Build with :func:`load_config`."""

    document: DocumentConfig = field(default_factory=DocumentConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    images: ImageConfig = field(default_factory=ImageConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    vector_store: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    context: ContextConfig = field(default_factory=ContextConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    project_root: Path = PROJECT_ROOT

    def path(self, relative: str | Path) -> Path:
        """Resolve a configured path against the project root."""
        p = Path(relative)
        return p if p.is_absolute() else self.project_root / p

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["project_root"] = str(self.project_root)
        return data

    def index_fingerprint(self) -> dict[str, Any]:
        """Settings that change the *content* of the index.

        If any of these differ from the persisted index fingerprint, the stored vectors are
        no longer comparable with new ones and the index must be rebuilt.
        """
        return {
            "embedding_model_id": self.embedding.model_id,
            "embedding_model_revision": self.embedding.revision,
            "max_text_tokens": self.embedding.max_text_tokens,
            "chunking": asdict(self.chunking),
            "images": {
                k: v for k, v in asdict(self.images).items() if k != "output_dir"
            },
        }

    def validate(self) -> None:
        """Raise :class:`ConfigurationError` listing every invalid setting."""
        errors: list[str] = []

        def check(cond: bool, msg: str) -> None:
            if not cond:
                errors.append(msg)

        d, c, im = self.document, self.chunking, self.images
        check(d.source_url.startswith(("http://", "https://")),
              "document.source_url must be an http(s) URL")
        check(bool(d.local_path), "document.local_path must be set")
        check(d.download_timeout_s > 0, "document.download_timeout_s must be > 0")
        check(1 <= d.download_max_retries <= 10, "document.download_max_retries must be in [1, 10]")
        check(d.max_download_bytes > 0, "document.max_download_bytes must be > 0")

        check(200 <= c.chunk_size_chars <= 4000, "chunking.chunk_size_chars must be in [200, 4000]")
        check(0 <= c.chunk_overlap_chars < c.chunk_size_chars // 2,
              "chunking.chunk_overlap_chars must be >= 0 and < chunk_size_chars / 2")
        check(c.parser in {"layout", "basic"}, "chunking.parser must be 'layout' or 'basic'")
        check(0 <= c.min_chunk_chars < c.chunk_size_chars,
              "chunking.min_chunk_chars must be >= 0 and < chunk_size_chars")

        check(72 <= im.render_dpi <= 300, "images.render_dpi must be in [72, 300]")
        check(1 <= im.min_image_side_px < im.max_image_side_px,
              "images.min_image_side_px must be >= 1 and < max_image_side_px")

        e = self.embedding
        check(bool(e.model_id), "embedding.model_id must be set")
        check(1 <= e.batch_size <= 256, "embedding.batch_size must be in [1, 256]")
        check(16 <= e.max_text_tokens <= 8192, "embedding.max_text_tokens must be in [16, 8192]")
        check(e.device in _DEVICES, f"embedding.device must be one of {sorted(_DEVICES)}")

        v = self.vector_store
        name = v.collection_name
        check(3 <= len(name) <= 512 and name[0].isalnum() and name[-1].isalnum()
              and all(ch.isalnum() or ch in "._-" for ch in name),
              "vector_store.collection_name must be 3-512 chars of [a-zA-Z0-9._-], "
              "starting and ending alphanumerically (ChromaDB rule)")

        r = self.retrieval
        check(r.mode in _RETRIEVAL_MODES, f"retrieval.mode must be one of {sorted(_RETRIEVAL_MODES)}")
        check(1 <= r.top_k_text <= 50, "retrieval.top_k_text must be in [1, 50]")
        check(0 <= r.top_k_image <= 20, "retrieval.top_k_image must be in [0, 20]")
        check(r.visual_scoring in {"pixel", "fused"}, "retrieval.visual_scoring must be 'pixel' or 'fused'")
        check(-1.0 <= r.min_visual_score <= 1.0, "retrieval.min_visual_score must be in [-1, 1]")
        check(1 <= r.top_k_sections <= 10, "retrieval.top_k_sections must be in [1, 10]")
        check(0 <= r.routed_text_slots <= r.top_k_text,
              "retrieval.routed_text_slots must be in [0, top_k_text]")

        ctx = self.context
        check(500 <= ctx.max_context_chars <= 100_000, "context.max_context_chars must be in [500, 100000]")
        check(0 <= ctx.max_images <= 4, "context.max_images must be in [0, 4]")

        g = self.generation
        check(bool(g.model_id), "generation.model_id must be set")
        check(g.device in _DEVICES, f"generation.device must be one of {sorted(_DEVICES)}")
        check(g.dtype in _DTYPES, f"generation.dtype must be one of {sorted(_DTYPES)}")
        check(1 <= g.num_crops <= 16, "generation.num_crops must be in [1, 16] (Phi-3-V HD transform)")
        check(1 <= g.max_new_tokens <= 4096, "generation.max_new_tokens must be in [1, 4096]")
        check(256 <= g.max_input_tokens <= 131072, "generation.max_input_tokens must be in [256, 131072]")
        check(g.temperature >= 0, "generation.temperature must be >= 0")
        check(not g.do_sample or g.temperature > 0, "generation.temperature must be > 0 when do_sample is true")
        check(g.repetition_penalty >= 1.0, "generation.repetition_penalty must be >= 1.0")
        check(g.attn_implementation in {"eager", "sdpa", "flash_attention_2"},
              "generation.attn_implementation must be eager, sdpa or flash_attention_2")

        lg = self.logging
        check(lg.level.upper() in _LOG_LEVELS, f"logging.level must be one of {sorted(_LOG_LEVELS)}")
        check(lg.max_bytes > 0 and lg.backup_count >= 0, "logging.max_bytes > 0 and backup_count >= 0")

        if errors:
            raise ConfigurationError("Invalid configuration:\n  - " + "\n  - ".join(errors))


_SECTION_CLASSES = {
    "document": DocumentConfig, "chunking": ChunkingConfig, "images": ImageConfig,
    "embedding": EmbeddingConfig, "vector_store": VectorStoreConfig,
    "retrieval": RetrievalConfig, "context": ContextConfig,
    "generation": GenerationConfig, "monitoring": MonitoringConfig, "logging": LoggingConfig,
}


def config_from_dict(raw: dict[str, Any], project_root: Path = PROJECT_ROOT) -> Config:
    """Build a :class:`Config` from a nested dict, rejecting unknown keys."""
    kwargs: dict[str, Any] = {}
    for section, values in raw.items():
        cls = _SECTION_CLASSES.get(section)
        if cls is None:
            raise ConfigurationError(f"Unknown config section '{section}'")
        if not isinstance(values, dict):
            raise ConfigurationError(f"Config section '{section}' must be an object")
        known = {f.name for f in fields(cls)}
        unknown = set(values) - known
        if unknown:
            raise ConfigurationError(f"Unknown keys in '{section}': {sorted(unknown)}")
        kwargs[section] = cls(**values)
    cfg = Config(**kwargs, project_root=project_root)
    cfg.validate()
    return cfg


def load_config(path: str | Path | None = None,
                overrides: dict[str, dict[str, Any]] | None = None) -> Config:
    """Load, merge overrides into, and validate the JSON configuration.

    Args:
        path: JSON file; defaults to ``config/config.json``.
        overrides: ``{"section": {"key": value}}`` applied on top of the file,
            e.g. from CLI flags. Overrides are explicit; nothing is overridden silently.
    """
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    try:
        raw = json.loads(cfg_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigurationError(f"Config file not found: {cfg_path}") from exc
    except json.JSONDecodeError as exc:
        raise ConfigurationError(f"Config file {cfg_path} is not valid JSON: {exc}") from exc
    for section, values in (overrides or {}).items():
        raw.setdefault(section, {}).update(values)
    return config_from_dict(raw)


def resolve_device(requested: str) -> str:
    """Map ``auto`` to the best available device; refuse unavailable explicit devices."""
    import torch

    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise ConfigurationError(
            "device 'cuda' was requested but CUDA is not available; set device to 'auto' or 'cpu'"
        )
    return requested
