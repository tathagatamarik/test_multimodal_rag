"""Domain-specific exception hierarchy.

Every module raises one of these (chaining the original exception with
``raise ... from exc``) so callers can distinguish *which stage* failed without
string-matching messages.
"""


class RAGError(Exception):
    """Base class for all errors raised by this project."""


class ConfigurationError(RAGError):
    """Invalid or inconsistent configuration. Never retried."""


class DocumentLoadError(RAGError):
    """The PDF could not be downloaded, found, opened or validated."""


class DocumentExtractionError(RAGError):
    """Text or image extraction from an opened PDF failed."""


class EmbeddingError(RAGError):
    """The embedding model failed to load or produced invalid vectors."""


class VectorStoreError(RAGError):
    """ChromaDB persistence, insertion or query failed."""


class IndexCompatibilityError(VectorStoreError):
    """The persisted index was built with a different document/model/config."""


class RetrievalError(RAGError):
    """A query could not be validated or answered from the index."""


class GenerationError(RAGError):
    """The generation model failed to load, run, or returned invalid output."""


class PipelineError(RAGError):
    """An end-to-end workflow failed in a way not covered by a stage error."""
