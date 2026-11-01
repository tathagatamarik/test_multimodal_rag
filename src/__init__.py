"""Multimodal RAG over "Attention Is All You Need" (Jina CLIP v1 + ChromaDB + Phi-3 Vision)."""

import warnings

# PyMuPDF >= 1.24 warns that the ``fitz`` import name is deprecated in favour of
# ``pymupdf``. The assignment's allow-list names ``fitz`` explicitly, so we keep
# that import and silence only this specific notice.
warnings.filterwarnings("ignore", message=r"The `fitz` API is deprecated", category=FutureWarning)
