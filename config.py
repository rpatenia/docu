"""
config.py

Section 14 of DocuMind: single source of truth for every tunable
constant across the pipeline. Previously scattered across
pdf_ingestion.py, indexing.py, retrieval.py, generation.py, and
storage.py — most importantly, EMBEDDING_MODEL_NAME was independently
duplicated in both pdf_ingestion.py (the chunking tokenizer) and
indexing.py (the embedding model itself). If those two ever drifted
apart, chunk token boundaries would silently stop matching the
embedding model's tokenizer, corrupting retrieval with no error raised
anywhere. That's the real bug this section closes.

Values are read from environment variables (via python-dotenv, pinned
since Section 1 but never actually wired in until now) with the same
defaults that were previously hardcoded — nothing changes at runtime
unless a variable is actually set.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # no-op if no .env file exists — safe in Colab and locally


def _env_str(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


# --- Embedding — shared by pdf_ingestion.py's chunker AND indexing.py's
# embedder. This is the single value that MUST stay identical between
# the two, which is exactly why it lives here instead of two places. ---
EMBEDDING_MODEL_NAME: str = _env_str("DOCUMIND_EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
# BGE models expect this instruction prefix on queries only, not
# passages — see indexing.py's Embedder for how it's applied.
BGE_QUERY_INSTRUCTION: str = "Represent this sentence for searching relevant passages: "

# --- Chunking (Section 2) ---
# bge-base-en-v1.5 has a 512-token max sequence length; CHUNK_SIZE_TOKENS
# leaves headroom for [CLS]/[SEP] special tokens and the query prefix.
CHUNK_SIZE_TOKENS: int = _env_int("DOCUMIND_CHUNK_SIZE_TOKENS", 400)
CHUNK_OVERLAP_TOKENS: int = _env_int("DOCUMIND_CHUNK_OVERLAP_TOKENS", 50)

# --- Retrieval (Section 4) ---
RERANKER_MODEL_NAME: str = _env_str(
    "DOCUMIND_RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
)
# RRF constant from Cormack et al. (2009) — dampens the influence of any
# single top rank so fusion isn't dominated by one retriever's #1 hit.
RRF_K: int = _env_int("DOCUMIND_RRF_K", 60)
DENSE_TOP_K: int = _env_int("DOCUMIND_DENSE_TOP_K", 20)
SPARSE_TOP_K: int = _env_int("DOCUMIND_SPARSE_TOP_K", 20)
FUSED_TOP_K: int = _env_int("DOCUMIND_FUSED_TOP_K", 10)
FINAL_TOP_K: int = _env_int("DOCUMIND_FINAL_TOP_K", 5)

# --- Generation (Section 5) ---
# A fixed, deliberate set — Section 6's evaluation compares exactly
# these three models on identical footing. NOT individually
# env-overridable; only which one loads by default is configurable.
MODEL_REGISTRY: dict[str, str] = {
    "qwen2.5-1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
    "llama-3.2-3b": "meta-llama/Llama-3.2-3B-Instruct",
    "mistral-7b": "mistralai/Mistral-7B-Instruct-v0.3",
}
DEFAULT_MODEL_KEY: str = _env_str("DOCUMIND_DEFAULT_MODEL", "qwen2.5-1.5b")
MAX_NEW_TOKENS: int = _env_int("DOCUMIND_MAX_NEW_TOKENS", 512)

# --- Storage (Section 8) ---
COLAB_DRIVE_ROOT: Path = Path(
    _env_str("DOCUMIND_DRIVE_ROOT", "/content/drive/MyDrive/DocuMind/indexes")
)
LOCAL_FALLBACK_ROOT: Path = Path(_env_str("DOCUMIND_LOCAL_ROOT", "./documind_indexes"))

# --- Resource limits (Section 15) ---
MAX_PDF_PAGES: int = _env_int("DOCUMIND_MAX_PDF_PAGES", 500)
MAX_UPLOAD_SIZE_MB: int = _env_int("DOCUMIND_MAX_UPLOAD_SIZE_MB", 50)
