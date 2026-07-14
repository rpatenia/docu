"""
verify_environment.py

Verifies that the DocuMind development environment (Google Colab) has all
required libraries installed with compatible versions, and that GPU/Drive
resources are available before any retrieval, reranking, or generation
code is run.

Run this as the last step of Section 1, immediately after the pip install
cell, and before writing any DocuMind pipeline code.
"""

from __future__ import annotations

import importlib
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# --------------------------------------------------------------------------- #
# Logging configuration
# --------------------------------------------------------------------------- #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("documind.setup")


@dataclass
class LibraryCheck:
    """Result of attempting to import and version-check a single library."""

    import_name: str
    display_name: str
    installed_version: Optional[str] = None
    ok: bool = False
    error: Optional[str] = None


# Libraries DocuMind depends on, grouped by pipeline stage.
# `import_name` is what you `import`, `display_name` is what pip installs.
REQUIRED_LIBRARIES: dict[str, list[tuple[str, str]]] = {
    "PDF ingestion": [
        ("pypdf", "pypdf"),
    ],
    "Embeddings & reranking": [
        ("sentence_transformers", "sentence-transformers"),
    ],
    "Sparse + dense retrieval": [
        ("rank_bm25", "rank-bm25"),
        ("faiss", "faiss-cpu"),
    ],
    "LLM inference": [
        ("torch", "torch"),
        ("transformers", "transformers"),
        ("accelerate", "accelerate"),
    ],
    "Orchestration": [
        ("langchain", "langchain"),
        ("langchain_community", "langchain-community"),
    ],
    "Evaluation": [
        ("ragas", "ragas"),
        ("rouge_score", "rouge-score"),
        ("bert_score", "bert-score"),
        ("datasets", "datasets"),
    ],
    "Deployment": [
        ("streamlit", "streamlit"),
    ],
    "Testing": [
        ("pytest", "pytest"),
    ],
    "Utilities": [
        ("numpy", "numpy"),
    ],
}


def check_library(import_name: str, display_name: str) -> LibraryCheck:
    """Attempt to import a library and read its __version__ if available."""
    result = LibraryCheck(import_name=import_name, display_name=display_name)
    try:
        module = importlib.import_module(import_name)
        result.installed_version = getattr(module, "__version__", "unknown")
        result.ok = True
    except ImportError as exc:
        result.error = str(exc)
        result.ok = False
    return result


def check_gpu() -> None:
    """Log GPU availability via torch.

    Does not raise: a GPU isn't required to finish Section 1, but every
    section from Section 3 onward (embedding + LLM inference) will be
    unusably slow without one.
    """
    try:
        import torch

        if torch.cuda.is_available():
            logger.info("GPU available: %s", torch.cuda.get_device_name(0))
        else:
            logger.warning(
                "No GPU detected. LLM inference (Qwen2.5-1.5B, Llama-3.2-3B, "
                "Mistral-7B) will be very slow on CPU. In Colab: "
                "Runtime > Change runtime type > GPU."
            )
    except ImportError:
        logger.error("torch is not installed — cannot check GPU.")


def check_google_drive() -> None:
    """Detect whether we're in Colab and, if so, whether Drive is mounted."""
    try:
        import google.colab  # noqa: F401

        if Path("/content/drive").exists():
            logger.info("Google Drive is mounted at /content/drive.")
        else:
            logger.warning(
                "Running in Colab but Google Drive is not mounted. Call "
                "`from google.colab import drive; drive.mount('/content/drive')` "
                "before saving models/indexes, or they will be lost when the "
                "runtime disconnects."
            )
    except ImportError:
        logger.info("Not running in Google Colab — skipping Drive check.")


def run_all_checks() -> bool:
    """Run every library check, log a report, and return overall pass/fail."""
    logger.info("Python version: %s", sys.version.split()[0])

    all_ok = True
    for stage, libraries in REQUIRED_LIBRARIES.items():
        logger.info("--- %s ---", stage)
        for import_name, display_name in libraries:
            result = check_library(import_name, display_name)
            if result.ok:
                logger.info(
                    "  [OK] %-24s version=%s", display_name, result.installed_version
                )
            else:
                all_ok = False
                logger.error(
                    "  [MISSING] %-20s -> pip install %s", display_name, display_name
                )

    check_gpu()
    check_google_drive()

    if all_ok:
        logger.info("All required libraries are installed. Environment is ready.")
    else:
        logger.error(
            "One or more libraries are missing. Install them before continuing "
            "to Section 2."
        )
    return all_ok


if __name__ == "__main__":
    success = run_all_checks()
    sys.exit(0 if success else 1)
