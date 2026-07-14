"""
conftest.py

Shared pytest fixtures for DocuMind's test suite. Hosts FakeEmbedder —
used by more than one test file — as a fixture. Anything used by only
one test file (like test_pdf_ingestion.py's FakeTokenizer) stays local
to that file instead of being shared here unnecessarily.
"""

from __future__ import annotations

import numpy as np
import pytest

from pdf_ingestion import Chunk


class FakeEmbedder:
    """A drop-in stand-in for indexing.Embedder that never loads a real
    model. Produces small, deterministic unit vectors from a hash of the
    input text — identical text always embeds identically, different
    text (almost always) embeds differently. Enough to exercise FAISS
    indexing/search logic without any ML dependency.
    """

    def __init__(self, dimension: int = 8) -> None:
        self.model_name = "fake-embedder"
        self.dimension = dimension

    def _embed_one(self, text: str) -> np.ndarray:
        rng = np.random.default_rng(abs(hash(text)) % (2**32))
        vector = rng.standard_normal(self.dimension).astype("float32")
        return vector / np.linalg.norm(vector)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return np.stack([self._embed_one(t) for t in texts])

    def embed_query(self, query: str) -> np.ndarray:
        return self._embed_one(f"query::{query}")[None, :]


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def sample_chunks() -> list[Chunk]:
    return [
        Chunk(
            chunk_id="doc.pdf::p1::c0",
            source="doc.pdf",
            page_number=1,
            text="Patient retention exceeded 92% across all three cohorts.",
            token_count=10,
        ),
        Chunk(
            chunk_id="doc.pdf::p2::c0",
            source="doc.pdf",
            page_number=2,
            text="Cohort B showed a slightly lower retention figure of 88%.",
            token_count=10,
        ),
        Chunk(
            chunk_id="doc.pdf::p3::c0",
            source="doc.pdf",
            page_number=3,
            text="Methodology used a randomized controlled trial design.",
            token_count=9,
        ),
    ]
