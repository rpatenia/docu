"""
test_indexing.py

Tests for indexing.py — HybridIndex build/save/load and the BM25
tokenizer, using FakeEmbedder (conftest.py) so no real model is loaded.
"""

from __future__ import annotations

import pytest

from conftest import FakeEmbedder
from indexing import HybridIndex, IndexingError, tokenize_for_bm25


def test_tokenize_for_bm25_lowercases_and_strips_punctuation():
    assert tokenize_for_bm25("Hello, World! Patient-92%.") == [
        "hello",
        "world",
        "patient",
        "92",
    ]


def test_tokenize_for_bm25_empty_string_returns_empty_list():
    assert tokenize_for_bm25("") == []


def test_build_raises_on_empty_chunk_list(fake_embedder):
    index = HybridIndex(fake_embedder)
    with pytest.raises(IndexingError):
        index.build([])


def test_build_produces_one_vector_per_chunk(fake_embedder, sample_chunks):
    index = HybridIndex(fake_embedder)
    index.build(sample_chunks)
    assert index.faiss_index.ntotal == len(sample_chunks)
    assert len(index.chunks) == len(sample_chunks)


def test_save_load_round_trip_preserves_chunks(tmp_path, fake_embedder, sample_chunks):
    index = HybridIndex(fake_embedder)
    index.build(sample_chunks)
    index.save(tmp_path / "my_index")

    reloaded = HybridIndex.load(tmp_path / "my_index", fake_embedder)

    assert [c.chunk_id for c in reloaded.chunks] == [c.chunk_id for c in sample_chunks]
    assert reloaded.faiss_index.ntotal == len(sample_chunks)


def test_load_rejects_mismatched_embedding_model(tmp_path, fake_embedder, sample_chunks):
    index = HybridIndex(fake_embedder)
    index.build(sample_chunks)
    index.save(tmp_path / "my_index")

    different_embedder = FakeEmbedder()
    different_embedder.model_name = "a-different-model"

    with pytest.raises(IndexingError, match="was built with embedding model"):
        HybridIndex.load(tmp_path / "my_index", different_embedder)


def test_save_raises_if_index_not_built(tmp_path, fake_embedder):
    index = HybridIndex(fake_embedder)
    with pytest.raises(IndexingError):
        index.save(tmp_path / "my_index")
