"""
test_retrieval.py

Tests for retrieval.py's pure fusion logic — Reciprocal Rank Fusion —
without running any real dense/sparse search or loading a reranker.
"""

from __future__ import annotations

import pytest

from retrieval import RRF_K, _reciprocal_rank_fusion


def test_rrf_boosts_chunk_found_by_both_retrievers():
    dense = [(1, 0.9), (2, 0.8), (3, 0.7)]
    sparse = [(2, 12.0), (4, 9.0), (1, 5.0)]

    fused_order = [idx for idx, _ in _reciprocal_rank_fusion([dense, sparse])]

    # chunk 2: rank 2 in dense, rank 1 in sparse -> highest combined score
    assert fused_order[0] == 2


def test_rrf_matches_hand_computed_score():
    dense = [(1, 0.9)]
    sparse = [(1, 5.0)]

    fused = _reciprocal_rank_fusion([dense, sparse], k=RRF_K)
    expected_score = 1 / (RRF_K + 1) + 1 / (RRF_K + 1)

    assert fused[0] == (1, pytest.approx(expected_score))


def test_rrf_includes_chunks_found_by_only_one_retriever():
    dense = [(1, 0.9)]
    sparse = [(2, 5.0)]

    fused_indices = {idx for idx, _ in _reciprocal_rank_fusion([dense, sparse])}
    assert fused_indices == {1, 2}
