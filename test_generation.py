"""
test_generation.py

Tests for generation.py's pure logic — prompt building and citation
extraction/validation — without loading any real LLM.
"""

from __future__ import annotations

from generation import _build_prompt, _extract_citations
from retrieval import ScoredChunk


def test_build_prompt_numbers_sources_from_one(sample_chunks):
    scored = [ScoredChunk(chunk=c, score=1.0 - i * 0.1) for i, c in enumerate(sample_chunks)]

    prompt, sources = _build_prompt("What was the retention rate?", scored)

    assert set(sources.keys()) == {1, 2, 3}
    assert "[1]" in prompt
    assert sample_chunks[0].text in prompt
    assert "What was the retention rate?" in prompt


def test_extract_citations_splits_valid_and_invalid():
    answer = "Retention was 92% [1]. A related figure was 88% [2]. See also [9]."
    valid, invalid = _extract_citations(answer, valid_numbers={1, 2, 3})

    assert valid == [1, 2]
    assert invalid == [9]


def test_extract_citations_no_markers_returns_empty_lists():
    valid, invalid = _extract_citations(
        "I don't have enough information.", valid_numbers={1, 2}
    )
    assert valid == []
    assert invalid == []


def test_extract_citations_deduplicates_repeated_markers():
    valid, _ = _extract_citations(
        "[1] confirms this. Again, [1] is the source.", valid_numbers={1}
    )
    assert valid == [1]
