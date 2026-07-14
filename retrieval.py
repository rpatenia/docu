"""
retrieval.py

Section 4 of DocuMind: hybrid retrieval over the indexes built in Section 3.
Runs dense (FAISS) and sparse (BM25) search, fuses the two ranked lists
with Reciprocal Rank Fusion, then reranks the fused candidates with a
cross-encoder for the final relevance ordering before generation
(Section 5+).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from sentence_transformers import CrossEncoder

from config import DENSE_TOP_K, FINAL_TOP_K, FUSED_TOP_K, RERANKER_MODEL_NAME, RRF_K, SPARSE_TOP_K
from indexing import Embedder, HybridIndex, tokenize_for_bm25
from pdf_ingestion import Chunk

logger = logging.getLogger("documind.retrieval")

# RERANKER_MODEL_NAME, RRF_K, and the four *_TOP_K tuning constants now
# live in config.py (Section 14) — env-overridable without touching code.


class RetrievalError(Exception):
    """Raised when retrieval is attempted against an unbuilt/empty index
    or a reranker model fails to load.
    """


@dataclass
class ScoredChunk:
    """A chunk with the cross-encoder relevance score that placed it in
    the final result set.
    """

    chunk: Chunk
    score: float


def _dense_search(
    index: HybridIndex, query_embedding: np.ndarray, top_k: int
) -> list[tuple[int, float]]:
    """Return (chunk_index, similarity_score) pairs from FAISS, best first."""
    scores, indices = index.faiss_index.search(query_embedding, top_k)
    return [
        (int(idx), float(score))
        for idx, score in zip(indices[0], scores[0])
        if idx != -1  # FAISS pads with -1 when top_k exceeds the index size
    ]


def _sparse_search(
    index: HybridIndex, tokenized_query: list[str], top_k: int
) -> list[tuple[int, float]]:
    """Return (chunk_index, bm25_score) pairs from BM25, best first."""
    scores = index.bm25.get_scores(tokenized_query)
    ranked = sorted(enumerate(scores), key=lambda pair: pair[1], reverse=True)
    return ranked[:top_k]


def _reciprocal_rank_fusion(
    ranked_lists: list[list[tuple[int, float]]], k: int = RRF_K
) -> list[tuple[int, float]]:
    """Fuse multiple ranked (chunk_index, score) lists into one ranking.

    Uses rank position, not raw score, so dense cosine similarity (~[-1,1])
    and unbounded BM25 scores never need to be normalized against each
    other — the two scales are simply never compared directly.
    """
    fused_scores: dict[int, float] = {}
    for ranked_list in ranked_lists:
        for rank, (chunk_index, _score) in enumerate(ranked_list, start=1):
            fused_scores[chunk_index] = fused_scores.get(chunk_index, 0.0) + 1.0 / (k + rank)

    return sorted(fused_scores.items(), key=lambda pair: pair[1], reverse=True)


class HybridRetriever:
    """Runs dense + sparse retrieval, fuses with RRF, then reranks the
    fused candidates with a cross-encoder for the final ordering.
    """

    def __init__(
        self,
        index: HybridIndex,
        embedder: Embedder,
        reranker_model_name: str = RERANKER_MODEL_NAME,
    ) -> None:
        if index.faiss_index is None or index.bm25 is None:
            raise RetrievalError(
                "Cannot retrieve from an index that hasn't been built or loaded."
            )

        self.index = index
        self.embedder = embedder
        try:
            self.reranker = CrossEncoder(reranker_model_name)
        except Exception as exc:
            raise RetrievalError(
                f"Failed to load reranker {reranker_model_name}: {exc}"
            ) from exc

    def retrieve(
        self,
        query: str,
        dense_top_k: int = DENSE_TOP_K,
        sparse_top_k: int = SPARSE_TOP_K,
        fused_top_k: int = FUSED_TOP_K,
        final_top_k: int = FINAL_TOP_K,
    ) -> list[ScoredChunk]:
        """Run the full hybrid retrieval + rerank pipeline for one query."""
        query_embedding = self.embedder.embed_query(query)
        tokenized_query = tokenize_for_bm25(query)

        dense_hits = _dense_search(self.index, query_embedding, dense_top_k)
        sparse_hits = _sparse_search(self.index, tokenized_query, sparse_top_k)
        logger.info(
            "Dense hits: %d, sparse hits: %d for query %r",
            len(dense_hits),
            len(sparse_hits),
            query,
        )

        fused = _reciprocal_rank_fusion([dense_hits, sparse_hits])[:fused_top_k]
        candidate_indices = [idx for idx, _ in fused]

        if not candidate_indices:
            logger.warning("No candidates found for query %r", query)
            return []

        pairs = [(query, self.index.chunks[idx].text) for idx in candidate_indices]
        rerank_scores = self.reranker.predict(pairs)

        reranked = sorted(
            zip(candidate_indices, rerank_scores), key=lambda pair: pair[1], reverse=True
        )[:final_top_k]

        results = [
            ScoredChunk(chunk=self.index.chunks[idx], score=float(score))
            for idx, score in reranked
        ]
        logger.info("Returning %d reranked chunks for query %r", len(results), query)
        return results


if __name__ == "__main__":
    import sys
    from pathlib import Path

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s"
    )

    if len(sys.argv) != 3:
        print("Usage: python retrieval.py <index_directory> <query>")
        sys.exit(1)

    query_embedder = Embedder()
    loaded_index = HybridIndex.load(Path(sys.argv[1]), query_embedder)
    retriever = HybridRetriever(loaded_index, query_embedder)

    for result in retriever.retrieve(sys.argv[2]):
        print(f"[{result.score:.3f}] {result.chunk.chunk_id}: {result.chunk.text[:120]}...")
