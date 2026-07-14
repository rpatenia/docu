"""
indexing.py

Section 3 of DocuMind: embeds chunks with BAAI/bge-base-en-v1.5, builds a
dense FAISS index and a sparse BM25 index over the same ordered chunk
list, and persists both — plus a model fingerprint — to disk so they
survive a Colab runtime restart and can't be silently reloaded against
the wrong embedding model.
"""

from __future__ import annotations

import json
import logging
import pickle
import re
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from config import BGE_QUERY_INSTRUCTION, EMBEDDING_MODEL_NAME
from pdf_ingestion import Chunk

logger = logging.getLogger("documind.indexing")

# EMBEDDING_MODEL_NAME and BGE_QUERY_INSTRUCTION now live in config.py
# (Section 14) — this used to be an independently duplicated string
# literal here AND in pdf_ingestion.py. If those two ever drifted apart,
# chunk token boundaries (Section 2's tokenizer) would silently stop
# matching this embedding model, corrupting retrieval with no error
# raised anywhere.


class IndexingError(Exception):
    """Raised when embedding, index construction, save, or load fails."""


def tokenize_for_bm25(text: str) -> list[str]:
    """Lowercase, alphanumeric-only tokenization for BM25.

    BM25 is a lexical/keyword method, so this stays deliberately simple —
    no stemming or stopword removal. Both can be added later as a tuning
    step without touching the dense side at all.

    Public (no leading underscore) because Section 4 reuses this exact
    function to tokenize queries — BM25 only matches correctly if queries
    and documents are tokenized identically.
    """
    return re.findall(r"[a-z0-9]+", text.lower())


class Embedder:
    """Wraps the BGE embedding model with the query/document instruction
    asymmetry bge-base-en-v1.5 was trained with.
    """

    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME) -> None:
        self.model_name = model_name
        try:
            self.model = SentenceTransformer(model_name)
        except Exception as exc:
            raise IndexingError(
                f"Failed to load embedding model {model_name}: {exc}"
            ) from exc

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """Embed passage/chunk text. No instruction prefix — BGE reserves
        that for queries only.
        """
        embeddings = self.model.encode(
            texts, normalize_embeddings=True, show_progress_bar=False
        )
        return np.asarray(embeddings, dtype="float32")

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a user query, with the BGE retrieval instruction prefix."""
        prefixed = f"{BGE_QUERY_INSTRUCTION}{query}"
        embedding = self.model.encode(
            [prefixed], normalize_embeddings=True, show_progress_bar=False
        )
        return np.asarray(embedding, dtype="float32")


class HybridIndex:
    """Owns a dense FAISS index and a sparse BM25 index over the same
    ordered list of chunks, plus the metadata needed to map a result
    back to its source document and page.
    """

    def __init__(self, embedder: Embedder) -> None:
        self.embedder = embedder
        self.chunks: list[Chunk] = []
        self.faiss_index: faiss.Index | None = None
        self.bm25: BM25Okapi | None = None

    def build(self, chunks: list[Chunk]) -> None:
        """Build both indexes from a flat list of chunks.

        Section 2's `ingest_pdf` returns chunks per PDF — concatenate the
        lists from every uploaded document before calling `build()` so one
        index covers the user's full document set.
        """
        if not chunks:
            raise IndexingError("Cannot build an index from zero chunks.")

        self.chunks = chunks
        texts = [c.text for c in chunks]

        logger.info("Embedding %d chunks with %s ...", len(texts), self.embedder.model_name)
        embeddings = self.embedder.embed_documents(texts)

        dimension = embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dimension)
        self.faiss_index.add(embeddings)
        logger.info(
            "FAISS index built: %d vectors, dimension %d", self.faiss_index.ntotal, dimension
        )

        tokenized_corpus = [tokenize_for_bm25(t) for t in texts]
        self.bm25 = BM25Okapi(tokenized_corpus)
        logger.info("BM25 index built over %d documents", len(tokenized_corpus))

    def save(self, directory: Path) -> None:
        """Persist both indexes, chunk metadata, and a model fingerprint."""
        if self.faiss_index is None or self.bm25 is None:
            raise IndexingError("Cannot save an index that hasn't been built yet.")

        directory.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.faiss_index, str(directory / "faiss.index"))

        with open(directory / "bm25.pkl", "wb") as f:
            pickle.dump(self.bm25, f)

        with open(directory / "chunks.json", "w", encoding="utf-8") as f:
            json.dump([asdict(c) for c in self.chunks], f, ensure_ascii=False, indent=2)

        meta = {
            "embedding_model": self.embedder.model_name,
            "dimension": self.faiss_index.d,
            "chunk_count": len(self.chunks),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        with open(directory / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        logger.info("Saved index (%d chunks) to %s", len(self.chunks), directory)

    @classmethod
    def load(cls, directory: Path, embedder: Embedder) -> "HybridIndex":
        """Reload a previously saved index — used at the start of every
        new Colab session instead of re-embedding everything from scratch.

        Raises:
            IndexingError: if any index file is missing, or if the saved
                index was built with a different embedding model than
                `embedder` — loading it anyway would silently corrupt
                every future search.
        """
        required = ["faiss.index", "bm25.pkl", "chunks.json", "meta.json"]
        for name in required:
            if not (directory / name).exists():
                raise IndexingError(f"Missing index file: {directory / name}")

        with open(directory / "meta.json", "r", encoding="utf-8") as f:
            meta = json.load(f)

        if meta["embedding_model"] != embedder.model_name:
            raise IndexingError(
                f"Index at {directory} was built with embedding model "
                f"'{meta['embedding_model']}', but the current embedder uses "
                f"'{embedder.model_name}'. Rebuild the index instead of loading it."
            )

        instance = cls(embedder)
        instance.faiss_index = faiss.read_index(str(directory / "faiss.index"))

        with open(directory / "bm25.pkl", "rb") as f:
            instance.bm25 = pickle.load(f)

        with open(directory / "chunks.json", "r", encoding="utf-8") as f:
            raw_chunks = json.load(f)
        instance.chunks = [Chunk(**c) for c in raw_chunks]

        logger.info("Loaded index (%d chunks) from %s", len(instance.chunks), directory)
        return instance


if __name__ == "__main__":
    import sys

    from pdf_ingestion import ingest_pdf

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s"
    )

    if len(sys.argv) != 2:
        print("Usage: python indexing.py <path_to_pdf>")
        sys.exit(1)

    pdf_chunks = ingest_pdf(Path(sys.argv[1]))
    doc_embedder = Embedder()
    index = HybridIndex(doc_embedder)
    index.build(pdf_chunks)
    index.save(Path("index_store"))
