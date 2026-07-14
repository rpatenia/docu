"""
pdf_ingestion.py

Section 2 of DocuMind: loads PDFs, extracts per-page text, and splits it
into token-bounded chunks sized for the BAAI/bge-base-en-v1.5 embedding
model (Section 3), while preserving page-level metadata so later answers
can cite the exact page a chunk came from.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from pypdf import PdfReader
from pypdf.errors import PdfReadError
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from config import CHUNK_OVERLAP_TOKENS, CHUNK_SIZE_TOKENS, EMBEDDING_MODEL_NAME, MAX_PDF_PAGES

logger = logging.getLogger("documind.ingestion")

# bge-base-en-v1.5 has a 512-token max sequence length; CHUNK_SIZE_TOKENS
# (config.py, Section 14) leaves headroom for [CLS]/[SEP] special tokens
# and any prompt prefix added at embed time.


class PDFIngestionError(Exception):
    """Raised when a PDF cannot be read or contains no extractable text."""


@dataclass
class PDFPage:
    """One page of extracted text from a source PDF."""

    source: str
    page_number: int  # 1-indexed, matches what a human sees in a PDF viewer
    text: str


@dataclass
class Chunk:
    """A token-bounded slice of a page's text, ready for embedding."""

    chunk_id: str
    source: str
    page_number: int
    text: str
    token_count: int


def load_pdf(file_path: Path) -> list[PDFPage]:
    """Extract text from every page of a PDF.

    Raises:
        PDFIngestionError: if the file is missing, encrypted with a real
            password, corrupt, exceeds MAX_PDF_PAGES (Section 15), or
            contains no extractable text at all (e.g. a scanned image
            PDF with no OCR layer).
    """
    if not file_path.exists():
        raise PDFIngestionError(f"PDF not found: {file_path}")

    try:
        reader = PdfReader(str(file_path))
    except PdfReadError as exc:
        raise PDFIngestionError(f"Could not read PDF {file_path.name}: {exc}") from exc

    if reader.is_encrypted:
        result = reader.decrypt("")  # try an empty password before giving up
        if result == 0:  # pypdf returns 0 (PasswordType.NOT_DECRYPTED) on failure
            raise PDFIngestionError(
                f"PDF {file_path.name} is password-protected and could not "
                f"be opened with an empty password."
            )

    page_count = len(reader.pages)
    if page_count > MAX_PDF_PAGES:
        raise PDFIngestionError(
            f"{file_path.name} has {page_count} pages, which exceeds the "
            f"{MAX_PDF_PAGES}-page limit. Split it into smaller files, or "
            f"raise DOCUMIND_MAX_PDF_PAGES if this is expected for your use case."
        )

    pages: list[PDFPage] = []
    for i, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append(PDFPage(source=file_path.name, page_number=i, text=text))
        else:
            logger.warning(
                "Page %d of %s has no extractable text (likely a scanned "
                "image with no OCR layer) — skipping.",
                i,
                file_path.name,
            )

    if not pages:
        raise PDFIngestionError(
            f"No extractable text found anywhere in {file_path.name}. "
            f"If this is a scanned document, it needs OCR before DocuMind "
            f"can index it."
        )

    logger.info("Loaded %d pages with text from %s", len(pages), file_path.name)
    return pages


class TokenChunker:
    """Splits page text into overlapping, token-bounded chunks sized for
    the embedding model, using that model's own tokenizer so chunk
    boundaries are measured in the same units the model will see.
    """

    def __init__(
        self,
        tokenizer_name: str = EMBEDDING_MODEL_NAME,
        chunk_size: int = CHUNK_SIZE_TOKENS,
        chunk_overlap: int = CHUNK_OVERLAP_TOKENS,
    ) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")

        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            tokenizer_name
        )
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_page(self, page: PDFPage) -> Iterator[Chunk]:
        """Yield token-bounded chunks for a single page, in order."""
        token_ids = self.tokenizer.encode(page.text, add_special_tokens=False)

        if not token_ids:
            return

        start = 0
        chunk_index = 0
        step = self.chunk_size - self.chunk_overlap

        while start < len(token_ids):
            end = min(start + self.chunk_size, len(token_ids))
            window = token_ids[start:end]
            text = self.tokenizer.decode(window, skip_special_tokens=True)

            yield Chunk(
                chunk_id=f"{page.source}::p{page.page_number}::c{chunk_index}",
                source=page.source,
                page_number=page.page_number,
                text=text,
                token_count=len(window),
            )

            chunk_index += 1
            if end == len(token_ids):
                break
            start += step

    def chunk_pages(self, pages: list[PDFPage]) -> list[Chunk]:
        """Chunk every page and return a flat, ordered list of chunks."""
        chunks: list[Chunk] = []
        for page in pages:
            chunks.extend(self.chunk_page(page))
        logger.info(
            "Produced %d chunks from %d pages (chunk_size=%d, overlap=%d tokens)",
            len(chunks),
            len(pages),
            self.chunk_size,
            self.chunk_overlap,
        )
        return chunks


def ingest_pdf(
    file_path: Path,
    chunk_size: int = CHUNK_SIZE_TOKENS,
    chunk_overlap: int = CHUNK_OVERLAP_TOKENS,
) -> list[Chunk]:
    """End-to-end: load a PDF and return its chunks, ready for Section 3
    (embedding + indexing).
    """
    pages = load_pdf(file_path)
    chunker = TokenChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return chunker.chunk_pages(pages)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s"
    )

    if len(sys.argv) != 2:
        print("Usage: python pdf_ingestion.py <path_to_pdf>")
        sys.exit(1)

    result_chunks = ingest_pdf(Path(sys.argv[1]))
    for c in result_chunks[:3]:
        print(f"[{c.chunk_id}] ({c.token_count} tokens): {c.text[:120]}...")
