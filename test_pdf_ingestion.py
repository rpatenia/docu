"""
test_pdf_ingestion.py

Tests for pdf_ingestion.py. load_pdf's error paths use a mocked
PdfReader (a third-party object this project doesn't own). TokenChunker
uses a local FakeTokenizer so its sliding-window logic is tested without
downloading the real BAAI/bge-base-en-v1.5 tokenizer.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

import pdf_ingestion
from pdf_ingestion import PDFIngestionError, PDFPage, TokenChunker, load_pdf


class FakeTokenizer:
    """Whitespace-based fake tokenizer — "tokens" are just words, which
    makes chunk boundaries trivial to predict and assert on in tests.
    """

    def encode(self, text: str, add_special_tokens: bool = False) -> list[str]:
        return text.split()

    def decode(self, token_ids: list[str], skip_special_tokens: bool = True) -> str:
        return " ".join(token_ids)


def test_load_pdf_raises_if_file_missing(tmp_path):
    with pytest.raises(PDFIngestionError, match="not found"):
        load_pdf(tmp_path / "does_not_exist.pdf")


def test_load_pdf_raises_if_no_extractable_text(tmp_path):
    fake_path = tmp_path / "scanned.pdf"
    fake_path.write_bytes(b"%PDF-1.4 fake bytes")

    fake_page = MagicMock()
    fake_page.extract_text.return_value = ""
    fake_reader = MagicMock(is_encrypted=False, pages=[fake_page])

    with patch("pdf_ingestion.PdfReader", return_value=fake_reader):
        with pytest.raises(PDFIngestionError, match="No extractable text"):
            load_pdf(fake_path)


def test_load_pdf_returns_pages_with_text(tmp_path):
    fake_path = tmp_path / "report.pdf"
    fake_path.write_bytes(b"%PDF-1.4 fake bytes")

    fake_page = MagicMock()
    fake_page.extract_text.return_value = "Retention exceeded 92%."
    fake_reader = MagicMock(is_encrypted=False, pages=[fake_page])

    with patch("pdf_ingestion.PdfReader", return_value=fake_reader):
        pages = load_pdf(fake_path)

    assert len(pages) == 1
    assert pages[0].page_number == 1
    assert pages[0].source == "report.pdf"
    assert "92%" in pages[0].text


def test_load_pdf_raises_if_encrypted_and_empty_password_fails(tmp_path):
    """Regression test for the pypdf.decrypt() return-value fix from
    Section 2 — this behavior was corrected after a fact-check found an
    unverified import (`pypdf.constants.PasswordType`); this test locks
    in the `== 0` check that replaced it.
    """
    fake_path = tmp_path / "locked.pdf"
    fake_path.write_bytes(b"%PDF-1.4 fake bytes")

    fake_reader = MagicMock(is_encrypted=True)
    fake_reader.decrypt.return_value = 0  # pypdf's PasswordType.NOT_DECRYPTED

    with patch("pdf_ingestion.PdfReader", return_value=fake_reader):
        with pytest.raises(PDFIngestionError, match="password-protected"):
            load_pdf(fake_path)


def test_chunk_page_respects_chunk_size_and_overlap(monkeypatch):
    monkeypatch.setattr(
        pdf_ingestion.AutoTokenizer, "from_pretrained", lambda name: FakeTokenizer()
    )
    chunker = TokenChunker(chunk_size=4, chunk_overlap=1)
    page = PDFPage(source="doc.pdf", page_number=1, text="one two three four five six seven")

    chunks = list(chunker.chunk_page(page))

    assert [c.text for c in chunks] == [
        "one two three four",
        "four five six seven",
    ]
    assert chunks[0].chunk_id == "doc.pdf::p1::c0"
    assert chunks[1].chunk_id == "doc.pdf::p1::c1"


def test_chunker_rejects_overlap_greater_or_equal_to_chunk_size(monkeypatch):
    monkeypatch.setattr(
        pdf_ingestion.AutoTokenizer, "from_pretrained", lambda name: FakeTokenizer()
    )
    with pytest.raises(ValueError):
        TokenChunker(chunk_size=4, chunk_overlap=4)
