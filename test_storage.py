"""
test_storage.py

Tests for storage.py's save/load/list/delete against a temp directory —
real filesystem I/O, no Google Drive, no real embedding model.
"""

from __future__ import annotations

import pytest

import storage
from indexing import HybridIndex
from storage import StorageError, _slugify, delete_index, list_indexes, load_index, save_index


@pytest.fixture(autouse=True)
def storage_root_in_tmp_path(tmp_path, monkeypatch):
    """Redirect storage.py's roots into pytest's tmp_path so tests never
    touch a real ./documind_indexes directory or assume Drive is mounted.
    """
    monkeypatch.setattr(storage, "LOCAL_FALLBACK_ROOT", tmp_path / "documind_indexes")
    monkeypatch.setattr(storage, "COLAB_DRIVE_ROOT", tmp_path / "unused_drive_root")


def test_slugify_normalizes_punctuation_and_case():
    assert _slugify("My Report! v1.0") == "my-report-v1-0"


def test_slugify_empty_name_falls_back_to_timestamp():
    assert _slugify("   ").startswith("index-")


def test_save_then_list_then_load_round_trip(fake_embedder, sample_chunks):
    index = HybridIndex(fake_embedder)
    index.build(sample_chunks)

    save_index(index, "Quarterly Report")

    entries = list_indexes()
    assert len(entries) == 1
    assert entries[0].name == "quarterly-report"
    assert entries[0].chunk_count == len(sample_chunks)

    reloaded = load_index("Quarterly Report", fake_embedder)
    assert len(reloaded.chunks) == len(sample_chunks)


def test_save_raises_on_name_collision(fake_embedder, sample_chunks):
    index = HybridIndex(fake_embedder)
    index.build(sample_chunks)
    save_index(index, "duplicate-name")

    with pytest.raises(StorageError, match="already exists"):
        save_index(index, "duplicate-name")


def test_delete_removes_saved_index(fake_embedder, sample_chunks):
    index = HybridIndex(fake_embedder)
    index.build(sample_chunks)
    save_index(index, "to-delete")

    delete_index("to-delete")

    assert list_indexes() == []


def test_delete_raises_if_name_not_found():
    with pytest.raises(StorageError, match="No saved index named"):
        delete_index("does-not-exist")
