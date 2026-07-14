"""
storage.py

Section 8 of DocuMind: persists HybridIndex objects (Section 3) to Google
Drive so a document, once ingested and embedded, never needs to be
re-processed in a later Colab/Streamlit session. Also provides a small
document library — list, load, and delete previously indexed document
sets.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

from config import COLAB_DRIVE_ROOT, LOCAL_FALLBACK_ROOT
from indexing import Embedder, HybridIndex, IndexingError

logger = logging.getLogger("documind.storage")

# COLAB_DRIVE_ROOT / LOCAL_FALLBACK_ROOT now live in config.py (Section
# 14), env-overridable. Section 9's test_storage.py monkeypatches these
# as module attributes of `storage` — that still works unchanged here:
# Python binds an imported name into the importing module's own
# namespace, and functions below look it up by that name at call time.


class StorageError(Exception):
    """Raised when saving, loading, or listing a persisted index fails."""


@dataclass
class LibraryEntry:
    """One saved document set, as summarized from its meta.json."""

    name: str
    directory: Path
    chunk_count: int
    embedding_model: str
    created_at: str


def get_storage_root() -> Path:
    """Return the Drive-backed root if Google Drive is mounted (Colab),
    otherwise a local fallback directory.

    Mirrors Section 1's Drive-mount check — applied here to decide where
    persistence actually writes, instead of just warning that it isn't
    mounted.
    """
    if Path("/content/drive").exists():
        root = COLAB_DRIVE_ROOT
    else:
        root = LOCAL_FALLBACK_ROOT
        logger.warning(
            "Google Drive not detected — persisting to local path %s instead. "
            "This will NOT survive a Colab runtime restart.",
            root,
        )
    root.mkdir(parents=True, exist_ok=True)
    return root


def _slugify(name: str) -> str:
    """Turn an arbitrary document-set name into a safe directory name."""
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", name.strip().lower()).strip("-")
    return slug or f"index-{int(time.time())}"


def save_index(index: HybridIndex, name: str) -> Path:
    """Save a built HybridIndex under `name` in the storage root.

    Raises:
        StorageError: if a saved index with this name already exists —
            call `delete_index` first if overwriting is intentional, so
            an accidental name collision can't silently destroy a
            previously indexed document set.
    """
    root = get_storage_root()
    directory = root / _slugify(name)

    if directory.exists():
        raise StorageError(
            f"An index named '{name}' already exists at {directory}. "
            f"Delete it first with delete_index() if you want to overwrite it."
        )

    index.save(directory)
    logger.info("Saved index '%s' to %s", name, directory)
    return directory


def load_index(name: str, embedder: Embedder) -> HybridIndex:
    """Load a previously saved index by name."""
    directory = get_storage_root() / _slugify(name)
    try:
        return HybridIndex.load(directory, embedder)
    except IndexingError as exc:
        raise StorageError(f"Could not load index '{name}': {exc}") from exc


def list_indexes() -> list[LibraryEntry]:
    """List every saved index in the storage root, newest first."""
    root = get_storage_root()
    entries: list[LibraryEntry] = []

    for directory in root.iterdir():
        if not directory.is_dir():
            continue
        meta_path = directory / "meta.json"
        if not meta_path.exists():
            continue

        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Skipping unreadable index at %s: %s", directory, exc)
            continue

        entries.append(
            LibraryEntry(
                name=directory.name,
                directory=directory,
                chunk_count=meta.get("chunk_count", 0),
                embedding_model=meta.get("embedding_model", "unknown"),
                created_at=meta.get("created_at", "unknown"),
            )
        )

    entries.sort(key=lambda e: e.created_at, reverse=True)
    return entries


def delete_index(name: str) -> None:
    """Permanently delete a saved index by name.

    No confirmation step lives in this function by design — it's a
    library-level primitive. Any UI that calls this must add its own
    explicit confirmation before invoking it; `shutil.rmtree` here is
    irreversible.
    """
    directory = get_storage_root() / _slugify(name)
    if not directory.exists():
        raise StorageError(f"No saved index named '{name}' at {directory}.")
    shutil.rmtree(directory)
    logger.info("Deleted index '%s' at %s", name, directory)
