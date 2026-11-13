from __future__ import annotations

from pathlib import Path

import pytest

from easyrag.store import StoreError, VectorStore

chromadb = pytest.importorskip("chromadb")

from chromadb.api.types import Documents, EmbeddingFunction, Embeddings  # noqa: E402


class HashEmbedding(EmbeddingFunction[Documents]):
    """Deterministic three-dimensional embedding, enough for ranking tests."""

    def __call__(self, input: Documents) -> Embeddings:  # noqa: A002 - Chroma's signature
        vectors = []
        for text in input:
            lowered = text.lower()
            vectors.append(
                [
                    float(lowered.count("alpha")),
                    float(lowered.count("beta")),
                    1.0,
                ]
            )
        return vectors

    @staticmethod
    def name() -> str:
        return "hash-embedding"

    @staticmethod
    def build_from_config(config: dict) -> "HashEmbedding":  # noqa: ANN001
        return HashEmbedding()

    def __init__(self) -> None:
        pass

    def get_config(self) -> dict:
        return {}


def make_store(tmp_path: Path) -> VectorStore:
    # A unique path per test keeps each Chroma system isolated.
    return VectorStore(tmp_path / "db", "test-collection", HashEmbedding())


def test_ingest_and_query(tmp_path: Path):
    store = make_store(tmp_path)
    document = tmp_path / "alpha.md"
    document.write_text("# Alpha\n\nalpha alpha alpha beta", encoding="utf-8")

    result = store.ingest_file(document, size=2000, overlap=0)
    assert result.error is None
    assert result.chunks == 1
    assert store.count() == 1

    hits = store.query("alpha", k=1)
    assert hits[0].source == str(document.resolve())
    assert hits[0].chunk == 0
    assert isinstance(hits[0].score, float)


def test_reingesting_replaces_old_chunks(tmp_path: Path):
    store = make_store(tmp_path)
    document = tmp_path / "note.txt"
    document.write_text("alpha", encoding="utf-8")
    store.ingest_file(document, size=2000, overlap=0)

    document.write_text("beta beta", encoding="utf-8")
    store.ingest_file(document, size=2000, overlap=0)

    assert store.count() == 1
    hits = store.query("beta", k=1)
    assert "beta" in hits[0].text


def test_sources_and_delete(tmp_path: Path):
    store = make_store(tmp_path)
    document = tmp_path / "doc.md"
    document.write_text("alpha" * 500, encoding="utf-8")
    store.ingest_file(document, size=300, overlap=50)
    assert store.count() > 1

    sources = store.sources()
    assert len(sources) == 1
    assert sources[0].path == str(document.resolve())
    assert sources[0].chunks == store.count()

    assert store.delete_source(str(document.resolve())) > 0
    assert store.count() == 0


def test_reset_clears_collection(tmp_path: Path):
    store = make_store(tmp_path)
    document = tmp_path / "doc.md"
    document.write_text("alpha", encoding="utf-8")
    store.ingest_file(document)
    store.reset()
    assert store.count() == 0


def test_ingest_text_uses_synthetic_source(tmp_path: Path):
    store = make_store(tmp_path)
    result = store.ingest_text("pasted alpha note", name="scratch")
    assert result.source == "<scratch>"
    assert store.sources()[0].path == "<scratch>"


def test_ingest_missing_file_reports_error(tmp_path: Path):
    store = make_store(tmp_path)
    result = store.ingest_file(tmp_path / "nope.md")
    assert result.chunks == 0
    assert result.error
