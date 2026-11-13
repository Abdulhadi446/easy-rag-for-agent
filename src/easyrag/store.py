"""Chroma-backed vector store.

The collection keeps one record per chunk. Metadata is kept flat (Chroma only stores
scalars) with ``source``, ``chunk`` and optionally ``page``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from .ingest import Chunk, chunk_documents, load_file


class StoreError(RuntimeError):
    """Anything that goes wrong while talking to Chroma."""


@dataclass
class Hit:
    """One retrieved chunk."""

    text: str
    source: str
    chunk: int
    score: float
    page: int | None = None

    @property
    def label(self) -> str:
        name = Path(self.source).name
        if self.page:
            return f"{name} p.{self.page}"
        if self.chunk:
            return f"{name} #{self.chunk + 1}"
        return name


@dataclass
class SourceInfo:
    path: str
    chunks: int
    size: int
    indexed_at: float


@dataclass
class IngestResult:
    source: str
    chunks: int
    skipped: bool = False
    error: str | None = None


class VectorStore:
    """Thin, synchronous wrapper around a persistent Chroma collection."""

    def __init__(
        self,
        path: Path,
        collection: str = "documents",
        embedding_function: Any | None = None,
        *,
        _client: Any | None = None,
    ) -> None:
        try:
            import chromadb
        except ImportError as exc:  # pragma: no cover
            raise StoreError("chromadb is not installed") from exc

        self.path = Path(path)
        self.collection_name = collection
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.client = _client or chromadb.PersistentClient(path=str(self.path))
            self.collection = self.client.get_or_create_collection(
                name=collection,
                embedding_function=embedding_function,
                metadata={"hnsw:space": "cosine", "app": "easy-rag"},
            )
        except Exception as exc:  # noqa: BLE001 - surfaced in the UI
            raise StoreError(f"could not open Chroma at {self.path}: {exc}") from exc

    # ------------------------------------------------------------------ writing
    def ingest_file(self, path: Path | str, size: int = 1200, overlap: int = 200) -> IngestResult:
        """Load, chunk and store one file, replacing any previous version of it."""
        path = Path(path).expanduser().resolve()
        try:
            documents = load_file(path)
        except Exception as exc:  # noqa: BLE001
            return IngestResult(source=str(path), chunks=0, error=str(exc))

        chunks = chunk_documents(documents, size=size, overlap=overlap)
        if not chunks:
            return IngestResult(source=str(path), chunks=0, error="nothing to index")

        self.delete_source(path)
        batch = 256
        try:
            for start in range(0, len(chunks), batch):
                window: Sequence[Chunk] = chunks[start : start + batch]
                self.collection.add(
                    ids=[chunk.id for chunk in window],
                    documents=[chunk.text for chunk in window],
                    metadatas=[
                        {**chunk.metadata, "indexed_at": time.time()} for chunk in window
                    ],
                )
        except Exception as exc:  # noqa: BLE001
            return IngestResult(source=str(path), chunks=0, error=f"Chroma: {exc}")
        return IngestResult(source=str(path), chunks=len(chunks))

    def ingest_text(self, text: str, name: str = "note", size: int = 1200, overlap: int = 200) -> IngestResult:
        """Store arbitrary text (a pasted note) under a synthetic source name."""
        from .ingest import Document

        source = name if name.startswith("<") else f"<{name}>"
        documents = [Document(source=source, text=text)]
        chunks = chunk_documents(documents, size=size, overlap=overlap)
        if not chunks:
            return IngestResult(source=source, chunks=0, error="nothing to index")
        self.delete_source(source)
        try:
            self.collection.add(
                ids=[chunk.id for chunk in chunks],
                documents=[chunk.text for chunk in chunks],
                metadatas=[{**chunk.metadata, "indexed_at": time.time()} for chunk in chunks],
            )
        except Exception as exc:  # noqa: BLE001
            return IngestResult(source=source, chunks=0, error=f"Chroma: {exc}")
        return IngestResult(source=source, chunks=len(chunks))

    def delete_source(self, source: Path | str) -> int:
        """Remove every chunk belonging to *source*."""
        key = str(source)
        try:
            existing = self.collection.get(where={"source": key}, include=[])
            ids = existing.get("ids") or []
            if ids:
                self.collection.delete(ids=ids)
            return len(ids)
        except Exception as exc:  # noqa: BLE001
            raise StoreError(f"could not delete {key}: {exc}") from exc

    # ------------------------------------------------------------------ reading
    def query(self, text: str, k: int = 5) -> list[Hit]:
        """Nearest chunks for *text* (cosine distance, lower is better)."""
        if not text.strip():
            return []
        try:
            result = self.collection.query(
                query_texts=[text],
                n_results=max(1, k),
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:  # noqa: BLE001
            raise StoreError(f"query failed: {exc}") from exc

        hits: list[Hit] = []
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        for document, metadata, distance in zip(documents, metadatas, distances):
            if document is None:
                continue
            metadata = metadata or {}
            hits.append(
                Hit(
                    text=document,
                    source=str(metadata.get("source", "?")),
                    chunk=int(metadata.get("chunk", 0) or 0),
                    page=int(metadata["page"]) if metadata.get("page") is not None else None,
                    score=float(distance),
                )
            )
        return hits

    def sources(self) -> list[SourceInfo]:
        """Every indexed source with its chunk count."""
        try:
            result = self.collection.get(include=["metadatas"])
        except Exception as exc:  # noqa: BLE001
            raise StoreError(f"could not list sources: {exc}") from exc

        grouped: dict[str, dict[str, Any]] = {}
        for metadata in result.get("metadatas") or []:
            if not metadata:
                continue
            source = str(metadata.get("source", "?"))
            entry = grouped.setdefault(
                source, {"chunks": 0, "indexed_at": 0.0, "size": 0}
            )
            entry["chunks"] += 1
            entry["indexed_at"] = max(entry["indexed_at"], float(metadata.get("indexed_at", 0) or 0))
        infos = []
        for source, entry in grouped.items():
            size = 0
            path = Path(source)
            if path.is_file():
                size = path.stat().st_size
            infos.append(
                SourceInfo(path=source, chunks=entry["chunks"], size=size, indexed_at=entry["indexed_at"])
            )
        return sorted(infos, key=lambda info: info.path.lower())

    def count(self) -> int:
        try:
            return int(self.collection.count())
        except Exception as exc:  # noqa: BLE001
            raise StoreError(f"could not count chunks: {exc}") from exc

    def reset(self) -> None:
        """Drop every chunk in the collection."""
        try:
            self.client.delete_collection(self.collection_name)
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine", "app": "easy-rag"},
            )
        except Exception as exc:  # noqa: BLE001
            raise StoreError(f"could not reset: {exc}") from exc


def ingest_many(
    store: VectorStore,
    paths: Iterable[Path],
    size: int = 1200,
    overlap: int = 200,
) -> list[IngestResult]:
    return [store.ingest_file(path, size=size, overlap=overlap) for path in paths]
