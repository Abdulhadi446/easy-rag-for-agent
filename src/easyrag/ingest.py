"""Document loading and chunking.

Supported formats: plain text, Markdown and PDF. Everything else is rejected with a
clear message rather than silently treated as text.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

TEXT_SUFFIXES = {".txt", ".text", ".md", ".markdown", ".org", ".rst"}
PDF_SUFFIXES = {".pdf"}
SUPPORTED_SUFFIXES = TEXT_SUFFIXES | PDF_SUFFIXES
IGNORED_DIRS = {
    ".git",
    ".venv",
    "venv",
    "node_modules",
    "__pycache__",
    "dist",
    "build",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
}


class UnsupportedFile(ValueError):
    """Raised when a file has no loader."""


@dataclass
class Document:
    """A whole file (or a single PDF page) before chunking."""

    source: str
    text: str
    page: int | None = None


@dataclass
class Chunk:
    """A slice of a document, ready for embedding."""

    id: str
    text: str
    source: str
    index: int
    page: int | None = None
    metadata: dict[str, object] = field(default_factory=dict)


def sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", "ignore")).hexdigest()


def normalise(text: str) -> str:
    """Collapse hard wraps and runs of blank lines while keeping paragraphs."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_file(path: Path | str) -> list[Document]:
    """Read a file into one or more :class:`Document` objects."""
    path = Path(path).expanduser()
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise UnsupportedFile(
            f"{path.name}: unsupported type '{suffix or 'no extension'}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_SUFFIXES))}"
        )
    if suffix in PDF_SUFFIXES:
        return _load_pdf(path)
    try:
        raw = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        raw = path.read_text(encoding="latin-1")
    text = normalise(raw)
    if not text:
        raise UnsupportedFile(f"{path.name}: file is empty")
    return [Document(source=str(path.resolve()), text=text)]


def _load_pdf(path: Path) -> list[Document]:
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - dependency is declared
        raise UnsupportedFile("PDF support needs the 'pypdf' package") from exc

    reader = PdfReader(str(path))
    documents: list[Document] = []
    for number, page in enumerate(reader.pages, start=1):
        try:
            text = normalise(page.extract_text() or "")
        except Exception:  # noqa: BLE001 - a broken page should not kill the import
            text = ""
        if text:
            documents.append(Document(source=str(path.resolve()), text=text, page=number))
    if not documents:
        raise UnsupportedFile(f"{path.name}: no extractable text (scanned PDF?)")
    return documents


def split_text(text: str, size: int = 1200, overlap: int = 200) -> list[str]:
    """Split text into overlapping chunks, preferring paragraph boundaries."""
    text = normalise(text)
    if not text:
        return []
    if len(text) <= size:
        return [text]

    overlap = max(0, min(overlap, size // 2))
    paragraphs = re.split(r"\n\s*\n", text)
    chunks: list[str] = []
    buffer = ""

    def flush() -> None:
        nonlocal buffer
        if buffer.strip():
            chunks.append(buffer.strip())
        buffer = ""

    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        # A single huge paragraph still has to be cut.
        while len(paragraph) > size:
            if buffer:
                flush()
            chunks.append(paragraph[:size].strip())
            paragraph = paragraph[size - overlap :]
            if len(paragraph) <= overlap:
                break
        if len(buffer) + len(paragraph) + 2 > size:
            flush()
            tail = chunks[-1][-overlap:] if overlap and chunks else ""
            buffer = (tail + "\n\n" + paragraph).strip() if tail else paragraph
        else:
            buffer = f"{buffer}\n\n{paragraph}" if buffer else paragraph
    flush()

    # Add a little context from the previous chunk when we started cold.
    return [chunk for chunk in chunks if chunk]


def chunk_documents(
    documents: list[Document], size: int = 1200, overlap: int = 200
) -> list[Chunk]:
    """Turn documents into chunks with stable ids derived from source + position."""
    chunks: list[Chunk] = []
    for document in documents:
        parts = split_text(document.text, size=size, overlap=overlap)
        for index, part in enumerate(parts):
            identifier = sha1(f"{document.source}:{document.page}:{index}")
            metadata: dict[str, object] = {"source": document.source, "chunk": index}
            if document.page is not None:
                metadata["page"] = document.page
            chunks.append(
                Chunk(
                    id=identifier,
                    text=part,
                    source=document.source,
                    index=index,
                    page=document.page,
                    metadata=metadata,
                )
            )
    return chunks


def collect_files(targets: list[Path], recursive: bool = True) -> list[Path]:
    """Expand directories into supported files, keeping order stable."""
    found: list[Path] = []
    for target in targets:
        target = Path(target).expanduser()
        if target.is_dir():
            iterator = target.rglob("*") if recursive else target.glob("*")
            for candidate in sorted(iterator):
                if not candidate.is_file():
                    continue
                relative = candidate.relative_to(target).parts
                if any(part.startswith(".") or part in IGNORED_DIRS for part in relative[:-1]):
                    continue
                if candidate.suffix.lower() in SUPPORTED_SUFFIXES:
                    found.append(candidate)
        elif target.is_file():
            found.append(target)
    return found
