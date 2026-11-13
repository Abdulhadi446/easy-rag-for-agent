from __future__ import annotations

from pathlib import Path

import pytest

from easyrag.ingest import (
    Chunk,
    Document,
    UnsupportedFile,
    chunk_documents,
    collect_files,
    load_file,
    normalise,
    split_text,
)


def test_normalise_collapses_whitespace_and_blank_lines():
    assert normalise("a  b\r\n\r\n\r\n\r\nc") == "a b\n\nc"


def test_load_markdown_and_text(tmp_path: Path):
    md = tmp_path / "note.md"
    md.write_text("# Title\n\nBody text.", encoding="utf-8")
    documents = load_file(md)
    assert len(documents) == 1
    assert documents[0].source == str(md.resolve())
    assert "Body text." in documents[0].text

    txt = tmp_path / "plain.txt"
    txt.write_text("hello", encoding="utf-8")
    assert load_file(txt)[0].text == "hello"


def test_unsupported_suffix_is_rejected(tmp_path: Path):
    binary = tmp_path / "image.png"
    binary.write_bytes(b"\x89PNG")
    with pytest.raises(UnsupportedFile) as excinfo:
        load_file(binary)
    assert "unsupported type" in str(excinfo.value)


def test_split_text_respects_size_and_overlaps():
    paragraphs = [f"Paragraph {index} " + ("word " * 40) for index in range(12)]
    text = "\n\n".join(paragraphs)
    chunks = split_text(text, size=400, overlap=80)
    assert len(chunks) > 3
    assert all(len(chunk) <= 420 for chunk in chunks)
    # every paragraph survives somewhere
    for index in range(12):
        assert any(f"Paragraph {index}" in chunk for chunk in chunks)


def test_split_short_text_is_single_chunk():
    assert split_text("just one line", size=100) == ["just one line"]


def test_split_empty():
    assert split_text("   \n\n  ") == []


def test_chunk_ids_are_stable_and_unique():
    documents = [Document(source="/tmp/a.md", text="alpha " * 500)]
    first = chunk_documents(documents, size=300, overlap=50)
    second = chunk_documents(documents, size=300, overlap=50)
    assert [chunk.id for chunk in first] == [chunk.id for chunk in second]
    assert len({chunk.id for chunk in first}) == len(first)
    assert all(isinstance(chunk, Chunk) for chunk in first)
    assert first[0].metadata["source"] == "/tmp/a.md"


def test_chunk_documents_keeps_pdf_pages_apart():
    documents = [
        Document(source="/tmp/doc.pdf", text="page one", page=1),
        Document(source="/tmp/doc.pdf", text="page two", page=2),
    ]
    chunks = chunk_documents(documents)
    assert [chunk.page for chunk in chunks] == [1, 2]
    assert all(chunk.metadata["page"] in (1, 2) for chunk in chunks)


def test_collect_files_walks_directories_and_skips_junk(tmp_path: Path):
    (tmp_path / "a.md").write_text("a", encoding="utf-8")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "b.txt").write_text("b", encoding="utf-8")
    (nested / "skip.bin").write_bytes(b"\x00")
    venv = tmp_path / ".venv"
    venv.mkdir()
    (venv / "c.md").write_text("c", encoding="utf-8")

    found = collect_files([tmp_path])
    names = sorted(path.name for path in found)
    assert names == ["a.md", "b.txt"]


def test_collect_files_accepts_single_files(tmp_path: Path):
    target = tmp_path / "one.md"
    target.write_text("x", encoding="utf-8")
    assert collect_files([target]) == [target]
