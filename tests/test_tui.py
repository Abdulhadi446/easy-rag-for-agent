from __future__ import annotations

from pathlib import Path

import pytest

from easyrag.config import Settings
from easyrag.rag import Event
from easyrag.store import Hit
from easyrag.tui.app import EasyRagApp, FileTree

pytestmark = pytest.mark.asyncio


class FakeStore:
    def __init__(self) -> None:
        self.chunks = 0
        self.paths: list[str] = []

    def count(self) -> int:
        return self.chunks

    def sources(self):  # noqa: ANN201
        from easyrag.store import SourceInfo

        return [SourceInfo(path=path, chunks=1, size=10, indexed_at=0) for path in self.paths]

    def query(self, text: str, k: int):  # noqa: ANN001
        return [Hit(text="context", source="/docs/a.md", chunk=0, score=0.24)]


def make_app(tmp_path: Path, engine=None) -> EasyRagApp:
    settings = Settings(api_key="test", db_path=tmp_path / "db", model="test/model")
    app = EasyRagApp(settings, root=tmp_path)
    if engine is not None:
        app.engine = engine
    return app


async def test_app_mounts_with_tree_and_chat(tmp_path: Path):
    (tmp_path / "note.md").write_text("hello", encoding="utf-8")
    (tmp_path / "skip.bin").write_bytes(b"\x00")
    app = make_app(tmp_path)
    app.engine.store = FakeStore()

    async with app.run_test() as pilot:
        await pilot.pause()
        tree = app.query_one("#tree", FileTree)
        assert tree is not None
        assert app.query_one("#prompt")
        assert app.query_one("#chat")
        status = str(app.query_one("#status").render())
        assert "0 chunks" in status


async def test_file_tree_filters_unsupported_types(tmp_path: Path):
    (tmp_path / "good.md").write_text("x", encoding="utf-8")
    (tmp_path / "bad.png").write_bytes(b"\x00")
    app = make_app(tmp_path)
    app.engine.store = FakeStore()

    async with app.run_test() as pilot:
        await pilot.pause()
        names = {path.name for path in app.query_one("#tree", FileTree).filter_paths(tmp_path.iterdir())}
        assert "good.md" in names
        assert "bad.png" not in names


async def test_question_streams_into_the_chat(tmp_path: Path):
    class Engine:
        store = FakeStore()
        settings = Settings()

        def sources(self):  # noqa: ANN201
            return []

        def ingest(self, paths):  # noqa: ANN001
            raise AssertionError("not used")

        def ask_stream(self, question, history):  # noqa: ANN001
            hits = [Hit(text="ctx", source="/docs/a.md", chunk=0, score=0.2)]
            yield Event.sources(hits)
            yield Event.token("Because ")
            yield Event.token("cosine.")
            yield Event.done("Because cosine.")

    app = make_app(tmp_path)
    app.engine = Engine()

    async with app.run_test() as pilot:
        await pilot.pause(0.3)
        prompt = app.query_one("#prompt")
        prompt.focus()
        await pilot.pause(0.05)
        prompt.value = "why cosine?"
        await pilot.press("enter")
        await pilot.pause(0.4)
        chat = app.query_one("#chat")
        text = "\n".join(str(line.text if hasattr(line, "text") else line) for line in chat.lines)
        assert "why cosine?" in text
        assert app.history[-1].content == "Because cosine."


async def test_space_marks_a_file(tmp_path: Path):
    (tmp_path / "a.md").write_text("x", encoding="utf-8")
    app = make_app(tmp_path)
    app.engine.store = FakeStore()

    async with app.run_test() as pilot:
        tree = app.query_one("#tree", FileTree)
        tree.focus()
        await pilot.pause()
        # move the cursor onto the file and mark it
        while tree.cursor_node is not None and tree.cursor_node.data and tree.cursor_node.data.path.is_dir():
            await pilot.press("down")
        if tree.cursor_node and tree.cursor_node.data and tree.cursor_node.data.path.is_file():
            await pilot.press("space")
            assert tree.marked


async def test_f2_and_f3_toggle_the_retrieval_modes(tmp_path: Path):
    app = make_app(tmp_path)
    app.engine.store = FakeStore()
    app.engine.warm_reranker = lambda: "fake/cross-encoder"  # type: ignore[method-assign]

    async with app.run_test() as pilot:
        await pilot.pause(0.3)
        assert "hybrid off" in str(app.query_one("#status").render())

        await pilot.press("f2")
        await pilot.press("f3")
        await pilot.pause(0.3)

        assert app.settings.hybrid is True
        assert app.settings.rerank is True
        status = str(app.query_one("#status").render())
        assert "hybrid on" in status
        assert "rerank on" in status
