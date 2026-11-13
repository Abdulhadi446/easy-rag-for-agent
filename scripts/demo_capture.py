"""Drive the real TUI and export frames for the README (screenshots + GIF).

Usage:  uv run python scripts/demo_capture.py [--size 120x40] [--out docs/demo]

The frames are real renders of the app with a temporary Chroma database, real
ingestion of ``sample-docs/`` and a scripted (fake) model stream so the capture
is offline and deterministic.
"""

from __future__ import annotations

import argparse
import asyncio
import shutil
import tempfile
import time
from pathlib import Path

from textual.widgets import Input

from easyrag.config import Settings
from easyrag.rag import Event
from easyrag.tui.app import EasyRagApp, FileTree

ROOT = Path(__file__).resolve().parents[1]
QUESTION = "How does easy-rag measure similarity, and how does it chunk documents?"

ANSWER = """easy-rag stores every chunk in Chroma with **cosine** distance, so a lower score means a closer match [1]. Documents are split into roughly 1,200-character chunks with 200 characters of overlap, and each chunk keeps its source path and page number in metadata [2].

Retrieval embeds the question with the same model, pulls the five closest chunks, and pastes them into the prompt as numbered context so the answer can cite them [2]."""


def _tokens(text: str, step: int = 7) -> list[str]:
    """Split into model-sized deltas without losing any characters."""
    return [text[index : index + step] for index in range(0, len(text), step)]


async def capture(size: tuple[int, int], out: Path) -> None:
    workdir = Path(tempfile.mkdtemp(prefix="easyrag-demo-"))
    settings = Settings(
        api_key="demo-key",
        model="openrouter/free",
        db_path=workdir / "chroma",
        collection="documents",
    )
    app = EasyRagApp(settings, root=ROOT)
    frames: list[Path] = []
    counter = 0

    def snap(label: str) -> None:
        nonlocal counter
        counter += 1
        path = out / f"f{counter:02d}-{label}.svg"
        path.write_text(app.export_screenshot(), encoding="utf-8")
        frames.append(path)
        print(f"  · {path.name}")

    try:
        async with app.run_test(size=size) as pilot:
            await pilot.pause(0.6)
            tree = app.query_one("#tree", FileTree)
            for node in tree.root.children:
                if node.data and Path(node.data.path).name == "sample-docs":
                    node.expand()
            await pilot.pause(0.3)
            snap("startup")

            docs = sorted((ROOT / "sample-docs").glob("*"))
            tree.marked = {path for path in docs if path.is_file()}
            tree.refresh()
            await pilot.pause(0.2)
            snap("marked")

            app.ingest(sorted(tree.marked))
            for _ in range(80):
                if not app._busy:
                    break
                await asyncio.sleep(0.1)
            await pilot.pause(0.3)
            snap("indexed")

            prompt = app.query_one("#prompt", Input)
            prompt.focus()
            prompt.value = QUESTION
            await pilot.pause(0.2)
            snap("question")
            prompt.value = ""

            hits = app.engine.retrieve(QUESTION, k=3)

            def fake_stream(question: str, history=None):  # noqa: ANN001, ANN202
                yield Event("sources", hits=hits)
                for token in _tokens(ANSWER):
                    time.sleep(0.09)
                    yield Event("token", text=token)
                yield Event("done", text=ANSWER)

            app.engine.ask_stream = fake_stream  # type: ignore[method-assign]
            app.ask(QUESTION)
            for _ in range(16):
                await asyncio.sleep(0.3)
                snap("typing")
            for _ in range(40):
                if not app._busy:
                    break
                await asyncio.sleep(0.1)
            await pilot.pause(0.4)
            snap("answer")
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    (out / "frames.txt").write_text("\n".join(p.name for p in frames), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", default="120x40")
    parser.add_argument("--out", default=str(ROOT / "docs" / "demo"))
    args = parser.parse_args()
    width, height = (int(part) for part in args.size.lower().split("x"))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("*.svg"):
        old.unlink()
    asyncio.run(capture((width, height), out))
    print(f"{len(list(out.glob('*.svg')))} frames in {out}")


if __name__ == "__main__":
    main()
