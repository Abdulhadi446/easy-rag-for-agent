"""The Textual UI: a file explorer on the left, a chat on the right.

Every widget is internal to the app - no fzf, no external picker, no shelling out.
"""

from __future__ import annotations

from pathlib import Path

from rich.markdown import Markdown
from rich.text import Text
from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import DirectoryTree, Footer, Header, Input, RichLog, Static

from ..config import Settings
from ..ingest import IGNORED_DIRS, SUPPORTED_SUFFIXES, collect_files
from ..llm import LLMError, Message
from ..rag import IngestSummary, RagEngine
from ..store import Hit, StoreError


def _chunks(count: int) -> str:
    return "1 chunk" if count == 1 else f"{count} chunks"


class FileTree(DirectoryTree):
    """A DirectoryTree limited to readable document types, with space-to-mark."""

    BINDINGS = [Binding("space", "toggle_mark", "Mark", show=False)]

    def __init__(self, root: Path, **kwargs: object) -> None:
        super().__init__(str(root), **kwargs)  # type: ignore[arg-type]
        self.marked: set[Path] = set()

    def filter_paths(self, paths):  # noqa: ANN001, ANN201 - Textual signature
        allowed = []
        for path in paths:
            if path.name.startswith(".") or path.name in IGNORED_DIRS:
                continue
            if path.is_dir() or path.suffix.lower() in SUPPORTED_SUFFIXES:
                allowed.append(path)
        return sorted(allowed, key=lambda item: (not item.is_dir(), item.name.lower()))

    def render_label(self, node, base_style, style):  # noqa: ANN001, ANN201
        label = super().render_label(node, base_style, style)
        try:
            path = node.data.path  # type: ignore[union-attr]
        except AttributeError:
            return label
        if path in self.marked:
            mark = Text("● ", style="bold yellow")
            mark.append_text(label)
            return mark
        return label

    def action_toggle_mark(self) -> None:
        node = self.cursor_node
        if node is None or node.data is None:
            return
        path = node.data.path  # type: ignore[union-attr]
        if path.is_dir():
            return
        if path in self.marked:
            self.marked.discard(path)
        else:
            self.marked.add(path)
        self.refresh()


class EasyRagApp(App[None]):
    """easy-rag terminal workbench."""

    TITLE = "easy-rag"
    SUB_TITLE = "index · retrieve · chat"

    CSS = """
    Screen {
        background: $surface;
    }
    #body {
        height: 1fr;
    }
    #left {
        width: 38%;
        min-width: 32;
        border-right: solid $border;
        background: $surface;
    }
    #right {
        width: 1fr;
    }
    .panel-title {
        padding: 0 1;
        height: 1;
        background: $boost;
        color: $text;
        text-style: bold;
    }
    #tree {
        height: 1fr;
        padding: 0 1;
        scrollbar-size-vertical: 1;
    }
    #indexed {
        height: auto;
        max-height: 40%;
        min-height: 4;
        border-top: solid $border;
        padding: 0 1;
        overflow-y: auto;
    }
    #chat {
        height: 1fr;
        padding: 1 2 0 2;
        scrollbar-size-vertical: 1;
        overflow-x: hidden;
    }
    #stream-wrap {
        height: auto;
        max-height: 18;
        display: none;
        padding: 0 2;
        background: $panel;
        border-top: solid $border;
    }
    #stream {
        padding: 1 0;
    }
    #prompt {
        dock: bottom;
        margin: 1 2;
        border: tall $accent;
    }
    #status {
        height: 1;
        background: $boost;
        color: $text-muted;
        padding: 0 1;
    }
    """

    BINDINGS = [
        Binding("ctrl+q", "quit", "Quit", priority=True),
        Binding("ctrl+i", "ingest", "Ingest"),
        Binding("ctrl+a", "ingest_all", "Ingest all"),
        Binding("ctrl+r", "refresh_all", "Refresh"),
        Binding("ctrl+l", "clear_chat", "Clear chat"),
        Binding("ctrl+u", "parent_dir", "Parent dir"),
        Binding("escape", "focus_tree", "Files", show=False),
    ]

    def __init__(self, settings: Settings, root: Path | None = None) -> None:
        super().__init__()
        self.settings = settings
        self.root = (root or Path.cwd()).resolve()
        self.engine = RagEngine(settings)
        self.history: list[Message] = []
        self._busy = False
        self._indexed_paths: set[str] = set()
        self._last_question: str = ""

    # ------------------------------------------------------------------ layout
    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="body"):
            with Vertical(id="left"):
                yield Static(" FILES  ·  space marks · enter ingests", classes="panel-title")
                yield FileTree(self.root, id="tree")
                yield Static(" INDEX", classes="panel-title")
                yield Static("", id="indexed")
            with Vertical(id="right"):
                yield RichLog(id="chat", markup=True, wrap=True, highlight=True, min_width=1)
                with VerticalScroll(id="stream-wrap"):
                    yield Static("", id="stream")
                yield Input(placeholder="Ask your documents…  (enter to send)", id="prompt")
        yield Static("", id="status")
        yield Footer()

    def on_mount(self) -> None:
        # Children are mounted after the App's own Mount event, so wait one refresh.
        self.call_after_refresh(self._startup)

    def _startup(self) -> None:
        self.query_one("#tree", FileTree).focus()
        self.chat = self.query_one("#chat", RichLog)
        self.prompt = self.query_one("#prompt", Input)
        self.stream_wrap = self.query_one("#stream-wrap", VerticalScroll)
        self.stream = self.query_one("#stream", Static)
        self.status = self.query_one("#status", Static)
        self._welcome()
        self.refresh_sources()
        self._set_status()

    # ------------------------------------------------------------------ helpers
    def _welcome(self) -> None:
        self.chat.write(
            Text.from_markup(
                "[bold]easy-rag[/]  ·  [dim]local Chroma index, chat with your files[/]\n"
                "[dim]1. pick files on the left (space marks, enter ingests)\n"
                "2. ctrl+a indexes everything under the current folder\n"
                "3. type a question below[/]"
            )
        )
        self.chat.write("")

    def _set_status(self, message: str | None = None) -> None:
        chunks = 0
        try:
            chunks = self.engine.store.count()
        except StoreError:
            pass
        files = len(self._indexed_paths)
        left = (
            f" {self.settings.model}  ·  {_chunks(chunks)}  ·  "
            f"{files} file{'s' if files != 1 else ''}"
        )
        text = Text(left, style="dim")
        if message:
            text.append("  ·  ", style="dim")
            text.append(message, style="bold")
        self.status.update(text)

    def refresh_sources(self) -> None:
        try:
            sources = self.engine.sources()
        except StoreError as exc:
            self.query_one("#indexed", Static).update(Text(str(exc), style="red"))
            return
        self._indexed_paths = {info.path for info in sources}
        panel = self.query_one("#indexed", Static)
        if not sources:
            panel.update(Text("nothing indexed yet", style="dim italic"))
            self._set_status()
            return
        lines = Text()
        for info in sources:
            name = Path(info.path).name
            exists = Path(info.path).exists()
            style = "white" if exists else "red"
            lines.append(f"{name}", style=style)
            lines.append(f"  {_chunks(info.chunks)}", style="dim")
            if not exists:
                lines.append("  missing", style="red")
            lines.append("\n")
        panel.update(lines)
        self._set_status()

    def _show_stream(self, visible: bool) -> None:
        self.stream_wrap.styles.display = "block" if visible else "none"

    def _describe_hits(self, hits: list[Hit]) -> Text:
        if not hits:
            return Text("no context retrieved", style="dim italic")
        parts = Text("sources: ", style="dim")
        for number, hit in enumerate(hits[:6], start=1):
            if number > 1:
                parts.append("  ", style="dim")
            parts.append(f"[{number}] {hit.label}", style="dim cyan")
        return parts

    # ------------------------------------------------------------------ events
    @on(DirectoryTree.FileSelected)
    def file_selected(self, event: DirectoryTree.FileSelected) -> None:
        path = Path(event.path)
        if path.suffix.lower() in SUPPORTED_SUFFIXES:
            self.ingest([path])

    @on(Input.Submitted)
    def submitted(self, event: Input.Submitted) -> None:
        question = event.value.strip()
        if not question or self._busy:
            return
        self.prompt.value = ""
        self.ask(question)

    # ------------------------------------------------------------------ actions
    def action_ingest(self) -> None:
        tree = self.query_one("#tree", FileTree)
        if tree.marked:
            paths = sorted(tree.marked)
            tree.marked.clear()
            tree.refresh()
        else:
            node = tree.cursor_node
            if node is None or node.data is None:
                self.notify("Nothing selected", severity="warning")
                return
            candidate = node.data.path  # type: ignore[union-attr]
            paths = [candidate] if candidate.is_file() else collect_files([candidate])
        if not paths:
            self.notify("No supported files selected", severity="warning")
            return
        self.ingest(paths)

    def action_ingest_all(self) -> None:
        paths = collect_files([self.root])
        if not paths:
            self.notify("No .md, .txt or .pdf files here", severity="warning")
            return
        self.ingest(paths)

    def action_refresh_all(self) -> None:
        self.query_one("#tree", FileTree).reload()
        self.refresh_sources()
        self.notify("Refreshed", timeout=1.5)

    def action_clear_chat(self) -> None:
        self.history.clear()
        self.chat.clear()
        self._welcome()

    def action_parent_dir(self) -> None:
        parent = self.root.parent
        if parent == self.root:
            return
        self.root = parent
        tree = self.query_one("#tree", FileTree)
        tree.path = parent
        tree.reload()
        self.notify(str(parent), timeout=1.5)

    def action_focus_tree(self) -> None:
        self.query_one("#tree", FileTree).focus()

    # ------------------------------------------------------------------ workers
    def ingest(self, paths: list[Path]) -> None:
        if self._busy:
            self.notify("Busy - wait for the current job", severity="warning")
            return
        self._busy = True
        self._set_status(f"indexing {len(paths)} file(s)…")
        self._ingest_worker(paths)

    @work(thread=True, exclusive=True, group="ingest")
    def _ingest_worker(self, paths: list[Path]) -> None:
        try:
            summary = self.engine.ingest(paths)
        except Exception as exc:  # noqa: BLE001
            self.call_from_thread(self._ingest_failed, str(exc))
            return
        self.call_from_thread(self._ingest_done, summary)

    def _ingest_done(self, summary: IngestSummary) -> None:
        self._busy = False
        ok, failed = summary.ok, summary.failed
        for item in ok:
            self.chat.write(
                Text(f"＋ indexed {Path(item.source).name}  ({_chunks(item.chunks)})", style="green")
            )
        for item in failed:
            self.chat.write(Text(f"✗ {Path(item.source).name}: {item.error}", style="red"))
        self.refresh_sources()

    def _ingest_failed(self, message: str) -> None:
        self._busy = False
        self.chat.write(Text(f"✗ indexing failed: {message}", style="red"))
        self._set_status("indexing failed")

    def ask(self, question: str) -> None:
        self._busy = True
        self._last_question = question
        self.chat.write(Text(f"\n› {question}", style="bold cyan"))
        self._show_stream(True)
        self.stream.update(Text("…", style="dim italic"))
        self._set_status("thinking…")
        self._ask_worker(question)

    @work(thread=True, exclusive=True, group="chat")
    def _ask_worker(self, question: str) -> None:
        buffer: list[str] = []
        hits: list[Hit] = []
        try:
            for event in self.engine.ask_stream(question, self.history):
                if event.kind == "sources":
                    hits = event.hits
                    self.call_from_thread(self._answer_opened, hits)
                elif event.kind == "token":
                    buffer.append(event.text)
                    self.call_from_thread(self._answer_token, "".join(buffer))
                elif event.kind == "done":
                    self.call_from_thread(self._answer_done, event.text, hits)
        except LLMError as exc:
            self.call_from_thread(self._answer_failed, str(exc))
        except Exception as exc:  # noqa: BLE001
            self.call_from_thread(self._answer_failed, f"{type(exc).__name__}: {exc}")

    def _answer_opened(self, hits: list[Hit]) -> None:
        self.stream.update(Text("", style="dim"))
        self._show_stream(True)

    def _answer_token(self, text: str) -> None:
        self.stream.update(Text(text))
        self.stream_wrap.scroll_end(animate=False)

    def _answer_done(self, answer: str, hits: list[Hit]) -> None:
        self._busy = False
        self._show_stream(False)
        self.stream.update("")
        if answer.strip():
            self.chat.write(Markdown(answer.strip()))
            self.chat.write(self._describe_hits(hits))
            self.history.append(Message("user", self._last_question))
            self.history.append(Message("assistant", answer.strip()))
            # keep the last few turns only
            self.history[:] = self.history[-8:]
        else:
            self.chat.write(Text("(empty answer - the model returned nothing)", style="yellow"))
        self._set_status()

    def _answer_failed(self, message: str) -> None:
        self._busy = False
        self._show_stream(False)
        self.stream.update("")
        self.chat.write(Text(f"✗ {message}", style="bold red"))
        self._set_status("error")
