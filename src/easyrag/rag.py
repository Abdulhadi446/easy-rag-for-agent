"""Retrieval-augmented generation: store documents, retrieve, then ask the model."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Sequence

from .config import Settings
from .ingest import SUPPORTED_SUFFIXES, collect_files
from .llm import LLMError, Message, stream_chat
from .store import Hit, IngestResult, SourceInfo, StoreError, VectorStore

SYSTEM_PROMPT = (
    "You are easy-rag, a careful research assistant answering questions about the user's "
    "own documents.\n"
    "Rules:\n"
    "- Use the numbered context below as your primary source.\n"
    "- Cite the snippets you use with their bracket number, like [1] or [2].\n"
    "- If the context does not contain the answer, say so plainly and offer what the "
    "documents do cover. Never invent facts or citations.\n"
    "- Be concise. Prefer short paragraphs or bullet points over long essays.\n"
    "- When the user asks for code or a summary, stay faithful to the source wording."
)

MAX_CONTEXT_CHARS = 12000


@dataclass
class Event:
    """One step of an answer: sources, a token, or the finished text."""

    kind: str
    text: str = ""
    hits: list[Hit] = field(default_factory=list)

    @classmethod
    def sources(cls, hits: list[Hit]) -> "Event":
        return cls(kind="sources", hits=hits)

    @classmethod
    def token(cls, text: str) -> "Event":
        return cls(kind="token", text=text)

    @classmethod
    def done(cls, text: str) -> "Event":
        return cls(kind="done", text=text)


@dataclass
class IngestSummary:
    results: list[IngestResult] = field(default_factory=list)

    @property
    def ok(self) -> list[IngestResult]:
        return [item for item in self.results if not item.error]

    @property
    def failed(self) -> list[IngestResult]:
        return [item for item in self.results if item.error]

    @property
    def chunks(self) -> int:
        return sum(item.chunks for item in self.results)


class RagEngine:
    """Owns the store and turns questions into grounded answers."""

    def __init__(self, settings: Settings, store: VectorStore | None = None) -> None:
        self.settings = settings
        self.store = store or VectorStore(settings.db_path, settings.collection)

    # ------------------------------------------------------------------ ingest
    def ingest(self, targets: Sequence[Path | str], recursive: bool = True) -> IngestSummary:
        paths = collect_files([Path(target) for target in targets], recursive=recursive)
        summary = IngestSummary()
        for path in paths:
            if path.suffix.lower() not in SUPPORTED_SUFFIXES:
                summary.results.append(
                    IngestResult(source=str(path), chunks=0, error="unsupported file type")
                )
                continue
            started = time.perf_counter()
            result = self.store.ingest_file(
                path, size=self.settings.chunk_size, overlap=self.settings.chunk_overlap
            )
            if not result.error:
                result.source = str(Path(result.source))
                _ = time.perf_counter() - started
            summary.results.append(result)
        return summary

    def ingest_text(self, text: str, name: str = "note") -> IngestResult:
        return self.store.ingest_text(
            text, name=name, size=self.settings.chunk_size, overlap=self.settings.chunk_overlap
        )

    # ------------------------------------------------------------------ search
    def retrieve(self, question: str, k: int | None = None) -> list[Hit]:
        return self.store.query(question, k=k or self.settings.top_k)

    def sources(self) -> list[SourceInfo]:
        return self.store.sources()

    # ------------------------------------------------------------------ answers
    def build_messages(
        self, question: str, hits: Sequence[Hit], history: Sequence[Message] = ()
    ) -> list[Message]:
        blocks: list[str] = []
        budget = MAX_CONTEXT_CHARS
        for number, hit in enumerate(hits, start=1):
            name = Path(hit.source).name
            where = f", page {hit.page}" if hit.page else f", chunk {hit.chunk}"
            block = f"[{number}] {name}{where}\n{hit.text.strip()}"
            if len(block) > budget:
                block = block[:budget]
            if budget <= 0:
                break
            blocks.append(block)
            budget -= len(block) + 2

        if blocks:
            context = "\n\n".join(blocks)
            prompt = (
                f"Context from the indexed documents:\n\n{context}\n\n"
                f"Question: {question}\n\n"
                "Answer using the context above and cite with [n]."
            )
        else:
            prompt = (
                f"Question: {question}\n\n"
                "No indexed context matched this question. Say that you could not find it in "
                "the indexed documents, then answer from general knowledge only if you clearly "
                "label it as such."
            )

        messages: list[Message] = [Message("system", SYSTEM_PROMPT)]
        messages.extend(history)
        messages.append(Message("user", prompt))
        return messages

    def ask_stream(
        self, question: str, history: Sequence[Message] = (), k: int | None = None
    ) -> Iterator[Event]:
        """Yield sources first, then tokens, then the complete answer."""
        try:
            hits = self.retrieve(question, k=k)
        except StoreError as exc:
            raise LLMError(str(exc)) from exc

        yield Event.sources(hits)
        messages = self.build_messages(question, hits, history)
        pieces: list[str] = []
        for token in stream_chat(messages, self.settings):
            pieces.append(token)
            yield Event.token(token)
        yield Event.done("".join(pieces))

    def ask(self, question: str, history: Sequence[Message] = (), k: int | None = None) -> tuple[str, list[Hit]]:
        """Blocking helper for the CLI."""
        answer = ""
        hits: list[Hit] = []
        for event in self.ask_stream(question, history, k=k):
            if event.kind == "sources":
                hits = event.hits
            elif event.kind == "done":
                answer = event.text
        return answer, hits
