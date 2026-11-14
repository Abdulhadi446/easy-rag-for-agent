"""Command line entry point.

``easy-rag`` with no arguments opens the TUI. The other subcommands are for scripting:
add files, ask one question, inspect or wipe the index.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import Settings, config_dir, data_dir
from .ingest import collect_files
from .llm import LLMError
from .rag import RagEngine
from .reranker import RerankError
from .store import StoreError, VectorStore


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="easy-rag",
        description="Index text, Markdown and PDF files in Chroma and chat with them.",
    )
    parser.add_argument("--version", action="store_true", help="print the version and exit")
    parser.add_argument("--path", type=Path, default=None, help="folder the TUI opens on")
    parser.add_argument("--env-file", type=Path, default=None, help="explicit .env to read")

    sub = parser.add_subparsers(dest="command")

    tui = sub.add_parser("tui", help="open the terminal interface (default)")
    tui.add_argument("path", nargs="?", type=Path, default=None)
    _add_retrieval_flags(tui)

    add = sub.add_parser("add", help="index files or folders")
    add.add_argument("paths", nargs="+", type=Path)
    add.add_argument("--no-recursive", action="store_true")

    ask = sub.add_parser("ask", help="ask one question and print the answer")
    ask.add_argument("question", nargs="+")
    ask.add_argument("-k", type=int, default=None, help="how many chunks to retrieve")
    _add_retrieval_flags(ask)

    sub.add_parser("ls", help="list indexed sources")
    sub.add_parser("doctor", help="check configuration, index and embeddings")

    rm = sub.add_parser("rm", help="remove one indexed source")
    rm.add_argument("source", type=str)

    sub.add_parser("reset", help="delete every indexed chunk")

    config = sub.add_parser("config", help="show or change settings")
    config.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    config.add_argument("--show", action="store_true")
    return parser


def _add_retrieval_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--hybrid",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="combine dense search with BM25 keyword search (RRF)",
    )
    parser.add_argument(
        "--rerank",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="rerank the candidate pool with a local cross-encoder",
    )


def _settings(args: argparse.Namespace) -> Settings:
    settings = Settings.load(env_file=args.env_file)
    overrides = {
        name: getattr(args, name)
        for name in ("hybrid", "rerank")
        if getattr(args, name, None) is not None
    }
    if overrides:
        settings.update(**overrides)
    return settings


def _chunks(count: int) -> str:
    return "1 chunk" if count == 1 else f"{count} chunks"


def _files(count: int) -> str:
    return "1 file" if count == 1 else f"{count} files"


def cmd_add(args: argparse.Namespace, settings: Settings) -> int:
    engine = RagEngine(settings)
    paths = collect_files(args.paths, recursive=not args.no_recursive)
    if not paths:
        print("no .md, .txt or .pdf files found", file=sys.stderr)
        return 1
    summary = engine.ingest(paths)
    for item in summary.results:
        if item.error:
            print(f"  ✗ {item.source}: {item.error}", file=sys.stderr)
        else:
            print(f"  + {item.source}  ({_chunks(item.chunks)})")
    print(f"{_files(len(summary.ok))}, {_chunks(summary.chunks)} indexed")
    return 0 if not summary.failed else 1


def cmd_ask(args: argparse.Namespace, settings: Settings) -> int:
    engine = RagEngine(settings)
    engine.on_status = lambda message: print(message, file=sys.stderr, flush=True)
    question = " ".join(args.question)
    try:
        answer, hits = engine.ask(question, k=args.k)
    except (LLMError, RerankError, StoreError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(answer.strip() or "(empty answer)")
    if hits:
        print("\nsources:", file=sys.stderr)
        for number, hit in enumerate(hits, start=1):
            precision = 4 if hit.metric == "rerank" else 3
            print(
                f"  [{number}] {hit.label}  ({hit.metric} {hit.score:.{precision}f})",
                file=sys.stderr,
            )
    return 0


def cmd_ls(settings: Settings) -> int:
    store = VectorStore(settings.db_path, settings.collection)
    sources = store.sources()
    if not sources:
        print("nothing indexed yet")
        return 0
    for info in sources:
        mark = "" if Path(info.path).exists() else "  (missing)"
        print(f"{_chunks(info.chunks):>9}  {info.path}{mark}")
    print(f"\n{len(sources)} source{'s' if len(sources) != 1 else ''}, {_chunks(store.count())}  ·  {settings.db_path}")
    return 0


def cmd_reset(settings: Settings) -> int:
    store = VectorStore(settings.db_path, settings.collection)
    store.reset()
    print("index cleared")
    return 0


def cmd_rm(args: argparse.Namespace, settings: Settings) -> int:
    store = VectorStore(settings.db_path, settings.collection)
    targets = [
        info.path
        for info in store.sources()
        if info.path == args.source or Path(info.path).name == args.source
    ]
    if not targets:
        print(f"no indexed source matches {args.source!r}", file=sys.stderr)
        return 1
    for target in targets:
        removed = store.delete_source(target)
        print(f"removed {removed} chunk(s) from {target}")
    return 0


def cmd_config(args: argparse.Namespace, settings: Settings) -> int:
    for item in args.set:
        if "=" not in item:
            print(f"ignoring {item!r}, expected KEY=VALUE", file=sys.stderr)
            continue
        key, _, value = item.partition("=")
        settings.update(**{key.strip(): value.strip()})
    if args.set:
        path = settings.save()
        print(f"saved {path}")
    if args.show or not args.set:
        print(json.dumps(settings.public_dict(), indent=2))
    return 0


def cmd_doctor(settings: Settings) -> int:
    ok = True
    print(f"config dir   {config_dir()}")
    print(f"data dir     {data_dir()}")
    print(f"database     {settings.db_path}")
    print(f"env file     {settings.env_file or '(none found)'}")
    print(f"model        {settings.model}")
    print(f"base url     {settings.base_url}")
    print(
        "retrieval    "
        f"hybrid {'on' if settings.hybrid else 'off'}  ·  "
        f"rerank {'on' if settings.rerank else 'off'} ({settings.rerank_model})  ·  "
        f"top {settings.top_k} of {settings.candidates}"
    )
    print(f"api key      {'set (' + settings.api_key[:6] + '…)' if settings.api_key else 'MISSING'}")
    if not settings.api_key:
        ok = False
        print("             → add OPENROUTER_API_KEY=... to .env")
    try:
        store = VectorStore(settings.db_path, settings.collection)
        print(f"chroma       ok, {store.count()} chunks, {len(store.sources())} sources")
    except StoreError as exc:
        ok = False
        print(f"chroma       FAILED: {exc}")
        return 1

    print("embeddings   testing (first run downloads a 80MB ONNX model)…")
    try:
        store.collection.query(query_texts=["ping"], n_results=1)
        print("embeddings   ok")
    except Exception as exc:  # noqa: BLE001
        ok = False
        print(f"embeddings   FAILED: {exc}")
    return 0 if ok else 1


def cmd_tui(args: argparse.Namespace, settings: Settings) -> int:
    from .tui.app import EasyRagApp

    root = args.path or Path.cwd()
    EasyRagApp(settings, root=root).run()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.version:
        from . import __version__

        print(__version__)
        return 0

    try:
        settings = _settings(args)
    except Exception as exc:  # noqa: BLE001
        print(f"could not load settings: {exc}", file=sys.stderr)
        return 1

    if args.command in (None, "tui"):
        if args.command == "tui":
            args.path = args.path or Path.cwd()
        return cmd_tui(args, settings)
    if args.command == "add":
        return cmd_add(args, settings)
    if args.command == "ask":
        return cmd_ask(args, settings)
    if args.command == "ls":
        return cmd_ls(settings)
    if args.command == "rm":
        return cmd_rm(args, settings)
    if args.command == "reset":
        return cmd_reset(settings)
    if args.command == "config":
        return cmd_config(args, settings)
    if args.command == "doctor":
        return cmd_doctor(settings)
    parser.print_help()
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
