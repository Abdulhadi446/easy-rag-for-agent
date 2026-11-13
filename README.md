<p align="center">
  <img src="docs/demo/demo.gif" alt="easy-rag terminal interface: mark files, index them into Chroma, ask a question and watch the answer stream in with citations" width="880">
</p>

<h1 align="center">easy-rag</h1>

<p align="center">
  A local-first RAG workbench that lives in your terminal.<br>
  Browse your files, index them into Chroma, and chat with them — no browser, no daemon, no glue code.
</p>

<p align="center">
  <a href="https://github.com/Abdulhadi446/easy-rag-for-agent/stargazers"><img src="https://img.shields.io/github/stars/Abdulhadi446/easy-rag-for-agent?style=flat-square&color=yellow" alt="stars"></a>
  <img src="https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue?style=flat-square" alt="python versions">
  <a href="https://textual.textualize.io/"><img src="https://img.shields.io/badge/UI-Textual-5c3cff?style=flat-square" alt="built with Textual"></a>
  <a href="https://www.trychroma.com/"><img src="https://img.shields.io/badge/vector%20store-Chroma-ff6b35?style=flat-square" alt="Chroma"></a>
  <img src="https://img.shields.io/badge/tests-29%20passing-brightgreen?style=flat-square" alt="29 tests passing">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green?style=flat-square" alt="MIT license"></a>
  <a href="https://openrouter.ai/"><img src="https://img.shields.io/badge/LLM-OpenAI--compatible-8a2be2?style=flat-square" alt="OpenAI compatible"></a>
</p>

---

## Why this exists

Most RAG demos are a notebook, a vector database you have to host, and a web UI you have to
babysit. **easy-rag** is the opposite: one `uv run` away from a finished tool. It ships its own
file explorer, its own indexer, and its own chat, all rendered with
[Textual](https://textual.textualize.io/) inside the terminal you already have open.

- **Everything is local** except the final chat completion. Files never leave your machine
  until the relevant chunks are sent to the model you configured.
- **No GPU, no Ollama, no local LLM.** Embeddings run on CPU through Chroma's bundled ONNX
  MiniLM model; the chat model is any OpenAI-compatible endpoint.
- **Citations built in.** Every answer is grounded in numbered chunks and shows the source files
  (with distances) underneath.
- **Two ways to drive it.** A full TUI with a mouse-free file explorer, and a scriptable CLI for
  agents and cron jobs.

```
❯ easy-rag
┌─ FILES ──────────────────────┬─ CHAT ─────────────────────────────────────────┐
│ ● chroma-notes.md            │ › How does easy-rag measure similarity?        │
│ ● rag-basics.txt             │ easy-rag uses cosine distance in Chroma, so a  │
│   README.md                  │ lower score means a closer match [1].          │
│                              │ sources: [1] chroma-notes.md                   │
└──────────────────────────────┴────────────────────────────────────────────────┘
 openrouter/free · 2 chunks · 2 files
```

## Screenshots

<p align="center">
  <img src="docs/demo/screenshot.png" alt="easy-rag answering a question with a cited, streamed answer" width="880">
</p>

<details>
<summary><b>More screenshots — marking files and the index panel</b></summary>
<br>
<p align="center">
  <img src="docs/demo/startup.png" alt="easy-rag startup: file tree on the left, chat on the right" width="880">
</p>
<p align="center">
  <img src="docs/demo/indexed.png" alt="two sample documents indexed, chunk counts shown in the INDEX panel" width="880">
</p>
</details>

---

## Quick start

### 1. Clone and install

```bash
git clone https://github.com/Abdulhadi446/easy-rag-for-agent.git
cd easy-rag-for-agent
uv sync
```

<details>
<summary>Prefer pip / an isolated global command?</summary>

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
# or install it once, globally:
uv tool install .
```

The `easy-rag` console script is what you get either way.
</details>

### 2. Add a model key

```bash
cp .env.example .env
$EDITOR .env          # set OPENROUTER_API_KEY=...
```

Any OpenAI-compatible endpoint works — OpenRouter, OpenAI, Groq, LM Studio, vLLM, Ollama's
`/v1` shim. The only hard requirement is a `/chat/completions` endpoint that supports streaming.

```dotenv
OPENROUTER_API_KEY=sk-or-v1-…
MODEL=openrouter/free

# point somewhere else if you like
# EASYRAG_BASE_URL=https://api.openai.com/v1
# MODEL=gpt-4o-mini
```

### 3. Index something and ask

```bash
uv run easy-rag add sample-docs
uv run easy-rag ask "What distance metric does easy-rag use?"
```

Or just open the workbench and do it visually:

```bash
uv run easy-rag
```

> **First run downloads ~80 MB** — Chroma fetches its ONNX embedding model once and caches it.
> After that, indexing two sample files takes about five seconds on a laptop CPU.

---

## The terminal interface

The left third is a real directory tree. The right side is the conversation.

| Key | Action |
| --- | --- |
| <kbd>↑</kbd> <kbd>↓</kbd> / type | Move through folders, expand with <kbd>→</kbd> |
| <kbd>space</kbd> | Mark / unmark a file (yellow ●) |
| <kbd>enter</kbd> | Index the highlighted file, or fold/unfold a folder |
| <kbd>ctrl</kbd>+<kbd>i</kbd> | Index every marked file (or the cursor target if nothing is marked) |
| <kbd>ctrl</kbd>+<kbd>a</kbd> | Index everything supported under the current root |
| <kbd>ctrl</kbd>+<kbd>r</kbd> | Reload the tree and the index panel |
| <kbd>ctrl</kbd>+<kbd>l</kbd> | Clear the chat and conversation memory |
| <kbd>ctrl</kbd>+<kbd>u</kbd> | Go to the parent directory |
| <kbd>ctrl</kbd>+<kbd>q</kbd> | Quit |

The **INDEX** panel lists every source already in the collection and how many chunks it
contributed, so you can tell at a glance what the model can actually see. Files that have been
deleted from disk show up red and marked `missing`.

Re-ingesting a file replaces its old chunks instead of duplicating them — the UI is safe to
hammer while you edit notes.

## Command line

```bash
easy-rag                          # open the TUI in the current folder
easy-rag --path ~/notes           # open the TUI somewhere else
easy-rag add ~/notes ~/papers/x.pdf
easy-rag ask "Summarise the architecture" -k 8
easy-rag ls                       # what is indexed
easy-rag rm chroma-notes.md       # drop one source
easy-rag reset                    # wipe the collection
easy-rag doctor                   # check key, database, embeddings
easy-rag config --show
easy-rag config --set model=gpt-4o-mini
```

<details>
<summary><b>Sample output</b></summary>

```console
$ easy-rag add sample-docs
  + /home/you/easy-rag-for-agent/sample-docs/chroma-notes.md  (1 chunk)
  + /home/you/easy-rag-for-agent/sample-docs/rag-basics.txt  (1 chunk)
2 files, 2 chunks indexed

$ easy-rag ask "What distance metric does easy-rag use?"
easy-rag stores chunks in Chroma with cosine distance, so a lower score means a closer
match [1].

sources:
  [1] chroma-notes.md  (distance 0.432)

$ easy-rag ls
   1 chunk  /home/you/easy-rag-for-agent/sample-docs/chroma-notes.md
   1 chunk  /home/you/easy-rag-for-agent/sample-docs/rag-basics.txt

2 sources, 2 chunks  ·  /home/you/.local/share/easy-rag/chroma

$ easy-rag doctor
config dir   /home/you/.config/easy-rag
data dir     /home/you/.local/share/easy-rag
database     /home/you/.local/share/easy-rag/chroma
env file     /home/you/easy-rag-for-agent/.env
model        openrouter/free
base url     https://openrouter.ai/api/v1
api key      set (sk-or-…)
chroma       ok, 2 chunks, 2 sources
embeddings   testing (first run downloads a 80MB ONNX model)…
embeddings   ok
```
</details>

## Configuration

Settings resolve in this order (last one wins):

```mermaid
flowchart LR
    A["built-in defaults"] --> B["~/.config/easy-rag/config.json"]
    B --> C[".env, walking up from cwd"]
    C --> D["process environment"]
    D --> E([Settings])
```

| Variable | Default | Meaning |
| --- | --- | --- |
| `EASYRAG_API_KEY` / `OPENROUTER_API_KEY` / `OPENAI_API_KEY` | — | Bearer token for the chat endpoint |
| `EASYRAG_BASE_URL` | `https://openrouter.ai/api/v1` | OpenAI-compatible base URL |
| `EASYRAG_MODEL` / `MODEL` | `openrouter/free` | Model id sent to that endpoint |
| `EASYRAG_DB` | `~/.local/share/easy-rag/chroma` | Chroma persistence directory |
| `EASYRAG_COLLECTION` | `documents` | Collection name (3–512 chars) |
| `EASYRAG_CHUNK_SIZE` | `1200` | Characters per chunk |
| `EASYRAG_CHUNK_OVERLAP` | `200` | Characters shared by neighbours |
| `EASYRAG_TOP_K` | `5` | Chunks retrieved per question |
| `EASYRAG_MAX_TOKENS` | `1600` | Answer budget |
| `EASYRAG_TEMPERATURE` | `0.2` | Sampling temperature |
| `EASYRAG_TIMEOUT` | `180` | HTTP timeout in seconds |

Anything you set with `easy-rag config --set key=value` is written to
`~/.config/easy-rag/config.json`; `easy-rag config --show` prints the effective values with the
key masked.

## How it works

### The pipeline

```mermaid
flowchart LR
    subgraph Disk
        F["txt · md · org · rst · pdf"]
        DB[("Chroma<br/>SQLite + HNSW")]
    end

    subgraph easy-rag
        L["load_file()<br/>utf-8 → latin-1 fallback<br/>pypdf per page"]
        C["split_text()<br/>1200 chars · 200 overlap<br/>paragraph aware"]
        EMB["Chroma embedding fn<br/>ONNX MiniLM (CPU)"]
        R["query(question, k=5)<br/>cosine distance"]
        P["build_messages()<br/>numbered context + history"]
        M["stream_chat()<br/>SSE tokens"]
    end

    F --> L --> C --> EMB --> DB
    DB --> R --> P --> M --> ANS["answer with [n] citations"]
```

### Indexing a file

```mermaid
sequenceDiagram
    autonumber
    actor U as You
    participant T as TUI / CLI
    participant I as ingest.py
    participant S as VectorStore
    participant C as Chroma

    U->>T: ctrl+i on marked files
    T->>I: collect_files()
    I->>I: load_file() → normalise() → split_text()
    I-->>T: chunks with sha1(source:page:index)
    T->>S: rag.ingest(paths)
    S->>C: delete ids where source = file
    S->>C: add(ids, documents, metadata) in batches of 256
    C-->>S: embeddings computed once, stored
    S-->>T: IngestResult(chunks, errors)
    T-->>U: ＋ indexed file (n chunks) + refreshed INDEX panel
```

### Asking a question

```mermaid
sequenceDiagram
    autonumber
    actor U as You
    participant T as TUI
    participant E as RagEngine
    participant C as Chroma
    participant L as LLM

    U->>T: "What distance metric does easy-rag use?"
    T->>E: ask_stream(question, history)
    E->>C: collection.query(query_texts=[q], n_results=5)
    C-->>E: chunks + cosine distances
    E-->>T: Event("sources", hits)
    Note over T: the streaming panel opens
    E->>L: system prompt + numbered context + last 8 messages
    L-->>E: SSE: delta.content …
    E-->>T: Event("token", text) per delta
    E-->>T: Event("done", full answer)
    T->>U: Markdown answer, then sources with distances
```

### Data model

```mermaid
erDiagram
    SOURCE ||--o{ CHUNK : "is split into"
    SOURCE {
        string path "absolute path on disk"
        int chunk_count
        float indexed_at
    }
    CHUNK {
        string id "sha1(path:page:index)"
        string document "the raw text"
        int chunk "0-based position"
        int page "PDF page, optional"
        string source "back-pointer"
    }
    COLLECTION ||--o{ CHUNK : stores
    COLLECTION {
        string name "documents"
        string space "cosine"
    }
```

### UI state machine

```mermaid
stateDiagram-v2
    [*] --> Browsing
    Browsing --> Marking: space
    Marking --> Browsing: space again
    Browsing --> Indexing: ctrl+i / enter
    Indexing --> Browsing: chunks written
    Browsing --> Retrieving: enter in prompt
    Retrieving --> Streaming: sources event
    Streaming --> Browsing: done event
    Streaming --> Browsing: error (shown in red)
```

### Design choices

| Decision | Why |
| --- | --- |
| Chroma's default ONNX MiniLM embeddings | Zero setup, CPU-only, no embedding API bills |
| Cosine distance | Scale-free and stable for sentence embeddings; lower = closer |
| Character-based chunking | Predictable cost across languages, no tokenizer dependency |
| Delete-then-add per source | Re-indexing is idempotent; no duplicate chunks |
| sha1(`source:page:index`) ids | Deterministic ids make the delete above trivial |
| SSE parsing that ignores `reasoning` | Free routers stream chain-of-thought before the answer |
| 12k-character context cap | Keeps prompts small enough for free tiers |
| Last 8 messages of history | Follow-up questions without unbounded prompt growth |

## Project layout

```
easy-rag-for-agent/
├── src/easyrag/
│   ├── config.py        # Settings, .env + config.json resolution, masking
│   ├── ingest.py        # loaders, normalisation, paragraph-aware chunking
│   ├── store.py         # Chroma wrapper: ingest, query, sources, reset
│   ├── llm.py           # OpenAI-compatible streaming client (httpx + SSE)
│   ├── rag.py           # prompt construction, ingestion summary, ask_stream
│   ├── cli.py           # argparse entry point for every subcommand
│   └── tui/
│       └── app.py       # Textual app: FileTree, RichLog chat, workers
├── tests/               # 29 tests: ingest, store, SSE parsing, TUI pilot
├── sample-docs/         # tiny corpus to try the app with
├── scripts/             # demo capture + GIF rendering
└── docs/demo/           # generated screenshots and demo.gif
```

## Development

```bash
uv sync --extra dev
uv run pytest -q            # 29 passed
uv run easy-rag --version
```

The test suite covers chunking edge cases, idempotent re-ingestion, delete-by-source, SSE
parsing with a mocked transport (including reasoning deltas and error events), prompt assembly,
and a real Textual pilot that types into the chat.

### Regenerating the demo assets

The GIF and screenshots in this README are generated, not hand-made:

```bash
uv run python scripts/demo_capture.py --size 126x40   # drives the real TUI, writes SVG frames
./scripts/render_demo.sh                              # SVG → PNG stills + docs/demo/demo.gif
```

`demo_capture.py` uses a throwaway Chroma database, indexes `sample-docs/` for real, and swaps in
a scripted model stream so the capture is deterministic and offline.

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `FreeUsageLimitError` / `Insufficient balance` | Your provider quota is used up. Switch `MODEL`, add credits, or point `EASYRAG_BASE_URL` at a local server. |
| `401` from the chat endpoint | Check the key in `.env`; `easy-rag doctor` prints whether it was found. |
| First ingest hangs for a minute | Chroma is downloading the ONNX embedding model (~80 MB). It is cached afterwards. |
| `no .md, .txt or .pdf files found` | The folder only had hidden files or unsupported extensions. |
| Answers ignore your newest notes | Re-ingest the file (`ctrl+i`); retrieval only sees what is in Chroma. |
| Scanned PDFs return nothing | There is no OCR. Run them through `ocrmypdf` first. |
| Collection name error | Chroma requires 3–512 characters for collection names. |
| TUI looks cramped | Resize the terminal; the layout needs roughly 80×24 minimum. |

## Limitations

- Supported inputs: `.txt`, `.text`, `.md`, `.markdown`, `.org`, `.rst`, `.pdf` (text-based).
- One collection at a time; switching collections means changing `EASYRAG_COLLECTION`.
- No reranker — the top-5 cosine hits go straight into the prompt.
- The chat model must be reachable over HTTP and stream OpenAI-style SSE.

## Roadmap

- [x] Text, Markdown and PDF ingestion
- [x] Persistent Chroma index with idempotent re-indexing
- [x] Internal file explorer TUI with marking and an index panel
- [x] Streaming answers with citations and conversation memory
- [x] CLI for agents (`add`, `ask`, `ls`, `rm`, `reset`, `doctor`, `config`)
- [ ] Pluggable embedding backends (OpenAI, Ollama) alongside the ONNX default
- [ ] Multiple named collections in the TUI
- [ ] Export a conversation to Markdown
- [ ] Optional reranking pass before generation

## Built with

[Textual](https://textual.textualize.io/) ·
[Chroma](https://www.trychroma.com/) ·
[Rich](https://rich.readthedocs.io/) ·
[httpx](https://www.python-httpx.org/) ·
[pypdf](https://pypdf.readthedocs.io/) ·
[uv](https://docs.astral.sh/uv/)

## License

MIT — see [LICENSE](LICENSE). The sample documents are original and free to reuse.

---

<p align="center">
  <a href="https://github.com/Abdulhadi446/easy-rag-for-agent">github.com/Abdulhadi446/easy-rag-for-agent</a><br>
  <sub>If this saved you an afternoon of wiring, a ⭐ helps.</sub>
</p>
