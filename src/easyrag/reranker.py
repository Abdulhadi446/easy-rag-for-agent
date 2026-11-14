"""A small local cross-encoder used as an optional reranking stage.

The retriever casts a wide net (``settings.candidates``), then this model scores each
(query, chunk) pair and keeps the best ``top_k``. It runs on CPU through onnxruntime,
needs no API key, and downloads a ~23 MB quantized model on first use.
"""

from __future__ import annotations

import os
import threading
from math import exp
from pathlib import Path
from typing import Sequence

from .config import DEFAULT_RERANK_MODEL

MAX_TOKENS = 512
BATCH_SIZE = 16


def _sigmoid(logit: float) -> float:
    """Stable 0..1 transform of a cross-encoder logit."""
    if logit >= 0.0:
        return 1.0 / (1.0 + exp(-logit))
    value = exp(logit)
    return value / (1.0 + value)


class RerankError(RuntimeError):
    """The reranker could not be loaded or run."""


def model_cache_dir() -> Path:
    """Where reranker weights live (kept out of the Chroma data directory)."""
    root = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return Path(os.environ.get("EASYRAG_CACHE_DIR", root / "easy-rag")) / "models"


class CrossEncoderReranker:
    """Lazy wrapper around a Hugging Face ONNX cross-encoder."""

    def __init__(
        self,
        model: str = DEFAULT_RERANK_MODEL,
        cache_dir: Path | None = None,
        threads: int = 2,
    ) -> None:
        self.model = model
        self.cache_dir = Path(cache_dir) if cache_dir else model_cache_dir()
        self.threads = max(1, threads)
        self._tokenizer = None
        self._session = None
        self._numpy = None
        self._input_names: set[str] = set()
        self._lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        return self._session is not None

    # ------------------------------------------------------------------ loading
    def _local_model(self) -> tuple[str, str] | None:
        """Resolve a local export directory, or return None for a Hub repo id."""
        root = Path(self.model).expanduser()
        if not root.is_dir():
            return None
        tokenizer = root / "tokenizer.json"
        for name in (
            "onnx/model_quantized.onnx",
            "onnx/model_int8.onnx",
            "onnx/model.onnx",
            "model_quantized.onnx",
            "model.onnx",
        ):
            weights = root / name
            if tokenizer.is_file() and weights.is_file():
                return str(tokenizer), str(weights)
        raise RerankError(
            f"{root} does not look like an ONNX export: need tokenizer.json and one of "
            "onnx/model_quantized.onnx, onnx/model_int8.onnx or onnx/model.onnx"
        )

    def _download(self, filename: str) -> str:
        from huggingface_hub import hf_hub_download

        kwargs = {
            "repo_id": self.model,
            "filename": filename,
            "cache_dir": str(self.cache_dir),
        }
        try:
            # Fast path: no network round-trip once the file is cached.
            return hf_hub_download(local_files_only=True, **kwargs)
        except Exception:  # noqa: BLE001 - not cached yet
            return hf_hub_download(**kwargs)

    def load(self) -> "CrossEncoderReranker":
        """Download (once) and initialise the tokenizer and ONNX session.

        Thread-safe: the TUI warms the model in a worker while a query may already
        be starting.
        """
        if self.loaded:
            return self
        with self._lock:
            if self.loaded:
                return self
            return self._load()

    def _load(self) -> "CrossEncoderReranker":
        try:
            import numpy as np
            import onnxruntime as ort
            from tokenizers import Tokenizer
        except ImportError as exc:  # pragma: no cover - declared dependencies
            raise RerankError(
                "reranking needs onnxruntime, tokenizers and huggingface-hub; "
                "run `uv sync` to install them"
            ) from exc

        try:
            local = self._local_model()
            if local is not None:
                tokenizer_path, model_path = local
            else:
                tokenizer_path = self._download("tokenizer.json")
                model_path = None
                errors: list[str] = []
                for filename in (
                    "onnx/model_quantized.onnx",
                    "onnx/model_int8.onnx",
                    "onnx/model.onnx",
                ):
                    try:
                        model_path = self._download(filename)
                        break
                    except Exception as exc:  # noqa: BLE001 - try the next variant
                        errors.append(f"{filename}: {exc}")
                if model_path is None:
                    raise RerankError(
                        f"no ONNX weights found for {self.model} ({'; '.join(errors)})"
                    )
        except RerankError:
            raise
        except Exception as exc:  # noqa: BLE001 - network, auth, disk
            raise RerankError(
                f"could not download reranker {self.model}: {exc}. "
                "Check your connection or point EASYRAG_RERANK_MODEL at a local ONNX export."
            ) from exc

        try:
            tokenizer = Tokenizer.from_file(tokenizer_path)
            tokenizer.enable_truncation(max_length=MAX_TOKENS)
            tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")
            options = ort.SessionOptions()
            options.intra_op_num_threads = self.threads
            session = ort.InferenceSession(
                model_path,
                sess_options=options,
                providers=["CPUExecutionProvider"],
            )
        except Exception as exc:  # noqa: BLE001
            raise RerankError(f"could not load reranker {self.model}: {exc}") from exc

        self._tokenizer = tokenizer
        self._session = session
        self._numpy = np
        self._input_names = {item.name for item in session.get_inputs()}
        return self

    # ------------------------------------------------------------------ scoring
    def score(self, query: str, documents: Sequence[str]) -> list[float]:
        """Relevance in 0..1, one per document (higher is better)."""
        if not documents:
            return []
        self.load()
        assert self._tokenizer is not None and self._session is not None
        results: list[float] = []
        for start in range(0, len(documents), BATCH_SIZE):
            window = documents[start : start + BATCH_SIZE]
            encodings = self._tokenizer.encode_batch(
                [(query, document.strip() or " ") for document in window]
            )
            feed = {
                "input_ids": self._numpy.array(
                    [encoding.ids for encoding in encodings], dtype=self._numpy.int64
                ),
                "attention_mask": self._numpy.array(
                    [encoding.attention_mask for encoding in encodings],
                    dtype=self._numpy.int64,
                ),
            }
            if "token_type_ids" in self._input_names:
                feed["token_type_ids"] = self._numpy.array(
                    [encoding.type_ids for encoding in encodings],
                    dtype=self._numpy.int64,
                )
            logits = self._session.run(None, feed)[0]
            results.extend(_sigmoid(float(row[0])) for row in logits)
        return results
