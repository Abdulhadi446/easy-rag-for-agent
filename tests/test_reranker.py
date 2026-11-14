from __future__ import annotations

from dataclasses import replace

import pytest

from easyrag.config import Settings
from easyrag.rag import RagEngine
from easyrag.reranker import CrossEncoderReranker, RerankError
from easyrag.store import Hit


class FakeStore:
    def __init__(self, hits: list[Hit]) -> None:
        self.hits = hits
        self.last_k = 0
        self.hybrid_args: tuple[int, int] | None = None

    def query(self, text: str, k: int) -> list[Hit]:
        self.last_k = k
        return [replace(hit) for hit in self.hits[:k]]

    def query_hybrid(self, text: str, k: int, candidates: int) -> list[Hit]:
        self.hybrid_args = (k, candidates)
        return [replace(hit, metric="rrf") for hit in self.hits[:k]]


class FakeReranker:
    model = "fake/cross-encoder"

    def __init__(self) -> None:
        self.loaded = False
        self.calls: list[tuple[str, list[str]]] = []

    def load(self) -> "FakeReranker":
        self.loaded = True
        return self

    def score(self, query: str, documents: list[str]) -> list[float]:
        self.calls.append((query, list(documents)))
        return [float(document.lower().count("zebra")) for document in documents]


def hit(text: str, score: float = 0.1) -> Hit:
    return Hit(text=text, source=f"/docs/{text}.md", chunk=0, score=score)


def test_rerank_reorders_the_candidate_pool_and_changes_the_metric():
    store = FakeStore([hit("apple"), hit("zebra facts"), hit("banana")])
    reranker = FakeReranker()
    settings = Settings(top_k=2, rerank=True, candidates=20)
    engine = RagEngine(settings, store=store, reranker=reranker)

    hits = engine.retrieve("tell me about zebra")

    assert store.last_k == 20
    assert [item.text for item in hits] == ["zebra facts", "apple"]
    assert all(item.metric == "rerank" for item in hits)
    assert hits[0].score == 1.0
    assert reranker.calls


def test_rerank_is_off_by_default():
    store = FakeStore([hit("apple")])
    reranker = FakeReranker()
    settings = Settings(top_k=3)
    engine = RagEngine(settings, store=store, reranker=reranker)

    hits = engine.retrieve("apple")

    assert store.last_k == 3
    assert not reranker.calls
    assert hits[0].metric == "distance"


def test_hybrid_retrieval_uses_the_wide_candidate_pool():
    store = FakeStore([hit("apple")])
    settings = Settings(top_k=3, hybrid=True, candidates=25)
    engine = RagEngine(settings, store=store)

    hits = engine.retrieve("apple")

    assert store.hybrid_args == (25, 25)
    assert hits[0].metric == "rrf"


def test_hybrid_then_rerank_composes():
    store = FakeStore([hit("apple"), hit("zebra facts")])
    reranker = FakeReranker()
    settings = Settings(top_k=2, hybrid=True, rerank=True, candidates=10)
    engine = RagEngine(settings, store=store, reranker=reranker)

    hits = engine.retrieve("zebra")

    assert store.hybrid_args == (10, 10)
    assert hits[0].text == "zebra facts"
    assert hits[0].metric == "rerank"


def test_warm_reranker_loads_without_a_query():
    store = FakeStore([])
    reranker = FakeReranker()
    engine = RagEngine(Settings(), store=store, reranker=reranker)

    assert engine.warm_reranker() == "fake/cross-encoder"
    assert reranker.loaded


def test_local_export_directory_is_detected(tmp_path):
    (tmp_path / "tokenizer.json").write_text("{}", encoding="utf-8")
    (tmp_path / "onnx").mkdir()
    weights = tmp_path / "onnx" / "model_quantized.onnx"
    weights.write_bytes(b"")

    reranker = CrossEncoderReranker(str(tmp_path))

    assert reranker._local_model() == (str(tmp_path / "tokenizer.json"), str(weights))


def test_local_export_without_weights_is_a_clear_error(tmp_path):
    (tmp_path / "tokenizer.json").write_text("{}", encoding="utf-8")

    with pytest.raises(RerankError, match="tokenizer.json"):
        CrossEncoderReranker(str(tmp_path))._local_model()


def test_hub_model_id_is_not_treated_as_a_local_path():
    assert CrossEncoderReranker("Xenova/ms-marco-MiniLM-L-6-v2")._local_model() is None
