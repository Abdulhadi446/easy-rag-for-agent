from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from easyrag import llm
from easyrag.config import Settings
from easyrag.llm import LLMError, Message, stream_chat
from easyrag.rag import Event, RagEngine
from easyrag.store import Hit


def test_settings_load_from_dotenv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    for name in ("EASYRAG_API_KEY", "OPENROUTER_API_KEY", "OPENAI_API_KEY", "MODEL"):
        monkeypatch.delenv(name, raising=False)
    env = tmp_path / ".env"
    env.write_text(
        "# comment\nOPENROUTER_API_KEY=sk-or-secret\nMODEL=openrouter/free\nEASYRAG_TOP_K=9\n"
        "EASYRAG_CANDIDATES=30\nEASYRAG_HYBRID=on\nEASYRAG_RERANK=0\n",
        encoding="utf-8",
    )
    settings = Settings.load(env_file=env, cwd=tmp_path)
    assert settings.api_key == "sk-or-secret"
    assert settings.model == "openrouter/free"
    assert settings.top_k == 9
    assert settings.candidates == 30
    assert settings.hybrid is True
    assert settings.rerank is False


def test_settings_are_not_configured_without_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("EASYRAG_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    settings = Settings.load(env_file=tmp_path / "missing.env", cwd=tmp_path)
    assert settings.configured is False


def test_settings_update_and_save_keep_retrieval_modes(tmp_path: Path):
    settings = Settings()
    settings.update(hybrid="yes", rerank="off", candidates="12", top_k="4")

    assert settings.hybrid is True
    assert settings.rerank is False
    assert settings.candidates == 12

    path = settings.save(tmp_path / "config.json")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["hybrid"] is True
    assert data["rerank"] is False
    assert data["candidates"] == 12
    assert data["rerank_model"]


def test_public_dict_hides_the_key(tmp_path: Path):
    settings = Settings(api_key="sk-or-very-secret")
    assert settings.public_dict()["api_key"] == "set"


def _sse(*chunks: str) -> bytes:
    lines = [f"data: {json.dumps(chunk)}\n\n" for chunk in chunks]
    lines.append("data: [DONE]\n\n")
    return "".join(lines).encode()


def test_stream_chat_parses_sse_and_skips_reasoning(monkeypatch: pytest.MonkeyPatch):
    body = _sse(
        {"choices": [{"delta": {"reasoning": "thinking hard"}}]},
        {"choices": [{"delta": {"content": "Hello"}}]},
        {"choices": [{"delta": {"content": " world"}}]},
    )

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url == "https://example.test/v1/chat/completions"
        assert request.headers["authorization"] == "Bearer sk-test"
        return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})

    transport = httpx.MockTransport(handler)
    real_client = httpx.Client

    def patched_client(*args, **kwargs):  # noqa: ANN002, ANN003
        kwargs["transport"] = transport
        return real_client(*args, **kwargs)

    monkeypatch.setattr(llm.httpx, "Client", patched_client)
    settings = Settings(api_key="sk-test", base_url="https://example.test/v1")
    result = "".join(stream_chat([Message("user", "hi")], settings))
    assert result == "Hello world"


def test_stream_chat_maps_provider_errors(monkeypatch: pytest.MonkeyPatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(402, json={"error": {"message": "Insufficient balance"}})

    transport = httpx.MockTransport(handler)
    real_client = httpx.Client

    def patched_client(*args, **kwargs):  # noqa: ANN002, ANN003
        kwargs["transport"] = transport
        return real_client(*args, **kwargs)

    monkeypatch.setattr(llm.httpx, "Client", patched_client)
    settings = Settings(api_key="sk-test", base_url="https://example.test/v1")
    with pytest.raises(LLMError) as excinfo:
        list(stream_chat([Message("user", "hi")], settings))
    assert "402" in str(excinfo.value)
    assert "Insufficient balance" in str(excinfo.value)


def test_stream_chat_requires_a_key():
    with pytest.raises(LLMError):
        list(stream_chat([Message("user", "hi")], Settings(api_key="")))


def test_build_messages_numbers_the_context():
    engine = RagEngine.__new__(RagEngine)  # no store needed for prompt building
    hits = [
        Hit(text="chroma uses cosine", source="/docs/chroma.md", chunk=2, score=0.1),
        Hit(text="page body", source="/docs/paper.pdf", chunk=0, page=4, score=0.2),
    ]
    messages = engine.build_messages("which metric?", hits)
    assert messages[0].role == "system"
    prompt = messages[-1].content
    assert "[1] chroma.md, chunk 2" in prompt
    assert "[2] paper.pdf, page 4" in prompt
    assert "which metric?" in prompt


def test_build_messages_flags_missing_context():
    engine = RagEngine.__new__(RagEngine)
    messages = engine.build_messages("anything?", [])
    assert "No indexed context matched" in messages[-1].content


def test_ask_stream_emits_sources_tokens_and_done(monkeypatch: pytest.MonkeyPatch):
    class FakeStore:
        def query(self, text, k):  # noqa: ANN001
            return [Hit(text="ctx", source="/a.md", chunk=0, score=0.1)]

        def count(self):
            return 1

    engine = RagEngine(Settings(api_key="k"), FakeStore())  # type: ignore[arg-type]
    monkeypatch.setattr("easyrag.rag.stream_chat", lambda *args, **kwargs: iter(["one ", "two"]))
    events = list(engine.ask_stream("hi"))
    assert isinstance(events[0], Event)
    assert events[0].kind == "sources"
    assert events[1].kind == "token"
    assert events[-1].kind == "done"
    assert events[-1].text == "one two"
