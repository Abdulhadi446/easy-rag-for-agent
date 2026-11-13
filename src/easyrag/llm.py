"""Minimal OpenAI-compatible chat client with streaming.

Works with OpenRouter, OpenAI, Groq, Together, local Ollama (`/v1`) and anything else that
speaks the same protocol. Only the standard library plus httpx.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Iterator, Sequence

import httpx

from .config import Settings


class LLMError(RuntimeError):
    """A provider error worth showing to the user verbatim."""


@dataclass
class Message:
    role: str
    content: str

    def as_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


def _friendly(status: int, body: str) -> str:
    snippet = body.strip()
    for prefix in ("data: ", "event: "):
        if snippet.startswith(prefix):
            snippet = snippet[len(prefix) :]
    try:
        payload = json.loads(snippet)
        error = payload.get("error") if isinstance(payload, dict) else None
        if isinstance(error, dict):
            snippet = error.get("message") or json.dumps(error)
        elif isinstance(error, str):
            snippet = error
        elif isinstance(payload, dict) and payload.get("message"):
            snippet = str(payload["message"])
    except (json.JSONDecodeError, AttributeError):
        pass
    snippet = snippet[:400]
    if status == 401:
        return "401 unauthorized - the API key was rejected. Check EASYRAG_API_KEY / OPENROUTER_API_KEY."
    if status == 402:
        return f"402 payment required - the provider wants credits. {snippet}"
    if status == 429:
        return f"429 rate limited - slow down or try another model. {snippet}"
    return f"{status} from provider: {snippet}"


def stream_chat(
    messages: Sequence[Message],
    settings: Settings,
    *,
    model: str | None = None,
    max_tokens: int | None = None,
    temperature: float | None = None,
) -> Iterator[str]:
    """Yield content deltas from a streaming chat completion.

    Reasoning deltas (``delta.reasoning``) are skipped: they are not part of the answer.
    """
    if not settings.api_key:
        raise LLMError(
            "No API key found. Put OPENROUTER_API_KEY=... in .env or export EASYRAG_API_KEY."
        )

    url = f"{settings.base_url}/chat/completions"
    payload: dict[str, object] = {
        "model": model or settings.model,
        "messages": [message.as_dict() for message in messages],
        "stream": True,
        "max_tokens": max_tokens or settings.max_tokens,
        "temperature": settings.temperature if temperature is None else temperature,
    }
    headers = {
        "Authorization": f"Bearer {settings.api_key}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
        # OpenRouter uses these for attribution; harmless elsewhere.
        "HTTP-Referer": "https://github.com/Abdulhadi446/easy-rag-for-agent",
        "X-Title": "easy-rag-for-agent",
    }
    headers.update(settings.extra_headers or {})

    try:
        with httpx.Client(timeout=httpx.Timeout(settings.timeout, connect=20.0)) as client:
            with client.stream("POST", url, json=payload, headers=headers) as response:
                if response.status_code >= 400:
                    raise LLMError(_friendly(response.status_code, response.read().decode("utf-8", "replace")))
                for line in response.iter_lines():
                    if not line:
                        continue
                    if line.startswith(":"):
                        continue  # keep-alive comment
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        return
                    try:
                        event = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(event, dict) and event.get("error"):
                        error = event["error"]
                        message = error.get("message") if isinstance(error, dict) else str(error)
                        raise LLMError(str(message))
                    for choice in event.get("choices") or []:
                        delta = choice.get("delta") or choice.get("message") or {}
                        content = delta.get("content")
                        if content:
                            yield content
    except httpx.TimeoutException as exc:
        raise LLMError(f"request timed out after {settings.timeout:.0f}s") from exc
    except httpx.RequestError as exc:
        raise LLMError(f"could not reach {url}: {exc}") from exc


def complete(messages: Sequence[Message], settings: Settings, **kwargs: object) -> str:
    """Collect a streamed answer into a single string (used by the CLI)."""
    return "".join(stream_chat(messages, settings, **kwargs))  # type: ignore[arg-type]
