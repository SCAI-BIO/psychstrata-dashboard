import io
import json
from urllib import error

import pytest

from app.clients.llm_client import LLMClientError
from app.clients.ollama_client import OllamaClient


class _FakeResponse:
    def __init__(self, body: bytes) -> None:
        self._body = body

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *_exc) -> bool:
        return False


def _patch_urlopen(monkeypatch: pytest.MonkeyPatch, handler):
    captured: dict[str, object] = {}

    def _fake_urlopen(req, timeout=None):
        captured["url"] = req.full_url
        captured["data"] = json.loads(req.data.decode("utf-8"))
        captured["headers"] = dict(req.header_items())
        captured["timeout"] = timeout
        return handler(req)

    monkeypatch.setattr("app.clients.llm_client.request.urlopen", _fake_urlopen)
    return captured


def test_complete_returns_stripped_message_content(monkeypatch: pytest.MonkeyPatch) -> None:
    body = json.dumps({"message": {"content": "  hello world  "}}).encode("utf-8")
    captured = _patch_urlopen(monkeypatch, lambda _req: _FakeResponse(body))

    client = OllamaClient(url="http://localhost:11434", model="llama3", secret=None)

    assert client.complete("prompt") == "hello world"
    assert captured["url"] == "http://localhost:11434/api/chat"
    assert captured["data"] == {
        "model": "llama3",
        "messages": [{"role": "user", "content": "prompt"}],
        "stream": False,
    }
    assert captured["timeout"] == 20


def test_complete_does_not_duplicate_api_chat_suffix(monkeypatch: pytest.MonkeyPatch) -> None:
    body = json.dumps({"message": {"content": "ok"}}).encode("utf-8")
    captured = _patch_urlopen(monkeypatch, lambda _req: _FakeResponse(body))

    client = OllamaClient(url="http://localhost:11434/api/chat/", model="llama3", secret=None)

    assert client.complete("prompt") == "ok"
    assert captured["url"] == "http://localhost:11434/api/chat"


def test_complete_raises_on_http_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise(_req):
        raise error.HTTPError(
            url="http://localhost:11434/api/chat",
            code=500,
            msg="boom",
            hdrs=None,
            fp=io.BytesIO(b"server exploded"),
        )

    _patch_urlopen(monkeypatch, _raise)
    client = OllamaClient(url="http://localhost:11434", model="llama3", secret=None)

    with pytest.raises(LLMClientError, match="HTTP 500") as excinfo:
        client.complete("prompt")
    assert "server exploded" in str(excinfo.value)


def test_complete_includes_http_error_body(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise(_req):
        raise error.HTTPError(
            url="http://localhost:11434/api/chat",
            code=400,
            msg="bad",
            hdrs=None,
            fp=io.BytesIO(b"x" * 2000),
        )

    _patch_urlopen(monkeypatch, _raise)
    client = OllamaClient(url="http://localhost:11434", model="llama3", secret=None)

    with pytest.raises(LLMClientError) as excinfo:
        client.complete("prompt")
    assert "x" * 2000 in str(excinfo.value)


def test_complete_raises_on_url_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise(_req):
        raise error.URLError("connection refused")

    _patch_urlopen(monkeypatch, _raise)
    client = OllamaClient(url="http://localhost:11434", model="llama3", secret=None)

    with pytest.raises(LLMClientError, match="could not be reached"):
        client.complete("prompt")


def test_complete_raises_on_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise(_req):
        raise TimeoutError()

    _patch_urlopen(monkeypatch, _raise)
    client = OllamaClient(url="http://localhost:11434", model="llama3", secret=None)

    with pytest.raises(LLMClientError, match="timed out"):
        client.complete("prompt")


def test_complete_raises_on_invalid_json(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_urlopen(monkeypatch, lambda _req: _FakeResponse(b"not json"))
    client = OllamaClient(url="http://localhost:11434", model="llama3", secret=None)

    with pytest.raises(LLMClientError, match="could not be parsed"):
        client.complete("prompt")


def test_complete_raises_on_empty_content(monkeypatch: pytest.MonkeyPatch) -> None:
    body = json.dumps({"message": {"content": ""}}).encode("utf-8")
    _patch_urlopen(monkeypatch, lambda _req: _FakeResponse(body))
    client = OllamaClient(url="http://localhost:11434", model="llama3", secret=None)

    with pytest.raises(LLMClientError, match="empty response"):
        client.complete("prompt")
