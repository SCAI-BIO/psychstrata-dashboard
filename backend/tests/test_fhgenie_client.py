import io
import json
from urllib import error

import pytest

from app.clients.fhgenie_client import FHGenieClient
from app.clients.llm_client import LLMClientError


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
        captured["authorization"] = req.get_header("Authorization")
        captured["timeout"] = timeout
        return handler(req)

    monkeypatch.setattr("app.clients.fhgenie_client.request.urlopen", _fake_urlopen)
    return captured


def test_complete_returns_not_configured_when_secret_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fail(_req):
        raise AssertionError("urlopen must not be called without a secret")

    monkeypatch.setattr("app.clients.fhgenie_client.request.urlopen", _fail)
    client = FHGenieClient(url="https://fhgenie.fraunhofer.de/v1", model="Qwen/Qwen3-Coder-Next-FP8", secret=None)

    text = client.complete("prompt")

    assert "not configured" in text


def test_complete_posts_chat_completions_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    body = json.dumps({"choices": [{"message": {"content": "  the explanation  "}}]}).encode("utf-8")
    captured = _patch_urlopen(monkeypatch, lambda _req: _FakeResponse(body))

    client = FHGenieClient(url="https://fhgenie.fraunhofer.de/v1", model="Qwen/Qwen3-Coder-Next-FP8", secret="secret")

    assert client.complete("prompt") == "the explanation"
    assert captured["url"] == "https://fhgenie.fraunhofer.de/v1/chat/completions"
    assert captured["authorization"] == "Bearer secret"
    assert captured["data"]["model"] == "Qwen/Qwen3-Coder-Next-FP8"
    assert captured["data"]["temperature"] == 0.2
    assert captured["data"]["max_tokens"] == 350
    assert captured["data"]["messages"][1] == {"role": "user", "content": "prompt"}
    assert captured["data"]["messages"][0]["role"] == "system"
    assert captured["timeout"] == 20


def test_complete_does_not_duplicate_chat_completions_suffix(monkeypatch: pytest.MonkeyPatch) -> None:
    body = json.dumps({"choices": [{"message": {"content": "ok"}}]}).encode("utf-8")
    captured = _patch_urlopen(monkeypatch, lambda _req: _FakeResponse(body))

    client = FHGenieClient(url="https://fhgenie.fraunhofer.de/v1/chat/completions/", model="m", secret="secret")

    assert client.complete("prompt") == "ok"
    assert captured["url"] == "https://fhgenie.fraunhofer.de/v1/chat/completions"


def test_complete_raises_on_empty_choices(monkeypatch: pytest.MonkeyPatch) -> None:
    body = json.dumps({"choices": []}).encode("utf-8")
    _patch_urlopen(monkeypatch, lambda _req: _FakeResponse(body))

    client = FHGenieClient(url="https://fhgenie.fraunhofer.de/v1", model="m", secret="secret")

    with pytest.raises(LLMClientError, match="empty response"):
        client.complete("prompt")


def test_complete_raises_on_http_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise(_req):
        raise error.HTTPError(
            url="https://fhgenie.fraunhofer.de/v1/chat/completions",
            code=401,
            msg="unauthorized",
            hdrs=None,
            fp=io.BytesIO(b"invalid key"),
        )

    _patch_urlopen(monkeypatch, _raise)
    client = FHGenieClient(url="https://fhgenie.fraunhofer.de/v1", model="m", secret="secret")

    with pytest.raises(LLMClientError, match="HTTP 401") as excinfo:
        client.complete("prompt")
    assert "invalid key" in str(excinfo.value)


def test_complete_raises_on_url_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise(_req):
        raise error.URLError("dns failure")

    _patch_urlopen(monkeypatch, _raise)
    client = FHGenieClient(url="https://fhgenie.fraunhofer.de/v1", model="m", secret="secret")

    with pytest.raises(LLMClientError, match="could not be reached"):
        client.complete("prompt")


def test_complete_raises_on_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise(_req):
        raise TimeoutError()

    _patch_urlopen(monkeypatch, _raise)
    client = FHGenieClient(url="https://fhgenie.fraunhofer.de/v1", model="m", secret="secret")

    with pytest.raises(LLMClientError, match="timed out"):
        client.complete("prompt")


def test_complete_raises_on_invalid_json(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_urlopen(monkeypatch, lambda _req: _FakeResponse(b"<html>"))
    client = FHGenieClient(url="https://fhgenie.fraunhofer.de/v1", model="m", secret="secret")

    with pytest.raises(LLMClientError, match="could not be parsed"):
        client.complete("prompt")
