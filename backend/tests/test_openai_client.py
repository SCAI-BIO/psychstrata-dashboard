import io
import json
from urllib import error

import pytest

from app.clients.llm_client import LLMClientError
from app.clients.openai_client import OpenAIClient


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

    monkeypatch.setattr("app.clients.llm_client.request.urlopen", _fake_urlopen)
    return captured


def test_complete_returns_not_configured_when_secret_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fail(_req):
        raise AssertionError("urlopen must not be called without a secret")

    monkeypatch.setattr("app.clients.llm_client.request.urlopen", _fail)
    client = OpenAIClient(url="https://api.openai.com/v1/responses", model="gpt-4.1-mini", secret=None)

    text = client.complete("prompt")

    assert "not configured" in text


def test_complete_uses_responses_api_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    body = json.dumps({"output_text": "  the explanation  "}).encode("utf-8")
    captured = _patch_urlopen(monkeypatch, lambda _req: _FakeResponse(body))

    client = OpenAIClient(url="https://api.openai.com/v1/responses", model="gpt-4.1-mini", secret="sk-secret")

    assert client.complete("prompt") == "the explanation"
    assert captured["url"] == "https://api.openai.com/v1/responses"
    assert captured["authorization"] == "Bearer sk-secret"
    assert captured["data"]["model"] == "gpt-4.1-mini"
    assert captured["data"]["temperature"] == 0.2
    assert captured["data"]["max_output_tokens"] == 350
    assert captured["data"]["input"][1]["content"][0] == {"type": "input_text", "text": "prompt"}


def test_complete_extracts_nested_output_text(monkeypatch: pytest.MonkeyPatch) -> None:
    body = json.dumps(
        {
            "output": [
                {"content": [{"type": "output_text", "text": "first"}]},
                {"content": [{"type": "output_text", "text": "second"}]},
            ]
        }
    ).encode("utf-8")
    _patch_urlopen(monkeypatch, lambda _req: _FakeResponse(body))

    client = OpenAIClient(url="https://api.openai.com/v1/responses", model="gpt-4.1-mini", secret="sk-secret")

    assert client.complete("prompt") == "first\nsecond"


def test_complete_raises_on_empty_output(monkeypatch: pytest.MonkeyPatch) -> None:
    body = json.dumps({"output": [{"content": [{"type": "output_text", "text": ""}]}]}).encode("utf-8")
    _patch_urlopen(monkeypatch, lambda _req: _FakeResponse(body))

    client = OpenAIClient(url="https://api.openai.com/v1/responses", model="gpt-4.1-mini", secret="sk-secret")

    with pytest.raises(LLMClientError, match="empty response"):
        client.complete("prompt")


def test_complete_raises_on_http_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise(_req):
        raise error.HTTPError(
            url="https://api.openai.com/v1/responses",
            code=429,
            msg="rate limited",
            hdrs=None,
            fp=io.BytesIO(b"slow down"),
        )

    _patch_urlopen(monkeypatch, _raise)
    client = OpenAIClient(url="https://api.openai.com/v1/responses", model="gpt-4.1-mini", secret="sk-secret")

    with pytest.raises(LLMClientError, match="HTTP 429") as excinfo:
        client.complete("prompt")
    assert "slow down" in str(excinfo.value)


def test_complete_raises_on_url_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise(_req):
        raise error.URLError("dns failure")

    _patch_urlopen(monkeypatch, _raise)
    client = OpenAIClient(url="https://api.openai.com/v1/responses", model="gpt-4.1-mini", secret="sk-secret")

    with pytest.raises(LLMClientError, match="could not be reached"):
        client.complete("prompt")


def test_complete_raises_on_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise(_req):
        raise TimeoutError()

    _patch_urlopen(monkeypatch, _raise)
    client = OpenAIClient(url="https://api.openai.com/v1/responses", model="gpt-4.1-mini", secret="sk-secret")

    with pytest.raises(LLMClientError, match="timed out"):
        client.complete("prompt")


def test_complete_raises_on_invalid_json(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_urlopen(monkeypatch, lambda _req: _FakeResponse(b"<html>"))
    client = OpenAIClient(url="https://api.openai.com/v1/responses", model="gpt-4.1-mini", secret="sk-secret")

    with pytest.raises(LLMClientError, match="could not be parsed"):
        client.complete("prompt")
