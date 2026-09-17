import pytest

from app.clients.client_factory import create_llm_client
from app.clients.ollama_client import OllamaClient
from app.clients.openai_client import OpenAIClient


def test_llm_client_factory_uses_openai_for_default_url() -> None:
    client = create_llm_client(
        url="https://api.openai.com/v1/responses",
        model="gpt-4.1-mini",
        secret=None,
    )

    assert isinstance(client, OpenAIClient)


@pytest.mark.parametrize("url", ["http://localhost:11434", "http://127.0.0.1:11434/api/chat"])
def test_llm_client_factory_uses_ollama_for_local_urls(url: str) -> None:
    client = create_llm_client(url=url, model="llama3", secret=None)

    assert isinstance(client, OllamaClient)


def test_llm_client_factory_rejects_unknown_urls() -> None:
    with pytest.raises(ValueError, match="Unsupported LLM client URL"):
        create_llm_client(url="https://example.com/llm", model="model", secret=None)
