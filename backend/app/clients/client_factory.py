from urllib.parse import urlparse

from .llm_client import LLMClient
from .ollama_client import OllamaClient
from .openai_client import OpenAIClient


def create_llm_client(*, url: str, model: str, secret: str | None) -> LLMClient:
    hostname = (urlparse(url).hostname or "").lower()
    if hostname == "api.openai.com":
        return OpenAIClient(url=url, model=model, secret=secret)
    elif hostname in {"localhost", "127.0.0.1", "::1"}:
        return OllamaClient(url=url, model=model, secret=secret)
    else:
        raise ValueError(f"Unsupported LLM client URL: {url}")
