from abc import ABC, abstractmethod


class LLMClientError(RuntimeError):
    pass


class LLMClient(ABC):
    def __init__(self, *, url: str, model: str, secret: str | None):
        self.url = url
        self.model = model
        self.secret = secret

    @abstractmethod
    def complete(self, prompt: str) -> str:
        """Return a model completion for the supplied prompt."""
