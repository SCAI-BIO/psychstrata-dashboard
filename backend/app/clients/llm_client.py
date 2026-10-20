import json
from abc import ABC, abstractmethod
from urllib import error, request


class LLMClientError(RuntimeError):
    pass


class LLMClient(ABC):
    """Base class for LLM providers.

    Subclasses only describe how a request is shaped and how the response is
    parsed; transport, auth, and error handling are shared here.
    """

    request_timeout = 20
    requires_secret = True
    temperature = 0.2
    max_tokens = 350
    system_prompt = (
        "You explain model predictions for a synthetic depression treatment-resistance demo. "
        "Base every statement only on the provided JSON payload. "
        "Use plain language, cite only supplied PMIDs, and avoid unsupported claims."
    )

    def __init__(self, *, url: str, model: str, secret: str | None):
        self.url = url
        self.model = model
        self.secret = secret

    @property
    def service_name(self) -> str:
        name = type(self).__name__.removesuffix("Client")
        return f"the {name} service" if name else "the language model service"

    def complete(self, prompt: str) -> str:
        """Return a model completion for the supplied prompt."""
        if self.requires_secret and not self.secret:
            return (
                "Prediction explanation unavailable: the language model service is not configured.\n\n"
                "The SHAP chart still shows which features pushed this prediction higher or lower."
            )

        req = request.Request(
            self._endpoint(),
            data=json.dumps(self._build_payload(prompt)).encode("utf-8"),
            headers=self._headers(),
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=self.request_timeout) as response:
                response_payload = json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise LLMClientError(
                f"Prediction explanation unavailable: {self.service_name} returned HTTP {exc.code}.\n\n"
                f"```text\n{detail[:800]}\n```"
            ) from exc
        except error.URLError as exc:
            raise LLMClientError(
                f"Prediction explanation unavailable: {self.service_name} could not be reached ({exc.reason})."
            ) from exc
        except TimeoutError as exc:
            raise LLMClientError(
                f"Prediction explanation unavailable: {self.service_name} timed out."
            ) from exc
        except json.JSONDecodeError as exc:
            raise LLMClientError(
                f"Prediction explanation unavailable: {self.service_name} response could not be parsed."
            ) from exc

        text = self._extract_text(response_payload).strip()
        if not text:
            raise LLMClientError(
                f"Prediction explanation unavailable: {self.service_name} returned an empty response."
            )
        return text

    def _endpoint(self) -> str:
        return self.url

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.secret:
            headers["Authorization"] = f"Bearer {self.secret}"
        return headers

    @abstractmethod
    def _build_payload(self, prompt: str) -> dict:
        """Return the provider-specific request body for the prompt."""

    @abstractmethod
    def _extract_text(self, response_payload: dict) -> str:
        """Return the completion text from the provider-specific response."""
