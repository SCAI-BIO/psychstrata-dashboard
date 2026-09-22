import json
from urllib import error, request

from .llm_client import LLMClient, LLMClientError


# TODO: test this
class OllamaClient(LLMClient):
    def complete(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
        }
        endpoint = self.url.rstrip("/")
        if not endpoint.endswith("/api/chat"):
            endpoint = f"{endpoint}/api/chat"
        req = request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=20) as response:
                response_payload = json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise LLMClientError(
                f"Prediction explanation unavailable: the Ollama service returned HTTP {exc.code}: {detail[:800]}"
            ) from exc
        except error.URLError as exc:
            raise LLMClientError(
                f"Prediction explanation unavailable: the Ollama service could not be reached ({exc.reason})."
            ) from exc
        except TimeoutError as exc:
            raise LLMClientError("Prediction explanation unavailable: the Ollama service timed out.") from exc
        except json.JSONDecodeError as exc:
            raise LLMClientError(
                "Prediction explanation unavailable: the Ollama service response could not be parsed."
            ) from exc

        text = response_payload.get("message", {}).get("content", "")
        if not text:
            raise LLMClientError(
                "Prediction explanation unavailable: the Ollama service returned an empty response."
            )
        return str(text).strip()
