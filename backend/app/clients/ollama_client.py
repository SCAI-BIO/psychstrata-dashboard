from .llm_client import LLMClient


# TODO: test this
class OllamaClient(LLMClient):
    requires_secret = False

    def _endpoint(self) -> str:
        endpoint = self.url.rstrip("/")
        if not endpoint.endswith("/api/chat"):
            endpoint = f"{endpoint}/api/chat"
        return endpoint

    def _build_payload(self, prompt: str) -> dict:
        return {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
        }

    def _extract_text(self, response_payload: dict) -> str:
        return str(response_payload.get("message", {}).get("content", ""))
