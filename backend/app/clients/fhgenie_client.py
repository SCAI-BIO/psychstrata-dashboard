from .llm_client import LLMClient


class FHGenieClient(LLMClient):
    def _endpoint(self) -> str:
        endpoint = self.url.rstrip("/")
        if not endpoint.endswith("/chat/completions"):
            endpoint = f"{endpoint}/chat/completions"
        return endpoint

    def _build_payload(self, prompt: str) -> dict:
        return {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

    def _extract_text(self, response_payload: dict) -> str:
        choices = response_payload.get("choices") or []
        if not choices:
            return ""
        return str(choices[0].get("message", {}).get("content") or "")

    def _truncation_detail(self, response_payload: dict) -> str | None:
        choices = response_payload.get("choices") or []
        if not choices:
            return None
        finish_reason = choices[0].get("finish_reason")
        if finish_reason != "length":
            return None
        usage = response_payload.get("usage") or {}
        reasoning_tokens = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")
        if reasoning_tokens:
            return f"hit the {self.max_tokens}-token limit while reasoning ({reasoning_tokens} reasoning tokens)"
        return f"hit the {self.max_tokens}-token limit"
