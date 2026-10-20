from .llm_client import LLMClient


class OpenAIClient(LLMClient):
    def _build_payload(self, prompt: str) -> dict:
        return {
            "model": self.model,
            "input": [
                {
                    "role": "system",
                    "content": [{"type": "input_text", "text": self.system_prompt}],
                },
                {"role": "user", "content": [{"type": "input_text", "text": prompt}]},
            ],
            "temperature": self.temperature,
            "max_output_tokens": self.max_tokens,
        }

    def _extract_text(self, response_payload: dict) -> str:
        if response_payload.get("output_text"):
            return str(response_payload["output_text"])
        return "\n".join(
            content.get("text", "")
            for output in response_payload.get("output", [])
            for content in output.get("content", [])
            if content.get("text")
        )
