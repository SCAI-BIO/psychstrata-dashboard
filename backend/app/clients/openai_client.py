import json
from urllib import error, request

from .llm_client import LLMClient, LLMClientError


class OpenAIClient(LLMClient):
    def complete(self, prompt: str) -> str:
        if not self.secret:
            return (
                "Prediction explanation unavailable: the language model service is not configured.\n\n"
                "The SHAP chart still shows which features pushed this prediction higher or lower."
            )

        payload = {
            "model": self.model,
            "input": [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "input_text",
                            "text": (
                                "You explain model predictions for a synthetic depression treatment-resistance demo. "
                                "Base every statement only on the provided JSON payload. "
                                "Use plain language, cite only supplied PMIDs, and avoid unsupported claims."
                            ),
                        }
                    ],
                },
                {"role": "user", "content": [{"type": "input_text", "text": prompt}]},
            ],
            "temperature": 0.2,
            "max_output_tokens": 350,
        }
        req = request.Request(
            self.url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Authorization": f"Bearer {self.secret}", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=20) as response:
                response_payload = json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise LLMClientError(
                f"Prediction explanation unavailable: the language model service returned HTTP {exc.code}.\n\n"
                f"```text\n{detail[:800]}\n```"
            ) from exc
        except error.URLError as exc:
            raise LLMClientError(
                f"Prediction explanation unavailable: the language model service could not be reached ({exc.reason})."
            ) from exc
        except TimeoutError as exc:
            raise LLMClientError("Prediction explanation unavailable: the language model service timed out.") from exc
        except json.JSONDecodeError as exc:
            raise LLMClientError(
                "Prediction explanation unavailable: the language model service response could not be parsed."
            ) from exc

        text = self._extract_output_text(response_payload)
        if not text:
            raise LLMClientError(
                "Prediction explanation unavailable: the language model service returned an empty response."
            )
        return text

    @staticmethod
    def _extract_output_text(response_payload: dict) -> str:
        if response_payload.get("output_text"):
            return str(response_payload["output_text"]).strip()
        return "\n".join(
            content.get("text", "")
            for output in response_payload.get("output", [])
            for content in output.get("content", [])
            if content.get("text")
        ).strip()
