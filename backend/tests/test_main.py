from fastapi.testclient import TestClient

from app.llm_summary import LLMServiceError
from app.main import app


client = TestClient(app)


def test_health_endpoint() -> None:
    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_summary_endpoint() -> None:
    response = client.get("/api/summary")

    assert response.status_code == 200
    body = response.json()
    assert body["title"] == "Treatment Resistance Classifier Demo"
    assert "React frontend" in body["message"]
    assert "not be used for clinical decisions" in body["disclaimer"]


def test_explain_propagates_llm_errors(monkeypatch) -> None:
    features_response = client.get("/api/features")
    assert features_response.status_code == 200
    defaults = features_response.json()["defaults"]

    def _raise_llm_error(*_args, **_kwargs):
        raise LLMServiceError("Prediction explanation unavailable: upstream LLM HTTP 500.")

    monkeypatch.setattr("app.main.generate_prediction_summary", _raise_llm_error)

    response = client.post(
        "/api/explain",
        json={
            "features": defaults,
            "confidence_level": 95,
        },
    )

    assert response.status_code == 502
    assert "Prediction explanation unavailable" in response.json()["detail"]
