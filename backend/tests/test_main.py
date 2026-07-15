from fastapi.testclient import TestClient
import pytest

from app.llm_summary import LLMServiceError
from app.main import BASIC_AUTH_PASSWORD_ENV, BASIC_AUTH_USERNAME_ENV, app


client = TestClient(app)


@pytest.fixture(autouse=True)
def _clear_auth_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(BASIC_AUTH_USERNAME_ENV, raising=False)
    monkeypatch.delenv(BASIC_AUTH_PASSWORD_ENV, raising=False)


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


def test_auth_status_endpoint_reports_disabled_by_default() -> None:
    response = client.get("/api/auth/status")

    assert response.status_code == 200
    assert response.json() == {"auth_enabled": False}


def test_auth_status_endpoint_reports_enabled_when_credentials_are_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(BASIC_AUTH_USERNAME_ENV, "dashboard-user")
    monkeypatch.setenv(BASIC_AUTH_PASSWORD_ENV, "dashboard-password")

    response = client.get("/api/auth/status")

    assert response.status_code == 200
    assert response.json() == {"auth_enabled": True}


def test_features_endpoint_requires_basic_auth_when_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(BASIC_AUTH_USERNAME_ENV, "dashboard-user")
    monkeypatch.setenv(BASIC_AUTH_PASSWORD_ENV, "dashboard-password")

    response = client.get("/api/features")

    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Basic"


def test_features_endpoint_accepts_valid_basic_auth(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(BASIC_AUTH_USERNAME_ENV, "dashboard-user")
    monkeypatch.setenv(BASIC_AUTH_PASSWORD_ENV, "dashboard-password")

    response = client.get("/api/features", auth=("dashboard-user", "dashboard-password"))

    assert response.status_code == 200


def test_auth_status_endpoint_reports_misconfiguration(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(BASIC_AUTH_USERNAME_ENV, "dashboard-user")

    response = client.get("/api/auth/status")

    assert response.status_code == 500
    assert "misconfigured" in response.json()["detail"]
