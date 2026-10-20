import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.io import feature_loader
from app.settings import _reset_backend_settings_for_tests

client = TestClient(app)


@pytest.fixture(autouse=True)
def _reset(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("BACKEND_BASIC_AUTH_USERNAME", raising=False)
    monkeypatch.delenv("BACKEND_BASIC_AUTH_PASSWORD", raising=False)
    monkeypatch.delenv("FEATURES_CONFIG_PATH", raising=False)
    monkeypatch.delenv("LLM_PROVIDER_KEY", raising=False)
    monkeypatch.delenv("LLM_CLIENT_SECRET", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    feature_loader._reset_feature_loader_for_tests()
    _reset_backend_settings_for_tests()
    yield
    feature_loader._reset_feature_loader_for_tests()
    _reset_backend_settings_for_tests()


def _defaults() -> dict:
    response = client.get("/api/features")
    assert response.status_code == 200
    return response.json()["defaults"]


def test_predict_returns_prediction_for_valid_payload() -> None:
    response = client.post("/api/predict", json={"features": _defaults(), "confidence_level": 95})

    assert response.status_code == 200
    body = response.json()
    assert body["prediction"]["predicted_class"] in {"Resistant", "Responsive"}
    assert body["model"]["type"] == "RandomForestClassifier"
    assert body["shap_values"]
    assert "disclaimer" in body


def test_predict_accepts_camel_case_confidence_level() -> None:
    response = client.post("/api/predict", json={"features": _defaults(), "confidenceLevel": 90})

    assert response.status_code == 200


def test_predict_rejects_non_object_body() -> None:
    response = client.post("/api/predict", json=[1, 2, 3])

    assert response.status_code == 400
    assert "JSON object" in response.json()["detail"]


def test_predict_rejects_unknown_field() -> None:
    payload = _defaults()
    payload["totally_unknown"] = 1

    response = client.post("/api/predict", json=payload)

    assert response.status_code == 400
    assert "Unknown fields" in response.json()["detail"]


def test_predict_rejects_missing_feature() -> None:
    payload = _defaults()
    payload.pop(next(iter(payload)))

    response = client.post("/api/predict", json={"features": payload})

    assert response.status_code == 400
    assert "Missing required features" in response.json()["detail"]


def test_predict_rejects_out_of_range_numeric() -> None:
    payload = _defaults()
    payload["age"] = 999

    response = client.post("/api/predict", json={"features": payload})

    assert response.status_code == 400
    assert "must be between" in response.json()["detail"]


def test_predict_rejects_non_integer_numeric() -> None:
    payload = _defaults()
    payload["age"] = 40.5

    response = client.post("/api/predict", json={"features": payload})

    assert response.status_code == 400
    assert "integer value" in response.json()["detail"]


def test_predict_rejects_non_numeric_value() -> None:
    payload = _defaults()
    payload["age"] = "old"

    response = client.post("/api/predict", json={"features": payload})

    assert response.status_code == 400
    assert "must be numeric" in response.json()["detail"]


@pytest.mark.parametrize("bad_level", [79, 100, 95.5, "95", True])
def test_predict_rejects_invalid_confidence_level(bad_level) -> None:
    response = client.post("/api/predict", json={"features": _defaults(), "confidence_level": bad_level})

    assert response.status_code == 400
    assert "Confidence level" in response.json()["detail"]


def test_predict_derives_age_from_date_of_birth() -> None:
    payload = _defaults()
    payload.pop("age")

    response = client.post(
        "/api/predict",
        json={"features": payload, "date_of_birth": "2000-01-01"},
    )

    assert response.status_code == 200


def test_predict_rejects_malformed_date_of_birth() -> None:
    response = client.post(
        "/api/predict",
        json={"features": _defaults(), "date_of_birth": "01/01/2000"},
    )

    assert response.status_code == 400
    assert "date_of_birth" in response.json()["detail"]


def test_explain_returns_explanation_for_valid_payload() -> None:
    response = client.post("/api/explain", json={"features": _defaults(), "confidence_level": 95})

    assert response.status_code == 200
    body = response.json()
    assert "explanation" in body
    assert body["prediction"]["predicted_class"] in {"Resistant", "Responsive"}


def test_explain_rejects_invalid_payload() -> None:
    response = client.post("/api/explain", json={"features": {"age": 999}})

    assert response.status_code == 400
