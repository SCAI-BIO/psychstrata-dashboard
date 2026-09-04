from collections.abc import Iterator
from datetime import date

from fastapi.testclient import TestClient
import pytest

from app.persistence.database import _reset_database_for_tests
from app.io.feature_loader import _reset_feature_loader_for_tests, get_model_feature_order
from app.main import app
from app.settings import _reset_backend_settings_for_tests
from app.utils.datetime import age_on_date


@pytest.fixture()
def client(monkeypatch: pytest.MonkeyPatch, tmp_path) -> Iterator[TestClient]:
    monkeypatch.setenv("BACKEND_DATABASE_URL", f"sqlite:///{tmp_path / 'test.sqlite3'}")
    monkeypatch.setenv("BACKEND_BASIC_AUTH_USERNAME", "clinician")
    monkeypatch.setenv("BACKEND_BASIC_AUTH_PASSWORD", "password")
    _reset_backend_settings_for_tests()
    _reset_feature_loader_for_tests()
    _reset_database_for_tests()
    with TestClient(app) as test_client:
        yield test_client
    _reset_backend_settings_for_tests()
    _reset_feature_loader_for_tests()
    _reset_database_for_tests()


def _auth() -> tuple[str, str]:
    return ("clinician", "password")


def _patient_payload(sex_at_birth: int = 2) -> dict:
    return {
        "first_name": "Ada",
        "last_name": "Lovelace",
        "clinical_data": {
            "date_of_birth": "1985-12-10",
            "diagnosis": "F33.1",
            "clinical_features": {
                "sex_at_birth": sex_at_birth,
                "phq9": 18,
                "duration_months": 8,
                "previous_failures": 1,
                "sleep_severity": 1,
                "substance_use": 0,
                "comorbid_anxiety": 1,
            },
            "genetics": {"available": True},
            "proteomics": {"available": False},
        },
    }


def _create_patient(client: TestClient) -> dict:
    response = client.post("/api/patients", json=_patient_payload(), auth=_auth())
    assert response.status_code == 201
    return response.json()


def _create_treatment_plan(client: TestClient, patient_id: str) -> dict:
    response = client.post(
        f"/api/patients/{patient_id}/treatment-plans",
        json={
            "medications": {
                "sertraline_mg": 100,
                "quetiapine_mg": 0,
                "lithium_mg": 300,
            },
            "adherence": {
                "adherence_pct": 85,
                "early_improvement": 1,
                "side_effects": 1,
            },
        },
        auth=_auth(),
    )
    assert response.status_code == 201
    return response.json()


def test_patient_endpoints_require_basic_auth(client: TestClient) -> None:
    response = client.post("/api/patients", json=_patient_payload())

    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid credentials."


def test_patient_aggregate_crud_and_config_defaults(client: TestClient) -> None:
    created = _create_patient(client)

    assert created["clinician_id"] == "clinician"
    assert created["clinical_data"]["clinical_features"]["sex_at_birth"] == 2
    assert created["clinical_data"]["genetics"] == {"available": True}

    list_response = client.get("/api/patients", auth=_auth())
    assert list_response.status_code == 200
    assert [patient["id"] for patient in list_response.json()] == [created["id"]]

    update_response = client.patch(
        f"/api/patients/{created['id']}",
        json={"last_name": "Byron"},
        auth=_auth(),
    )
    assert update_response.status_code == 200
    assert update_response.json()["last_name"] == "Byron"


def test_rejects_unknown_and_out_of_range_features(client: TestClient) -> None:
    unknown_payload = _patient_payload()
    unknown_payload["clinical_data"]["clinical_features"]["unknown"] = 1
    unknown_response = client.post("/api/patients", json=unknown_payload, auth=_auth())
    assert unknown_response.status_code == 422
    assert "Unknown clinical values" in unknown_response.json()["detail"]

    invalid_payload = _patient_payload()
    invalid_payload["clinical_data"]["clinical_features"]["previous_failures"] = 10
    invalid_response = client.post("/api/patients", json=invalid_payload, auth=_auth())
    assert invalid_response.status_code == 422
    assert "between 0 and 5" in invalid_response.json()["detail"]


def test_clinical_data_is_immutable(client: TestClient) -> None:
    patient = _create_patient(client)

    response = client.patch(
        f"/api/patients/{patient['id']}",
        json={"clinical_data": {"diagnosis": "F33.2"}},
        auth=_auth(),
    )

    assert response.status_code == 422
    assert response.json()["detail"][0]["type"] == "extra_forbidden"


def test_treatment_plan_crud_and_persisted_prediction(client: TestClient) -> None:
    patient = _create_patient(client)
    treatment_plan = _create_treatment_plan(client, patient["id"])

    assert treatment_plan["medications"]["lithium_mg"] == 300
    assert treatment_plan["adherence"]["adherence_pct"] == 85

    update_response = client.patch(
        f"/api/treatment-plans/{treatment_plan['id']}",
        json={
            "medications": {"lithium_mg": 600},
            "adherence": {"side_effects": 2},
        },
        auth=_auth(),
    )
    assert update_response.status_code == 200
    assert update_response.json()["medications"]["lithium_mg"] == 600
    assert update_response.json()["medications"]["sertraline_mg"] == 100
    assert update_response.json()["adherence"]["side_effects"] == 2

    prediction_response = client.post(
        f"/api/treatment-plans/{treatment_plan['id']}/predict",
        auth=_auth(),
    )
    assert prediction_response.status_code == 200
    features = prediction_response.json()["features"]
    assert list(features) == get_model_feature_order()
    assert features["sex_at_birth"] == 2
    assert features["lithium_mg"] == 600
    assert features["age"] == age_on_date(date(1985, 12, 10))


def test_deleting_patient_cascades_aggregate(client: TestClient) -> None:
    patient = _create_patient(client)
    treatment_plan = _create_treatment_plan(client, patient["id"])

    delete_response = client.delete(f"/api/patients/{patient['id']}", auth=_auth())
    assert delete_response.status_code == 204

    assert client.get(f"/api/patients/{patient['id']}", auth=_auth()).status_code == 404
    assert client.get(f"/api/treatment-plans/{treatment_plan['id']}", auth=_auth()).status_code == 404


def test_age_is_derived_from_date_of_birth() -> None:
    assert age_on_date(date(2000, 9, 5), date(2026, 9, 4)) == 25
    assert age_on_date(date(2000, 9, 4), date(2026, 9, 4)) == 26
