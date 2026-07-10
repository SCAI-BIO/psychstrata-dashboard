from fastapi.testclient import TestClient

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
