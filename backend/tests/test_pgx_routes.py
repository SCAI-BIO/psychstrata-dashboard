from collections.abc import Iterator

from fastapi.testclient import TestClient
import pytest

from app.main import app
from app.security.rate_limit import limiter
from app.settings import _reset_backend_settings_for_tests


@pytest.fixture()
def client(monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.setenv("BACKEND_BASIC_AUTH_USERNAME", "clinician")
    monkeypatch.setenv("BACKEND_BASIC_AUTH_PASSWORD", "password")
    monkeypatch.setenv("RATE_LIMIT_STORAGE_URI", "memory://")
    _reset_backend_settings_for_tests()
    limiter.reset()
    with TestClient(app) as test_client:
        yield test_client
    limiter.reset()
    _reset_backend_settings_for_tests()


def _auth() -> tuple[str, str]:
    return ("clinician", "password")


def _interpret(client: TestClient, payload: dict) -> dict:
    response = client.post("/api/pgx/interpret", json=payload, auth=_auth())
    assert response.status_code == 200, response.text
    return response.json()


def test_pgx_endpoints_require_basic_auth(client: TestClient) -> None:
    assert client.get("/api/pgx/options").status_code == 401
    assert client.post("/api/pgx/interpret", json={}).status_code == 401


def test_options_expose_genes_alleles_and_drugs(client: TestClient) -> None:
    response = client.get("/api/pgx/options", auth=_auth())

    assert response.status_code == 200
    body = response.json()
    genes = {gene["gene"]: gene for gene in body["genes"]}
    assert set(genes) == {"CYP2C19", "CYP2D6"}
    assert "*17" in genes["CYP2C19"]["alleles"]
    assert "*1x2" in genes["CYP2D6"]["common_alleles"]
    assert genes["CYP2D6"]["phenotypes"]
    drugs = {drug["drug"]: drug for drug in body["drugs"]}
    assert drugs["amitriptyline"]["genes"] == ["CYP2C19", "CYP2D6"]
    assert drugs["sertraline"]["genes"] == ["CYP2C19"]


def test_interpret_returns_phenotypes_and_recommendations(client: TestClient) -> None:
    body = _interpret(
        client,
        {
            "cyp2c19": {"allele1": "*1", "allele2": "*2"},
            "cyp2d6": {"allele1": "*4", "allele2": "*4"},
            "drugs": ["amitriptyline", "sertraline"],
        },
    )

    genes = {gene["gene"]: gene for gene in body["genes"]}
    assert genes["CYP2C19"]["phenotype"] == "Intermediate Metabolizer"
    assert genes["CYP2C19"]["diplotype"] == "*1/*2"
    assert genes["CYP2C19"]["activity_score"] is None
    assert genes["CYP2C19"]["priority"]
    assert genes["CYP2D6"]["phenotype"] == "Poor Metabolizer"
    assert genes["CYP2D6"]["activity_score"] == 0.0
    assert genes["CYP2D6"]["is_indeterminate"] is False

    drugs = {drug["drug"]: drug for drug in body["drug_recommendations"]}
    assert set(drugs) == {"Amitriptyline", "Sertraline"}
    assert drugs["Amitriptyline"]["status"] == "available"
    assert drugs["Amitriptyline"]["required_genes"] == ["CYP2C19", "CYP2D6"]
    assert "amitriptyline" in drugs["Amitriptyline"]["recommendation"].lower()
    assert drugs["Amitriptyline"]["cpic_level"] == "A"
    assert drugs["Amitriptyline"]["pmids"]
    assert drugs["Sertraline"]["status"] == "available"
    assert "disclaimer" in body


def test_interpret_flags_missing_second_gene(client: TestClient) -> None:
    body = _interpret(
        client,
        {"cyp2c19": {"allele1": "*1", "allele2": "*1"}, "drugs": ["amitriptyline"]},
    )

    recommendation = body["drug_recommendations"][0]
    assert recommendation["status"] == "unavailable"
    assert recommendation["recommendation"] is None
    assert "CYP2D6" in recommendation["unavailable_reason"]
    assert recommendation["provided_genes"] == ["CYP2C19"]


def test_interpret_handles_indeterminate_phenotype(client: TestClient) -> None:
    body = _interpret(
        client,
        {"cyp2d6": {"allele1": "*1", "allele2": "*22"}, "drugs": ["atomoxetine"]},
    )

    gene = body["genes"][0]
    assert gene["phenotype"] == "Indeterminate"
    assert gene["is_indeterminate"] is True
    assert gene["priority"] is None
    recommendation = body["drug_recommendations"][0]
    assert recommendation["status"] == "unavailable"
    assert recommendation["unavailable_reason"] == "No CPIC guidance for this phenotype combination"


def test_interpret_supports_structural_variants(client: TestClient) -> None:
    body = _interpret(
        client,
        {"cyp2d6": {"allele1": "*1x2", "allele2": "*4"}, "drugs": ["atomoxetine"]},
    )

    gene = body["genes"][0]
    assert gene["diplotype"] == "*1x2/*4"
    assert gene["activity_score"] == 2.0
    assert gene["phenotype"] == "Normal Metabolizer"


def test_interpret_rejects_unknown_allele(client: TestClient) -> None:
    response = client.post(
        "/api/pgx/interpret",
        json={"cyp2c19": {"allele1": "*1", "allele2": "*999"}},
        auth=_auth(),
    )

    assert response.status_code == 422
    assert "Unknown CYP2C19 allele" in response.json()["detail"]


def test_interpret_rejects_unknown_gene_and_drug(client: TestClient) -> None:
    empty = client.post("/api/pgx/interpret", json={}, auth=_auth())
    assert empty.status_code == 422
    assert "at least one gene" in empty.json()["detail"]

    bad_drug = client.post(
        "/api/pgx/interpret",
        json={"cyp2d6": {"allele1": "*1", "allele2": "*1"}, "drugs": ["lithium"]},
        auth=_auth(),
    )
    assert bad_drug.status_code == 422
    assert "Unsupported drug" in bad_drug.json()["detail"]


def test_interpret_rejects_unexpected_fields(client: TestClient) -> None:
    response = client.post(
        "/api/pgx/interpret",
        json={"cyp2d6": {"allele1": "*1", "allele2": "*1", "allele3": "*4"}},
        auth=_auth(),
    )

    assert response.status_code == 422
