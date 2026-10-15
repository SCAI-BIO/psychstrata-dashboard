import pytest

from app.io import feature_loader
from app.settings import _reset_backend_settings_for_tests
from app.services import feature_service
from app.services.feature_service import (
    feature_schema,
    get_features_response,
    parse_prediction_payload,
)


@pytest.fixture(autouse=True)
def _reset(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("FEATURES_CONFIG_PATH", raising=False)
    feature_loader._reset_feature_loader_for_tests()
    _reset_backend_settings_for_tests()
    yield
    feature_loader._reset_feature_loader_for_tests()
    _reset_backend_settings_for_tests()


def test_get_features_response_shape() -> None:
    response = get_features_response()

    assert set(response) == {
        "features",
        "feature_groups",
        "defaults",
        "model_feature_order",
        "confidence_level",
        "model",
    }
    assert set(response["feature_groups"]) == {"clinical", "medications", "adherence"}
    assert response["confidence_level"] == {"default": 95, "min": 80, "max": 99, "step": 1}
    assert response["model"]["type"] == "RandomForestClassifier"
    assert response["model"]["synthetic"] is True
    assert response["model_feature_order"] == response["model"]["feature_order"]


def test_feature_schema_numeric_includes_bounds() -> None:
    numeric = feature_loader.get_features_by_id()["age"]

    schema = feature_schema(numeric)

    assert schema["dtype"] == "numeric"
    assert {"min", "max", "step"} <= set(schema)
    assert "options" not in schema


def test_feature_schema_categorical_includes_options() -> None:
    categorical = next(
        feature
        for feature in feature_loader.get_features_by_id().values()
        if feature.dtype == "categorical"
    )

    schema = feature_schema(categorical)

    assert schema["dtype"] == "categorical"
    assert schema["options"]
    assert "min" not in schema


def test_parse_prediction_payload_wraps_bare_feature_fields() -> None:
    defaults = feature_loader.get_feature_defaults()

    values, confidence = parse_prediction_payload(defaults)

    assert values == defaults
    assert confidence == 95


def test_parse_prediction_payload_accepts_explicit_features_key() -> None:
    defaults = feature_loader.get_feature_defaults()

    values, confidence = parse_prediction_payload({"features": defaults, "confidence_level": 85})

    assert values == defaults
    assert confidence == 85


def test_parse_prediction_payload_rejects_non_dict() -> None:
    with pytest.raises(ValueError, match="JSON object"):
        parse_prediction_payload(["nope"])


def test_parse_prediction_payload_rejects_unknown_fields() -> None:
    defaults = feature_loader.get_feature_defaults()
    defaults["bogus"] = 1

    with pytest.raises(ValueError, match="Unknown fields"):
        parse_prediction_payload(defaults)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (95, 95),
        (95.0, 95),
        (80, 80),
        (99, 99),
    ],
)
def test_extract_confidence_level_accepts_valid(raw: object, expected: int) -> None:
    assert feature_service._extract_confidence_level({"confidence_level": raw}) == expected


@pytest.mark.parametrize("raw", [True, False, "95", None, 95.5, 79, 100])
def test_extract_confidence_level_rejects_invalid(raw: object) -> None:
    with pytest.raises(ValueError):
        feature_service._extract_confidence_level({"confidence_level": raw})


def test_extract_confidence_level_defaults_when_absent() -> None:
    assert feature_service._extract_confidence_level({}) == 95


def test_extract_confidence_level_accepts_camel_case() -> None:
    assert feature_service._extract_confidence_level({"confidenceLevel": 88}) == 88
