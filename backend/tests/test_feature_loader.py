import json
from datetime import date
from pathlib import Path

import pytest

from app.io import feature_loader
from app.services.features import get_features_response, parse_prediction_payload
from app.settings import _reset_backend_settings_for_tests


@pytest.fixture(autouse=True)
def _reset_feature_loader(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("FEATURES_CONFIG_PATH", raising=False)
    feature_loader._reset_feature_loader_for_tests()
    _reset_backend_settings_for_tests()
    yield
    feature_loader._reset_feature_loader_for_tests()
    _reset_backend_settings_for_tests()


def test_feature_loader_uses_default_config_when_not_configured() -> None:
    response = get_features_response()

    assert response["features"]
    assert all("params" not in feature for feature in response["features"])
    assert all(
        "params" not in feature
        for features in response["feature_groups"].values()
        for feature in features
    )
    assert feature_loader.feature_source() == "default"


def test_feature_loader_loads_json_config_from_disk(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_path = tmp_path / "features.json"
    config_path.write_text(
        json.dumps(
            {
                "clinical": [
                    {
                        "id": "age",
                        "label": "Age custom",
                        "dtype": "numeric",
                        "default": 50,
                        "min": 18,
                        "max": 80,
                        "step": 1,
                    },
                    {
                        "id": "sex_at_birth",
                        "label": "Sex custom",
                        "dtype": "categorical",
                        "default": 0,
                        "options": [
                            {"label": "Male", "value": 0},
                            {"label": "Female", "value": 1},
                        ],
                    },
                ],
                "medications": [
                    {
                        "id": "sertraline_mg",
                        "label": "Sertraline",
                        "dtype": "numeric",
                        "default": 0,
                        "min": 0,
                        "max": 200,
                    }
                ],
                "adherence": [
                    {
                        "id": "adherence_pct",
                        "label": "Adherence",
                        "dtype": "numeric",
                        "default": 80,
                        "min": 0,
                        "max": 100,
                    }
                ],
                "model_feature_order": ["age", "sex_at_birth", "sertraline_mg", "adherence_pct"],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("FEATURES_CONFIG_PATH", str(config_path))

    response = get_features_response()

    assert feature_loader.feature_source() == "file"
    assert response["features"][0]["label"] == "Age custom"
    assert response["defaults"]["age"] == 50
    assert response["feature_groups"]["medications"][0]["id"] == "sertraline_mg"


def test_feature_loader_raises_when_path_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FEATURES_CONFIG_PATH", "/tmp/does-not-exist-features.json")

    with pytest.raises(RuntimeError, match="file does not exist"):
        get_features_response()


def test_prediction_payload_derives_age_from_date_of_birth() -> None:
    response = get_features_response()

    features, _ = parse_prediction_payload(
        {
            "features": response["defaults"],
            "date_of_birth": "2000-01-01",
        }
    )

    assert features["age"] == date.today().year - 2000
