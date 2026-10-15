import json
from pathlib import Path

import pytest

from app.io.feature_config import load_feature_config


def _valid_config() -> dict:
    return {
        "clinical": [
            {"id": "age", "label": "Age", "dtype": "numeric", "default": 40, "min": 18, "max": 80, "step": 1},
            {"id": "sex", "label": "Sex", "dtype": "categorical", "default": 0, "options": [{"label": "Male", "value": 0}, {"label": "Female", "value": 1}]},
        ],
        "medications": [
            {"id": "sertraline", "label": "Sertraline", "dtype": "numeric", "default": 100, "min": 0, "max": 200, "step": 5},
        ],
        "adherence": [
            {"id": "adherence", "label": "Adherence", "dtype": "numeric", "default": 80, "min": 0, "max": 100, "step": 1},
        ],
        "model_feature_order": ["age", "sex", "sertraline", "adherence"],
    }


def _write(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "features.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_load_feature_config_round_trip(tmp_path: Path) -> None:
    path = _write(tmp_path, _valid_config())

    features_by_category, order = load_feature_config(path)

    assert set(features_by_category) == {"clinical", "medications", "adherence"}
    assert order == ["age", "sex", "sertraline", "adherence"]
    age = features_by_category["clinical"][0]
    assert age.params == {"min": 18, "max": 80, "step": 1}
    sex = features_by_category["clinical"][1]
    assert sex.params == {"options": [{"label": "Male", "value": 0}, {"label": "Female", "value": 1}]}
    assert sex.category == "clinical"


def test_load_feature_config_wraps_validation_error(tmp_path: Path) -> None:
    payload = _valid_config()
    payload["clinical"][0]["default"] = 999

    path = _write(tmp_path, payload)

    with pytest.raises(RuntimeError, match="Invalid feature configuration"):
        load_feature_config(path)


def test_load_feature_config_rejects_unknown_top_level_key(tmp_path: Path) -> None:
    payload = _valid_config()
    payload["unexpected"] = []

    path = _write(tmp_path, payload)

    with pytest.raises(RuntimeError, match="Invalid feature configuration"):
        load_feature_config(path)


def test_load_feature_config_rejects_duplicate_feature_ids(tmp_path: Path) -> None:
    payload = _valid_config()
    payload["medications"].append(
        {"id": "age", "label": "Dup", "dtype": "numeric", "default": 40, "min": 18, "max": 80}
    )
    payload["model_feature_order"] = ["age", "sex", "age", "adherence"]

    path = _write(tmp_path, payload)

    with pytest.raises(RuntimeError, match="duplicate feature ids"):
        load_feature_config(path)


def test_load_feature_config_rejects_duplicate_model_order(tmp_path: Path) -> None:
    payload = _valid_config()
    payload["model_feature_order"] = ["age", "sex", "sertraline", "age"]

    path = _write(tmp_path, payload)

    with pytest.raises(RuntimeError, match="model_feature_order contains duplicate ids"):
        load_feature_config(path)


def test_load_feature_config_requires_all_features_in_model_order(tmp_path: Path) -> None:
    payload = _valid_config()
    payload["model_feature_order"] = ["age", "sex", "sertraline"]

    path = _write(tmp_path, payload)

    with pytest.raises(RuntimeError, match="every configured feature exactly once"):
        load_feature_config(path)


def test_load_feature_config_rejects_categorical_default_not_in_options(tmp_path: Path) -> None:
    payload = _valid_config()
    payload["clinical"][1]["default"] = 9

    path = _write(tmp_path, payload)

    with pytest.raises(RuntimeError, match="Invalid feature configuration"):
        load_feature_config(path)


def test_load_feature_config_rejects_non_numeric_numeric_default(tmp_path: Path) -> None:
    payload = _valid_config()
    payload["clinical"][0]["default"] = "forty"

    path = _write(tmp_path, payload)

    with pytest.raises(RuntimeError, match="Invalid feature configuration"):
        load_feature_config(path)
