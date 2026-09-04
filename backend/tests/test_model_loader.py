import pickle
from pathlib import Path

import pandas as pd
import pytest

from app.io import model_loader
from app.settings import _reset_backend_settings_for_tests


class _DummyModel:
    def __init__(self, feature_cols: list[str] | None = None) -> None:
        self.feature_cols = feature_cols or ["age"]
        self.auc = 0.9
        self.X = pd.DataFrame({"age": [1]})

    def predict_proba(self, X_row: pd.DataFrame) -> float:
        return 0.5

    def get_shap_values(self, X_row: pd.DataFrame):
        return [0.0]

    def approximate_tsne_position(self, X_row: pd.DataFrame) -> tuple[float, float]:
        return (0.0, 0.0)

    def get_conformal_prediction(self, X_row: pd.DataFrame, ci_level: int) -> dict[str, int]:
        return {"confidence_level": ci_level}

    def tsne_points(self) -> list[dict[str, float]]:
        return [{"x": 0.0, "y": 0.0}]


@pytest.fixture(autouse=True)
def _reset_model_loader(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MODEL_ARTIFACT_PATH", raising=False)
    model_loader._reset_model_loader_for_tests()
    _reset_backend_settings_for_tests()
    yield
    model_loader._reset_model_loader_for_tests()
    _reset_backend_settings_for_tests()


def test_model_loader_uses_synthetic_when_not_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    synthetic_model = object()
    monkeypatch.setattr("app.io.model_loader._load_synthetic_model", lambda: synthetic_model)

    loaded_model = model_loader.get_model()

    assert loaded_model is synthetic_model
    assert model_loader.model_source() == "synthetic"


def test_model_loader_loads_model_from_disk(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    artifact_path = tmp_path / "model.pkl"
    with artifact_path.open("wb") as handle:
        pickle.dump(_DummyModel(), handle)
    monkeypatch.setenv("MODEL_ARTIFACT_PATH", str(artifact_path))

    loaded_model = model_loader.get_model()

    assert isinstance(loaded_model, _DummyModel)
    assert model_loader.model_source() == "disk"


def test_model_loader_raises_when_configured_path_is_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MODEL_ARTIFACT_PATH", "/tmp/does-not-exist-model.pkl")

    with pytest.raises(RuntimeError, match="file does not exist"):
        model_loader.get_model()


def test_model_loader_rejects_features_absent_from_configuration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact_path = tmp_path / "model.pkl"
    with artifact_path.open("wb") as handle:
        pickle.dump(_DummyModel(["unknown_feature"]), handle)
    monkeypatch.setenv("MODEL_ARTIFACT_PATH", str(artifact_path))

    with pytest.raises(RuntimeError, match="absent from feature configuration"):
        model_loader.get_model()
