import os
import pickle
from pathlib import Path
from typing import Any


MODEL_ARTIFACT_PATH_ENV = "MODEL_ARTIFACT_PATH"

_model_instance: Any | None = None
_model_source: str | None = None


def _validate_loaded_model(model: Any) -> None:
    required_attributes = ("feature_cols", "auc", "X")
    missing_attributes = [name for name in required_attributes if not hasattr(model, name)]
    if missing_attributes:
        raise RuntimeError(f"Model is missing required attributes: {', '.join(missing_attributes)}.")

    required_methods = (
        "predict_proba",
        "get_shap_values",
        "approximate_tsne_position",
        "get_conformal_prediction",
        "tsne_points",
    )
    missing_methods = [name for name in required_methods if not callable(getattr(model, name, None))]
    if missing_methods:
        raise RuntimeError(f"Model is missing required methods: {', '.join(missing_methods)}.")


def _load_model_from_disk(path: Path) -> Any:
    with path.open("rb") as handle:
        loaded_model = pickle.load(handle)
    _validate_loaded_model(loaded_model)
    return loaded_model


def _load_synthetic_model() -> Any:
    from .model import model

    _validate_loaded_model(model)
    return model


def _configured_model_path() -> str | None:
    raw_value = os.getenv(MODEL_ARTIFACT_PATH_ENV)
    if raw_value is None:
        return None
    stripped = raw_value.strip()
    return stripped or None


def get_model() -> Any:
    global _model_instance, _model_source
    if _model_instance is not None:
        return _model_instance

    configured_path = _configured_model_path()
    if configured_path is None:
        _model_instance = _load_synthetic_model()
        _model_source = "synthetic"
        return _model_instance

    model_path = Path(configured_path)
    if not model_path.exists():
        raise RuntimeError(
            f"{MODEL_ARTIFACT_PATH_ENV} is set but file does not exist: {configured_path}."
        )
    if not model_path.is_file():
        raise RuntimeError(
            f"{MODEL_ARTIFACT_PATH_ENV} must point to a file: {configured_path}."
        )

    _model_instance = _load_model_from_disk(model_path)
    _model_source = "disk"
    return _model_instance


def model_source() -> str:
    configured_path = _configured_model_path()
    if _model_source is not None:
        return _model_source
    if configured_path is None:
        return "synthetic"
    return "disk"


def _reset_model_loader_for_tests() -> None:
    global _model_instance, _model_source
    _model_instance = None
    _model_source = None

