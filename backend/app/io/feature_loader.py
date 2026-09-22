from pathlib import Path
from typing import Any

from ..domain.feature import Feature, FeatureCategory
from ..settings import get_backend_settings
from .feature_config import FEATURE_CATEGORIES, load_feature_config

_BUILTIN_FEATURES_PATH = Path(__file__).resolve().parents[1] / "defaults" / "feature_definitions.json"
DERIVED_FEATURE_IDS = {"age"}


_features_by_category: dict[str, list[Feature]] | None = None
_features_by_id: dict[str, Feature] | None = None
_feature_defaults: dict[str, Any] | None = None
_feature_option_labels: dict[str, dict[Any, str]] | None = None
_model_feature_order: list[str] | None = None
_feature_source: str | None = None


def _configured_features_path() -> str | None:
    return get_backend_settings().features_config_path


def _ensure_loaded() -> None:
    global _features_by_category, _features_by_id, _feature_defaults
    global _feature_option_labels, _model_feature_order, _feature_source
    if _features_by_category is not None:
        return
    configured_path = _configured_features_path()
    feature_path = _BUILTIN_FEATURES_PATH if configured_path is None else Path(configured_path)
    if not feature_path.exists():
        raise RuntimeError(f"FEATURES_CONFIG_PATH is set but file does not exist: {configured_path}.")
    if not feature_path.is_file():
        raise RuntimeError(f"FEATURES_CONFIG_PATH must point to a file: {configured_path}.")

    _features_by_category, _model_feature_order = load_feature_config(feature_path)
    all_features = [feature for category in FEATURE_CATEGORIES for feature in _features_by_category[category]]
    _features_by_id = {feature.id: feature for feature in all_features}
    _feature_defaults = {feature.id: feature.default for feature in all_features}
    _feature_option_labels = {
        feature.id: {option["value"]: option["label"] for option in feature.params["options"]}
        for feature in all_features
        if feature.dtype == "categorical"
    }
    _feature_source = "default" if configured_path is None else "file"


def get_features_by_category(category: FeatureCategory) -> list[Feature]:
    if category not in FEATURE_CATEGORIES:
        raise ValueError(f"Unknown feature category: {category}.")
    _ensure_loaded()
    return list((_features_by_category or {})[category])


def get_features_ui() -> list[Feature]:
    _ensure_loaded()
    features_by_id = _features_by_id or {}
    return [features_by_id[feature_id] for feature_id in (_model_feature_order or [])]


def get_model_feature_order() -> list[str]:
    _ensure_loaded()
    return list(_model_feature_order or [])


def get_features_by_id() -> dict[str, Feature]:
    _ensure_loaded()
    return dict(_features_by_id or {})


def get_feature_defaults() -> dict[str, Any]:
    _ensure_loaded()
    return dict(_feature_defaults or {})


def get_feature_option_labels() -> dict[str, dict[Any, str]]:
    _ensure_loaded()
    return dict(_feature_option_labels or {})


def feature_source() -> str:
    _ensure_loaded()
    return _feature_source or "default"


def _coerce_feature_value(feature: Feature, value: Any) -> Any:
    if feature.dtype == "numeric":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"Feature '{feature.id}' must be numeric.")
        numeric_value = float(value)
        if not numeric_value.is_integer():
            raise ValueError(f"Feature '{feature.id}' must be an integer value.")
        integer_value = int(numeric_value)
        if integer_value < feature.params["min"] or integer_value > feature.params["max"]:
            raise ValueError(
                f"Feature '{feature.id}' must be between {feature.params['min']} and {feature.params['max']}."
            )
        return integer_value

    valid_values = {option["value"] for option in feature.params["options"]}
    if value not in valid_values:
        raise ValueError(f"Feature '{feature.id}' has an invalid categorical value.")
    return value


def validate_feature_values(
    category: FeatureCategory,
    values: dict[str, Any],
    *,
    include_defaults: bool = False,
    exclude_derived: bool = True,
) -> dict[str, Any]:
    configured = get_features_by_category(category)
    allowed = {
        feature.id: feature
        for feature in configured
        if not (exclude_derived and feature.id in DERIVED_FEATURE_IDS)
    }
    unknown = sorted(set(values) - set(allowed))
    if unknown:
        raise ValueError(f"Unknown {category} values: {', '.join(unknown)}.")

    result: dict[str, Any] = {}
    for feature_id, feature in allowed.items():
        if feature_id not in values:
            if include_defaults:
                result[feature_id] = feature.default
            continue
        result[feature_id] = _coerce_feature_value(feature, values[feature_id])
    return result


def validate_model_feature_values(values: dict[str, Any]) -> dict[str, Any]:
    _ensure_loaded()
    features_by_id = _features_by_id or {}
    missing = [feature_id for feature_id in (_model_feature_order or []) if feature_id not in values]
    if missing:
        raise ValueError(f"Missing required features: {', '.join(missing)}.")
    unknown = sorted(set(values) - set(features_by_id))
    if unknown:
        raise ValueError(f"Unknown features provided: {', '.join(unknown)}.")
    return {
        feature_id: _coerce_feature_value(features_by_id[feature_id], values[feature_id])
        for feature_id in (_model_feature_order or [])
    }


def _reset_feature_loader_for_tests() -> None:
    global _features_by_category, _features_by_id, _feature_defaults
    global _feature_option_labels, _model_feature_order, _feature_source
    _features_by_category = None
    _features_by_id = None
    _feature_defaults = None
    _feature_option_labels = None
    _model_feature_order = None
    _feature_source = None
