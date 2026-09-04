import json
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError, field_validator, model_validator

from ..domain.feature import Feature, FeatureCategory
from ..settings import get_backend_settings

_BUILTIN_FEATURES_PATH = Path(__file__).resolve().parents[1] / "defaults" / "feature_definitions.json"
FEATURE_CATEGORIES: tuple[FeatureCategory, ...] = ("clinical", "medications", "adherence")
DERIVED_FEATURE_IDS = {"age"}


class _FeatureOptionInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: str = Field(min_length=1)
    value: Any


class _FeatureBaseInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    label: str = Field(min_length=1)
    default: Any
    @field_validator("id", "label")
    @classmethod
    def _strip_text(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("must not be empty.")
        return stripped


class _NumericFeatureInput(_FeatureBaseInput):
    dtype: Literal["numeric"]
    min: int | float
    max: int | float
    step: int | float = 1

    @field_validator("default", "min", "max", "step")
    @classmethod
    def _require_numeric(cls, value: Any) -> int | float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("must be numeric.")
        return value

    @model_validator(mode="after")
    def _validate_bounds(self) -> "_NumericFeatureInput":
        if self.max < self.min:
            raise ValueError("max must be greater than or equal to min.")
        if not self.min <= self.default <= self.max:
            raise ValueError("default must be within configured bounds.")
        return self


class _CategoricalFeatureInput(_FeatureBaseInput):
    dtype: Literal["categorical"]
    options: list[_FeatureOptionInput] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_options(self) -> "_CategoricalFeatureInput":
        option_values = [option.value for option in self.options]
        if self.default not in option_values:
            raise ValueError("default must match one of the configured option values.")
        return self


FeatureInput = Annotated[_NumericFeatureInput | _CategoricalFeatureInput, Field(discriminator="dtype")]
FEATURE_INPUT_LIST_ADAPTER = TypeAdapter(list[FeatureInput])


class _FeatureConfigInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    clinical: list[dict[str, Any]] = Field(min_length=1)
    medications: list[dict[str, Any]] = Field(min_length=1)
    adherence: list[dict[str, Any]] = Field(min_length=1)
    model_feature_order: list[str] = Field(min_length=1)


_features_by_category: dict[str, list[Feature]] | None = None
_features_by_id: dict[str, Feature] | None = None
_feature_defaults: dict[str, Any] | None = None
_feature_option_labels: dict[str, dict[Any, str]] | None = None
_model_feature_order: list[str] | None = None
_feature_source: str | None = None


def _configured_features_path() -> str | None:
    return get_backend_settings().features_config_path


def _to_feature(feature_input: FeatureInput, category: FeatureCategory) -> Feature:
    if isinstance(feature_input, _NumericFeatureInput):
        params = {"min": feature_input.min, "max": feature_input.max, "step": feature_input.step}
    else:
        params = {
            "options": [
                {
                    "label": option.label,
                    "value": option.value,
                }
                for option in feature_input.options
            ]
        }
    return Feature(
        id=feature_input.id,
        label=feature_input.label,
        dtype=feature_input.dtype,
        default=feature_input.default,
        params=params,
        category=category,
    )


def _load_features_from_json(path: Path) -> tuple[dict[str, list[Feature]], list[str]]:
    with path.open("r", encoding="utf-8") as handle:
        raw_payload = json.load(handle)
    try:
        payload = _FeatureConfigInput.model_validate(raw_payload)
        features_by_category = {
            category: [
                _to_feature(feature, category)
                for feature in FEATURE_INPUT_LIST_ADAPTER.validate_python(getattr(payload, category))
            ]
            for category in FEATURE_CATEGORIES
        }
    except ValidationError as exc:
        raise RuntimeError(f"Invalid feature configuration in {path}: {exc}") from exc

    all_features = [feature for category in FEATURE_CATEGORIES for feature in features_by_category[category]]
    feature_ids = [feature.id for feature in all_features]
    if len(feature_ids) != len(set(feature_ids)):
        raise RuntimeError("Feature config contains duplicate feature ids.")
    if len(payload.model_feature_order) != len(set(payload.model_feature_order)):
        raise RuntimeError("Feature config model_feature_order contains duplicate ids.")
    if set(payload.model_feature_order) != set(feature_ids):
        raise RuntimeError("Feature config model_feature_order must contain every configured feature exactly once.")
    return features_by_category, payload.model_feature_order


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

    _features_by_category, _model_feature_order = _load_features_from_json(feature_path)
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
        value = values[feature_id]
        if feature.dtype == "numeric":
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"Feature '{feature_id}' must be numeric.")
            numeric_value = float(value)
            if not numeric_value.is_integer():
                raise ValueError(f"Feature '{feature_id}' must be an integer value.")
            integer_value = int(numeric_value)
            if integer_value < feature.params["min"] or integer_value > feature.params["max"]:
                raise ValueError(
                    f"Feature '{feature_id}' must be between {feature.params['min']} and {feature.params['max']}."
                )
            value = integer_value
        elif value not in {option["value"] for option in feature.params["options"]}:
            raise ValueError(f"Feature '{feature_id}' has an invalid categorical value.")
        result[feature_id] = value
    return result


def _reset_feature_loader_for_tests() -> None:
    global _features_by_category, _features_by_id, _feature_defaults
    global _feature_option_labels, _model_feature_order, _feature_source
    _features_by_category = None
    _features_by_id = None
    _feature_defaults = None
    _feature_option_labels = None
    _model_feature_order = None
    _feature_source = None
