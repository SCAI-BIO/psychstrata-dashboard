import json
from pathlib import Path
from typing import Any, Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError, field_validator, model_validator

from ..domain.feature import Feature
from ..settings import get_backend_settings


_BUILTIN_FEATURES_PATH = Path(__file__).resolve().parents[1] / "defaults" / "feature_definitions.json"

_features_ui: list[Feature] | None = None
_features_by_id: dict[str, Feature] | None = None
_feature_defaults: dict[str, Any] | None = None
_feature_option_labels: dict[str, dict[Any, str]] | None = None
_feature_source: str | None = None


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
    def _strip_non_empty_text(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("must not be empty.")
        return stripped


class _NumericFeatureInput(_FeatureBaseInput):
    kind: Literal["numeric"]
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
        return self


class _CategoricalFeatureInput(_FeatureBaseInput):
    kind: Literal["categorical"]
    options: list[_FeatureOptionInput] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_default_in_options(self) -> "_CategoricalFeatureInput":
        option_values = {option.value for option in self.options}
        if self.default not in option_values:
            raise ValueError("default must match one of the configured option values.")
        return self


FeatureInput = Annotated[_NumericFeatureInput | _CategoricalFeatureInput, Field(discriminator="kind")]
FEATURE_INPUT_LIST_ADAPTER = TypeAdapter(list[FeatureInput])


def _configured_features_path() -> str | None:
    return get_backend_settings().features_config_path


def _to_feature(feature_input: FeatureInput) -> Feature:
    if isinstance(feature_input, _NumericFeatureInput):
        params = {"min": feature_input.min, "max": feature_input.max, "step": feature_input.step}
        return Feature(feature_input.id, feature_input.label, "numeric", feature_input.default, params)
    params = {"options": [{"label": option.label, "value": option.value} for option in feature_input.options]}
    return Feature(feature_input.id, feature_input.label, "categorical", feature_input.default, params)


def _load_features_from_json(path: Path) -> list[Feature]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    features_rows = payload.get("features")
    try:
        feature_inputs = FEATURE_INPUT_LIST_ADAPTER.validate_python(features_rows)
    except ValidationError as exc:
        raise RuntimeError(f"Invalid feature configuration in {path}: {exc}") from exc
    if not feature_inputs:
        raise RuntimeError("Feature config must define a non-empty 'features' array.")
    features_ui = [_to_feature(feature_input) for feature_input in feature_inputs]
    feature_ids = [feature.id for feature in features_ui]
    if len(feature_ids) != len(set(feature_ids)):
        raise RuntimeError("Feature config contains duplicate feature ids.")
    return features_ui


def _materialize_feature_maps(features_ui: list[Feature]) -> tuple[
    dict[str, Feature],
    dict[str, Any],
    dict[str, dict[Any, str]],
]:
    by_id = {cfg.id: cfg for cfg in features_ui}
    defaults = {cfg.id: cfg.default for cfg in features_ui}
    option_labels = {
        cfg.id: {option["value"]: option["label"] for option in cfg.params["options"]}
        for cfg in features_ui
        if cfg.kind == "categorical"
    }
    return by_id, defaults, option_labels


def _load_defaults() -> tuple[list[Feature], dict[str, Feature], dict[str, Any], dict[str, dict[Any, str]]]:
    features_ui = _load_features_from_json(_BUILTIN_FEATURES_PATH)
    by_id, defaults, option_labels = _materialize_feature_maps(features_ui)
    return features_ui, by_id, defaults, option_labels


def _ensure_loaded() -> None:
    global _features_ui, _features_by_id, _feature_defaults, _feature_option_labels, _feature_source
    if _features_ui is not None:
        return
    configured_path = _configured_features_path()
    if configured_path is None:
        _features_ui, _features_by_id, _feature_defaults, _feature_option_labels = _load_defaults()
        _feature_source = "default"
        return
    feature_path = Path(configured_path)
    if not feature_path.exists():
        raise RuntimeError(f"FEATURES_CONFIG_PATH is set but file does not exist: {configured_path}.")
    if not feature_path.is_file():
        raise RuntimeError(f"FEATURES_CONFIG_PATH must point to a file: {configured_path}.")
    _features_ui = _load_features_from_json(feature_path)
    _features_by_id, _feature_defaults, _feature_option_labels = _materialize_feature_maps(_features_ui)
    _feature_source = "file"


def get_features_ui() -> list[Feature]:
    _ensure_loaded()
    return _features_ui or []


def get_features_by_id() -> dict[str, Feature]:
    _ensure_loaded()
    return _features_by_id or {}


def get_feature_defaults() -> dict[str, Any]:
    _ensure_loaded()
    return _feature_defaults or {}


def get_feature_option_labels() -> dict[str, dict[Any, str]]:
    _ensure_loaded()
    return _feature_option_labels or {}


def feature_source() -> str:
    configured_path = _configured_features_path()
    if _feature_source is not None:
        return _feature_source
    if configured_path is None:
        return "default"
    return "file"


def _reset_feature_loader_for_tests() -> None:
    global _features_ui, _features_by_id, _feature_defaults, _feature_option_labels, _feature_source
    _features_ui = None
    _features_by_id = None
    _feature_defaults = None
    _feature_option_labels = None
    _feature_source = None
