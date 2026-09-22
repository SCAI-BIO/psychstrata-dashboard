import json
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from ..domain.feature import Feature, FeatureCategory


FEATURE_CATEGORIES: tuple[FeatureCategory, ...] = ("clinical", "medications", "adherence")


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
        if self.default not in [option.value for option in self.options]:
            raise ValueError("default must match one of the configured option values.")
        return self


FeatureInput = Annotated[_NumericFeatureInput | _CategoricalFeatureInput, Field(discriminator="dtype")]


class _FeatureConfigInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    clinical: list[FeatureInput] = Field(min_length=1)
    medications: list[FeatureInput] = Field(min_length=1)
    adherence: list[FeatureInput] = Field(min_length=1)
    model_feature_order: list[str] = Field(min_length=1)


def _to_feature(feature_input: FeatureInput, category: FeatureCategory) -> Feature:
    params = (
        {"min": feature_input.min, "max": feature_input.max, "step": feature_input.step}
        if isinstance(feature_input, _NumericFeatureInput)
        else {"options": [{"label": option.label, "value": option.value} for option in feature_input.options]}
    )
    return Feature(
        id=feature_input.id,
        label=feature_input.label,
        dtype=feature_input.dtype,
        default=feature_input.default,
        params=params,
        category=category,
    )


def load_feature_config(path: Path) -> tuple[dict[str, list[Feature]], list[str]]:
    with path.open("r", encoding="utf-8") as handle:
        raw_payload = json.load(handle)
    try:
        payload = _FeatureConfigInput.model_validate(raw_payload)
        features_by_category = {
            category: [_to_feature(feature, category) for feature in getattr(payload, category)]
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
