from dataclasses import dataclass
from typing import Any, Literal

FeatureCategory = Literal["clinical", "medications", "adherence"]
FeatureDataType = Literal["numeric", "categorical"]


@dataclass(frozen=True)
class Feature:
    id: str
    label: str
    dtype: FeatureDataType
    default: Any
    params: dict[str, Any]
    category: FeatureCategory
