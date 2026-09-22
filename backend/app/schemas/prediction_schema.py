from typing import Any, Literal

from pydantic import BaseModel


class ConformalPrediction(BaseModel):
    confidence_level: int
    alpha: float
    label: Literal["Responsive", "Resistant", "Uncertain"]
    included_classes: list[str]


class ShapValue(BaseModel):
    feature_id: str
    feature_label: str
    selected_value: float
    selected_value_label: str
    shap_value: float
    abs_shap_value: float
    direction: Literal["raises", "lowers", "neutral"]


class Contributor(BaseModel):
    feature_id: str
    feature_label: str
    selected_value: str
    shap_value: float
    direction: Literal["raises", "lowers"]


class PredictionFields(BaseModel):
    probability_resistance: float
    predicted_class: Literal["Responsive", "Resistant"]
    conformal_prediction: ConformalPrediction


class PredictionModelMetadata(BaseModel):
    type: str
    auc: float
    feature_order: list[str]
    training_rows: int
    synthetic: bool


class SelectedTsnePosition(BaseModel):
    x: float
    y: float


class TsnePoint(BaseModel):
    x: float
    y: float
    class_value: Literal[0, 1]
    class_label: Literal["Responsive", "Resistant"]


class TsneModelMetadata(BaseModel):
    source: str
    rows: int


class PredictionResponse(BaseModel):
    features: dict[str, float]
    prediction: PredictionFields
    shap_values: list[ShapValue]
    top_contributors: dict[str, list[Contributor]]
    tsne: dict[str, SelectedTsnePosition]
    model: PredictionModelMetadata
    disclaimer: str


class ExplainResponse(BaseModel):
    features: dict[str, float]
    prediction: PredictionFields
    top_contributors: dict[str, list[Contributor]]
    explanation: str


class TsneResponse(BaseModel):
    points: list[TsnePoint]
    classes: list[dict[str, Any]]
    model: TsneModelMetadata
