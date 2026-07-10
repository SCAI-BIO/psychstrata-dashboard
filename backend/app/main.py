import os
from typing import Any

import pandas as pd
from fastapi import Body, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .config import FEATURE_DEFAULTS, FEATURES_BY_ID, FEATURES_UI
from .llm_summary import (
    LLMServiceError,
    format_feature_value,
    generate_prediction_summary,
    select_influential_features,
)


DEFAULT_CORS_ORIGINS = ("http://localhost:3000", "http://localhost:5173")
CONFIDENCE_LEVEL_DEFAULT = 95
CONFIDENCE_LEVEL_MIN = 80
CONFIDENCE_LEVEL_MAX = 99
MODEL_FEATURE_ORDER = [cfg.id for cfg in FEATURES_UI]


class HealthResponse(BaseModel):
    status: str


class SummaryResponse(BaseModel):
    title: str
    message: str
    disclaimer: str


def get_cors_origins() -> list[str]:
    raw_origins = os.getenv("BACKEND_CORS_ORIGINS")
    if raw_origins is None:
        return list(DEFAULT_CORS_ORIGINS)

    return [origin.strip() for origin in raw_origins.split(",") if origin.strip()]


app = FastAPI(title="PsychStrata Dashboard API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_cors_origins(),
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


def _get_model():
    from .model import model

    return model


def _feature_schema(cfg) -> dict[str, Any]:
    schema = {
        "id": cfg.id,
        "label": cfg.label,
        "kind": cfg.kind,
        "default": cfg.default,
        "params": cfg.params,
    }
    if cfg.kind == "numeric":
        schema["min"] = cfg.params["min"]
        schema["max"] = cfg.params["max"]
        schema["step"] = cfg.params.get("step", 1)
    else:
        schema["options"] = cfg.params["options"]
    return schema


def _coerce_feature_value(cfg, raw_value: Any) -> Any:
    if cfg.kind == "numeric":
        if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
            raise ValueError(f"Feature '{cfg.id}' must be numeric.")

        numeric_value = float(raw_value)
        if not numeric_value.is_integer():
            raise ValueError(f"Feature '{cfg.id}' must be an integer value.")

        integer_value = int(numeric_value)
        if integer_value < cfg.params["min"] or integer_value > cfg.params["max"]:
            raise ValueError(f"Feature '{cfg.id}' must be between {cfg.params['min']} and {cfg.params['max']}.")
        return integer_value

    valid_values = {option["value"] for option in cfg.params["options"]}
    if raw_value not in valid_values:
        raise ValueError(f"Feature '{cfg.id}' must be one of {sorted(valid_values)}.")
    return raw_value


def _validate_features(features_payload: Any) -> dict[str, Any]:
    if not isinstance(features_payload, dict):
        raise ValueError("Request field 'features' must be a JSON object.")

    missing = [cfg.id for cfg in FEATURES_UI if cfg.id not in features_payload]
    if missing:
        raise ValueError(f"Missing required features: {', '.join(missing)}.")

    unknown = sorted(set(features_payload.keys()) - set(FEATURES_BY_ID.keys()))
    if unknown:
        raise ValueError(f"Unknown features provided: {', '.join(unknown)}.")

    return {cfg.id: _coerce_feature_value(cfg, features_payload[cfg.id]) for cfg in FEATURES_UI}


def _extract_features_payload(payload: dict[str, Any]) -> Any:
    if "features" in payload:
        return payload["features"]

    allowed_control_keys = {"confidence_level", "confidenceLevel"}
    unknown = sorted(set(payload.keys()) - set(FEATURES_BY_ID.keys()) - allowed_control_keys)
    if unknown:
        raise ValueError(f"Unknown fields provided: {', '.join(unknown)}.")

    return {feature_id: payload[feature_id] for feature_id in FEATURES_BY_ID if feature_id in payload}


def _extract_confidence_level(payload: dict[str, Any]) -> int:
    raw_level = payload.get("confidence_level", payload.get("confidenceLevel", CONFIDENCE_LEVEL_DEFAULT))
    if isinstance(raw_level, bool) or not isinstance(raw_level, (int, float)):
        raise ValueError("Confidence level must be numeric.")

    numeric_level = float(raw_level)
    if not numeric_level.is_integer():
        raise ValueError("Confidence level must be an integer value.")

    ci_level = int(numeric_level)
    if ci_level < CONFIDENCE_LEVEL_MIN or ci_level > CONFIDENCE_LEVEL_MAX:
        raise ValueError(
            f"Confidence level must be between {CONFIDENCE_LEVEL_MIN} and {CONFIDENCE_LEVEL_MAX}."
        )
    return ci_level


def _pack_instance(values_dict: dict[str, Any], feature_cols: list[str]) -> pd.DataFrame:
    row = {col: values_dict[col] for col in feature_cols}
    return pd.DataFrame([row], columns=feature_cols)


def _shap_entries(values_dict: dict[str, Any], shap_values, feature_cols: list[str]) -> list[dict[str, Any]]:
    entries = []
    for feature_id, shap_value in zip(feature_cols, shap_values):
        rounded_value = round(float(shap_value), 6)
        entries.append(
            {
                "feature_id": feature_id,
                "feature_label": FEATURES_BY_ID[feature_id].label,
                "selected_value": values_dict[feature_id],
                "selected_value_label": format_feature_value(feature_id, values_dict[feature_id]),
                "shap_value": rounded_value,
                "abs_shap_value": round(abs(float(shap_value)), 6),
                "direction": "raises" if shap_value > 0 else "lowers" if shap_value < 0 else "neutral",
            }
        )

    return sorted(entries, key=lambda entry: entry["abs_shap_value"], reverse=True)


def _build_prediction_response(values_dict: dict[str, Any], confidence_level: int) -> dict[str, Any]:
    treatment_model = _get_model()
    X_row = _pack_instance(values_dict, treatment_model.feature_cols)
    probability = treatment_model.predict_proba(X_row)
    shap_values = treatment_model.get_shap_values(X_row)
    selected_x, selected_y = treatment_model.approximate_tsne_position(X_row)

    return {
        "features": values_dict,
        "prediction": {
            "probability_resistance": round(float(probability), 6),
            "predicted_class": "Resistant" if probability >= 0.5 else "Responsive",
            "conformal_prediction": treatment_model.get_conformal_prediction(X_row, confidence_level),
        },
        "shap_values": _shap_entries(values_dict, shap_values, treatment_model.feature_cols),
        "top_contributors": select_influential_features(values_dict, shap_values),
        "tsne": {
            "selected": {"x": round(selected_x, 4), "y": round(selected_y, 4)},
        },
        "model": {
            "type": "RandomForestClassifier",
            "auc": round(treatment_model.auc, 6),
            "feature_order": treatment_model.feature_cols,
            "training_rows": len(treatment_model.X),
            "synthetic": True,
        },
        "disclaimer": (
            "This demo uses synthetic data for illustration purposes only. "
            "It is not a medical device and must not be used for diagnosis or treatment decisions."
        ),
    }


def _parse_prediction_payload(payload: Any) -> tuple[dict[str, Any], int]:
    if not isinstance(payload, dict):
        raise ValueError("Request body must be a JSON object.")

    features_payload = _extract_features_payload(payload)
    values_dict = _validate_features(features_payload)
    confidence_level = _extract_confidence_level(payload)
    return values_dict, confidence_level


@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.get("/api/summary", response_model=SummaryResponse)
def summary() -> SummaryResponse:
    return SummaryResponse(
        title="Treatment Resistance Classifier Demo",
        message="The React frontend is connected to the FastAPI backend.",
        disclaimer=(
            "This demo uses synthetic data for illustration purposes only. "
            "It is not a medical device and must not be used for clinical decisions."
        ),
    )


@app.get("/api/features")
def features() -> dict[str, Any]:
    return {
        "features": [_feature_schema(cfg) for cfg in FEATURES_UI],
        "defaults": FEATURE_DEFAULTS,
        "model_feature_order": MODEL_FEATURE_ORDER,
        "confidence_level": {
            "default": CONFIDENCE_LEVEL_DEFAULT,
            "min": CONFIDENCE_LEVEL_MIN,
            "max": CONFIDENCE_LEVEL_MAX,
            "step": 1,
        },
        "model": {
            "type": "RandomForestClassifier",
            "feature_order": MODEL_FEATURE_ORDER,
            "synthetic": True,
        },
    }


@app.post("/api/predict")
def predict(payload: Any = Body(...)) -> dict[str, Any]:
    try:
        values_dict, confidence_level = _parse_prediction_payload(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return _build_prediction_response(values_dict, confidence_level)


@app.post("/api/explain")
def explain(payload: Any = Body(...)) -> dict[str, Any]:
    try:
        values_dict, confidence_level = _parse_prediction_payload(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    treatment_model = _get_model()
    X_row = _pack_instance(values_dict, treatment_model.feature_cols)
    probability = treatment_model.predict_proba(X_row)
    shap_values = treatment_model.get_shap_values(X_row)
    try:
        explanation = generate_prediction_summary(values_dict, probability, shap_values)
    except LLMServiceError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return {
        "features": values_dict,
        "prediction": {
            "probability_resistance": round(float(probability), 6),
            "predicted_class": "Resistant" if probability >= 0.5 else "Responsive",
            "conformal_prediction": treatment_model.get_conformal_prediction(X_row, confidence_level),
        },
        "top_contributors": select_influential_features(values_dict, shap_values),
        "explanation": explanation,
    }


@app.get("/api/tsne")
def tsne() -> dict[str, Any]:
    points = _get_model().tsne_points()
    return {
        "points": points,
        "classes": [
            {"value": 0, "label": "Responsive"},
            {"value": 1, "label": "Resistant"},
        ],
        "model": {
            "source": "synthetic_training_population",
            "rows": len(points),
        },
    }
