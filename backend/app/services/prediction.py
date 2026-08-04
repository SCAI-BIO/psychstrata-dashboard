import logging
import os
from datetime import datetime, timezone
from typing import Any

import pandas as pd
from fastapi import HTTPException

from ..config import FEATURES_BY_ID
from ..llm_summary import (
    LLMServiceError,
    format_feature_value,
    generate_prediction_summary,
    select_influential_features,
)
from ..model_loader import get_model, model_source


EXPLAIN_GLOBAL_DAILY_CAP = int(os.getenv("EXPLAIN_GLOBAL_DAILY_CAP", "5000"))
_explain_usage = {"day": "", "count": 0}

logger = logging.getLogger("psychstrata.api")


def _check_global_cap() -> None:
    if EXPLAIN_GLOBAL_DAILY_CAP <= 0:  # 0 = deactivated
        return
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if _explain_usage["day"] != today:
        _explain_usage.update(day=today, count=0)
    _explain_usage["count"] += 1
    if _explain_usage["count"] > EXPLAIN_GLOBAL_DAILY_CAP:
        logger.warning("Global explain cap reached (%s)", EXPLAIN_GLOBAL_DAILY_CAP)
        raise HTTPException(
            status_code=503,
            detail="Daily demo budget reached. Please try again tomorrow.",
            headers={"Retry-After": "3600"},
        )


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


def build_prediction_response(values_dict: dict[str, Any], confidence_level: int) -> dict[str, Any]:
    treatment_model = get_model()
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
            "synthetic": model_source() == "synthetic",
        },
        "disclaimer": (
            "This demo uses synthetic data for illustration purposes only. "
            "It is not a medical device and must not be used for diagnosis or treatment decisions."
        ),
    }


def build_explanation_response(values_dict: dict[str, Any], confidence_level: int) -> dict[str, Any]:
    _check_global_cap()

    treatment_model = get_model()
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


def get_tsne_response() -> dict[str, Any]:
    points = get_model().tsne_points()
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
