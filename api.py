from typing import Any, Dict, List

import pandas as pd
from flask import jsonify, request

from config import FEATURES_UI
from model import model


FEATURES_BY_ID = {cfg.id: cfg for cfg in FEATURES_UI}


def _pack_instance(values_dict: Dict[str, Any]) -> pd.DataFrame:
    row = {col: values_dict[col] for col in model.feature_cols}
    return pd.DataFrame([row], columns=model.feature_cols)


def _feature_schema(cfg) -> Dict[str, Any]:
    schema = {
        "id": cfg.id,
        "label": cfg.label,
        "kind": cfg.kind,
        "default": cfg.default,
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
            raise ValueError(
                f"Feature '{cfg.id}' must be between {cfg.params['min']} and {cfg.params['max']}."
            )
        return integer_value

    valid_values = {opt["value"] for opt in cfg.params["options"]}
    if raw_value not in valid_values:
        raise ValueError(f"Feature '{cfg.id}' must be one of {sorted(valid_values)}.")
    return raw_value


def _validate_features(payload: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("Request body must be a JSON object.")

    missing = [cfg.id for cfg in FEATURES_UI if cfg.id not in payload]
    if missing:
        raise ValueError(f"Missing required features: {', '.join(missing)}.")

    unknown = sorted(set(payload.keys()) - set(FEATURES_BY_ID.keys()))
    if unknown:
        raise ValueError(f"Unknown features provided: {', '.join(unknown)}.")

    return {cfg.id: _coerce_feature_value(cfg, payload[cfg.id]) for cfg in FEATURES_UI}


def _top_contributors(values_dict: Dict[str, Any], shap_values) -> Dict[str, List[Dict[str, Any]]]:
    ranked = sorted(
        zip(model.feature_cols, shap_values),
        key=lambda item: abs(float(item[1])),
        reverse=True,
    )
    positive, negative = [], []

    for feature_id, shap_value in ranked:
        entry = {
            "feature": feature_id,
            "label": FEATURES_BY_ID[feature_id].label,
            "selected_value": values_dict[feature_id],
            "shap_value": round(float(shap_value), 6),
        }
        if shap_value > 0 and len(positive) < 3:
            positive.append(entry)
        elif shap_value < 0 and len(negative) < 3:
            negative.append(entry)

        if len(positive) >= 3 and len(negative) >= 3:
            break

    return {"positive": positive, "negative": negative}


def register_api(server):
    @server.get("/api/health")
    def api_health():
        return jsonify({"status": "ok"})

    @server.get("/api/features")
    def api_features():
        return jsonify(
            {
                "features": [_feature_schema(cfg) for cfg in FEATURES_UI],
                "model_feature_order": model.feature_cols,
            }
        )

    @server.post("/api/predict")
    def api_predict():
        payload = request.get_json(silent=True)
        if payload is None:
            return jsonify({"error": "Request body must be valid JSON."}), 400
        if not isinstance(payload, dict):
            return jsonify({"error": "Request body must be a JSON object."}), 400

        features_payload = payload.get("features", payload)
        try:
            values_dict = _validate_features(features_payload)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400

        X_row = _pack_instance(values_dict)
        probability = model.predict_proba(X_row)
        shap_values = model.get_shap_values(X_row)
        conformal_label, _, _, _ = model.get_conformal_prediction(X_row, 95)

        return jsonify(
            {
                "features": values_dict,
                "prediction": {
                    "probability_resistance": round(float(probability), 6),
                    "predicted_class": "Resistant" if probability >= 0.5 else "Responsive",
                    "conformal_prediction": {
                        "confidence_level": 95,
                        "label": conformal_label,
                    },
                },
                "shap_values": {
                    feature: round(float(value), 6)
                    for feature, value in zip(model.feature_cols, shap_values)
                },
                "top_contributors": _top_contributors(values_dict, shap_values),
            }
        )
