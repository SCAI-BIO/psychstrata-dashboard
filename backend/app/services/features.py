from typing import Any

from ..io.feature_loader import get_feature_defaults, get_features_by_id, get_features_ui
from ..io.model_loader import model_source


CONFIDENCE_LEVEL_DEFAULT = 95
CONFIDENCE_LEVEL_MIN = 80
CONFIDENCE_LEVEL_MAX = 99


def feature_schema(cfg) -> dict[str, Any]:
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


def get_features_response() -> dict[str, Any]:
    features_ui = get_features_ui()
    feature_defaults = get_feature_defaults()
    model_feature_order = [cfg.id for cfg in features_ui]
    return {
        "features": [feature_schema(cfg) for cfg in features_ui],
        "defaults": feature_defaults,
        "model_feature_order": model_feature_order,
        "confidence_level": {
            "default": CONFIDENCE_LEVEL_DEFAULT,
            "min": CONFIDENCE_LEVEL_MIN,
            "max": CONFIDENCE_LEVEL_MAX,
            "step": 1,
        },
        "model": {
            "type": "RandomForestClassifier",
            "feature_order": model_feature_order,
            "synthetic": model_source() == "synthetic",
        },
    }


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
    features_ui = get_features_ui()
    features_by_id = get_features_by_id()
    if not isinstance(features_payload, dict):
        raise ValueError("Request field 'features' must be a JSON object.")

    missing = [cfg.id for cfg in features_ui if cfg.id not in features_payload]
    if missing:
        raise ValueError(f"Missing required features: {', '.join(missing)}.")

    unknown = sorted(set(features_payload.keys()) - set(features_by_id.keys()))
    if unknown:
        raise ValueError(f"Unknown features provided: {', '.join(unknown)}.")

    return {cfg.id: _coerce_feature_value(cfg, features_payload[cfg.id]) for cfg in features_ui}


def _extract_features_payload(payload: dict[str, Any]) -> Any:
    features_by_id = get_features_by_id()
    if "features" in payload:
        return payload["features"]

    allowed_control_keys = {"confidence_level", "confidenceLevel"}
    unknown = sorted(set(payload.keys()) - set(features_by_id.keys()) - allowed_control_keys)
    if unknown:
        raise ValueError(f"Unknown fields provided: {', '.join(unknown)}.")

    return {feature_id: payload[feature_id] for feature_id in features_by_id if feature_id in payload}


def _extract_confidence_level(payload: dict[str, Any]) -> int:
    raw_level = payload.get("confidence_level", payload.get("confidenceLevel", CONFIDENCE_LEVEL_DEFAULT))
    if isinstance(raw_level, bool) or not isinstance(raw_level, (int, float)):
        raise ValueError("Confidence level must be numeric.")

    numeric_level = float(raw_level)
    if not numeric_level.is_integer():
        raise ValueError("Confidence level must be an integer value.")

    ci_level = int(numeric_level)
    if ci_level < CONFIDENCE_LEVEL_MIN or ci_level > CONFIDENCE_LEVEL_MAX:
        raise ValueError(f"Confidence level must be between {CONFIDENCE_LEVEL_MIN} and {CONFIDENCE_LEVEL_MAX}.")
    return ci_level


def parse_prediction_payload(payload: Any) -> tuple[dict[str, Any], int]:
    if not isinstance(payload, dict):
        raise ValueError("Request body must be a JSON object.")

    features_payload = _extract_features_payload(payload)
    values_dict = _validate_features(features_payload)
    confidence_level = _extract_confidence_level(payload)
    return values_dict, confidence_level
