import json
from typing import Any

import numpy as np

from ..defaults.feature_evidence import FEATURE_EVIDENCE
from ..io.feature_loader import get_feature_option_labels, get_features_by_id
from ..settings import get_backend_settings
from ..clients.client_factory import create_llm_client


TOP_FEATURES_PER_DIRECTION = 3


def format_feature_value(feature_id: str, value: Any) -> str:
    feature_option_labels = get_feature_option_labels()
    if feature_id in feature_option_labels:
        option_label = feature_option_labels[feature_id].get(value, str(value))
        if feature_id == "early_improvement":
            return "Yes" if value == 1 else "No"
        return option_label

    integer_value = int(round(float(value)))
    if feature_id == "age":
        return f"{integer_value} years"
    if feature_id == "phq9":
        return f"{integer_value}/27"
    if feature_id == "duration_months":
        return f"{integer_value} months"
    if feature_id == "previous_failures":
        return str(integer_value)
    if feature_id == "adherence_pct":
        return f"{integer_value}%"
    if feature_id.endswith("_mg"):
        return f"{integer_value} mg/day"
    return str(integer_value)


def _build_feature_item(feature_id: str, value: Any, shap_value: float) -> dict[str, Any]:
    evidence = FEATURE_EVIDENCE.get(feature_id)
    return {
        "feature_id": feature_id,
        "feature_label": get_features_by_id()[feature_id].label,
        "selected_value": format_feature_value(feature_id, value),
        "shap_value": round(float(shap_value), 4),
        "direction": "raises" if shap_value > 0 else "lowers",
        "evidence_association": (
            evidence.association
            if evidence
            else "No supporting literature entry was provided in the README for this feature."
        ),
        "evidence_pmids": evidence.pmids if evidence else [],
        "evidence_note": (
            evidence.note
            if evidence
            else "Treat this as a model-specific pattern rather than a literature-supported claim."
        ),
    }


def select_influential_features(
    values_dict: dict[str, Any],
    shap_vals: np.ndarray,
    feature_order: list[str] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    ordered_features = feature_order or list(get_features_by_id())
    ranked = sorted(
        zip(ordered_features, np.asarray(shap_vals, dtype=float)),
        key=lambda item: abs(item[1]),
        reverse=True,
    )
    positive, negative = [], []
    for feature_id, shap_value in ranked:
        if shap_value > 0 and len(positive) < TOP_FEATURES_PER_DIRECTION:
            positive.append(_build_feature_item(feature_id, values_dict[feature_id], shap_value))
        elif shap_value < 0 and len(negative) < TOP_FEATURES_PER_DIRECTION:
            negative.append(_build_feature_item(feature_id, values_dict[feature_id], shap_value))
    return {"positive": positive, "negative": negative}


def build_llm_prompt(
    values_dict: dict[str, Any],
    probability: float,
    shap_vals: np.ndarray,
    feature_order: list[str] | None = None,
) -> str:
    influential = select_influential_features(values_dict, shap_vals, feature_order)
    prompt_payload = {
        "task": "Summarize why the model predicted this resistance probability using only the supplied SHAP-based feature list and evidence notes.",
        "prediction_probability_of_resistance": round(float(probability), 4),
        "features_pushing_higher": influential["positive"],
        "features_pushing_lower": influential["negative"],
        "rules": [
            "Use plain language and keep the explanation concise.",
            "Only mention features listed in the payload.",
            "Treat SHAP direction as the source of truth for whether a feature pushed the current prediction up or down.",
            "Only cite PMIDs that appear in evidence_pmids for the same feature.",
            "Do not invent papers, PMIDs, mechanisms, or unsupported clinical facts.",
            "If the evidence note says the evidence is weak, mixed, indirect, or treatment-focused rather than predictor-focused, say that clearly.",
            "If a feature has no PMIDs, say that no supporting PMID was provided and avoid a literature-backed claim.",
            "Do not give medical advice and do not claim causality.",
            "Return markdown with: one short opening sentence, then a 'Factors pushing higher' bullet list, then a 'Factors pushing lower' bullet list.",
        ],
    }
    return json.dumps(prompt_payload, indent=2)


def fetch_llm_summary(prompt: str) -> str:
    settings = get_backend_settings()
    return create_llm_client(
        url=settings.llm_client_url,
        model=settings.llm_client_model,
        secret=settings.llm_client_secret,
    ).complete(prompt)


def generate_prediction_summary(
    values_dict: dict[str, Any],
    probability: float,
    shap_vals: np.ndarray,
    feature_order: list[str] | None = None,
) -> str:
    return fetch_llm_summary(build_llm_prompt(values_dict, probability, shap_vals, feature_order))
