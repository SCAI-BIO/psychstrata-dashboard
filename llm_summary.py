import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List
from urllib import error, request

import numpy as np

from config import FEATURES_UI


OPENAI_API_URL = "https://api.openai.com/v1/responses"
DEFAULT_OPENAI_MODEL = "gpt-4.1-mini"
TOP_FEATURES_PER_DIRECTION = 3


@dataclass(frozen=True)
class FeatureEvidence:
    association: str
    pmids: List[str]
    note: str = ""


FEATURE_LOOKUP = {cfg.id: cfg for cfg in FEATURES_UI}
FEATURE_OPTION_LABELS = {
    cfg.id: {opt["value"]: opt["label"] for opt in cfg.params["options"]}
    for cfg in FEATURES_UI
    if cfg.kind == "categorical"
}

FEATURE_EVIDENCE: Dict[str, FeatureEvidence] = {
    "phq9": FeatureEvidence(
        association="Higher baseline depression severity is associated with higher treatment resistance risk.",
        pmids=["17685743"],
    ),
    "duration_months": FeatureEvidence(
        association="Longer current episode duration is used in staging higher treatment resistance burden.",
        pmids=["19192471"],
    ),
    "previous_failures": FeatureEvidence(
        association="Multiple adequate prior antidepressant failures define or increase treatment resistance staging.",
        pmids=["17444078", "19192471"],
    ),
    "adherence_pct": FeatureEvidence(
        association="Poor adherence can create apparent treatment resistance and may reflect pseudo-resistance.",
        pmids=["11480879", "33779973"],
        note="This is conceptual evidence about pseudo-resistance rather than a direct treatment-resistance predictor.",
    ),
    "sertraline_mg": FeatureEvidence(
        association="Lower licensed SSRI dose ranges tend to balance efficacy and tolerability best in acute depression treatment, but dose is not a treatment-resistance predictor.",
        pmids=["31178367", "29477251"],
        note="Treatment dosing evidence rather than predictor evidence.",
    ),
    "quetiapine_mg": FeatureEvidence(
        association="Quetiapine augmentation improves response or remission in difficult-to-treat depression, but it is a treatment variable rather than a predictor.",
        pmids=["34986373", "35993319"],
        note="Treatment evidence rather than predictor evidence.",
    ),
    "lithium_mg": FeatureEvidence(
        association="Lithium augmentation is evidence-supported in inadequate antidepressant response, but it is a treatment variable rather than a predictor.",
        pmids=["24825489", "34986373"],
        note="Treatment evidence rather than predictor evidence.",
    ),
    "early_improvement": FeatureEvidence(
        association="Lack of early improvement is associated with higher later non-response risk.",
        pmids=["19254516"],
        note="This is stronger for later non-response than for treatment resistance specifically.",
    ),
    "sleep_severity": FeatureEvidence(
        association="Sleep disturbance is common in depression and can complicate treatment response, but evidence as a treatment-resistance predictor is weak and indirect.",
        pmids=["22681161", "28791566"],
        note="Evidence is weaker and more indirect than for core resistance predictors.",
    ),
    "substance_use": FeatureEvidence(
        association="Comorbid substance use can complicate depression treatment, but direct evidence for treatment-resistance prediction is limited.",
        pmids=["15100209"],
        note="This is indirect evidence from co-occurring depression and substance-use treatment outcomes.",
    ),
    "comorbid_anxiety": FeatureEvidence(
        association="Comorbid anxiety is associated with higher treatment resistance risk.",
        pmids=["17685743"],
    ),
    "side_effects": FeatureEvidence(
        association="Side effect burden may contribute to dose reduction or poor adherence, but it is not well established as a direct predictor of treatment resistance.",
        pmids=["31178367", "33779973"],
        note="Indirect evidence rather than a well-established predictor.",
    ),
    "sex_female": FeatureEvidence(
        association="Sex differences in antidepressant response exist, but they are not large enough to guide care alone and do not establish a strong standalone treatment-resistance predictor.",
        pmids=["16012273"],
        note="Use cautious wording because this is about antidepressant response differences, not a direct treatment-resistance predictor.",
    ),
}


def _format_feature_value(feature_id: str, value: Any) -> str:
    if feature_id in FEATURE_OPTION_LABELS:
        option_label = FEATURE_OPTION_LABELS[feature_id].get(value, str(value))
        if feature_id == "early_improvement":
            return "Yes" if value == 1 else "No"
        return option_label

    numeric_value = float(value)
    integer_value = int(round(numeric_value))
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


def _build_feature_item(feature_id: str, value: Any, shap_value: float) -> Dict[str, Any]:
    evidence = FEATURE_EVIDENCE.get(feature_id)
    return {
        "feature_id": feature_id,
        "feature_label": FEATURE_LOOKUP[feature_id].label,
        "selected_value": _format_feature_value(feature_id, value),
        "shap_value": round(float(shap_value), 4),
        "direction": "raises" if shap_value > 0 else "lowers",
        "evidence_association": evidence.association if evidence else "No supporting literature entry was provided in the README for this feature.",
        "evidence_pmids": evidence.pmids if evidence else [],
        "evidence_note": evidence.note if evidence else "Treat this as a model-specific pattern rather than a literature-supported claim.",
    }


def _select_influential_features(values_dict: Dict[str, Any], shap_vals: np.ndarray) -> Dict[str, List[Dict[str, Any]]]:
    ranked = sorted(
        zip(FEATURE_LOOKUP.keys(), np.asarray(shap_vals, dtype=float)),
        key=lambda item: abs(item[1]),
        reverse=True,
    )
    positive, negative = [], []
    for feature_id, shap_value in ranked:
        if shap_value > 0 and len(positive) < TOP_FEATURES_PER_DIRECTION:
            positive.append(_build_feature_item(feature_id, values_dict[feature_id], shap_value))
        elif shap_value < 0 and len(negative) < TOP_FEATURES_PER_DIRECTION:
            negative.append(_build_feature_item(feature_id, values_dict[feature_id], shap_value))

        if len(positive) >= TOP_FEATURES_PER_DIRECTION and len(negative) >= TOP_FEATURES_PER_DIRECTION:
            break
    return {"positive": positive, "negative": negative}


def build_llm_prompt(values_dict: Dict[str, Any], probability: float, shap_vals: np.ndarray) -> str:
    influential = _select_influential_features(values_dict, shap_vals)
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


def _extract_output_text(response_payload: Dict[str, Any]) -> str:
    if response_payload.get("output_text"):
        return str(response_payload["output_text"]).strip()

    parts: List[str] = []
    for output_item in response_payload.get("output", []):
        for content_item in output_item.get("content", []):
            text = content_item.get("text")
            if text:
                parts.append(text)
    return "\n".join(parts).strip()


def fetch_llm_summary(prompt: str) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    model_name = os.getenv("OPENAI_MODEL", DEFAULT_OPENAI_MODEL)
    if not api_key:
        return (
            "Prediction explanation unavailable: the language model service is not configured.\n\n"
            "The SHAP chart still shows which features pushed this prediction higher or lower."
        )

    payload = {
        "model": model_name,
        "input": [
            {
                "role": "system",
                "content": [
                    {
                        "type": "input_text",
                        "text": (
                            "You explain model predictions for a synthetic depression treatment-resistance demo. "
                            "Base every statement only on the provided JSON payload. "
                            "Use plain language, cite only supplied PMIDs, and avoid unsupported claims."
                        ),
                    }
                ],
            },
            {"role": "user", "content": [{"type": "input_text", "text": prompt}]},
        ],
        "temperature": 0.2,
        "max_output_tokens": 350,
    }
    req = request.Request(
        OPENAI_API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with request.urlopen(req, timeout=20) as response:
            response_payload = json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        return f"Prediction explanation unavailable: the language model service returned HTTP {exc.code}.\n\n```text\n{detail[:800]}\n```"
    except error.URLError as exc:
        reason = getattr(exc, "reason", exc)
        return f"Prediction explanation unavailable: the language model service could not be reached ({reason})."
    except TimeoutError:
        return "Prediction explanation unavailable: the language model service timed out."
    except json.JSONDecodeError:
        return "Prediction explanation unavailable: the language model service response could not be parsed."

    text = _extract_output_text(response_payload)
    if not text:
        return "Prediction explanation unavailable: the language model service returned an empty response."
    return text


def generate_prediction_summary(values_dict: Dict[str, Any], probability: float, shap_vals: np.ndarray) -> str:
    prompt = build_llm_prompt(values_dict, probability, shap_vals)
    return fetch_llm_summary(prompt)
