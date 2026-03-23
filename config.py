from typing import List
from data_synth import FeatureConfig

CARD = {
    "backgroundColor": "white",
    "borderRadius": "12px",
    "boxShadow": "0 2px 10px rgba(0,0,0,0.06)",
    "padding": "16px",
}

PILL = {
    "display": "inline-block",
    "padding": "6px 12px",
    "borderRadius": "999px",
    "fontSize": "14px",
    "fontWeight": "700",
    "border": "1px solid #e5e7eb",
    "backgroundColor": "#eef2ff",
    "color": "#111827",
}

SECTION_TITLE = {"margin": "0 0 8px 0", "fontSize": "18px"}
SUBHEADING = {"fontSize": "12px", "color": "#6b7280", "fontWeight": "600", "marginTop": "2px"}
INFO_TEXT_STYLE = {"fontSize": "12px", "color": "#6b7280", "marginTop": "6px"}

FEATURES_UI: List[FeatureConfig] = [
    FeatureConfig("age", "Age (years)", "numeric", 40, {"min": 18, "max": 80, "step": 1}),
    FeatureConfig("sex_female", "Sex", "categorical", 0, {"options": [
        {"label": "Male", "value": 0},
        {"label": "Female", "value": 1},
    ]}),
    FeatureConfig("phq9", "PHQ-9 (0-27)", "numeric", 18, {"min": 0, "max": 27, "step": 1}),
    FeatureConfig("duration_months", "Current episode duration (months)", "numeric", 6, {"min": 0, "max": 60, "step": 1}),
    FeatureConfig("previous_failures", "Previous adequate treatment failures", "numeric", 1, {"min": 0, "max": 5, "step": 1}),
    FeatureConfig("adherence_pct", "Adherence (%)", "numeric", 80, {"min": 0, "max": 100, "step": 1}),
    FeatureConfig("sertraline_mg", "Sertraline dose (mg/day)", "numeric", 100, {"min": 0, "max": 200, "step": 5}),
    FeatureConfig("quetiapine_mg", "Quetiapine augmentation (mg/day)", "numeric", 0, {"min": 0, "max": 300, "step": 25}),
    FeatureConfig("lithium_mg", "Lithium dose (mg/day)", "numeric", 0, {"min": 0, "max": 1200, "step": 100}),
    FeatureConfig("early_improvement", "Early improvement at 2 weeks", "categorical", 0, {"options": [
        {"label": "No", "value": 0},
        {"label": "Yes", "value": 1},
    ]}),
    FeatureConfig("sleep_severity", "Sleep disturbance", "categorical", 1, {"options": [
        {"label": "None", "value": 0},
        {"label": "Mild", "value": 1},
        {"label": "Severe", "value": 2},
    ]}),
    FeatureConfig("substance_use", "Substance use", "categorical", 0, {"options": [
        {"label": "None", "value": 0},
        {"label": "Occasional", "value": 1},
        {"label": "Regular", "value": 2},
    ]}),
    FeatureConfig("comorbid_anxiety", "Comorbid anxiety", "categorical", 0, {"options": [
        {"label": "No", "value": 0},
        {"label": "Yes", "value": 1},
    ]}),
    FeatureConfig("side_effects", "Side effect burden", "categorical", 1, {"options": [
        {"label": "None", "value": 0},
        {"label": "Mild", "value": 1},
        {"label": "Moderate", "value": 2},
        {"label": "Severe", "value": 3},
    ]}),
]