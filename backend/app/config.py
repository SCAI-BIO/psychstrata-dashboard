from .data_synth import FeatureConfig


BRAND_BLUE = "#006fb9"
BRAND_BLUE_DARK = "#00558d"
BRAND_BLUE_LIGHT = "#e6f4fb"
BRAND_CYAN = "#37aee2"

CALM_BLUE = "#4C78A8"
CALM_TEAL = "#72B7B2"
CALM_LAVENDER = "#A78BFA"
CALM_AMBER = "#D4A72C"
CALM_SLATE = "#64748B"
CALM_BLUE_LIGHT = "#EAF2FB"
CALM_TEAL_LIGHT = "#E3F4F2"
CALM_LAVENDER_LIGHT = "#F1EAFE"
CALM_AMBER_LIGHT = "#FBF3D1"
CALM_SLATE_LIGHT = "#E2E8F0"

FEATURES_UI: list[FeatureConfig] = [
    FeatureConfig("age", "Age (years)", "numeric", 40, {"min": 18, "max": 80, "step": 1}),
    FeatureConfig(
        "sex_female",
        "Sex",
        "categorical",
        0,
        {
            "options": [
                {"label": "Male", "value": 0},
                {"label": "Female", "value": 1},
            ]
        },
    ),
    FeatureConfig("phq9", "PHQ-9 (0-27)", "numeric", 18, {"min": 0, "max": 27, "step": 1}),
    FeatureConfig(
        "duration_months",
        "Current episode duration (months)",
        "numeric",
        6,
        {"min": 0, "max": 60, "step": 1},
    ),
    FeatureConfig(
        "previous_failures",
        "Previous adequate treatment failures",
        "numeric",
        1,
        {"min": 0, "max": 5, "step": 1},
    ),
    FeatureConfig("adherence_pct", "Adherence (%)", "numeric", 80, {"min": 0, "max": 100, "step": 1}),
    FeatureConfig(
        "sertraline_mg",
        "Sertraline dose (mg/day)",
        "numeric",
        100,
        {"min": 0, "max": 200, "step": 5},
    ),
    FeatureConfig(
        "quetiapine_mg",
        "Quetiapine augmentation (mg/day)",
        "numeric",
        0,
        {"min": 0, "max": 300, "step": 25},
    ),
    FeatureConfig(
        "lithium_mg",
        "Lithium dose (mg/day)",
        "numeric",
        0,
        {"min": 0, "max": 1200, "step": 100},
    ),
    FeatureConfig(
        "early_improvement",
        "Early improvement at 2 weeks",
        "categorical",
        0,
        {
            "options": [
                {"label": "No", "value": 0},
                {"label": "Yes", "value": 1},
            ]
        },
    ),
    FeatureConfig(
        "sleep_severity",
        "Sleep disturbance",
        "categorical",
        1,
        {
            "options": [
                {"label": "None", "value": 0},
                {"label": "Mild", "value": 1},
                {"label": "Severe", "value": 2},
            ]
        },
    ),
    FeatureConfig(
        "substance_use",
        "Substance use",
        "categorical",
        0,
        {
            "options": [
                {"label": "None", "value": 0},
                {"label": "Occasional", "value": 1},
                {"label": "Regular", "value": 2},
            ]
        },
    ),
    FeatureConfig(
        "comorbid_anxiety",
        "Comorbid anxiety",
        "categorical",
        0,
        {
            "options": [
                {"label": "No", "value": 0},
                {"label": "Yes", "value": 1},
            ]
        },
    ),
    FeatureConfig(
        "side_effects",
        "Side effect burden",
        "categorical",
        1,
        {
            "options": [
                {"label": "None", "value": 0},
                {"label": "Mild", "value": 1},
                {"label": "Moderate", "value": 2},
                {"label": "Severe", "value": 3},
            ]
        },
    ),
]

FEATURES_BY_ID = {cfg.id: cfg for cfg in FEATURES_UI}
FEATURE_DEFAULTS = {cfg.id: cfg.default for cfg in FEATURES_UI}
FEATURE_OPTION_LABELS = {
    cfg.id: {option["value"]: option["label"] for option in cfg.params["options"]}
    for cfg in FEATURES_UI
    if cfg.kind == "categorical"
}
