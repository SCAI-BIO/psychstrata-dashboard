from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output

from data_synth import generate_synthetic_dataset, FeatureConfig

X, y = generate_synthetic_dataset(n=2500, random_state=42)
feature_cols = X.columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

rf = RandomForestClassifier(
    n_estimators=350,
    max_depth=None,
    min_samples_leaf=3,
    random_state=42,
    n_jobs=-1,
)
rf.fit(X_train, y_train)
auc = roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1])

app = Dash(__name__)
app.title = "Treatment Resistance Classifier (Demo)"

# Feature definitions for UI
features_ui: List[FeatureConfig] = [
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

def feature_input_component(cfg: FeatureConfig) -> html.Div:
    label_id = f"label-{cfg.id}"
    input_id = f"input-{cfg.id}"
    base_style = {
        "marginBottom": "12px",
        "padding": "10px",
        "border": "1px solid #e1e1e1",
        "borderRadius": "8px",
        "backgroundColor": "#fafafa",
    }
    label_style = {"fontWeight": "600", "marginBottom": "6px", "color": "#444"}
    if cfg.kind == "numeric":
        control = dcc.Slider(
            id=input_id,
            min=cfg.params["min"],
            max=cfg.params["max"],
            step=cfg.params.get("step", 1),
            value=cfg.default,
            tooltip={"always_visible": False, "placement": "bottom"},
            marks=None,
        )
        value_display = html.Div(id=f"value-{cfg.id}", style={"fontSize": "12px", "color": "#666", "marginTop": "4px"})
        return html.Div(
            [
                html.Div(cfg.label, id=label_id, style=label_style),
                control,
                value_display,
            ],
            id=f"container-{cfg.id}",
            style=base_style,
        )
    else:
        control = dcc.Dropdown(
            id=input_id,
            options=cfg.params["options"],
            value=cfg.default,
            clearable=False,
        )
        return html.Div(
            [
                html.Div(cfg.label, id=label_id, style=label_style),
                control,
            ],
            id=f"container-{cfg.id}",
            style=base_style,
        )

# Layout
# Optional style helpers
CARD = {
    "backgroundColor": "white",
    "borderRadius": "12px",
    "boxShadow": "0 2px 10px rgba(0,0,0,0.06)",
    "padding": "16px",
}
PILL = {
    "display": "inline-block",
    "padding": "4px 10px",
    "borderRadius": "999px",
    "backgroundColor": "#eef2ff",
    "color": "#3730a3",
    "fontSize": "12px",
    "fontWeight": "600",
}
SECTION_TITLE = {"margin": "0 0 8px 0", "fontSize": "18px"}

# Updated layout
app.layout = html.Div(
    [
        html.Div(  # page container
            [
                # Header
                html.Div(
                    [
                        html.Div(
                            [
                                html.H2(
                                    "Treatment Resistance Classifier",
                                    style={"margin": 0}
                                ),
                                html.Div(
                                    "Demo • Not medical advice",
                                    style={"color": "#6b7280", "fontSize": "12px", "marginTop": "2px"}
                                ),
                            ],
                            style={"display": "flex", "flexDirection": "column"}
                        ),
                        html.Div(
                            [
                                html.Span("Model AUC", style={"marginRight": "8px", "color": "#6b7280", "fontSize": "12px"}),
                                html.Span(f"{auc:.3f}", style=PILL),
                            ],
                            style={"display": "flex", "alignItems": "center", "gap": "4px"}
                        ),
                    ],
                    style={**CARD, "display": "flex", "justifyContent": "space-between", "alignItems": "center"}
                ),

                # Main content
                html.Div(
                    [
                        # Left: feature inputs
                        html.Div(
                            [
                                html.H4("Enter patient features", style=SECTION_TITLE),
                                html.Div(
                                    [feature_input_component(cfg) for cfg in features_ui],
                                    style={
                                        "display": "grid",
                                        "gridTemplateColumns": "repeat(auto-fit, minmax(280px, 1fr))",
                                        "gap": "12px",
                                    },
                                ),
                            ],
                            style={**CARD, "flex": "1.7", "minWidth": "320px"}
                        ),

                        # Right: prediction card
                        html.Div(
                            [
                                html.H4("Prediction", style=SECTION_TITLE),
                                dcc.Graph(
                                    id="pred-indicator",
                                    config={"displayModeBar": False},
                                    style={"height": "200px", "marginTop": "4px"}
                                ),
                                html.Div(
                                    [
                                        html.Div(
                                            "Confidence interval (%)",
                                            style={"fontSize": "12px", "color": "#444", "marginBottom": "4px"}
                                        ),
                                        dcc.Slider(
                                            id="ci-slider",
                                            min=80, max=99, step=1, value=95,
                                            marks={x: str(x) for x in [80, 85, 90, 95, 99]},
                                        ),
                                    ],
                                    style={
                                        "marginTop": "6px",
                                        "padding": "10px",
                                        "backgroundColor": "#f9fafb",
                                        "borderRadius": "8px",
                                        "border": "1px solid #eee",
                                    }
                                ),
                            ],
                            style={**CARD, "flex": "1", "minWidth": "360px"}
                        ),
                    ],
                    style={"display": "flex", "gap": "20px", "marginTop": "20px", "flexWrap": "wrap"}
                ),

                # Footer
                html.Div(
                    "This demo is for educational purposes only and not a medical device.",
                    style={
                        "fontSize": "12px",
                        "color": "#6b7280",
                        "textAlign": "center",
                        "marginTop": "16px"
                    }
                ),
            ],
            style={"maxWidth": "1200px", "margin": "0 auto"}
        ),
    ],
    style={"padding": "24px", "backgroundColor": "#f6f7fb", "fontFamily": "Inter, Arial, sans-serif"}
)

# -----------------------------
# Callback helpers
# -----------------------------
def pack_instance_from_inputs(values_dict: Dict[str, Any]) -> pd.DataFrame:
    row = {col: values_dict[col] for col in feature_cols}
    return pd.DataFrame([row], columns=feature_cols)

def rf_tree_proba_distribution(model: RandomForestClassifier, X_row: pd.DataFrame) -> np.ndarray:
    # Use numpy array to avoid "feature names" warnings on individual trees
    X_np = X_row.to_numpy()
    probs = [est.predict_proba(X_np)[0, 1] for est in model.estimators_]
    return np.array(probs)

def ci_from_distribution(dist: np.ndarray, ci_level: int) -> Tuple[float, float]:
    alpha = 1 - (ci_level / 100.0)
    lower = np.percentile(dist, 100 * (alpha / 2.0))
    upper = np.percentile(dist, 100 * (1 - alpha / 2.0))
    return float(lower), float(upper)

def indicator_figure(prob: float, ci: Tuple[float, float]) -> go.Figure:
    cls = "Resistant" if prob >= 0.5 else "Responsive"
    color = "#d62728" if prob >= 0.5 else "#2ca02c"
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            number={"suffix": "%", "font": {"size": 28}},
            title={"text": f"Predicted: {cls}<br><span style='font-size:13px;color:#666'>Probability of resistance</span>"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": color},
                "steps": [
                    {"range": [0, 50], "color": "#e6f4ea"},
                    {"range": [50, 100], "color": "#fdecea"},
                ],
                "threshold": {"line": {"color": "#444", "width": 2}, "thickness": 0.75, "value": 50},
            },
            domain={"x": [0, 1], "y": [0, 1]},
        )
    )
    ci_txt = f"CI: {int(round(ci[0]*100))}% – {int(round(ci[1]*100))}%"
    fig.add_annotation(
        text=ci_txt,
        x=0.5, y=-0.15, xref="paper", yref="paper",
        showarrow=False, font={"size": 12, "color": "#444"},
    )
    fig.update_layout(margin=dict(l=20, r=20, t=40, b=40))
    return fig

# -----------------------------
# Dynamic callback wiring
# -----------------------------
input_ids = [f"input-{cfg.id}" for cfg in features_ui]
value_ids = [f"value-{cfg.id}" for cfg in features_ui if cfg.kind == "numeric"]

dash_inputs = [Input(comp_id, "value") for comp_id in input_ids]
dash_inputs.append(Input("ci-slider", "value"))

dash_outputs = [
    Output("pred-indicator", "figure"),
]
for vid in value_ids:
    dash_outputs.append(Output(vid, "children"))

@app.callback(dash_outputs, dash_inputs)
def update_predictions(*args):
    values = args[:-1]
    ci_level = args[-1]

    values_dict = {}
    for cfg, v in zip(features_ui, values):
        values_dict[cfg.id] = v

    X_row = pack_instance_from_inputs(values_dict)

    prob = float(rf.predict_proba(X_row)[0, 1])

    dist = rf_tree_proba_distribution(rf, X_row)
    ci = ci_from_distribution(dist, ci_level)

    pred_fig = indicator_figure(prob, ci)

    numeric_displays = []
    for cfg in features_ui:
        if cfg.kind == "numeric":
            v = values_dict[cfg.id]
            numeric_displays.append(f"Selected: {v}")

    outputs = [pred_fig]
    outputs.extend(numeric_displays)
    return outputs


if __name__ == "__main__":
    # To run: python app.py
    app.run(debug=True, port=8050)