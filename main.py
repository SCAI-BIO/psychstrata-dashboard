from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output

from mapie.classification import MapieClassifier
import shap

from data_synth import generate_synthetic_dataset, FeatureConfig

# -----------------------------
# Data and model
# -----------------------------
X, y = generate_synthetic_dataset(n=2500, random_state=42)
feature_cols = X.columns.tolist()

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
# Split training in Fit/Calibration (for MAPIE)
X_fit, X_calib, y_fit, y_calib = train_test_split(
    X_train, y_train, test_size=0.25, random_state=42, stratify=y_train
)

# Fit RF
rf = RandomForestClassifier(
    n_estimators=350,
    max_depth=None,
    min_samples_leaf=3,
    random_state=42,
    n_jobs=-1,
)
rf.fit(X_fit, y_fit)

# AUC on test
auc = roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1])

# MAPIE (Conformal)
mapie = MapieClassifier(estimator=rf, method="score", cv="prefit")
mapie.fit(X_calib, y_calib)

# SHAP explainer: probability output with interventional + background data
background = X_fit.sample(n=min(200, len(X_fit)), random_state=42)
shap_explainer = shap.TreeExplainer(
    rf,
    data=background,
    model_output="probability",
    feature_perturbation="interventional",
)

# -----------------------------
# App
# -----------------------------
app = Dash(__name__)
app.title = "Treatment Resistance Classifier (Demo)"

# Feature definitions (UI)
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

# -----------------------------
# Layout styles
# -----------------------------
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

# -----------------------------
# Layout
# -----------------------------
app.layout = html.Div(
    [
        html.Div(
            [
                # Header
                html.Div(
                    [
                        html.Div(
                            [
                                html.H2("Treatment Resistance Classifier", style={"margin": 0}),
                                html.Div("Demo • Not medical advice", style={"color": "#6b7280", "fontSize": "12px", "marginTop": "2px"}),
                            ],
                            style={"display": "flex", "flexDirection": "column"}
                        ),
                        html.Div(
                            [
                                html.Span("Model AUC", style={"marginRight": "8px", "color": "#6b7280", "fontSize": "12px"}),
                                html.Span(f"{auc:.3f}", style={
                                    "display": "inline-block",
                                    "padding": "4px 10px",
                                    "borderRadius": "999px",
                                    "backgroundColor": "#eef2ff",
                                    "color": "#3730a3",
                                    "fontSize": "12px",
                                    "fontWeight": "600",
                                }),
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

                        # Right column: stack Prediction card and SHAP card
                        html.Div(
                            [
                                # Prediction card
                                html.Div(
                                    [
                                        html.H4("Prediction", style=SECTION_TITLE),
                                        html.Div("Random Forest Base Prediction", style=SUBHEADING),
                                        dcc.Graph(
                                            id="pred-indicator",
                                            config={"displayModeBar": False},
                                            style={"height": "200px", "marginTop": "4px"}
                                        ),
                                        html.Div(
                                            [
                                                html.Div("Conformal Prediction Guarantees", style={**SUBHEADING, "marginBottom": "6px"}),
                                                html.Div("Confidence interval (%)", style={"fontSize": "12px", "color": "#444", "marginBottom": "4px"}),
                                                dcc.Slider(
                                                    id="ci-slider",
                                                    min=80, max=99, step=1, value=95,
                                                    marks={x: str(x) for x in [80, 85, 90, 95, 99]},
                                                ),
                                                html.Div(
                                                    [html.Div(id="prediction-badge", style={**PILL}, children="")],
                                                    style={"display": "flex", "justifyContent": "center", "marginTop": "10px"}
                                                ),
                                            ],
                                            style={
                                                "marginTop": "10px",
                                                "padding": "10px",
                                                "backgroundColor": "#f9fafb",
                                                "borderRadius": "8px",
                                                "border": "1px solid #eee",
                                            }
                                        ),
                                    ],
                                    style={**CARD}
                                ),

                                # SHAP card (separate component with small gap above via parent gap)
                                html.Div(
                                    [
                                        html.H4("Feature contributions (SHAP)", style={**SECTION_TITLE, "textAlign": "center"}),
                                        dcc.Graph(
                                            id="shap-bar",
                                            config={"displayModeBar": False},
                                            style={"height": "360px", "margin": "6px auto", "width": "95%"}
                                        ),
                                    ],
                                    style={**CARD}
                                ),
                            ],
                            style={"flex": "1", "minWidth": "360px", "display": "flex", "flexDirection": "column", "gap": "12px"}
                        ),
                    ],
                    style={"display": "flex", "gap": "20px", "marginTop": "20px", "flexWrap": "wrap"}
                ),

                # Footer
                html.Div(
                    "This demo is for educational purposes only and not a medical device.",
                    style={"fontSize": "12px", "color": "#6b7280", "textAlign": "center", "marginTop": "16px"}
                ),
            ],
            style={"maxWidth": "1200px", "margin": "0 auto"}
        ),
    ],
    style={"padding": "24px", "backgroundColor": "#f6f7fb", "fontFamily": "Inter, Arial, sans-serif"}
)

# -----------------------------
# Helpers
# -----------------------------
def pack_instance_from_inputs(values_dict: Dict[str, Any]) -> pd.DataFrame:
    row = {col: values_dict[col] for col in feature_cols}
    return pd.DataFrame([row], columns=feature_cols)

def indicator_figure(prob: float) -> go.Figure:
    color = "#d62728" if prob >= 0.5 else "#2ca02c"
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            number={"suffix": "%", "font": {"size": 28}},
            title={"text": "Probability of resistance"},
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
    fig.update_layout(margin=dict(l=20, r=20, t=50, b=30))
    return fig

def conformal_badge_from_mapie(X_row: pd.DataFrame, ci_level: int) -> Tuple[str, Dict[str, Any]]:
    alpha = 1.0 - (ci_level / 100.0)
    _, y_ps = mapie.predict(X_row, alpha=alpha)
    if y_ps.ndim == 3:
        y_ps = y_ps[:, :, 0]
    included = y_ps[0].astype(bool)
    classes = list(rf.classes_)
    num_included = int(included.sum())

    if num_included == 0 or num_included == 2:
        label = "Uncertain"
        bg, border, color = "#fef3c7", "1px solid #fcd34d", "#92400e"
    else:
        idx = int(np.where(included)[0][0])
        cls_val = classes[idx]
        if cls_val == 1:
            label = "Resistant"
            bg, border, color = "#fee2e2", "1px solid #fca5a5", "#991b1b"
        else:
            label = "Responsive"
            bg, border, color = "#dcfce7", "1px solid #86efac", "#14532d"

    style = {
        "display": "inline-block",
        "padding": "6px 12px",
        "borderRadius": "999px",
        "fontSize": "14px",
        "fontWeight": "700",
        "backgroundColor": bg,
        "border": border,
        "color": color,
    }
    return label, style

def shap_bar_figure(shap_vals: np.ndarray, feature_names: List[str]) -> go.Figure:
    s = pd.Series(shap_vals, index=feature_names)
    s = s.sort_values(key=lambda x: x.abs(), ascending=True)
    colors = ["#d62728" if v > 0 else "#1f77b4" for v in s.values]  # red=increases risk, blue=decreases risk

    fig = go.Figure(
        data=go.Bar(
            x=s.values,
            y=s.index.tolist(),
            orientation="h",
            marker_color=colors,
            hovertemplate="Feature: %{y}<br>SHAP: %{x:.4f}<extra></extra>",
        )
    )
    fig.update_layout(
        title={"text": "SHAP contributions to P(Resistance)", "x": 0.5},
        xaxis_title="SHAP value (Δ probability)",
        yaxis_title="Feature",
        margin=dict(l=90, r=20, t=50, b=20),
        plot_bgcolor="white",
    )
    # Vertical zero line
    fig.add_shape(type="line", x0=0, x1=0, y0=-0.5, y1=len(s)-0.5, line=dict(color="#9ca3af", width=1))
    return fig

def shap_values_for_positive_class(X_row: pd.DataFrame) -> np.ndarray:
    sv = shap_explainer.shap_values(X_row)

    # Convert to array (list -> choose last class; else direct)
    if isinstance(sv, list):
        arr = sv[-1]
    else:
        arr = sv
    arr = np.array(arr)

    # Handle shapes robustly
    if arr.ndim == 3:
        # (n_samples, n_features, n_outputs) -> take class 1 (last) for first sample
        vals = arr[0, :, -1]
    elif arr.ndim == 2 and arr.shape[0] == 1 and arr.shape[1] == len(feature_cols):
        vals = arr[0]
    elif arr.ndim == 2 and arr.shape[0] == len(feature_cols) and arr.shape[1] >= 2:
        vals = arr[:, 1]
    elif arr.ndim == 2 and arr.shape[1] == len(feature_cols):
        vals = arr[0]
    else:
        vals = np.squeeze(arr)
        if vals.ndim > 1:
            vals = vals.reshape(-1)[:len(feature_cols)]
    return np.asarray(vals, dtype=float)

# -----------------------------
# Callback wiring
# -----------------------------
input_ids = [f"input-{cfg.id}" for cfg in features_ui]
value_ids = [f"value-{cfg.id}" for cfg in features_ui if cfg.kind == "numeric"]

dash_inputs = [Input(comp_id, "value") for comp_id in input_ids]
dash_inputs.append(Input("ci-slider", "value"))

dash_outputs = [
    Output("pred-indicator", "figure"),
    Output("prediction-badge", "children"),
    Output("prediction-badge", "style"),
    Output("shap-bar", "figure"),
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

    # Base RF probability and gauge
    prob = float(rf.predict_proba(X_row)[0, 1])
    pred_fig = indicator_figure(prob)

    # MAPIE badge
    badge_text, badge_style = conformal_badge_from_mapie(X_row, ci_level)

    # SHAP: feature contributions for positive class (resistance)
    shap_vals = shap_values_for_positive_class(X_row)
    shap_fig = shap_bar_figure(shap_vals, feature_cols)

    # Numeric displays
    numeric_displays = []
    for cfg in features_ui:
        if cfg.kind == "numeric":
            numeric_displays.append(f"Selected: {values_dict[cfg.id]}")

    outputs = [pred_fig, badge_text, badge_style, shap_fig]
    outputs.extend(numeric_displays)
    return outputs


if __name__ == "__main__":
    app.run(debug=True, port=8050)