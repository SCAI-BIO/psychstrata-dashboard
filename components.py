from dash import dcc, html
from data_synth import FeatureConfig
from config import CARD, SECTION_TITLE, SUBHEADING, INFO_TEXT_STYLE


def create_feature_input(cfg: FeatureConfig) -> html.Div:
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
        value_display = html.Div(
            id=f"value-{cfg.id}",
            style={"fontSize": "12px", "color": "#666", "marginTop": "4px"}
        )
        return html.Div(
            [html.Div(cfg.label, style=label_style), control, value_display],
            id=f"container-{cfg.id}",
            style=base_style,
        )
    
    control = dcc.Dropdown(
        id=input_id,
        options=cfg.params["options"],
        value=cfg.default,
        clearable=False,
    )
    return html.Div(
        [html.Div(cfg.label, style=label_style), control],
        id=f"container-{cfg.id}",
        style=base_style,
    )


def create_info_details(summary: str, content: str, style_override: dict = None) -> html.Details:
    base_style = {"marginTop": "8px"}
    if style_override:
        base_style.update(style_override)
    
    return html.Details(
        [
            html.Summary(summary),
            html.Div(content, style=INFO_TEXT_STYLE),
        ],
        open=False,
        style=base_style,
    )


def create_header_card(auc: float) -> html.Div:
    return html.Div(
        [
            html.Div(
                [
                    html.H2("Treatment Resistance Classifier", style={"margin": 0}),
                    html.Div(
                        "Demo • Not medical advice",
                        style={"color": "#6b7280", "fontSize": "12px", "marginTop": "2px"}
                    ),
                    html.Div(
                        "Important: This demo uses fully synthetic (non-real) data created for illustration. "
                        "It does not reflect actual patient information, clinical outcomes, or treatment insights. "
                        "It is not a medical device and should not be used for diagnosis or treatment decisions. "
                        "For medical guidance, please consult a qualified clinician.",
                        style={
                            "color": "#6b7280", "fontSize": "12px", "marginTop": "6px",
                            "maxWidth": "820px", "lineHeight": "1.4"
                        }
                    ),
                ],
                style={"display": "flex", "flexDirection": "column"}
            ),
            html.Div(
                [
                    html.Span("Model AUC", style={"marginRight": "8px", "color": "#6b7280", "fontSize": "12px"}),
                    html.Span(f"{auc:.3f}", style={
                        "display": "inline-block", "padding": "4px 10px", "borderRadius": "999px",
                        "backgroundColor": "#eef2ff", "color": "#3730a3", "fontSize": "12px", "fontWeight": "600",
                    }),
                ],
                style={"display": "flex", "alignItems": "center", "gap": "4px"}
            ),
        ],
        style={**CARD, "display": "flex", "justifyContent": "space-between", "alignItems": "flex-start"}
    )


def create_prediction_card() -> html.Div:
    return html.Div(
        [
            html.H4("Prediction", style=SECTION_TITLE),
            html.Div("Random Forest Base Prediction", style=SUBHEADING),
            dcc.Graph(id="pred-indicator", config={"displayModeBar": False}, style={"height": "200px", "marginTop": "4px"}),
            html.Div(
                [
                    html.Div("Conformal Prediction Guarantees", style={**SUBHEADING, "marginBottom": "6px"}),
                    html.Div("Confidence interval (%)", style={"fontSize": "12px", "color": "#444", "marginBottom": "4px"}),
                    dcc.Slider(id="ci-slider", min=80, max=99, step=1, value=95, marks={x: str(x) for x in [80, 85, 90, 95, 99]}),
                    html.Div(
                        [html.Div(id="prediction-badge", children="")],
                        style={"display": "flex", "justifyContent": "center", "marginTop": "10px"}
                    ),
                ],
                style={"marginTop": "10px", "padding": "10px", "backgroundColor": "#f9fafb", "borderRadius": "8px", "border": "1px solid #eee"}
            ),
            create_info_details(
                "What's this?",
                "This gauge shows the estimated chance of treatment resistance. Green is lower risk, red is higher risk. "
                "The badge uses a statistical method to indicate how confident the model is."
            ),
        ],
        style=CARD
    )


def create_shap_card() -> html.Div:
    return html.Div(
        [
            html.H4("Feature contributions (SHAP)", style={**SECTION_TITLE, "textAlign": "center"}),
            dcc.Graph(id="shap-bar", config={"displayModeBar": False}, style={"height": "360px", "margin": "6px auto", "width": "95%"}),
            create_info_details(
                "What's this?",
                "Each bar shows how a feature pushed the prediction. Green bars lower resistance risk. Red bars raise resistance risk.",
                {"width": "95%", "marginLeft": "auto", "marginRight": "auto"}
            ),
        ],
        style=CARD
    )


def create_tsne_card() -> html.Div:
    return html.Div(
        [
            html.H4("Population map (t-SNE)", style={**SECTION_TITLE, "textAlign": "center"}),
            dcc.Graph(id="tsne-scatter", config={"displayModeBar": False}, style={"height": "380px", "margin": "6px auto", "width": "95%"}),
            create_info_details(
                "What's this?",
                "This map places similar patients close together. Green dots are patients who responded; red dots are patients who were resistant. The blue dot shows the current selection.",
                {"width": "95%", "marginLeft": "auto", "marginRight": "auto"}
            ),
        ],
        style=CARD
    )


def create_llm_placeholder_card() -> html.Div:
    return html.Div(
        [
            html.H4("Prediction explanation (LLM - preview)", style=SECTION_TITLE),
            html.Div(
                "This section will soon provide an AI-generated explanation of your predicted risk and the most influential features (via SHAP).",
                style={
                    "backgroundColor": "#f9fafb", "border": "1px solid #eee", "borderRadius": "8px",
                    "padding": "12px", "color": "#374151", "fontSize": "14px", "lineHeight": "1.5",
                },
            ),
            create_info_details(
                "What's this?",
                "A simple explanation (written in plain language) of why the model predicts a certain risk, based on the most important features. Coming soon."
            ),
        ],
        style={**CARD, "flex": 1}
    )