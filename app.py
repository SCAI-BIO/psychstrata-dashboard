from dash import Dash, Input, Output, dcc, html

from auth import configure_auth
from config import (
    BRAND_BLUE,
    BRAND_BLUE_DARK,
    BRAND_BLUE_LIGHT,
    CALM_AMBER,
    CALM_AMBER_LIGHT,
    CALM_BLUE,
    CALM_BLUE_LIGHT,
    CALM_TEAL,
    CALM_TEAL_LIGHT,
    CARD,
    FEATURES_UI,
)
from model import model
from api import register_api
from components import (
    create_feature_input,
    create_header_card,
    create_prediction_card,
    create_shap_card,
    create_tsne_card,
    create_llm_summary_card,
)
from callbacks import register_callbacks


app = Dash(__name__, suppress_callback_exceptions=True)
app.title = "Treatment Resistance Classifier (Demo)"
server = app.server
configure_auth(server)

PSYCH_STRATA_URL = "https://psych-strata.eu/"
PSYCH_STRATA_LOGO_URL = "https://psych-strata.eu/wp-content/uploads/2023/05/logo_footer_blue.png"
PATIENT_VIEW_ICON_URL = "https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/icons/person-badge.svg"
CLINICIAN_VIEW_ICON_URL = "https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/icons/clipboard2-pulse.svg"


def primary_cta_style() -> dict:
    return {
        "marginTop": "14px",
        "display": "inline-block",
        "padding": "8px 12px",
        "borderRadius": "8px",
        "backgroundColor": BRAND_BLUE,
        "color": "white",
        "fontWeight": "700",
        "fontSize": "13px",
        "boxShadow": "0 2px 8px rgba(0, 111, 185, 0.18)",
    }


def create_feature_section():
    features_grid = html.Div(
        [create_feature_input(cfg) for cfg in FEATURES_UI],
        style={"display": "grid", "gridTemplateColumns": "repeat(auto-fit, minmax(280px, 1fr))", "gap": "12px"},
    )

    return html.Div(
        [
            html.H4("Enter patient features", style={"margin": "0 0 8px 0", "fontSize": "18px"}),
            html.Div(
                "Use these values to represent the current case before reviewing the dashboard together.",
                style={"color": "#6b7280", "fontSize": "13px", "lineHeight": "1.5", "marginBottom": "10px"},
            ),
            features_grid,
        ],
        style=CARD,
    )


def create_footer():
    return html.Div(
        "This demo is for educational purposes only and not a medical device.",
        style={"fontSize": "12px", "color": "#6b7280", "textAlign": "center", "marginTop": "auto", "paddingTop": "16px"}
    )


def create_view_switcher(active_view: str) -> html.Div:
    def nav_item(label: str, href: str, view: str) -> dcc.Link:
        is_active = active_view == view
        return dcc.Link(
            label,
            href=href,
            style={
                "display": "inline-block",
                "padding": "8px 12px",
                "borderRadius": "999px",
                "border": f"1px solid {BRAND_BLUE}" if is_active else "1px solid #dbe2f0",
                "backgroundColor": BRAND_BLUE if is_active else "white",
                "color": "white" if is_active else BRAND_BLUE_DARK,
                "fontSize": "13px",
                "fontWeight": "700" if is_active else "600",
                "textDecoration": "none",
            },
        )

    return html.Div(
        [
            nav_item("Home", "/", "home"),
            nav_item("Patient view", "/patients", "patients"),
            nav_item("Clinician view", "/clinicians", "clinicians"),
        ],
        style={
            "display": "flex",
            "gap": "8px",
            "flexWrap": "wrap",
            "alignItems": "center",
            "marginTop": "14px",
            "padding": "6px",
            "borderRadius": "999px",
            "backgroundColor": BRAND_BLUE_LIGHT,
            "border": "1px solid #cbe8f5",
            "width": "fit-content",
        },
    )


def create_info_card(title: str, items: list[str]) -> html.Div:
    return html.Div(
        [
            html.H4(title, style={"margin": "0 0 10px 0", "fontSize": "18px"}),
            html.Ul(
                [html.Li(item, style={"marginBottom": "8px", "color": "#4b5563", "lineHeight": "1.5"}) for item in items],
                style={"margin": 0, "paddingLeft": "20px"},
            ),
        ],
        style={**CARD, "height": "100%"},
    )


def prediction_label_style(label: str) -> dict:
    styles = {
        "Responsive": {"backgroundColor": CALM_TEAL_LIGHT, "border": f"1px solid {CALM_TEAL}", "color": "#1F5F5B"},
        "Resistant": {"backgroundColor": CALM_BLUE_LIGHT, "border": f"1px solid {CALM_BLUE}", "color": "#1F3B63"},
        "Uncertain": {"backgroundColor": CALM_AMBER_LIGHT, "border": f"1px solid {CALM_AMBER}", "color": "#7A5C00"},
    }
    return {
        "display": "inline-block",
        "padding": "4px 10px",
        "borderRadius": "999px",
        "fontSize": "12px",
        "fontWeight": "700",
        **styles.get(label, {"backgroundColor": "#eef2ff", "border": "1px solid #c7d2fe", "color": "#3730a3"}),
    }


def create_component_guide_card(
    title: str,
    image_src: str,
    overview: str,
    detail_paragraphs: list[str],
    interpretation_note: str,
    spotlight_title: str | None = None,
    spotlight_paragraphs: list[str] | None = None,
    label_items: list[tuple[str, str]] | None = None,
    style_override: dict | None = None,
    image_style_override: dict | None = None,
) -> html.Div:
    label_items = label_items or []
    card_style = {**CARD}
    if style_override:
        card_style.update(style_override)
    image_style = {"width": "100%", "borderRadius": "10px", "border": "1px solid #e5e7eb", "display": "block"}
    if image_style_override:
        image_style.update(image_style_override)

    return html.Div(
        [
            html.H4(title, style={"margin": "0 0 10px 0", "fontSize": "18px"}),
            html.Img(
                src=image_src,
                alt=title,
                style=image_style,
            ),
            html.Div(
                [
                    html.P(
                        overview,
                        style={"margin": 0, "color": "#374151", "lineHeight": "1.6"},
                    ),
                    html.Div(
                        [
                            html.P(
                                paragraph,
                                style={"margin": "10px 0 0 0", "color": "#4b5563", "lineHeight": "1.6"},
                            )
                            for paragraph in detail_paragraphs
                        ],
                    ),
                    html.Div(
                        [
                            html.Div(spotlight_title, style={"fontWeight": "700", "marginBottom": "6px"}),
                            html.Div(
                                [
                                    html.P(
                                        paragraph,
                                        style={"margin": "6px 0 0 0", "color": "#374151", "lineHeight": "1.5"},
                                    )
                                    for paragraph in (spotlight_paragraphs or [])
                                ],
                            ),
                        ],
                        style={
                            "marginTop": "10px",
                            "padding": "8px",
                            "borderRadius": "8px",
                            "backgroundColor": "#f0f9ff",
                            "border": "1px solid #bae6fd",
                            "color": "#0f172a",
                            "fontSize": "13px",
                            "display": "block" if spotlight_title else "none",
                        },
                    ),
                    html.Div(
                        [
                            html.Div("Possible prediction labels", style={"fontWeight": "700", "marginBottom": "8px"}),
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            html.Div(html.Span(label, style=prediction_label_style(label)), style={"marginBottom": "8px"}),
                                            html.Div(description, style={"color": "#4b5563", "lineHeight": "1.5"}),
                                        ],
                                        style={
                                            "padding": "8px",
                                            "borderRadius": "8px",
                                            "border": "1px solid #e5e7eb",
                                            "backgroundColor": "white",
                                        },
                                    )
                                    for label, description in label_items
                                ],
                                style={"display": "grid", "gridTemplateColumns": "repeat(auto-fit, minmax(160px, 1fr))", "gap": "8px"},
                            ),
                        ],
                        style={
                            "marginTop": "10px",
                            "padding": "8px",
                            "borderRadius": "8px",
                            "backgroundColor": "#f8fafc",
                            "border": "1px solid #e5e7eb",
                            "fontSize": "13px",
                            "display": "block" if label_items else "none",
                        },
                    ),
                    html.Div(
                        [
                            html.Span("Interpretation note: ", style={"fontWeight": "700"}),
                            interpretation_note,
                        ],
                        style={
                            "marginTop": "10px",
                            "padding": "8px",
                            "borderRadius": "8px",
                            "backgroundColor": "#f8fafc",
                            "border": "1px solid #e5e7eb",
                            "color": "#374151",
                            "lineHeight": "1.5",
                            "fontSize": "13px",
                        },
                    ),
                ],
                style={"marginTop": "12px"},
            ),
        ],
        style=card_style,
    )


def create_input_guide_card() -> html.Div:
    return html.Div(
        [
            html.Div(
                [
                    html.H4("Input profile panel", style={"margin": "0 0 10px 0", "fontSize": "18px"}),
                    html.P(
                        "The input profile panel contains the values used by the model. It is available in the Clinician view and hidden in the Patient view so the simpler view stays focused on the result.",
                        style={"margin": 0, "color": "#374151", "lineHeight": "1.6"},
                    ),
                    html.P(
                        "Numeric values are adjusted with sliders, while categorical values are selected with dropdown menus. Changing an input changes the profile that the prediction, explanation, SHAP chart, and t-SNE map refer to.",
                        style={"margin": "10px 0 0 0", "color": "#4b5563", "lineHeight": "1.6"},
                    ),
                    html.Div(
                        [
                            html.Span("Interpretation note: ", style={"fontWeight": "700"}),
                            "The values define the demonstration scenario. They are not automatically verified clinical records in this demo.",
                        ],
                        style={
                            "marginTop": "10px",
                            "padding": "10px",
                            "borderRadius": "8px",
                            "backgroundColor": "#f8fafc",
                            "border": "1px solid #e5e7eb",
                            "color": "#374151",
                            "lineHeight": "1.5",
                            "fontSize": "14px",
                        },
                    ),
                ],
                style={"flex": "1.2", "minWidth": "280px"},
            ),
            html.Img(
                src="/assets/tutorial-inputs.png",
                alt="Input profile panel",
                style={
                    "width": "58%",
                    "maxWidth": "760px",
                    "minWidth": "520px",
                    "height": "100%",
                    "minHeight": "350px",
                    "maxHeight": "380px",
                    "objectFit": "cover",
                    "objectPosition": "top left",
                    "borderRadius": "10px",
                    "border": "1px solid #e5e7eb",
                    "display": "block",
                },
            ),
        ],
        style={
            **CARD,
            "gridColumn": "2 / span 3",
            "display": "flex",
            "gap": "18px",
            "alignItems": "stretch",
            "justifyContent": "space-between",
            "minHeight": "390px",
        },
    )


def create_tutorial_section() -> html.Div:
    return html.Div(
        [
            html.H3("What each panel means", style={"margin": "0 0 12px 0"}),
            html.Div(
                "This guide explains the dashboard components in neutral, non-technical language. It describes what each panel represents and how the information can be read.",
                style={"color": "#4b5563", "lineHeight": "1.5", "marginBottom": "14px"},
            ),
            html.Div(
                [
                    create_component_guide_card(
                        "Prediction with uncertainty slider",
                        "/assets/tutorial-prediction.png",
                        "This panel shows the model's estimated probability of treatment resistance for the current profile. It also shows how certain or uncertain the model is when asked to make a prediction at a selected confidence level.",
                        [
                            "The gauge is the risk estimate: closer to 0% means lower estimated resistance risk; closer to 100% means higher estimated risk.",
                            "The slider sets the confidence interval for the label. For example, 90% means the method is calibrated to include the correct class (Resistant / Responsive) about 90 out of 100 times in similar future cases.",
                            "Higher confidence levels are more conservative and may produce 'Uncertain' labels.",
                        ],
                        "The probability and the uncertainty label describe model output only. They do not establish diagnosis, prognosis, or treatment response on their own.",
                        spotlight_title="How the uncertainty slider changes the label",
                        spotlight_paragraphs=[
                            "The slider does not change the underlying patient profile. It changes how strict the model is when deciding whether the patient will respond to treatment or not.",
                            "At 80-90%, Responsive / Resistant labels may appear more often. At 95-99%, the model is more cautious and may show 'Uncertain' when evidence is not strong enough to determine resistance.",
                        ],
                        label_items=[
                            ("Responsive", "Patient is estimated to respond to treatment."),
                            ("Resistant", "Patient is estimated to be resistant to treatment."),
                            ("Uncertain", "Both classes remain plausible at the selected confidence level."),
                        ],
                        style_override={"gridColumn": "1", "gridRow": "1 / span 2", "padding": "12px", "alignSelf": "start"},
                        image_style_override={"maxHeight": "399px", "objectFit": "cover", "objectPosition": "top"},
                    ),
                    create_component_guide_card(
                        "Text explanation component",
                        "/assets/tutorial-explanation.png",
                        "This component summarizes the current model result in plain language. It is a readable explanation of the selected profile and the model output, rather than a technical chart.",
                        [
                            "The explanation is generated for the current feature values. If those values change, the text should be refreshed so that it matches the current profile.",
                            "The text may mention factors that contributed to a higher or lower estimate and may include supporting literature references where relevant.",
                            "Because it is generated text, it should be read as a summary of the dashboard output rather than an independent source of truth.",
                        ],
                        "The explanation should stay consistent with the prediction and visual panels. If it appears too strong or unclear, the underlying dashboard outputs are the reference point.",
                        style_override={"gridColumn": "2"},
                    ),
                    create_component_guide_card(
                        "SHAP view",
                        "/assets/tutorial-shap.png",
                        "SHAP is a model-explanation method. In simple terms, it breaks down the prediction and shows how much each feature contributed to moving the estimate up or down for this specific profile.",
                        [
                            "Each bar represents one feature, such as symptom severity, adherence, or treatment history. Longer bars indicate a larger influence within the model for this specific profile.",
                            "Bars in right direction increase the estimated treatment-resistance risk; bars in the left direction decrease it.",
                            "SHAP values are local explanations: they describe the prediction for the current profile, not every possible patient profile.",
                        ],
                        "SHAP does not prove cause and effect. It explains how the model used the information, not why a person will or will not respond to treatment.",
                        style_override={"gridColumn": "3"},
                    ),
                    create_component_guide_card(
                        "t-SNE population map",
                        "/assets/tutorial-tsne.png",
                        "This map gives a visual overview of similarity between synthetic profiles in the demo dataset. The current profile is shown in relation to other examples.",
                        [
                            "Each dot is one synthetic example. Dots that appear close together correspond to similar patient profiles according to the projection.",
                            "The highlighted dot marks the current profile. Its position can show whether the profile lies near many similar examples or in a sparser region of the map.",
                            "The projection method used here is t-SNE, a method for drawing high-dimensional data as a two-dimensional map.",
                        ],
                        "Read this as a context picture only. Similar-looking profiles can still have different clinical stories and outcomes.",
                        style_override={"gridColumn": "4"},
                    ),
                    create_input_guide_card(),
                ],
                className="component-tutorial-grid",
            ),
        ],
        style={**CARD, "marginTop": "18px"},
    )


def create_landing_page():
    tile_style = {
        **CARD,
        "textDecoration": "none",
        "color": "#111827",
        "display": "block",
        "flex": "1",
        "minWidth": "420px",
        "padding": "22px",
        "minHeight": "220px",
        "border": f"2px solid {BRAND_BLUE_LIGHT}",
        "cursor": "pointer",
        "boxShadow": "0 6px 16px rgba(0, 111, 185, 0.10)",
        "transition": "transform 120ms ease-out",
    }

    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Img(
                                        src=PSYCH_STRATA_LOGO_URL,
                                        alt="Psych-STRATA logo",
                                        style={"height": "72px", "width": "auto"},
                                    ),
                                    html.Div(
                                        [
                                            html.H2("Psych-STRATA Demonstrator", style={"margin": "0 0 6px 0"}),
                                            html.Div(
                                                "Treatment Resistance Classifier • Research demo • Not medical advice",
                                                style={"color": "#6b7280", "fontSize": "13px"},
                                            ),
                                        ]
                                    ),
                                ],
                                style={"display": "flex", "gap": "14px", "alignItems": "center", "flexWrap": "wrap"},
                            ),
                            html.P(
                                "Psych-STRATA is an EU-funded research project focused on personalised mental health care. "
                                "This dashboard is a prototype that demonstrates treatment-resistance prediction "
                                "using synthetic (non-real) data.",
                                style={"margin": "12px 0 0 0", "color": "#374151", "lineHeight": "1.6"},
                            ),
                            html.P(
                                "It is intended for research communication and software evaluation only. "
                                "It is not a medical device and must not be used for diagnosis or treatment decisions.",
                                style={"margin": "8px 0 0 0", "color": "#4b5563", "lineHeight": "1.6"},
                            ),
                            html.A(
                                "Visit the Psych-STRATA project website",
                                href=PSYCH_STRATA_URL,
                                target="_blank",
                                rel="noopener noreferrer",
                                style={
                                    "display": "inline-block",
                                    "marginTop": "12px",
                                    "color": BRAND_BLUE_DARK,
                                    "fontWeight": "600",
                                    "textDecoration": "none",
                                },
                            ),
                        ]
                    ),
                ],
                style={**CARD, "padding": "20px"},
            ),
            html.Div(
                [
                    dcc.Link(
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.Img(
                                            src=PATIENT_VIEW_ICON_URL,
                                            alt="Patient view icon",
                                            style={"height": "24px", "width": "24px"},
                                        ),
                                        html.H4("Patient view", style={"margin": 0}),
                                    ],
                                    style={"display": "flex", "alignItems": "center", "gap": "8px", "marginBottom": "6px"},
                                ),
                                html.Div(
                                    (
                                        "A simplified view for shared conversations. It keeps the focus on the prediction, "
                                        "uncertainty, and plain-language explanation, while leaving out advanced charts that "
                                        "can distract from the main result."
                                    ),
                                    style={"color": "#4b5563", "lineHeight": "1.6", "marginBottom": "8px"},
                                ),
                                html.Div(
                                    (
                                        "This view is useful when the goal is to understand the result without showing all "
                                        "technical details at once."
                                    ),
                                    style={"color": "#4b5563", "lineHeight": "1.6"},
                                ),
                                html.Div(
                                    "Open Patient view ->",
                                    style=primary_cta_style(),
                                ),
                            ]
                        ),
                        href="/patients",
                        style={**tile_style, "textDecoration": "none"},
                    ),
                    dcc.Link(
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.Img(
                                            src=CLINICIAN_VIEW_ICON_URL,
                                            alt="Clinician view icon",
                                            style={"height": "24px", "width": "24px"},
                                        ),
                                        html.H4("Clinician view", style={"margin": 0}),
                                    ],
                                    style={"display": "flex", "alignItems": "center", "gap": "8px", "marginBottom": "6px"},
                                ),
                                html.Div(
                                    (
                                        "A fuller dashboard for reviewing how the result was produced. It shows the input "
                                        "profile, prediction, uncertainty, feature contributions, and population context in "
                                        "one place."
                                    ),
                                    style={"color": "#4b5563", "lineHeight": "1.6", "marginBottom": "8px"},
                                ),
                                html.Div(
                                    (
                                        "This view provides more detail for exploring why an estimate changes and how each "
                                        "component relates to the selected profile."
                                    ),
                                    style={"color": "#4b5563", "lineHeight": "1.6"},
                                ),
                                html.Div(
                                    "Open Clinician view ->",
                                    style=primary_cta_style(),
                                ),
                            ]
                        ),
                        href="/clinicians",
                        style={**tile_style, "textDecoration": "none"},
                    ),
                ],
                style={"display": "flex", "gap": "18px", "marginTop": "18px", "flexWrap": "wrap"},
            ),
            html.Div(
                create_tutorial_section(),
            ),
            create_footer(),
        ],
        style={"width": "100%", "minHeight": "calc(100vh - 48px)", "display": "flex", "flexDirection": "column"},
    )


def create_patient_page():
    hidden_feature_section = html.Div(
        create_feature_section(),
        style={"display": "none"},
        **{"aria-hidden": "true"},
    )

    left_column = html.Div(
        [create_llm_summary_card()],
        style={"flex": "1", "minWidth": "360px", "display": "flex", "flexDirection": "column", "gap": "12px"},
    )

    right_column = html.Div(
        [
            create_prediction_card(),
            dcc.Graph(id="shap-bar", style={"display": "none"}),
            dcc.Graph(id="tsne-scatter", style={"display": "none"}),
        ],
        style={"flex": "1", "minWidth": "360px", "display": "flex", "flexDirection": "column", "gap": "12px"},
    )

    return html.Div(
        [
            create_header_card(model.auc),
            create_view_switcher("patients"),
            hidden_feature_section,
            html.Div(
                [left_column, right_column],
                style={"display": "flex", "gap": "20px", "marginTop": "20px", "flexWrap": "wrap", "alignItems": "stretch"},
            ),
            html.Div(
                [
                    create_info_card(
                        "About this patient view",
                        [
                            "This page keeps the discussion focused on the main result and plain-language explanation.",
                            "The same profile values are used as in the fuller dashboard, but the input form is hidden here.",
                            "This view is intended for reviewing the result without extra technical detail.",
                        ],
                    ),
                    create_info_card(
                        "Need changes?",
                        [
                            "Profile changes are made in the Clinician view.",
                            "After changes, return here to review the simpler result and explanation."
                        ],
                    ),
                ],
                style={"display": "grid", "gridTemplateColumns": "repeat(auto-fit, minmax(360px, 1fr))", "gap": "20px", "marginTop": "20px"},
            ),
            create_footer(),
        ],
        style={"width": "100%", "minHeight": "calc(100vh - 48px)", "display": "flex", "flexDirection": "column"},
    )


def create_clinician_page():
    left_column = html.Div(
        [create_feature_section(), create_llm_summary_card()],
        style={"flex": "1.7", "minWidth": "320px", "display": "flex", "flexDirection": "column", "gap": "12px"},
    )

    right_column = html.Div(
        [create_prediction_card(), create_shap_card(), create_tsne_card()],
        style={"flex": "1", "minWidth": "360px", "display": "flex", "flexDirection": "column", "gap": "12px"},
    )

    return html.Div(
        [
            create_header_card(model.auc),
            create_view_switcher("clinicians"),
            html.Div(
                [left_column, right_column],
                style={"display": "flex", "gap": "20px", "marginTop": "20px", "flexWrap": "wrap", "alignItems": "stretch"},
            ),
            create_footer(),
        ],
        style={"width": "100%", "minHeight": "calc(100vh - 48px)", "display": "flex", "flexDirection": "column"},
    )


def create_layout():
    return html.Div(
        [
            dcc.Location(id="url", refresh=False),
            html.Div(
                id="page-content",
                style={"width": "100%", "minHeight": "calc(100vh - 48px)"}
            ),
        ],
        style={
            "padding": "24px",
            "backgroundColor": "#f6f7fb",
            "fontFamily": "Inter, Arial, sans-serif",
            "minHeight": "100vh",
            "boxSizing": "border-box",
        }
    )


app.layout = create_layout()
register_api(server)
register_callbacks(app)


@app.callback(Output("page-content", "children"), Input("url", "pathname"))
def render_page(pathname: str):
    if pathname in ("/patients",):
        return create_patient_page()
    if pathname in ("/clinicians", "/clinicans"):
        return create_clinician_page()
    return create_landing_page()


if __name__ == "__main__":
    app.run(debug=True, port=8050)
