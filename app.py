from dash import Dash, html

from config import CARD, FEATURES_UI
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


app = Dash(__name__)
app.title = "Treatment Resistance Classifier (Demo)"
server = app.server


def create_layout():
    features_grid = html.Div(
        [create_feature_input(cfg) for cfg in FEATURES_UI],
        style={"display": "grid", "gridTemplateColumns": "repeat(auto-fit, minmax(280px, 1fr))", "gap": "12px"},
    )
    
    left_column = html.Div(
        [
            html.Div(
                [
                    html.H4("Enter patient features", style={"margin": "0 0 8px 0", "fontSize": "18px"}),
                    features_grid,
                ],
                style=CARD
            ),
            create_llm_summary_card(),
        ],
        style={"flex": "1.7", "minWidth": "320px", "display": "flex", "flexDirection": "column", "gap": "12px"}
    )
    
    right_column = html.Div(
        [create_prediction_card(), create_shap_card(), create_tsne_card()],
        style={"flex": "1", "minWidth": "360px", "display": "flex", "flexDirection": "column", "gap": "12px"}
    )
    
    main_content = html.Div(
        [left_column, right_column],
        style={"display": "flex", "gap": "20px", "marginTop": "20px", "flexWrap": "wrap", "alignItems": "stretch"}
    )
    
    footer = html.Div(
        "This demo is for educational purposes only and not a medical device.",
        style={"fontSize": "12px", "color": "#6b7280", "textAlign": "center", "marginTop": "16px"}
    )
    
    return html.Div(
        [
            html.Div(
                [create_header_card(model.auc), main_content, footer],
                style={"maxWidth": "1200px", "margin": "0 auto"}
            ),
        ],
        style={"padding": "24px", "backgroundColor": "#f6f7fb", "fontFamily": "Inter, Arial, sans-serif"}
    )


app.layout = create_layout()
register_api(server)
register_callbacks(app)


if __name__ == "__main__":
    app.run(debug=True, port=8050)
