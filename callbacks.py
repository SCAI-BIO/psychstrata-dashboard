from typing import Dict, Any
import pandas as pd
from dash import Input, Output

from model import model
from config import FEATURES_UI, PILL
from visualization import create_indicator_figure, create_shap_bar_figure, create_tsne_scatter_figure


def pack_instance(values_dict: Dict[str, Any]) -> pd.DataFrame:
    row = {col: values_dict[col] for col in model.feature_cols}
    return pd.DataFrame([row], columns=model.feature_cols)


def create_badge_style(bg: str, border: str, color: str) -> dict:
    return {
        **PILL,
        "backgroundColor": bg,
        "border": border,
        "color": color,
    }


def register_callbacks(app):
    input_ids = [f"input-{cfg.id}" for cfg in FEATURES_UI]
    value_ids = [f"value-{cfg.id}" for cfg in FEATURES_UI if cfg.kind == "numeric"]
    
    dash_inputs = [Input(comp_id, "value") for comp_id in input_ids]
    dash_inputs.append(Input("ci-slider", "value"))
    
    dash_outputs = [
        Output("pred-indicator", "figure"),
        Output("prediction-badge", "children"),
        Output("prediction-badge", "style"),
        Output("shap-bar", "figure"),
        Output("tsne-scatter", "figure"),
    ]
    for vid in value_ids:
        dash_outputs.append(Output(vid, "children"))
    
    @app.callback(dash_outputs, dash_inputs)
    def update_predictions(*args):
        values = args[:-1]
        ci_level = args[-1]
        
        values_dict = {cfg.id: v for cfg, v in zip(FEATURES_UI, values)}
        X_row = pack_instance(values_dict)
        
        prob = model.predict_proba(X_row)
        pred_fig = create_indicator_figure(prob)
        
        label, bg, border, color = model.get_conformal_prediction(X_row, ci_level)
        badge_style = create_badge_style(bg, border, color)
        
        shap_vals = model.get_shap_values(X_row)
        shap_fig = create_shap_bar_figure(shap_vals, model.feature_cols)
        
        sel_x, sel_y = model.approximate_tsne_position(X_row)
        tsne_fig = create_tsne_scatter_figure(model.tsne_df, sel_x, sel_y)
        
        numeric_displays = [f"Selected: {values_dict[cfg.id]}" for cfg in FEATURES_UI if cfg.kind == "numeric"]
        
        return [pred_fig, label, badge_style, shap_fig, tsne_fig] + numeric_displays