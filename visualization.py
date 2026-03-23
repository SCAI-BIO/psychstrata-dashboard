from typing import List
import numpy as np
import pandas as pd
import plotly.graph_objects as go


def create_indicator_figure(prob: float) -> go.Figure:
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


def create_shap_bar_figure(shap_vals: np.ndarray, feature_names: List[str]) -> go.Figure:
    s = pd.Series(shap_vals, index=feature_names)
    s = s.sort_values(key=lambda x: x.abs(), ascending=True)
    colors = ["#d62728" if v > 0 else "#2ca02c" for v in s.values]
    
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
    fig.add_shape(
        type="line", x0=0, x1=0, y0=-0.5, y1=len(s)-0.5,
        line=dict(color="#9ca3af", width=1)
    )
    return fig


def create_tsne_scatter_figure(tsne_df: pd.DataFrame, sel_x: float, sel_y: float) -> go.Figure:
    mask_resistant = tsne_df["y"].values == 1
    mask_responsive = tsne_df["y"].values == 0
    
    trace_resp = go.Scattergl(
        x=tsne_df.loc[mask_responsive, "tsne_1"],
        y=tsne_df.loc[mask_responsive, "tsne_2"],
        mode="markers",
        name="Responsive",
        marker=dict(color="#2ca02c", size=6, opacity=0.75),
        hovertemplate="Class: Responsive<extra></extra>",
    )
    trace_resi = go.Scattergl(
        x=tsne_df.loc[mask_resistant, "tsne_1"],
        y=tsne_df.loc[mask_resistant, "tsne_2"],
        mode="markers",
        name="Resistant",
        marker=dict(color="#d62728", size=6, opacity=0.75),
        hovertemplate="Class: Resistant<extra></extra>",
    )
    trace_sel = go.Scattergl(
        x=[sel_x],
        y=[sel_y],
        mode="markers",
        name="Current selection",
        marker=dict(color="#1f77b4", size=12, line=dict(color="white", width=1.5)),
        hovertemplate="Current selection<extra></extra>",
    )
    
    fig = go.Figure(data=[trace_resp, trace_resi, trace_sel])
    fig.update_layout(
        xaxis_title="t-SNE 1",
        yaxis_title="t-SNE 2",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        margin=dict(l=40, r=20, t=40, b=40),
        plot_bgcolor="white",
    )
    return fig