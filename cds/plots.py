# Plotly chart builders
# Dark-mode, monochrome white/grey palette

from __future__ import annotations
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# Colour palette
PALETTE = {
    "line1": "#FFFFFF",
    "line2": "#A0A0A0",
    "line3": "#666666",
    "marker": "#FFFFFF",
    "bar1": "#CCCCCC",
    "bar2": "#777777",
    "bg": "rgba(0,0,0,0)",
    "grid": "#333333",
    "text": "#FAFAFA",
}

LAYOUT_DEFAULTS = dict(
    template="plotly_dark",
    paper_bgcolor=PALETTE["bg"],
    plot_bgcolor=PALETTE["bg"],
    font=dict(family="Inter, sans-serif", color=PALETTE["text"]),
    margin=dict(l=60, r=30, t=50, b=50),
    xaxis=dict(gridcolor=PALETTE["grid"], zeroline=False),
    yaxis=dict(gridcolor=PALETTE["grid"], zeroline=False),
)


def _apply_layout(fig, title, **kwargs):
    fig.update_layout(title=title, **LAYOUT_DEFAULTS, **kwargs)
    return fig


# -- Survival & default curves --

def plot_survival_default(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["time"], y=df["survival_prob"],
        mode="lines", name="Survival Q(t)",
        line=dict(color=PALETTE["line1"], width=2),
    ))
    fig.add_trace(go.Scatter(
        x=df["time"], y=df["default_prob"],
        mode="lines", name="Default PD(t)",
        line=dict(color=PALETTE["line2"], width=2, dash="dash"),
    ))
    return _apply_layout(
        fig, "Survival & Default Probabilities",
        xaxis_title="Time (years)", yaxis_title="Probability",
        yaxis_range=[0, 1.05],
    )


# -- 1D sensitivity --

def plot_1d_sensitivity(df, x_col, y_col, x_label, y_label, title, color="line1"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df[x_col], y=df[y_col],
        mode="lines",
        line=dict(color=PALETTE.get(color, color), width=2),
        name=y_label,
    ))
    return _apply_layout(fig, title, xaxis_title=x_label, yaxis_title=y_label)


def plot_multi_sensitivity(df, x_col, y_cols, x_label, title, colors=None):
    if colors is None:
        colors = ["line1", "line2", "line3"]
    fig = go.Figure()
    for i, col in enumerate(y_cols):
        c = colors[i % len(colors)]
        fig.add_trace(go.Scatter(
            x=df[x_col], y=df[col],
            mode="lines",
            line=dict(color=PALETTE.get(c, c), width=2),
            name=col.replace("_", " ").title(),
        ))
    return _apply_layout(fig, title, xaxis_title=x_label)


# -- Heatmap --

def plot_heatmap(x, y, z, x_label, y_label, title, colorscale="Greys", z_label=""):
    fig = go.Figure(data=go.Heatmap(
        x=np.round(x, 4), y=np.round(y, 4), z=z,
        colorscale=colorscale,
        colorbar=dict(title=z_label),
    ))
    return _apply_layout(fig, title, xaxis_title=x_label, yaxis_title=y_label)


# -- Bootstrap: hazard term structure --

def plot_hazard_term_structure(df):
    tenors = df["tenor"].values
    rates = df["hazard_rate"].values

    # Build step coordinates
    x_steps, y_steps = [0.0], [rates[0]]
    for i in range(len(tenors)):
        x_steps.append(tenors[i])
        y_steps.append(rates[i])
        if i < len(tenors) - 1:
            x_steps.append(tenors[i])
            y_steps.append(rates[i + 1])

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_steps, y=y_steps, mode="lines",
        line=dict(color=PALETTE["line1"], width=2), name="λ(t)",
    ))
    fig.add_trace(go.Scatter(
        x=tenors, y=rates, mode="markers",
        marker=dict(color=PALETTE["marker"], size=8,
                    line=dict(color=PALETTE["line2"], width=1)),
        name="Knots",
    ))
    return _apply_layout(
        fig, "Bootstrapped Hazard Rate Term Structure",
        xaxis_title="Maturity (years)", yaxis_title="Hazard Rate λ",
    )


# -- Bootstrap: survival curve --

def plot_bootstrap_survival(df_curve):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_curve["time"], y=df_curve["survival_prob"],
        mode="lines", name="Q(t)",
        line=dict(color=PALETTE["line1"], width=2),
    ))
    fig.add_trace(go.Scatter(
        x=df_curve["time"], y=df_curve["default_prob"],
        mode="lines", name="PD(t)",
        line=dict(color=PALETTE["line2"], width=2, dash="dash"),
    ))
    return _apply_layout(
        fig, "Bootstrapped Survival & Default Curve",
        xaxis_title="Time (years)", yaxis_title="Probability",
        yaxis_range=[0, 1.05],
    )


# -- Bootstrap: spread comparison --

def plot_spread_comparison(df):
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df["tenor"].astype(str) + "Y", y=df["market_spread_bps"],
        name="Market Spread", marker_color=PALETTE["bar1"],
    ))
    fig.add_trace(go.Bar(
        x=df["tenor"].astype(str) + "Y", y=df["model_spread_bps"],
        name="Model Spread", marker_color=PALETTE["bar2"],
    ))
    return _apply_layout(
        fig, "Market vs Model CDS Spreads",
        xaxis_title="Tenor", yaxis_title="Spread (bps)", barmode="group",
    )
