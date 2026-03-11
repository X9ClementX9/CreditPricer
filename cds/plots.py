# Plotly chart builders
# Dark-mode, monochrome white/grey palette

from __future__ import annotations
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



# -- Basket CDS charts --

def plot_basket_pd(labels: list, probs: list):
    # Bar chart of default probability per tranche
    fig = go.Figure(go.Bar(
        x=labels, y=[p * 100 for p in probs],
        marker_color=PALETTE["bar1"],
        text=[f"{p:.1%}" for p in probs],
        textposition="outside",
        textfont=dict(color=PALETTE["text"], size=11),
    ))
    return _apply_layout(
        fig, "P(default < T)",
        xaxis_title="Tranche", yaxis_title="%",
        yaxis_range=[0, 105],
    )


def plot_basket_spread(labels: list, spreads: list):
    # Bar chart of fair spread per tranche
    fig = go.Figure(go.Bar(
        x=labels, y=spreads,
        marker_color=PALETTE["bar1"],
        text=[f"{s:.0f}" for s in spreads],
        textposition="outside",
        textfont=dict(color=PALETTE["text"], size=11),
    ))
    return _apply_layout(
        fig, "Fair Spread",
        xaxis_title="Tranche", yaxis_title="bps",
    )


def plot_basket_default_count(n_defaults: list, probs: list):
    # Bar chart of default count distribution
    labels = [str(n) for n in n_defaults]
    fig = go.Figure(go.Bar(
        x=labels, y=[p * 100 for p in probs],
        marker_color=PALETTE["bar1"],
        text=[f"{p:.1%}" for p in probs],
        textposition="outside",
        textfont=dict(color=PALETTE["text"], size=11),
    ))
    return _apply_layout(
        fig, "Default Count Distribution",
        xaxis_title="Number of Defaults", yaxis_title="%",
        yaxis_range=[0, 105],
    )
