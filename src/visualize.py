"""Plotly charts for posterior densities, lift, forest plots, and expected loss."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.bayesian import PosteriorResult

CONTROL_COLOR = "#64748B"
TREATMENT_COLOR = "#2563EB"
LIFT_COLOR = "#0F766E"
NEGATIVE_COLOR = "#DC2626"
POSITIVE_COLOR = "#16A34A"


def _kde_xy(samples: np.ndarray, n_grid: int = 256) -> tuple[np.ndarray, np.ndarray]:
    samples = np.asarray(samples, dtype=float)
    samples = samples[np.isfinite(samples)]
    if samples.size == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 0.0])
    lo, hi = np.quantile(samples, [0.001, 0.999])
    if lo == hi:
        lo, hi = lo - 1e-6, hi + 1e-6
    pad = 0.05 * (hi - lo)
    grid = np.linspace(lo - pad, hi + pad, n_grid)
    std = samples.std()
    bw = 1.06 * std * (samples.size ** -0.2) if std > 0 else (hi - lo) / 50
    bw = max(bw, 1e-12)
    z = (grid[:, None] - samples[None, :]) / bw
    dens = np.exp(-0.5 * z**2).mean(axis=1) / (bw * np.sqrt(2 * np.pi))
    return grid, dens


def posterior_overlay(result: PosteriorResult) -> go.Figure:
    x_c, y_c = _kde_xy(result.control_samples)
    x_t, y_t = _kde_xy(result.treatment_samples)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x_c,
            y=y_c,
            fill="tozeroy",
            name="Control",
            line=dict(color=CONTROL_COLOR, width=2),
            hovertemplate="Control %{x:.4g}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x_t,
            y=y_t,
            fill="tozeroy",
            name="Treatment",
            line=dict(color=TREATMENT_COLOR, width=2),
            hovertemplate="Treatment %{x:.4g}<extra></extra>",
        )
    )
    fig.add_vline(
        x=result.control_posterior_mean,
        line_dash="dot",
        line_color=CONTROL_COLOR,
        annotation_text="E[control]",
        annotation_position="top left",
    )
    fig.add_vline(
        x=result.treatment_posterior_mean,
        line_dash="dot",
        line_color=TREATMENT_COLOR,
        annotation_text="E[treatment]",
        annotation_position="top right",
    )
    fig.update_layout(
        title=f"Posterior of {result.metric_name}",
        xaxis_title="Metric mean",
        yaxis_title="Posterior density",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        margin=dict(l=40, r=20, t=60, b=40),
        height=380,
    )
    return fig


def lift_distribution(result: PosteriorResult, cred_mass: float = 0.90) -> go.Figure:
    samples = result.relative_lift_samples
    x, y = _kde_xy(samples)
    lo, hi = result.relative_lift_hdi
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            fill="tozeroy",
            name="Relative lift",
            line=dict(color=LIFT_COLOR, width=2),
            hovertemplate="Lift %{x:.2%}<extra></extra>",
        )
    )
    mask = (x >= lo) & (x <= hi)
    if mask.any():
        fig.add_trace(
            go.Scatter(
                x=x[mask],
                y=y[mask],
                fill="tozeroy",
                name=f"{int(cred_mass * 100)}% HDI",
                line=dict(color="rgba(15,118,110,0.01)"),
                fillcolor="rgba(15,118,110,0.35)",
                hoverinfo="skip",
            )
        )
    fig.add_vline(x=0, line_color=NEGATIVE_COLOR, line_width=2, annotation_text="No lift")
    fig.add_vline(
        x=result.expected_relative_lift,
        line_dash="dash",
        line_color=LIFT_COLOR,
        annotation_text="Expected lift",
    )
    fig.update_layout(
        title=f"Posterior lift for {result.metric_name}",
        xaxis_title="Relative lift (treatment − control) / control",
        xaxis_tickformat=".1%",
        yaxis_title="Density",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        margin=dict(l=40, r=20, t=60, b=40),
        height=380,
    )
    return fig


def expected_loss_bars(result: PosteriorResult) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=["Ship treatment", "Hold on control"],
            y=[result.expected_loss_ship, result.expected_loss_hold],
            marker_color=[TREATMENT_COLOR, CONTROL_COLOR],
            text=[f"{result.expected_loss_ship:.4g}", f"{result.expected_loss_hold:.4g}"],
            textposition="outside",
        )
    )
    fig.update_layout(
        title="Expected loss of each decision",
        yaxis_title=f"E[loss] on {result.metric_name}",
        template="plotly_white",
        height=340,
        margin=dict(l=40, r=20, t=60, b=40),
        showlegend=False,
    )
    return fig


def probability_gauge(result: PosteriorResult) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=100 * result.prob_improvement,
            number={"suffix": "%", "font": {"size": 28}},
            title={"text": "P(treatment > control)"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": TREATMENT_COLOR},
                "steps": [
                    {"range": [0, 80], "color": "#FEE2E2"},
                    {"range": [80, 95], "color": "#FEF3C7"},
                    {"range": [95, 100], "color": "#DCFCE7"},
                ],
                "threshold": {"line": {"color": POSITIVE_COLOR, "width": 3}, "value": 95},
            },
        )
    )
    fig.update_layout(height=260, margin=dict(l=30, r=30, t=50, b=20), template="plotly_white")
    return fig


def forest_plot(results: list[PosteriorResult], cred_mass: float = 0.90) -> go.Figure:
    names = [r.metric_name for r in results][::-1]
    means = [r.expected_relative_lift for r in results][::-1]
    lows = [r.relative_lift_hdi[0] for r in results][::-1]
    highs = [r.relative_lift_hdi[1] for r in results][::-1]
    colors = [POSITIVE_COLOR if m > 0 else NEGATIVE_COLOR for m in means]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=means,
            y=names,
            mode="markers",
            marker=dict(size=12, color=colors),
            error_x=dict(
                type="data",
                symmetric=False,
                array=[h - m for h, m in zip(highs, means)],
                arrayminus=[m - lo for m, lo in zip(means, lows)],
                color="#94A3B8",
                thickness=2,
                width=5,
            ),
            hovertemplate="%{y}: %{x:.2%}<extra></extra>",
        )
    )
    fig.add_vline(x=0, line_dash="dash", line_color="#94A3B8")
    fig.update_layout(
        title=f"Expected relative lift with {int(cred_mass * 100)}% HDI",
        xaxis_title="Relative lift",
        xaxis_tickformat=".1%",
        template="plotly_white",
        height=max(280, 80 * len(results) + 80),
        margin=dict(l=120, r=30, t=60, b=40),
        showlegend=False,
    )
    return fig


def results_table(results: list[PosteriorResult], cred_mass: float = 0.90) -> pd.DataFrame:
    rows = []
    for r in results:
        rows.append(
            {
                "Metric": r.metric_name,
                "Type": r.metric_type.value,
                "n control": r.n_control,
                "n treatment": r.n_treatment,
                "Control mean": r.observed_control,
                "Treatment mean": r.observed_treatment,
                "Observed lift": r.observed_lift,
                "P(better)": r.prob_improvement,
                "Expected lift": r.expected_relative_lift,
                f"{int(cred_mass * 100)}% HDI low": r.relative_lift_hdi[0],
                f"{int(cred_mass * 100)}% HDI high": r.relative_lift_hdi[1],
                "P(lift > MDE)": r.prob_beat_mde,
                "P(harm > MDE)": r.prob_harm_beyond_mde,
                "ROPE P": r.rope_probability,
                "E[loss] ship": r.expected_loss_ship,
                "E[loss] hold": r.expected_loss_hold,
                "Frequentist p": r.frequentist_pvalue,
            }
        )
    return pd.DataFrame(rows)


def format_results_table(table: pd.DataFrame, cred_mass: float = 0.90) -> pd.DataFrame:
    """String-format a results table so Streamlit does not need pandas Styler/jinja2."""
    display = table.copy()
    mass = int(cred_mass * 100)
    percent_signed = [
        "Observed lift",
        "Expected lift",
        f"{mass}% HDI low",
        f"{mass}% HDI high",
    ]
    percent_plain = ["P(better)", "P(lift > MDE)", "P(harm > MDE)", "ROPE P"]
    four_g = ["Control mean", "Treatment mean", "E[loss] ship", "E[loss] hold", "Frequentist p"]
    for col in percent_signed:
        if col in display.columns:
            display[col] = display[col].map(lambda x: "" if pd.isna(x) else f"{x:+.1%}")
    for col in percent_plain:
        if col in display.columns:
            display[col] = display[col].map(lambda x: "" if pd.isna(x) else f"{x:.1%}")
    for col in four_g:
        if col in display.columns:
            display[col] = display[col].map(lambda x: "" if pd.isna(x) else f"{x:.4g}")
    return display


def overview_figure(results: list[PosteriorResult]) -> go.Figure:
    """Compact dashboard: forest + probability bars."""
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Probability treatment is better", "Expected relative lift"),
        horizontal_spacing=0.12,
    )
    names = [r.metric_name for r in results]
    fig.add_trace(
        go.Bar(
            x=[r.prob_improvement for r in results],
            y=names,
            orientation="h",
            marker_color=TREATMENT_COLOR,
            name="P(better)",
            hovertemplate="%{y}: %{x:.1%}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_vline(x=0.95, line_dash="dot", line_color=POSITIVE_COLOR, row=1, col=1)
    colors = [POSITIVE_COLOR if r.expected_relative_lift >= 0 else NEGATIVE_COLOR for r in results]
    fig.add_trace(
        go.Bar(
            x=[r.expected_relative_lift for r in results],
            y=names,
            orientation="h",
            marker_color=colors,
            name="Expected lift",
            hovertemplate="%{y}: %{x:.2%}<extra></extra>",
        ),
        row=1,
        col=2,
    )
    fig.update_xaxes(tickformat=".0%", row=1, col=1, range=[0, 1])
    fig.update_xaxes(tickformat=".1%", row=1, col=2)
    fig.update_layout(template="plotly_white", height=max(320, 70 * len(results) + 80), showlegend=False)
    return fig
