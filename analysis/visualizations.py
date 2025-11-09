"""
Reusable visualisation helpers for OI back-analysis.

The functions return Plotly figures so they can be embedded in notebooks,
exported as HTML, or integrated into the Flask UI in future iterations.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from .strings import STRINGS


def plot_underlying_vs_oi(
    summary_df: pd.DataFrame,
    *,
    exchange: str,
    include_pcr: bool = True,
) -> go.Figure:
    """
    Plot underlying price alongside call/put OI totals.

    Parameters
    ----------
    summary_df:
        Output from `OIBackAnalysisPipeline.build_summary_table`.
    exchange:
        Exchange identifier used in the title.
    include_pcr:
        If True, overlays the PCR on a secondary axis.
    """

    if summary_df.empty:
        raise ValueError("summary_df is empty – run the pipeline before plotting.")

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=summary_df["timestamp"],
            y=summary_df["underlying_price"],
            mode="lines",
            name=STRINGS["chart_underlying_axis_title"],
            line=dict(color="#FFA500", width=2),
            yaxis="y1",
        )
    )

    if "call_oi" in summary_df.columns:
        fig.add_trace(
            go.Scatter(
                x=summary_df["timestamp"],
                y=summary_df["call_oi"],
                mode="lines",
                name=STRINGS["chart_call_oi_label"],
                line=dict(color="#1f77b4", dash="dash"),
                yaxis="y2",
            )
        )

    if "put_oi" in summary_df.columns:
        fig.add_trace(
            go.Scatter(
                x=summary_df["timestamp"],
                y=summary_df["put_oi"],
                mode="lines",
                name=STRINGS["chart_put_oi_label"],
                line=dict(color="#d62728", dash="dot"),
                yaxis="y2",
            )
        )

    include_pcr_axis = include_pcr and "pcr" in summary_df.columns

    if include_pcr_axis:
        fig.add_trace(
            go.Scatter(
                x=summary_df["timestamp"],
                y=summary_df["pcr"],
                mode="lines",
                name=STRINGS["chart_pcr_label"],
                line=dict(color="#2ca02c", width=1.5),
                yaxis="y3",
                opacity=0.6,
            )
        )

    fig.update_layout(
        title=f"{STRINGS['chart_underlying_vs_oi_title']} – {exchange.upper()}",
        xaxis=dict(title=STRINGS["chart_timestamp_axis_title"]),
        yaxis=dict(
            title=STRINGS["chart_underlying_axis_title"],
            titlefont=dict(color="#FFA500"),
            tickfont=dict(color="#FFA500"),
        ),
        yaxis2=dict(
            title=STRINGS["chart_oi_axis_title"],
            titlefont=dict(color="#1f77b4"),
            tickfont=dict(color="#1f77b4"),
            overlaying="y",
            side="right",
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=60, r=80, t=60, b=40),
    )

    if include_pcr_axis:
        fig.update_layout(
            yaxis3=dict(
                title=STRINGS["chart_pcr_label"],
                titlefont=dict(color="#2ca02c"),
                tickfont=dict(color="#2ca02c"),
                anchor="free",
                overlaying="y",
                side="right",
                position=1.0,
            )
        )

    return fig


def plot_net_oi_vs_underlying_change(
    summary_df: pd.DataFrame,
    *,
    exchange: str,
    scatter_alpha: float = 0.7,
) -> go.Figure:
    """
    Scatter plot comparing net OI change against underlying return.

    Useful for spotting lead/lag tendencies between OI surges and price moves.
    """

    if summary_df.empty:
        raise ValueError("summary_df is empty – run the pipeline before plotting.")

    if "net_oi" not in summary_df.columns:
        raise ValueError("summary_df must include 'net_oi' column.")

    if "underlying_return_pct" not in summary_df.columns:
        raise ValueError("summary_df must include 'underlying_return_pct' column.")

    df = summary_df.copy()
    df["net_oi_change"] = df["net_oi"].diff()

    fig = px.scatter(
        df,
        x="net_oi_change",
        y="underlying_return_pct",
        color="underlying_return_pct",
        color_continuous_scale="RdYlGn",
        opacity=scatter_alpha,
        title=f"{STRINGS['chart_net_oi_label']} Δ vs Underlying Return – {exchange.upper()}",
        labels={
            "net_oi_change": f"{STRINGS['chart_net_oi_label']} Δ",
            "underlying_return_pct": f"{STRINGS['chart_underlying_axis_title']} % Δ",
        },
    )

    fig.update_traces(marker=dict(size=8, line=dict(width=0.5, color="#333333")))
    fig.update_layout(margin=dict(l=60, r=40, t=70, b=60))

    return fig


def plot_heatmap(
    heatmap_df: pd.DataFrame,
    *,
    title_suffix: Optional[str] = None,
) -> go.Figure:
    """
    Visualise strike vs time data as a heatmap.

    Parameters
    ----------
    heatmap_df:
        Output from `OIBackAnalysisPipeline.build_strike_heatmap`.
    title_suffix:
        Optional string appended to the chart title (e.g., exchange name).
    """

    if heatmap_df.empty:
        raise ValueError("heatmap_df is empty – generate the matrix before plotting.")

    title = STRINGS["chart_heatmap_title"]
    if title_suffix:
        title = f"{title} – {title_suffix}"

    fig = go.Figure(
        data=go.Heatmap(
            z=heatmap_df.values,
            x=heatmap_df.columns,
            y=heatmap_df.index,
            colorscale="RdBu",
            colorbar=dict(title=STRINGS["chart_heatmap_colorbar"]),
            reversescale=True,
        )
    )

    fig.update_layout(
        title=title,
        xaxis=dict(title=STRINGS["chart_timestamp_axis_title"]),
        yaxis=dict(title="Strike"),
        margin=dict(l=80, r=40, t=70, b=60),
    )

    return fig

