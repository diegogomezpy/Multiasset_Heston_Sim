"""
app/charts.py
-------------
Every Plotly figure in the app as a standalone function.

All functions:
- take plain numpy / pandas arguments
- return a plotly.graph_objects.Figure
- accept a Translator instance for axis/legend labels
- have no Streamlit calls — tested independently

Usage
-----
from app.charts import (
    build_call_prob_curve, build_irr_distribution,
    build_fan_chart, build_wof_fan, build_corr_heatmaps,
    build_backtest_irr_scatter, build_backtest_outcome_bar,
    build_worst_asset_pie, build_historical_prices, build_historical_wof_path,
)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from translations import Translator

# Consistent colour palette
_GREEN_DARK   = "#145214"
_GREEN_MID    = "#1a6b1a"
_GREEN_LIGHT  = "#2ecc71"
_RED          = "#c0392b"
_GREY         = "#888888"
_WHITE        = "white"

_OUTCOME_COLORS = {
    # keys are English labels — caller maps translated labels to these
    "Called at 3M":  _GREEN_LIGHT,
    "Called at 6M":  _GREEN_MID,
    "Called at 9M":  _GREEN_DARK,
    "Maturity":      "#3498db",
    "Floor Applied": "#e74c3c",
}


def _plain_layout(fig: go.Figure) -> go.Figure:
    fig.update_layout(plot_bgcolor=_WHITE, paper_bgcolor=_WHITE)
    return fig


# ---------------------------------------------------------------------------
# Sidebar — issuer call probability curve
# ---------------------------------------------------------------------------

def build_call_prob_curve(
    floor_level:    float,
    call_steepness: float,
    tr:             Translator,
) -> go.Figure:
    perf = np.linspace(0.80, 1.20, 300)
    prob = 1.0 / (1.0 + np.exp(-call_steepness * (perf - floor_level)))
    prob[perf < floor_level] = 0.0

    fig = px.line(
        pd.DataFrame({
            tr("worst_of_perf_axis"): perf,
            tr("p_issuer_calls"):     prob,
        }),
        x=tr("worst_of_perf_axis"),
        y=tr("p_issuer_calls"),
    )
    fig.add_vline(
        x=floor_level,
        line_dash="dash",
        line_color=_GREEN_MID,
        annotation_text=f"Call Strike ({floor_level:.0%})",
        annotation_position="top right",
    )
    fig.update_layout(
        yaxis=dict(tickformat=".0%", range=[0, 1.05]),
        xaxis=dict(tickformat=".0%"),
        height=280,
        margin=dict(t=10, b=20),
    )
    return _plain_layout(fig)


# ---------------------------------------------------------------------------
# Tab 1 — IRR distribution
# ---------------------------------------------------------------------------

def build_irr_distribution(
    annualized_returns: np.ndarray,
    autocall_events:    np.ndarray,
    expected_irr:       float,
    coupon_rate_pa:     float,      # p.a. coupon for reference line
    tr:                 Translator,
) -> go.Figure:
    """
    Histogram of compound annualised IRR across all paths.
    Compound IRR: (1 + total_return)^(1/t) - 1
    Splits into autocalled vs maturity paths for clarity.
    """
    # Recompute compound IRR from simple IRR stored in annualized_returns
    # (note: price_note stores simple IRR — convert to compound here for display)
    # We show compound: already stored as simple, so just display as-is with label
    irr_all      = annualized_returns
    irr_called   = irr_all[autocall_events > 0]
    irr_maturity = irr_all[autocall_events == 0]

    fig = go.Figure()

    if len(irr_called) > 0:
        fig.add_trace(go.Histogram(
            x=irr_called,
            nbinsx=50,
            name="Autocalled paths",
            opacity=0.7,
            marker_color=_GREEN_MID,
            histnorm="probability",
        ))
    if len(irr_maturity) > 0:
        fig.add_trace(go.Histogram(
            x=irr_maturity,
            nbinsx=50,
            name="Maturity paths",
            opacity=0.6,
            marker_color=_GREEN_DARK,
            histnorm="probability",
        ))

    # Reference lines
    fig.add_vline(
        x=expected_irr, line_dash="dash", line_color=_GREEN_MID,
        annotation_text=f"Mean IRR ({expected_irr:.2%})",
        annotation_position="top right",
    )
    fig.add_vline(
        x=coupon_rate_pa, line_dash="dot", line_color=_GREY,
        annotation_text=f"Coupon p.a. ({coupon_rate_pa:.2%})",
        annotation_position="top left",
    )
    fig.add_vline(
        x=0, line_dash="solid", line_color=_RED,
        annotation_text="Break-even",
        annotation_position="bottom right",
    )

    fig.update_layout(
        title="Annualised IRR Distribution — All Simulated Paths",
        xaxis=dict(title="Annualised IRR (simple)", tickformat=".1%"),
        yaxis=dict(title="Probability", tickformat=".1%"),
        barmode="overlay",
        legend=dict(x=0.01, y=0.99),
        hovermode="x unified",
    )
    return _plain_layout(fig)


# ---------------------------------------------------------------------------
# Tab 2 — per-asset fan chart
# ---------------------------------------------------------------------------

def build_fan_chart(
    paths:      np.ndarray,   # (n_paths, N+1) — single asset
    asset_name: str,
    t_grid:     np.ndarray,
    obs_labels: list[tuple[str, float]],  # [(label, t), ...] e.g. [("3M", 0.25)]
    tr:         Translator,
) -> go.Figure:
    S0 = paths[:, 0].mean()
    pcts = [5, 25, 50, 75, 95]
    bands = np.percentile(paths, pcts, axis=0)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=np.concatenate([t_grid, t_grid[::-1]]),
        y=np.concatenate([bands[4], bands[0][::-1]]),
        fill="toself", fillcolor="rgba(26,107,26,0.08)",
        line=dict(color="rgba(0,0,0,0)"),
        name=tr("pct_5_95"),
    ))
    fig.add_trace(go.Scatter(
        x=np.concatenate([t_grid, t_grid[::-1]]),
        y=np.concatenate([bands[3], bands[1][::-1]]),
        fill="toself", fillcolor="rgba(26,107,26,0.20)",
        line=dict(color="rgba(0,0,0,0)"),
        name=tr("pct_25_75"),
    ))
    fig.add_trace(go.Scatter(
        x=t_grid, y=bands[2],
        mode="lines", name=tr("median"),
        line=dict(color=_GREEN_MID, width=2),
    ))
    fig.add_hline(y=S0, line_dash="dash", line_color=_GREY,
                  annotation_text="S₀", annotation_position="right")

    for label, t_val in obs_labels:
        fig.add_vline(x=t_val, line_dash="dot", line_color="#aaa",
                      annotation_text=label, annotation_position="top")

    fig.update_layout(
        title=f"{asset_name} — {tr('simulated_price_dist')}",
        xaxis=dict(title=tr("time_years"), tickformat=".2f"),
        yaxis=dict(title=tr("price")),
        hovermode="x unified",
        legend=dict(x=0.01, y=0.99),
    )
    return _plain_layout(fig)


# ---------------------------------------------------------------------------
# Tab 2 — worst-of fan chart
# ---------------------------------------------------------------------------

def build_wof_fan(
    worst_of_paths: np.ndarray,
    t_grid:         np.ndarray,
    floor_level:    float,
    obs_labels:     list[tuple[float, str]],
    tr:             Translator,
) -> go.Figure:
    pcts = [5, 25, 50, 75, 95]
    bands = np.percentile(worst_of_paths, pcts, axis=0)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=np.concatenate([t_grid, t_grid[::-1]]),
        y=np.concatenate([bands[4], bands[0][::-1]]),
        fill="toself", fillcolor="rgba(26,107,26,0.08)",
        line=dict(color="rgba(0,0,0,0)"), name=tr("pct_5_95"),
    ))
    fig.add_trace(go.Scatter(
        x=np.concatenate([t_grid, t_grid[::-1]]),
        y=np.concatenate([bands[3], bands[1][::-1]]),
        fill="toself", fillcolor="rgba(26,107,26,0.20)",
        line=dict(color="rgba(0,0,0,0)"), name=tr("pct_25_75"),
    ))
    fig.add_trace(go.Scatter(
        x=t_grid, y=bands[2],
        mode="lines", name=tr("median"),
        line=dict(color=_GREEN_MID, width=2),
    ))
    fig.add_hline(
        y=floor_level, line_dash="dash", line_color=_RED,
        annotation_text=f"Floor / Call Strike ({floor_level:.0%})",
        annotation_position="bottom right",
    )
    for label, t_val in obs_labels:
        fig.add_vline(x=t_val, line_dash="dot", line_color="#aaa",
                      annotation_text=label, annotation_position="top")

    fig.update_layout(
        xaxis=dict(title=tr("time_years"), tickformat=".2f"),
        yaxis=dict(title=tr("perf_vs_initial"), tickformat=".0%"),
        hovermode="x unified",
        legend=dict(x=0.01, y=0.01),
    )
    return _plain_layout(fig)


# ---------------------------------------------------------------------------
# Tab 3 — path explorer: price paths
# ---------------------------------------------------------------------------

def build_path_price_chart(
    path_prices: pd.DataFrame,
    path_num:    int,
    obs_steps:   list[int],
    obs_labels:  list[str],
    tr:          Translator,
) -> go.Figure:
    fig = px.line(
        path_prices,
        title=tr("asset_price_paths", n=path_num),
        labels={
            "value": tr("price_label"),
            "index": tr("time_step"),
        },
        color_discrete_sequence=[_GREEN_MID, _GREEN_LIGHT, _GREEN_DARK],
    )
    for step, label in zip(obs_steps, obs_labels):
        fig.add_vline(x=step, line_dash="dot", line_color="#aaa",
                      annotation_text=label, annotation_position="top")
    return _plain_layout(fig)


def build_path_wof_chart(
    worst_path:      np.ndarray,
    autocall_q:      int,
    obs_steps:       list[int],
    obs_labels:      list[str],
    floor_level:     float,
    path_num:        int,
    tr:              Translator,
    asset_paths:     np.ndarray | None = None,   # (N+1, n_assets) optional
    asset_names:     list[str] | None  = None,
) -> go.Figure:
    asset_colors = [_GREEN_MID, _GREEN_LIGHT, _GREEN_DARK, "#f39c12", "#9b59b6"]
    fig = go.Figure()

    # Per-asset lines behind worst-of (if provided)
    if asset_paths is not None and asset_names is not None:
        for i, name in enumerate(asset_names):
            fig.add_trace(go.Scatter(
                y=asset_paths[:, i], mode="lines", name=name,
                line=dict(color=asset_colors[i % len(asset_colors)], width=1.2, dash="dot"),
                opacity=0.65,
            ))

    fig.add_trace(go.Scatter(
        y=worst_path, mode="lines",
        name="Worst-of",
        line=dict(color=_GREEN_DARK, width=2.5),
    ))
    fig.add_hline(
        y=floor_level, line_dash="dash", line_color=_RED,
        annotation_text=f"Call Strike / Floor ({floor_level:.0%})",
        annotation_position="bottom right",
    )
    for i, (step, label) in enumerate(zip(obs_steps, obs_labels)):
        called_here  = (autocall_q == i + 1)
        marker_color = _GREEN_MID if called_here else _GREY
        marker_sym   = "star"     if called_here else "circle"
        suffix       = tr("called_label") if called_here else tr("continued_label")
        fig.add_trace(go.Scatter(
            x=[step], y=[worst_path[step]],
            mode="markers",
            marker=dict(size=12, color=marker_color, symbol=marker_sym),
            name=f"{label} {suffix}",
        ))
        fig.add_vline(x=step, line_dash="dot", line_color="#aaa",
                      annotation_text=label, annotation_position="top")

    fig.update_layout(
        title=tr("wof_path_title", n=path_num),
        yaxis=dict(title=tr("perf_vs_initial"), tickformat=".0%"),
        xaxis=dict(title=tr("time_step")),
        hovermode="x unified",
    )
    return _plain_layout(fig)


# ---------------------------------------------------------------------------
# Tab 4 — correlation heatmaps
# ---------------------------------------------------------------------------

def build_corr_heatmap(
    matrix:     np.ndarray,
    asset_names: list[str],
    title:      str,
    zmin:       float = -1.0,
    zmax:       float =  1.0,
) -> go.Figure:
    df = pd.DataFrame(matrix, index=asset_names, columns=asset_names)
    fig = px.imshow(
        df, text_auto=".3f",
        color_continuous_scale=[[0, _RED], [0.5, _WHITE], [1, _GREEN_MID]],
        zmin=zmin, zmax=zmax,
        title=title, aspect="auto",
    )
    fig.update_layout(coloraxis_showscale=False, paper_bgcolor=_WHITE)
    return fig


# ---------------------------------------------------------------------------
# Backtest — outcome bar chart
# ---------------------------------------------------------------------------

def build_backtest_outcome_bar(
    bt:         pd.DataFrame,
    color_map:  dict[str, str],
    tr:         Translator,
) -> go.Figure:
    outcome_counts = bt["Outcome"].value_counts().reset_index()
    outcome_counts.columns = ["Outcome", tr("count")]
    fig = px.bar(
        outcome_counts,
        x="Outcome", y=tr("count"),
        color="Outcome",
        color_discrete_map=color_map,
        title=tr("outcome_dist"),
        text=tr("count"),
    )
    fig.update_layout(showlegend=False)
    return _plain_layout(fig)


# ---------------------------------------------------------------------------
# Backtest — worst asset pie
# ---------------------------------------------------------------------------

def build_worst_asset_pie(
    bt:  pd.DataFrame,
    tr:  Translator,
) -> go.Figure:
    maturity_bt = bt[bt["Call Quarter"] == 0]
    if maturity_bt.empty:
        return go.Figure()

    wc = maturity_bt["Worst Asset"].value_counts().reset_index()
    wc.columns = [tr("asset"), tr("count")]
    fig = px.pie(
        wc,
        names=tr("asset"),
        values=tr("count"),
        title=tr("worst_asset_at_mat"),
        hole=0.4,
        color_discrete_sequence=[_GREEN_MID, _GREEN_LIGHT, _GREEN_DARK],
    )
    fig.update_layout(paper_bgcolor=_WHITE)
    return fig


# ---------------------------------------------------------------------------
# Backtest — IRR scatter by issue date
# ---------------------------------------------------------------------------

def build_backtest_irr_scatter(
    bt:        pd.DataFrame,
    color_map: dict[str, str],
    tr:        Translator,
) -> go.Figure:
    fig = px.scatter(
        bt, x="Issue Date", y="IRR", color="Outcome",
        color_discrete_map=color_map,
        hover_data=["Payout", "Worst Asset", "Worst Final Perf"],
        title=tr("realised_irr_title"),
    )
    fig.add_hline(
        y=0, line_dash="dash", line_color=_GREY,
        annotation_text=tr("break_even"),
        annotation_position="right",
    )
    fig.update_layout(yaxis=dict(tickformat=".1%"))
    return _plain_layout(fig)


# ---------------------------------------------------------------------------
# Backtest — historical normalised price paths
# ---------------------------------------------------------------------------

def build_historical_prices(
    hist_prices:  pd.DataFrame,
    bt_start:     pd.Timestamp,
    bt_end:       pd.Timestamp,
    tr:           Translator,
) -> go.Figure:
    colors = {"SPX": _GREEN_MID, "SX5E": _GREEN_LIGHT, "SMI": _GREEN_DARK}
    fig = go.Figure()
    for col in hist_prices.columns:
        normed = hist_prices[col] / hist_prices[col].iloc[0] * 100
        fig.add_trace(go.Scatter(
            x=hist_prices.index, y=normed,
            mode="lines", name=col,
            line=dict(color=colors.get(col, _GREY), width=1.5),
        ))
    fig.add_vline(x=bt_start, line_dash="dot", line_color=_GREY,
                  annotation_text=tr("backtest_start"), annotation_position="top right")
    fig.add_vline(x=bt_end,   line_dash="dot", line_color=_GREY,
                  annotation_text=tr("backtest_end"),   annotation_position="top left")
    fig.update_layout(
        xaxis=dict(title=tr("date_axis")),
        yaxis=dict(title=tr("normalised_level")),
        hovermode="x unified",
        legend=dict(x=0.01, y=0.99),
    )
    return _plain_layout(fig)


# ---------------------------------------------------------------------------
# Backtest — historical worst-of performance path for a specific issue date
# ---------------------------------------------------------------------------

def build_historical_wof_path(
    hist_prices:   pd.DataFrame,
    issue_date:    pd.Timestamp,
    maturity_days: int,
    obs_day_offsets: list[int],
    knock_in_barrier: float,
    autocall_barrier: float,
    call_quarter:  int,             # 0 = maturity, else period number
    tr:            Translator,
) -> go.Figure:
    """
    Show per-asset performance + worst-of line for one historical issue date.
    Vertical dotted lines at each observation date.
    Markers show whether each observation was autocalled, coupon paid, or missed.
    """
    issue_idx = hist_prices.index.searchsorted(issue_date)
    end_idx   = min(issue_idx + maturity_days + 1, len(hist_prices))
    slice_    = hist_prices.iloc[issue_idx:end_idx]
    dates     = slice_.index
    S0        = hist_prices.iloc[issue_idx].values.astype(float)

    # Normalise each asset to 1.0 at issue date
    perf = slice_.values / S0[np.newaxis, :]    # (days, n_assets)
    wof  = perf.min(axis=1)

    asset_colors = [_GREEN_MID, _GREEN_LIGHT, _GREEN_DARK, "#f39c12", "#9b59b6"]
    asset_names  = list(hist_prices.columns)

    fig = go.Figure()

    # Per-asset lines (lighter, dashed)
    for i, name in enumerate(asset_names):
        fig.add_trace(go.Scatter(
            x=dates, y=perf[:, i],
            mode="lines", name=name,
            line=dict(color=asset_colors[i % len(asset_colors)], width=1.2, dash="dot"),
            opacity=0.65,
        ))

    # Worst-of line (solid, prominent)
    fig.add_trace(go.Scatter(
        x=dates, y=wof,
        mode="lines", name="Worst-of",
        line=dict(color=_GREEN_DARK, width=2.5),
    ))

    # Barriers
    fig.add_hline(y=knock_in_barrier, line_dash="dash", line_color=_RED,
                  annotation_text=f"Knock-in / Coupon barrier ({knock_in_barrier:.0%})",
                  annotation_position="bottom right")
    fig.add_hline(y=autocall_barrier, line_dash="dot", line_color=_GREY,
                  annotation_text=f"Autocall barrier ({autocall_barrier:.0%})",
                  annotation_position="top right")

    # Observation markers
    for q, offset in enumerate(obs_day_offsets):
        obs_idx_local = offset
        if obs_idx_local >= len(dates):
            break
        obs_date = dates[obs_idx_local]
        wof_val  = float(wof[obs_idx_local])
        is_call  = (call_quarter == q + 1)
        color    = _GREEN_MID if wof_val >= knock_in_barrier else _RED
        symbol   = "star" if is_call else "circle"
        size     = 14 if is_call else 9
        label    = f"P{q+1} {'← CALLED' if is_call else ''}"

        fig.add_trace(go.Scatter(
            x=[obs_date], y=[wof_val],
            mode="markers",
            marker=dict(size=size, color=color, symbol=symbol,
                        line=dict(width=1.5, color="white")),
            name=label, showlegend=True,
        ))
        fig.add_vline(x=obs_date.isoformat(), line_dash="dot",
                      line_color="#cccccc",
                      annotation_text=f"P{q+1}", annotation_position="top")

    outcome = "Autocalled" if call_quarter > 0 else "Maturity"
    fig.update_layout(
        title=f"Historical Worst-of Path — Issue: {issue_date.date()} · Outcome: {outcome} P{call_quarter}" if call_quarter > 0
              else f"Historical Worst-of Path — Issue: {issue_date.date()} · Outcome: Maturity",
        xaxis=dict(title="Date"),
        yaxis=dict(title="Performance vs Issue Date", tickformat=".0%"),
        hovermode="x unified",
        legend=dict(x=1.01, y=1, xanchor="left"),
    )
    return _plain_layout(fig)