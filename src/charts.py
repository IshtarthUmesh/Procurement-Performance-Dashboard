"""Shared Plotly chart builders and page chrome.

All charts draw from one visual system: fixed categorical hue order, a single
sequential blue ramp for magnitude, reserved status colors for risk tiers,
hairline gridlines, 2px lines and thin bars. Keeping every figure here means
the five pages stay visually consistent.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.config import (
    AXIS_LINE,
    CATEGORICAL,
    FONT_FAMILY,
    GRIDLINE,
    INK_MUTED,
    INK_PRIMARY,
    INK_SECONDARY,
    PAGE_BG,
    SEQUENTIAL_BLUES,
    STATUS,
    SURFACE,
)

TIER_COLORS = {"Low": STATUS["good"], "Medium": STATUS["warning"], "High": STATUS["critical"]}
TIER_ICONS = {"Low": "✓", "Medium": "⚠", "High": "✖"}

BASE_LAYOUT = dict(
    paper_bgcolor=SURFACE,
    plot_bgcolor=SURFACE,
    font=dict(family=FONT_FAMILY, color=INK_SECONDARY, size=13),
    title=dict(font=dict(color=INK_PRIMARY, size=15)),
    colorway=CATEGORICAL,
    margin=dict(l=10, r=10, t=40, b=10),
    hoverlabel=dict(bgcolor="#ffffff", font=dict(family=FONT_FAMILY, color=INK_PRIMARY)),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)

AXIS_STYLE = dict(
    gridcolor=GRIDLINE,
    gridwidth=1,
    linecolor=AXIS_LINE,
    tickfont=dict(color=INK_MUTED, size=12),
    title_font=dict(color=INK_MUTED, size=12),
    zeroline=False,
)


def _finish(fig: go.Figure, **layout) -> go.Figure:
    fig.update_layout(**BASE_LAYOUT | layout)
    fig.update_xaxes(**AXIS_STYLE)
    fig.update_yaxes(**AXIS_STYLE)
    return fig


def money(value: float) -> str:
    """Compact money format: $4.2M, $312K, $87."""
    v = float(value)
    for cutoff, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if abs(v) >= cutoff:
            return f"${v / cutoff:,.1f}{suffix}"
    return f"${v:,.0f}"


def setup_page(title: str) -> None:
    """Page config + light CSS so every page shares the same chrome."""
    st.set_page_config(page_icon="📦", layout="wide",
                       initial_sidebar_state="expanded")
    st.markdown(
        f"""
        <style>
            .stApp {{ background-color: {PAGE_BG}; }}
            section[data-testid="stSidebar"] {{ background-color: {SURFACE}; }}
            div[data-testid="stMetric"] {{
                background-color: {SURFACE};
                border: 1px solid rgba(11, 11, 11, 0.10);
                border-radius: 10px;
                padding: 14px 18px;
            }}
            div[data-testid="stMetricLabel"] p {{ color: {INK_SECONDARY}; }}
            h1, h2, h3 {{ color: {INK_PRIMARY}; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Chart builders
# ---------------------------------------------------------------------------


def monthly_spend_line(monthly: pd.DataFrame) -> go.Figure:
    fig = go.Figure(
        go.Scatter(
            x=monthly["month"],
            y=monthly["spend"],
            mode="lines",
            line=dict(color=CATEGORICAL[0], width=2, shape="spline", smoothing=0.4),
            fill="tozeroy",
            fillcolor="rgba(42, 120, 214, 0.10)",
            name="Monthly spend",
            hovertemplate="%{x|%b %Y}<br>$%{y:,.0f}<extra></extra>",
        )
    )
    return _finish(fig, title="Monthly spend", showlegend=False, height=340)


def category_spend_bar(cat: pd.DataFrame) -> go.Figure:
    cat = cat.sort_values("spend")
    fig = go.Figure(
        go.Bar(
            x=cat["spend"],
            y=cat["category"],
            orientation="h",
            marker=dict(color=CATEGORICAL[0]),
            width=0.55,
            hovertemplate="%{y}<br>$%{x:,.0f}<extra></extra>",
        )
    )
    return _finish(fig, title="Spend by category", showlegend=False, height=380)


def top_suppliers_bar(top: pd.DataFrame) -> go.Figure:
    top = top.sort_values("spend")
    fig = go.Figure(
        go.Bar(
            x=top["spend"],
            y=top["supplier_name"],
            orientation="h",
            marker=dict(color=CATEGORICAL[1]),
            width=0.55,
            hovertemplate="%{y}<br>$%{x:,.0f}<extra></extra>",
        )
    )
    return _finish(fig, title="Top suppliers by spend", showlegend=False, height=380)


def country_spend_bar(country: pd.DataFrame) -> go.Figure:
    country = country.sort_values("spend")
    fig = go.Figure(
        go.Bar(
            x=country["spend"],
            y=country["supplier_country"],
            orientation="h",
            marker=dict(color=CATEGORICAL[4]),
            width=0.55,
            hovertemplate="%{y}<br>$%{x:,.0f}<extra></extra>",
        )
    )
    return _finish(fig, title="Spend by supplier country", showlegend=False, height=380)


def pareto_chart(pareto: pd.DataFrame, top_n: int = 40) -> go.Figure:
    """Supplier concentration. Bars and line share one % axis (no dual axis):
    bars are each supplier's share of spend, the line is the running total."""
    p = pareto.head(top_n)
    fig = go.Figure()
    fig.add_bar(
        x=p["rank"],
        y=p["share"] * 100,
        name="Share of spend",
        marker=dict(color=CATEGORICAL[0]),
        customdata=p[["supplier_name", "spend"]],
        hovertemplate=(
            "#%{x} %{customdata[0]}<br>Share: %{y:.1f}%<br>"
            "Spend: $%{customdata[1]:,.0f}<extra></extra>"
        ),
    )
    fig.add_scatter(
        x=p["rank"],
        y=p["cumulative_share"] * 100,
        mode="lines",
        name="Cumulative share",
        line=dict(color=CATEGORICAL[5], width=2),
        customdata=p[["supplier_name"]],
        hovertemplate="Top %{x} suppliers<br>Cumulative: %{y:.1f}%<extra></extra>",
    )
    fig.add_hline(y=80, line=dict(color=INK_MUTED, width=1, dash="dot"))
    fig.add_annotation(x=top_n * 0.98, y=83, text="80% of spend", showarrow=False,
                       font=dict(color=INK_MUTED, size=11), xanchor="right")
    fig = _finish(
        fig,
        title=f"Supplier spend concentration (top {min(top_n, len(pareto))} of {len(pareto)})",
        height=400,
    )
    fig.update_yaxes(title_text="% of total spend", range=[0, 105])
    fig.update_xaxes(title_text="Supplier rank")
    return fig


def price_index_lines(price_idx: pd.DataFrame, categories: list[str]) -> go.Figure:
    """Price index (first month = 100) for the selected categories."""
    fig = go.Figure()
    for i, cat in enumerate(categories[:8]):
        sub = price_idx[price_idx["category"] == cat]
        fig.add_scatter(
            x=sub["month"],
            y=sub["price_index"],
            mode="lines",
            name=cat,
            line=dict(color=CATEGORICAL[i], width=2),
            hovertemplate=cat + "<br>%{x|%b %Y}: %{y:.1f}<extra></extra>",
        )
    fig.add_hline(y=100, line=dict(color=INK_MUTED, width=1, dash="dot"))
    fig = _finish(fig, title="Unit-price index by category (start = 100)", height=400)
    fig.update_yaxes(title_text="Price index")
    return fig


def category_treemap(df: pd.DataFrame) -> go.Figure:
    """Category → item spend treemap, colored by magnitude (sequential blues)."""
    by_item = df.groupby(["category", "item"], as_index=False)["line_total"].sum()
    by_cat = df.groupby("category", as_index=False)["line_total"].sum()

    labels = by_cat["category"].tolist() + by_item["item"].tolist()
    parents = [""] * len(by_cat) + by_item["category"].tolist()
    values = by_cat["line_total"].tolist() + by_item["line_total"].tolist()

    # Color carries item-level magnitude. Category headers take their items'
    # mean so the (much larger) branch totals don't stretch the scale.
    item_means = by_item.groupby("category")["line_total"].mean()
    colors = [float(item_means[c]) for c in by_cat["category"]] + by_item[
        "line_total"
    ].tolist()

    fig = go.Figure(
        go.Treemap(
            labels=labels,
            parents=parents,
            values=values,
            branchvalues="total",
            maxdepth=2,
            root_color=SURFACE,
            pathbar=dict(visible=False),
            tiling=dict(pad=3),
            marker=dict(
                colors=colors,
                colorscale=[[i / (len(SEQUENTIAL_BLUES) - 1), c]
                            for i, c in enumerate(SEQUENTIAL_BLUES)],
                line=dict(width=2, color=SURFACE),
            ),
            textfont=dict(family=FONT_FAMILY, size=13),
            hovertemplate="%{label}<br>$%{value:,.0f}<extra></extra>",
        )
    )
    return _finish(fig, title="Spend by category and item", height=480)


def maverick_bu_bar(by_bu: pd.DataFrame) -> go.Figure:
    b = by_bu.sort_values("off_contract_share")
    fig = go.Figure(
        go.Bar(
            x=b["off_contract_share"] * 100,
            y=b["business_unit"],
            orientation="h",
            marker=dict(color=CATEGORICAL[7]),
            width=0.55,
            customdata=b[["off_contract_spend"]],
            hovertemplate="%{y}<br>Off-contract: %{x:.1f}% ($%{customdata[0]:,.0f})<extra></extra>",
        )
    )
    fig = _finish(fig, title="Off-contract spend share by business unit",
                  showlegend=False, height=320)
    fig.update_xaxes(title_text="% of BU spend off contract")
    return fig


def risk_scatter(scored: pd.DataFrame) -> go.Figure:
    """Spend vs risk. Top-right quadrant = high spend AND high risk."""
    fig = go.Figure()
    for tier in ["Low", "Medium", "High"]:
        sub = scored[scored["risk_tier"] == tier]
        if sub.empty:
            continue
        fig.add_scatter(
            x=sub["total_spend"],
            y=sub["risk_score"],
            mode="markers",
            name=f"{TIER_ICONS[tier]} {tier} risk",
            marker=dict(
                color=TIER_COLORS[tier],
                size=10,
                line=dict(width=2, color=SURFACE),
            ),
            customdata=sub[["supplier_name", "category"]],
            hovertemplate="%{customdata[0]} (%{customdata[1]})<br>"
                          "Spend: $%{x:,.0f}<br>Risk score: %{y:.0f}<extra></extra>",
        )
    median_spend = float(scored["total_spend"].median())
    fig.add_vline(x=median_spend, line=dict(color=INK_MUTED, width=1, dash="dot"))
    fig.add_hline(y=70, line=dict(color=INK_MUTED, width=1, dash="dot"))
    fig = _finish(fig, title="Supplier spend vs risk score", height=460)
    fig.update_xaxes(type="log", title_text="Total spend (log scale)")
    fig.update_yaxes(title_text="Risk score", range=[0, 100])
    return fig


def risk_factor_bar(row: pd.Series, spec: dict) -> go.Figure:
    """Composite-score contribution breakdown for one supplier."""
    names = {
        "late_rate": "Late delivery rate",
        "avg_delay_days": "Average delay (days)",
        "reject_rate": "Quality reject rate",
        "price_volatility": "Price volatility",
        "off_contract_share": "Off-contract share",
    }
    factors, points = [], []
    for col, s in spec.items():
        contribution = min(float(row[col]) / s["cap"], 1.0) * s["weight"] * 100
        factors.append(names[col])
        points.append(round(contribution, 1))
    frame = pd.DataFrame({"factor": factors, "points": points}).sort_values("points")
    fig = go.Figure(
        go.Bar(
            x=frame["points"],
            y=frame["factor"],
            orientation="h",
            marker=dict(color=CATEGORICAL[5]),
            width=0.5,
            hovertemplate="%{y}: %{x:.1f} pts<extra></extra>",
        )
    )
    fig = _finish(fig, title="Composite risk contribution (points)",
                  showlegend=False, height=300)
    return fig


def tier_distribution_bar(scored: pd.DataFrame) -> go.Figure:
    counts = scored["risk_tier"].value_counts().reindex(["Low", "Medium", "High"]).fillna(0)
    fig = go.Figure(
        go.Bar(
            x=[f"{TIER_ICONS[t]} {t}" for t in counts.index],
            y=counts.values,
            marker=dict(color=[TIER_COLORS[t] for t in counts.index]),
            width=0.45,
            text=[int(v) for v in counts.values],
            textposition="outside",
            textfont=dict(color=INK_SECONDARY),
            hovertemplate="%{x}: %{y} suppliers<extra></extra>",
        )
    )
    fig = _finish(fig, title="Suppliers by risk tier", showlegend=False, height=320)
    fig.update_yaxes(title_text="Suppliers")
    return fig


def supplier_monthly_trend(df: pd.DataFrame, supplier: str) -> go.Figure:
    sub = df[df["supplier_name"] == supplier]
    monthly = sub.groupby("month", as_index=False)["line_total"].sum()
    fig = go.Figure(
        go.Scatter(
            x=monthly["month"],
            y=monthly["line_total"],
            mode="lines",
            line=dict(color=CATEGORICAL[0], width=2),
            fill="tozeroy",
            fillcolor="rgba(42, 120, 214, 0.10)",
            hovertemplate="%{x|%b %Y}<br>$%{y:,.0f}<extra></extra>",
        )
    )
    return _finish(fig, title=f"Monthly spend — {supplier}", showlegend=False, height=300)


def delay_distribution(df: pd.DataFrame, supplier: str) -> go.Figure:
    sub = df[df["supplier_name"] == supplier]
    fig = go.Figure(
        go.Histogram(
            x=sub["delay_days"],
            marker=dict(color=CATEGORICAL[2], line=dict(width=2, color=SURFACE)),
            hovertemplate="Delay: %{x} days<br>Orders: %{y}<extra></extra>",
        )
    )
    fig = _finish(fig, title=f"Delivery delay distribution — {supplier}",
                  showlegend=False, height=300)
    fig.update_xaxes(title_text="Days late (0 = on time)")
    fig.update_yaxes(title_text="Orders")
    return fig
