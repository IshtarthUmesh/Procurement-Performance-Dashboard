"""Core analytics: KPIs, trends, Pareto, price variance and savings logic.

Every function takes a purchase-order DataFrame (see ``data_loader``) and
returns plain pandas objects, so the module is fully unit-testable without
Streamlit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import (
    DUPLICATE_WINDOW_DAYS,
    MAVERICK_RECOVERY_RATE,
    MIN_SUPPLIERS_FOR_BENCHMARK,
    PARETO_TAIL_THRESHOLD,
    PRICE_BENCHMARK_QUANTILE,
    TAIL_CONSOLIDATION_RATE,
)

# ---------------------------------------------------------------------------
# Headline KPIs
# ---------------------------------------------------------------------------


def kpi_summary(df: pd.DataFrame) -> dict:
    """Headline numbers for the overview page."""
    total_spend = float(df["line_total"].sum())
    po_count = int(df["po_id"].nunique())
    suppliers = int(df["supplier_name"].nunique())
    avg_po_value = total_spend / po_count if po_count else 0.0
    return {
        "total_spend": total_spend,
        "po_count": po_count,
        "supplier_count": suppliers,
        "avg_po_value": avg_po_value,
        "on_time_rate": float(df["on_time"].mean()) if len(df) else 0.0,
        "yoy_change": yoy_spend_change(df),
    }


def yoy_spend_change(df: pd.DataFrame) -> float | None:
    """Spend in the trailing 12 months vs the 12 months before that.

    Returns a fraction (0.08 = +8%) or None when there is not enough history.
    """
    if df.empty:
        return None
    end = df["order_date"].max()
    last_12_start = end - pd.DateOffset(months=12)
    prev_12_start = end - pd.DateOffset(months=24)

    last = df.loc[df["order_date"] > last_12_start, "line_total"].sum()
    prev = df.loc[
        (df["order_date"] > prev_12_start) & (df["order_date"] <= last_12_start),
        "line_total",
    ].sum()
    if prev <= 0:
        return None
    return float(last / prev - 1)


# ---------------------------------------------------------------------------
# Trends and breakdowns
# ---------------------------------------------------------------------------


def monthly_spend(df: pd.DataFrame) -> pd.DataFrame:
    out = df.groupby("month", as_index=False)["line_total"].sum()
    return out.rename(columns={"line_total": "spend"})


def spend_by_category(df: pd.DataFrame) -> pd.DataFrame:
    out = (
        df.groupby("category", as_index=False)["line_total"]
        .sum()
        .rename(columns={"line_total": "spend"})
        .sort_values("spend", ascending=False)
    )
    total = out["spend"].sum()
    out["share"] = out["spend"] / total if total else 0.0
    return out


def spend_by_country(df: pd.DataFrame) -> pd.DataFrame:
    out = (
        df.groupby("supplier_country", as_index=False)["line_total"]
        .sum()
        .rename(columns={"line_total": "spend"})
        .sort_values("spend", ascending=False)
    )
    return out


def top_suppliers(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    out = (
        df.groupby("supplier_name", as_index=False)["line_total"]
        .sum()
        .rename(columns={"line_total": "spend"})
        .sort_values("spend", ascending=False)
        .head(n)
    )
    return out


def supplier_pareto(df: pd.DataFrame) -> pd.DataFrame:
    """Supplier spend ranked descending, with individual and cumulative share."""
    out = (
        df.groupby("supplier_name", as_index=False)["line_total"]
        .sum()
        .rename(columns={"line_total": "spend"})
        .sort_values("spend", ascending=False)
        .reset_index(drop=True)
    )
    total = out["spend"].sum()
    out["share"] = out["spend"] / total if total else 0.0
    out["cumulative_share"] = out["share"].cumsum()
    out["rank"] = np.arange(1, len(out) + 1)
    return out


def supplier_scorecard(df: pd.DataFrame) -> pd.DataFrame:
    """One row per supplier: spend, volume, delivery and quality performance."""
    grouped = df.groupby(["supplier_name", "supplier_country", "category"])
    out = grouped.agg(
        spend=("line_total", "sum"),
        po_count=("po_id", "nunique"),
        on_time_rate=("on_time", "mean"),
        avg_delay_days=("delay_days", "mean"),
        reject_rate=("quality_rejected", "mean"),
        off_contract_share=("on_contract", lambda s: 1 - s.mean()),
    ).reset_index()
    return out.sort_values("spend", ascending=False).reset_index(drop=True)


def category_price_index(df: pd.DataFrame) -> pd.DataFrame:
    """Spend-weighted unit-price index per category (first month = 100).

    Each item is indexed against its own average price in its first three
    observed months, then items are combined into a category index weighted
    by item spend — so mix effects don't masquerade as inflation.
    """
    item_month = (
        df.groupby(["category", "item", "month"])
        .agg(spend=("line_total", "sum"), quantity=("quantity", "sum"))
        .reset_index()
        .sort_values("month")
    )
    item_month["avg_price"] = item_month["spend"] / item_month["quantity"]

    baseline = (
        item_month.groupby(["category", "item"])["avg_price"]
        .agg(lambda s: s.head(3).mean())
        .rename("base_price")
        .reset_index()
    )
    item_month = item_month.merge(baseline, on=["category", "item"])
    item_month["price_index"] = item_month["avg_price"] / item_month["base_price"] * 100

    # Spend-weighted average of item indices per category-month.
    item_month["weighted"] = item_month["price_index"] * item_month["spend"]
    out = (
        item_month.groupby(["category", "month"])
        .agg(weighted=("weighted", "sum"), spend=("spend", "sum"))
        .reset_index()
    )
    out["price_index"] = out["weighted"] / out["spend"]
    return out[["category", "month", "price_index"]]


def item_price_benchmark(df: pd.DataFrame) -> pd.DataFrame:
    """Per item: what was paid vs a negotiable benchmark price.

    Benchmark = the ``PRICE_BENCHMARK_QUANTILE`` unit price observed for that
    item across all suppliers (a price someone actually achieved, not a
    theoretical floor). Savings = what would have been saved buying every
    unit at the benchmark. Items need >= MIN_SUPPLIERS_FOR_BENCHMARK
    suppliers to qualify — one supplier is not a negotiation.
    """
    stats = (
        df.groupby(["category", "item"])
        .agg(
            spend=("line_total", "sum"),
            quantity=("quantity", "sum"),
            suppliers=("supplier_name", "nunique"),
            benchmark_price=("unit_price", lambda s: s.quantile(PRICE_BENCHMARK_QUANTILE)),
        )
        .reset_index()
    )
    stats = stats[stats["suppliers"] >= MIN_SUPPLIERS_FOR_BENCHMARK].copy()
    stats["avg_price"] = stats["spend"] / stats["quantity"]

    merged = df.merge(
        stats[["category", "item", "benchmark_price"]], on=["category", "item"]
    )
    merged["line_savings"] = (
        (merged["unit_price"] - merged["benchmark_price"]).clip(lower=0)
        * merged["quantity"]
    )
    savings = (
        merged.groupby(["category", "item"])["line_savings"].sum().rename("savings")
    )
    out = stats.merge(savings, on=["category", "item"])
    out["savings_pct_of_spend"] = out["savings"] / out["spend"]
    return out.sort_values("savings", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Savings opportunities
# ---------------------------------------------------------------------------


def price_variance_savings(df: pd.DataFrame) -> float:
    """Total addressable savings from closing item price variance."""
    bench = item_price_benchmark(df)
    return float(bench["savings"].sum())


def tail_spend_analysis(df: pd.DataFrame) -> dict:
    """Suppliers beyond the Pareto threshold: candidates for consolidation."""
    pareto = supplier_pareto(df)
    tail = pareto[pareto["cumulative_share"] > PARETO_TAIL_THRESHOLD]
    tail_spend = float(tail["spend"].sum())
    return {
        "tail_suppliers": int(len(tail)),
        "total_suppliers": int(len(pareto)),
        "tail_spend": tail_spend,
        "estimated_savings": tail_spend * TAIL_CONSOLIDATION_RATE,
    }


def maverick_spend_analysis(df: pd.DataFrame) -> dict:
    """Off-contract ("maverick") purchasing, overall and by business unit."""
    off = df[~df["on_contract"]]
    by_bu = df.groupby("business_unit", as_index=False)["line_total"].sum()
    by_bu = by_bu.rename(columns={"line_total": "spend"})
    off_by_bu = off.groupby("business_unit", as_index=False)["line_total"].sum()
    off_by_bu = off_by_bu.rename(columns={"line_total": "off_contract_spend"})
    by_bu = by_bu.merge(off_by_bu, on="business_unit", how="left")
    by_bu["off_contract_spend"] = by_bu["off_contract_spend"].fillna(0.0)
    by_bu["off_contract_share"] = by_bu["off_contract_spend"] / by_bu["spend"]
    total_off = float(off["line_total"].sum())
    return {
        "off_contract_spend": total_off,
        "off_contract_share": total_off / df["line_total"].sum() if len(df) else 0.0,
        "by_business_unit": by_bu.sort_values("off_contract_share", ascending=False),
        "estimated_savings": total_off * MAVERICK_RECOVERY_RATE,
    }


def duplicate_payment_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Potential duplicate invoice lines.

    Two lines from the same supplier for the same item, quantity and amount,
    ordered within DUPLICATE_WINDOW_DAYS of each other but under different
    PO ids, are flagged. Every line after the first in such a group is
    counted as at-risk spend.
    """
    keys = ["supplier_name", "item", "quantity", "line_total"]
    df = df.sort_values(keys + ["order_date"]).copy()
    grp = df.groupby(keys)
    df["prev_date"] = grp["order_date"].shift(1)
    df["prev_po"] = grp["po_id"].shift(1)
    df["gap_days"] = (df["order_date"] - df["prev_date"]).dt.days

    dupes = df[
        df["gap_days"].notna()
        & (df["gap_days"] <= DUPLICATE_WINDOW_DAYS)
        & (df["po_id"] != df["prev_po"])
    ]
    cols = [
        "po_id", "prev_po", "order_date", "supplier_name", "item",
        "quantity", "line_total", "gap_days",
    ]
    return dupes[cols].sort_values("line_total", ascending=False).reset_index(drop=True)


def savings_summary(df: pd.DataFrame) -> pd.DataFrame:
    """All savings opportunities in one ranked table."""
    dupes = duplicate_payment_flags(df)
    tail = tail_spend_analysis(df)
    maverick = maverick_spend_analysis(df)
    rows = [
        {
            "opportunity": "Price variance consolidation",
            "description": "Buy each item at its 10th-percentile achieved price",
            "estimated_savings": price_variance_savings(df),
        },
        {
            "opportunity": "Maverick spend recovery",
            "description": "Route off-contract purchases back onto negotiated contracts",
            "estimated_savings": maverick["estimated_savings"],
        },
        {
            "opportunity": "Tail supplier consolidation",
            "description": (
                f"Consolidate {tail['tail_suppliers']} tail suppliers holding "
                "the last 5% of spend"
            ),
            "estimated_savings": tail["estimated_savings"],
        },
        {
            "opportunity": "Duplicate payment recovery",
            "description": f"Review {len(dupes)} suspected duplicate invoice lines",
            "estimated_savings": float(dupes["line_total"].sum()),
        },
    ]
    out = pd.DataFrame(rows).sort_values("estimated_savings", ascending=False)
    return out.reset_index(drop=True)


def single_source_categories(df: pd.DataFrame, threshold: float = 0.85) -> pd.DataFrame:
    """Categories dominated by one supplier — concentration risk."""
    spend = df.groupby(["category", "supplier_name"])["line_total"].sum().reset_index()
    top = spend.sort_values("line_total", ascending=False).groupby("category").head(1)
    totals = df.groupby("category")["line_total"].sum().rename("category_spend")
    out = top.merge(totals, on="category")
    out["top_supplier_share"] = out["line_total"] / out["category_spend"]
    out = out[out["top_supplier_share"] >= threshold]
    out = out.rename(columns={"supplier_name": "dominant_supplier", "line_total": "supplier_spend"})
    return out.sort_values("top_supplier_share", ascending=False).reset_index(drop=True)
