"""Unit tests for the analytics module against a small hand-built dataset."""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import analytics
from src.config import DATA_PATH
from src.data_loader import load_purchase_orders
from src.risk_features import (
    FEATURE_COLS,
    build_supplier_features,
    composite_score,
    risk_tier,
)


@pytest.fixture()
def df() -> pd.DataFrame:
    """Six PO lines: two suppliers, two items, one duplicate pair."""
    rows = [
        # Alpha sells Widget cheap, Beta sells the same Widget dear.
        dict(po_id="PO-1", order_date="2025-01-10", promised="2025-01-20",
             actual="2025-01-20", supplier="Alpha", country="USA",
             category="MRO", item="Widget", bu="Manufacturing",
             unit_price=10.0, quantity=100, on_contract=True, rejected=False),
        dict(po_id="PO-2", order_date="2025-02-10", promised="2025-02-20",
             actual="2025-02-20", supplier="Alpha", country="USA",
             category="MRO", item="Widget", bu="Manufacturing",
             unit_price=10.0, quantity=100, on_contract=True, rejected=False),
        dict(po_id="PO-3", order_date="2025-01-15", promised="2025-01-25",
             actual="2025-02-04", supplier="Beta", country="India",
             category="MRO", item="Widget", bu="R&D",
             unit_price=20.0, quantity=100, on_contract=False, rejected=True),
        dict(po_id="PO-4", order_date="2025-03-01", promised="2025-03-10",
             actual="2025-03-10", supplier="Gamma", country="UK",
             category="MRO", item="Widget", bu="Corporate",
             unit_price=12.0, quantity=50, on_contract=True, rejected=False),
        # Duplicate pair: same supplier/item/qty/amount, 1 day apart.
        dict(po_id="PO-5", order_date="2025-04-01", promised="2025-04-11",
             actual="2025-04-11", supplier="Alpha", country="USA",
             category="MRO", item="Bolt", bu="Manufacturing",
             unit_price=5.0, quantity=200, on_contract=True, rejected=False),
        dict(po_id="PO-6", order_date="2025-04-02", promised="2025-04-12",
             actual="2025-04-12", supplier="Alpha", country="USA",
             category="MRO", item="Bolt", bu="Manufacturing",
             unit_price=5.0, quantity=200, on_contract=True, rejected=False),
    ]
    frame = pd.DataFrame(rows).rename(
        columns={
            "promised": "promised_delivery_date",
            "actual": "actual_delivery_date",
            "supplier": "supplier_name",
            "country": "supplier_country",
            "bu": "business_unit",
            "rejected": "quality_rejected",
        }
    )
    for col in ["order_date", "promised_delivery_date", "actual_delivery_date"]:
        frame[col] = pd.to_datetime(frame[col])
    frame["supplier_id"] = frame["supplier_name"]
    frame["line_total"] = frame["unit_price"] * frame["quantity"]
    frame["month"] = frame["order_date"].dt.to_period("M").dt.to_timestamp()
    frame["year"] = frame["order_date"].dt.year
    frame["delay_days"] = (
        frame["actual_delivery_date"] - frame["promised_delivery_date"]
    ).dt.days
    frame["on_time"] = frame["delay_days"] <= 0
    return frame


def test_kpi_summary(df):
    kpis = analytics.kpi_summary(df)
    assert kpis["total_spend"] == pytest.approx(6600.0)
    assert kpis["po_count"] == 6
    assert kpis["supplier_count"] == 3
    # Beta's one order was 10 days late; the other five were on time.
    assert kpis["on_time_rate"] == pytest.approx(5 / 6)


def test_monthly_spend_sums_to_total(df):
    monthly = analytics.monthly_spend(df)
    assert monthly["spend"].sum() == pytest.approx(df["line_total"].sum())
    assert monthly["month"].is_monotonic_increasing


def test_supplier_pareto(df):
    pareto = analytics.supplier_pareto(df)
    assert list(pareto["supplier_name"]) == ["Alpha", "Beta", "Gamma"]
    assert pareto["cumulative_share"].iloc[-1] == pytest.approx(1.0)
    assert pareto["share"].iloc[0] == pytest.approx(4000 / 6600)


def test_item_price_benchmark_finds_widget_variance(df):
    bench = analytics.item_price_benchmark(df)
    # Only Widget has >= 3 suppliers; Bolt (1 supplier) is excluded.
    assert list(bench["item"]) == ["Widget"]
    row = bench.iloc[0]
    # Benchmark = 10th percentile of [10, 10, 12, 20] = 10.0 (interpolated).
    assert row["benchmark_price"] == pytest.approx(10.0, abs=0.01)
    # Beta paid 20 and Gamma paid 12 vs the 10.0 benchmark:
    # (20-10)*100 + (12-10)*50 = 1100.
    assert row["savings"] == pytest.approx(1100.0)


def test_duplicate_payment_flags(df):
    dupes = analytics.duplicate_payment_flags(df)
    assert len(dupes) == 1
    assert dupes.iloc[0]["po_id"] == "PO-6"
    assert dupes.iloc[0]["prev_po"] == "PO-5"


def test_maverick_analysis(df):
    maverick = analytics.maverick_spend_analysis(df)
    # Beta's PO-3 is the only off-contract line: 20 * 100 = 2000.
    assert maverick["off_contract_spend"] == pytest.approx(2000.0)
    by_bu = maverick["by_business_unit"].set_index("business_unit")
    assert by_bu.loc["R&D", "off_contract_share"] == pytest.approx(1.0)
    assert by_bu.loc["Manufacturing", "off_contract_share"] == pytest.approx(0.0)


def test_savings_summary_structure(df):
    summary = analytics.savings_summary(df)
    assert len(summary) == 4
    assert (summary["estimated_savings"] >= 0).all()
    # Ranked descending by value.
    assert summary["estimated_savings"].is_monotonic_decreasing


def test_risk_features_flag_bad_supplier(df):
    features = build_supplier_features(df)
    assert set(FEATURE_COLS) <= set(features.columns)
    scores = composite_score(features)
    by_name = dict(zip(features["supplier_name"], scores, strict=True))
    # Beta was late, rejected and off-contract: must outscore Alpha.
    assert by_name["Beta"] > by_name["Alpha"]


def test_real_dataset_loads():
    """Integration: the committed CSV loads with all expected columns."""
    real = load_purchase_orders(DATA_PATH)
    expected = {
        "po_id", "order_date", "supplier_name", "category", "item",
        "business_unit", "unit_price", "quantity", "line_total",
        "on_contract", "quality_rejected", "month", "delay_days", "on_time",
    }
    assert expected <= set(real.columns)
    assert len(real) > 40_000
    assert real["line_total"].min() >= 0


def test_risk_tier_thresholds_are_inclusive():
    tiers = risk_tier(pd.Series([44.9, 45.0, 69.9, 70.0])).astype("string").tolist()
    assert tiers == ["Low", "Medium", "Medium", "High"]


def test_item_average_price_is_quantity_weighted(df):
    bench = analytics.item_price_benchmark(df)
    widget = bench.loc[bench["item"] == "Widget"].iloc[0]
    expected = df.loc[df["item"] == "Widget", "line_total"].sum() / df.loc[
        df["item"] == "Widget", "quantity"
    ].sum()
    assert widget["avg_price"] == pytest.approx(expected)


def test_empty_breakdowns_do_not_divide_by_zero(df):
    empty = df.iloc[0:0]
    assert analytics.spend_by_category(empty).empty
    assert analytics.supplier_pareto(empty).empty


def test_loader_rejects_missing_columns(tmp_path):
    path = tmp_path / "invalid.csv"
    pd.DataFrame({"po_id": ["PO-1"]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing columns"):
        load_purchase_orders(path)


def test_loader_rejects_inconsistent_line_total(df, tmp_path):
    path = tmp_path / "invalid-total.csv"
    raw = df.drop(columns=["month", "year", "delay_days", "on_time"]).copy()
    raw["payment_terms"] = "Net 30"
    raw.loc[raw.index[0], "line_total"] += 10
    raw.to_csv(path, index=False)
    with pytest.raises(ValueError, match="Line totals"):
        load_purchase_orders(path)
