"""Data loading, derived columns and the shared sidebar filter block."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from src.config import DATA_PATH

DATE_COLS = ["order_date", "promised_delivery_date", "actual_delivery_date"]
REQUIRED_COLUMNS = {
    "po_id",
    *DATE_COLS,
    "supplier_id",
    "supplier_name",
    "supplier_country",
    "category",
    "item",
    "business_unit",
    "unit_price",
    "quantity",
    "line_total",
    "payment_terms",
    "on_contract",
    "quality_rejected",
}
BOOLEAN_COLUMNS = ["on_contract", "quality_rejected"]
NUMERIC_COLUMNS = ["unit_price", "quantity", "line_total"]


def _parse_boolean(column: pd.Series, name: str) -> pd.Series:
    """Parse common CSV boolean values and reject ambiguous input."""
    if pd.api.types.is_bool_dtype(column):
        return column.astype(bool)

    normalized = column.astype("string").str.strip().str.lower()
    mapping = {"true": True, "false": False, "1": True, "0": False}
    invalid = normalized.notna() & ~normalized.isin(mapping)
    if invalid.any():
        values = ", ".join(sorted(normalized[invalid].dropna().unique())[:3])
        raise ValueError(f"Column '{name}' contains invalid boolean values: {values}")
    return normalized.map(mapping).astype("boolean")


def load_purchase_orders(path: str | Path = DATA_PATH) -> pd.DataFrame:
    """Load, validate, and enrich a purchase-order CSV."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Purchase-order dataset not found: {path}")

    df = pd.read_csv(path)
    missing = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing:
        raise ValueError(f"Purchase-order dataset is missing columns: {', '.join(missing)}")
    if df.empty:
        raise ValueError("Purchase-order dataset contains no rows")

    for column in DATE_COLS:
        df[column] = pd.to_datetime(df[column], errors="raise")
    for column in NUMERIC_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="raise")
    for column in BOOLEAN_COLUMNS:
        df[column] = _parse_boolean(df[column], column)

    required_values = REQUIRED_COLUMNS - {"actual_delivery_date"}
    null_columns = sorted(column for column in required_values if df[column].isna().any())
    if null_columns:
        raise ValueError(
            "Purchase-order dataset contains missing required values in: "
            + ", ".join(null_columns)
        )
    if (df[["unit_price", "quantity", "line_total"]] < 0).any().any():
        raise ValueError("Prices, quantities, and line totals must be non-negative")
    expected_total = df["unit_price"] * df["quantity"]
    if not np.allclose(df["line_total"], expected_total, rtol=1e-6, atol=0.01):
        raise ValueError("Line totals must equal unit price multiplied by quantity")

    df["month"] = df["order_date"].dt.to_period("M").dt.to_timestamp()
    df["year"] = df["order_date"].dt.year
    df["delay_days"] = (df["actual_delivery_date"] - df["promised_delivery_date"]).dt.days
    df["on_time"] = (df["delay_days"] <= 0).astype("boolean")
    return df


@st.cache_data(show_spinner="Loading purchase orders...")
def get_data() -> pd.DataFrame:
    return load_purchase_orders()


def render_sidebar_filters(df: pd.DataFrame, key_prefix: str = "flt") -> pd.DataFrame:
    """Render the global sidebar filters and return the filtered frame.

    Used by every page so filtering behaves identically across the app.
    """
    st.sidebar.header("Filters")

    min_d, max_d = df["order_date"].min().date(), df["order_date"].max().date()
    date_range = st.sidebar.date_input(
        "Order date range",
        value=(min_d, max_d),
        min_value=min_d,
        max_value=max_d,
        key=f"{key_prefix}_dates",
    )

    categories = st.sidebar.multiselect(
        "Category",
        options=sorted(df["category"].unique()),
        key=f"{key_prefix}_cat",
    )
    business_units = st.sidebar.multiselect(
        "Business unit",
        options=sorted(df["business_unit"].unique()),
        key=f"{key_prefix}_bu",
    )
    suppliers = st.sidebar.multiselect(
        "Supplier",
        options=sorted(df["supplier_name"].unique()),
        key=f"{key_prefix}_sup",
    )

    out = df
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start, end = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
        out = out[(out["order_date"] >= start) & (out["order_date"] <= end)]
    if categories:
        out = out[out["category"].isin(categories)]
    if business_units:
        out = out[out["business_unit"].isin(business_units)]
    if suppliers:
        out = out[out["supplier_name"].isin(suppliers)]

    st.sidebar.caption(
        f"{len(out):,} of {len(df):,} PO lines selected"
    )
    return out
