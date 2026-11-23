"""Category Spend — composition, price inflation and item benchmarking."""

import streamlit as st

from src import analytics, charts
from src.data_loader import get_data, render_sidebar_filters

st.title("Category Spend")
st.caption(
    "What the money buys: category and item composition, unit-price inflation "
    "against a start-of-period baseline, and cross-supplier price benchmarks."
)

df = get_data()
filtered = render_sidebar_filters(df, key_prefix="category")

if filtered.empty:
    st.warning("No purchase orders match the current filters.")
    st.stop()

st.plotly_chart(charts.category_treemap(filtered), width="stretch")

st.subheader("Price inflation watch")
price_idx = analytics.category_price_index(filtered)
available = sorted(price_idx["category"].unique())
default = [c for c in ["Electronics", "Chemicals", "Raw Materials", "Office Supplies"]
           if c in available] or available[:4]
selected = st.multiselect(
    "Categories to plot (max 8)",
    options=available,
    default=default,
    max_selections=8,
)
if selected:
    st.plotly_chart(charts.price_index_lines(price_idx, selected),
                    width="stretch")
    st.caption(
        "Each category's index is the spend-weighted average of its items' "
        "unit prices, indexed to that item's first three observed months "
        "(start = 100) — so a rising line is genuine price inflation, not a "
        "shift in purchase mix."
    )
else:
    st.info("Select at least one category to plot the price index.")

st.subheader("Item price benchmarking")
st.caption(
    "For every item bought from 3+ suppliers: the average price paid vs the "
    "10th-percentile price actually achieved. The gap is negotiable."
)
bench = analytics.item_price_benchmark(filtered)
st.dataframe(
    bench,
    width="stretch",
    hide_index=True,
    column_config={
        "category": "Category",
        "item": "Item",
        "spend": st.column_config.NumberColumn("Spend", format="dollar"),
        "quantity": st.column_config.NumberColumn("Units bought"),
        "suppliers": st.column_config.NumberColumn("Suppliers"),
        "avg_price": st.column_config.NumberColumn("Avg price paid", format="dollar"),
        "benchmark_price": st.column_config.NumberColumn("Benchmark price", format="dollar"),
        "savings": st.column_config.NumberColumn("Addressable savings", format="dollar"),
        "savings_pct_of_spend": st.column_config.NumberColumn(
            "Savings % of spend", format="percent"
        ),
    },
)
