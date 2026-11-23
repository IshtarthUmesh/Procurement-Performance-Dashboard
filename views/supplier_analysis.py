"""Supplier Analysis — concentration, scorecards and per-supplier drill-down."""

import math

import streamlit as st

from src import analytics, charts
from src.data_loader import get_data, render_sidebar_filters

st.title("Supplier Analysis")
st.caption(
    "Who the spend goes to, how concentrated it is, and how each supplier "
    "performs on delivery and quality."
)

df = get_data()
filtered = render_sidebar_filters(df, key_prefix="suppliers")

if filtered.empty:
    st.warning("No purchase orders match the current filters.")
    st.stop()

pareto = analytics.supplier_pareto(filtered)
top20_count = max(1, math.ceil(len(pareto) * 0.2))
top20_share = pareto["cumulative_share"].iloc[top20_count - 1]

c1, c2, c3 = st.columns(3)
c1.metric("Suppliers in selection", f"{len(pareto):,}")
c2.metric("Top-20% suppliers hold", f"{top20_share:.0%} of spend")
suppliers_to_80 = int((pareto["cumulative_share"] < 0.8).sum()) + 1
c3.metric("Suppliers covering 80% of spend", f"{suppliers_to_80}")

st.plotly_chart(charts.pareto_chart(pareto), width="stretch")

st.subheader("Supplier scorecard")
st.caption("Sortable performance table — click a column header to rank by it.")
scorecard = analytics.supplier_scorecard(filtered)
st.dataframe(
    scorecard,
    width="stretch",
    hide_index=True,
    column_config={
        "supplier_name": "Supplier",
        "supplier_country": "Country",
        "category": "Category",
        "spend": st.column_config.NumberColumn("Spend", format="dollar"),
        "po_count": st.column_config.NumberColumn("POs"),
        "on_time_rate": st.column_config.ProgressColumn(
            "On-time rate", format="percent", min_value=0, max_value=1
        ),
        "avg_delay_days": st.column_config.NumberColumn("Avg delay (days)", format="%.1f"),
        "reject_rate": st.column_config.NumberColumn("Reject rate", format="percent"),
        "off_contract_share": st.column_config.NumberColumn("Off-contract", format="percent"),
    },
)

st.subheader("Supplier drill-down")
supplier = st.selectbox(
    "Choose a supplier",
    options=scorecard["supplier_name"].tolist(),
)
row = scorecard[scorecard["supplier_name"] == supplier].iloc[0]

d1, d2, d3, d4 = st.columns(4)
d1.metric("Spend", charts.money(row["spend"]))
d2.metric("Purchase orders", f"{row['po_count']:,}")
d3.metric("On-time delivery", f"{row['on_time_rate']:.1%}")
d4.metric("Quality reject rate", f"{row['reject_rate']:.1%}")

left, right = st.columns(2)
with left:
    st.plotly_chart(charts.supplier_monthly_trend(filtered, supplier),
                    width="stretch")
with right:
    st.plotly_chart(charts.delay_distribution(filtered, supplier),
                    width="stretch")
