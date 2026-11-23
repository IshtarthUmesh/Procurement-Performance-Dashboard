"""Overview — executive summary of purchasing activity."""

import streamlit as st

from src import analytics, charts
from src.data_loader import get_data, render_sidebar_filters

st.title("Procurement Spend Analysis")
st.caption(
    "Executive view of purchasing activity: spend levels, trends, categories "
    "and supplier concentration. Use the sidebar filters to slice any view."
)

df = get_data()
filtered = render_sidebar_filters(df, key_prefix="overview")

if filtered.empty:
    st.warning("No purchase orders match the current filters.")
    st.stop()

kpis = analytics.kpi_summary(filtered)
savings = analytics.savings_summary(filtered)["estimated_savings"].sum()

c1, c2, c3, c4, c5, c6 = st.columns(6)
yoy = kpis["yoy_change"]
c1.metric("Total spend", charts.money(kpis["total_spend"]),
          delta=f"{yoy:+.1%} vs prior 12 mo" if yoy is not None else None,
          delta_color="inverse")
c2.metric("Purchase orders", f"{kpis['po_count']:,}")
c3.metric("Active suppliers", f"{kpis['supplier_count']:,}")
c4.metric("Avg PO value", charts.money(kpis["avg_po_value"]))
c5.metric("On-time delivery", f"{kpis['on_time_rate']:.1%}")
c6.metric("Identified savings", charts.money(savings),
          help="Total addressable savings — see the Savings Opportunities page")

st.plotly_chart(charts.monthly_spend_line(analytics.monthly_spend(filtered)),
                width="stretch")

left, right = st.columns(2)
with left:
    st.plotly_chart(charts.category_spend_bar(analytics.spend_by_category(filtered)),
                    width="stretch")
with right:
    st.plotly_chart(charts.top_suppliers_bar(analytics.top_suppliers(filtered)),
                    width="stretch")

left2, right2 = st.columns(2)
with left2:
    st.plotly_chart(charts.country_spend_bar(analytics.spend_by_country(filtered)),
                    width="stretch")
with right2:
    st.subheader("Concentration warnings")
    single = analytics.single_source_categories(filtered)
    if single.empty:
        st.success("No category is single-source dominated at the 85% threshold.")
    else:
        for row in single.itertuples():
            st.warning(
                f"**{row.category}** — {row.top_supplier_share:.0%} of category spend "
                f"sits with **{row.dominant_supplier}** "
                f"({charts.money(row.supplier_spend)}). Single-source dependency."
            )
    st.caption(
        "A category is flagged when one supplier holds ≥ 85% of its spend — "
        "a supply-continuity and negotiating-leverage risk."
    )

with st.expander("Browse filtered purchase orders"):
    st.dataframe(
        filtered.drop(columns=["month", "year"]).sort_values("order_date", ascending=False),
        width="stretch",
        hide_index=True,
    )
