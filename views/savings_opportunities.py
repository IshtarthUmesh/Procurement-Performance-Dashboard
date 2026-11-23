"""Savings Opportunities — quantified, ranked cost-reduction levers."""

import streamlit as st

from src import analytics, charts
from src.data_loader import get_data, render_sidebar_filters

st.title("Savings Opportunities")
st.caption(
    "Every lever is quantified from the transaction data itself and ranked "
    "by addressable value. Assumptions are stated under each section."
)

df = get_data()
filtered = render_sidebar_filters(df, key_prefix="savings")

if filtered.empty:
    st.warning("No purchase orders match the current filters.")
    st.stop()

summary = analytics.savings_summary(filtered)
total = summary["estimated_savings"].sum()
spend = filtered["line_total"].sum()

st.metric(
    "Total addressable savings",
    charts.money(total),
    delta=f"{total / spend:.1%} of selected spend",
    delta_color="off",
)

st.dataframe(
    summary,
    width="stretch",
    hide_index=True,
    column_config={
        "opportunity": "Opportunity",
        "description": "How it's realised",
        "estimated_savings": st.column_config.NumberColumn(
            "Estimated savings", format="dollar"
        ),
    },
)

# ---------------------------------------------------------------------------
st.subheader("1 · Price variance consolidation")
st.caption(
    "Benchmark: the 10th-percentile unit price actually paid for the same "
    "item. Savings assume every future unit is bought at that benchmark."
)
bench = analytics.item_price_benchmark(filtered).head(15)
st.dataframe(
    bench[["category", "item", "spend", "avg_price", "benchmark_price", "savings"]],
    width="stretch",
    hide_index=True,
    column_config={
        "category": "Category",
        "item": "Item",
        "spend": st.column_config.NumberColumn("Spend", format="dollar"),
        "avg_price": st.column_config.NumberColumn("Avg price", format="dollar"),
        "benchmark_price": st.column_config.NumberColumn("Benchmark", format="dollar"),
        "savings": st.column_config.NumberColumn("Savings", format="dollar"),
    },
)

# ---------------------------------------------------------------------------
st.subheader("2 · Maverick (off-contract) spend")
maverick = analytics.maverick_spend_analysis(filtered)
m1, m2, m3 = st.columns(3)
m1.metric("Off-contract spend", charts.money(maverick["off_contract_spend"]))
m2.metric("Share of all spend", f"{maverick['off_contract_share']:.1%}")
m3.metric("Recovery potential", charts.money(maverick["estimated_savings"]),
          help="Assumes 5% price uplift is paid on off-contract purchases")
st.plotly_chart(charts.maverick_bu_bar(maverick["by_business_unit"]),
                width="stretch")

# ---------------------------------------------------------------------------
st.subheader("3 · Tail supplier consolidation")
tail = analytics.tail_spend_analysis(filtered)
t1, t2, t3 = st.columns(3)
t1.metric("Tail suppliers", f"{tail['tail_suppliers']} of {tail['total_suppliers']}",
          help="Suppliers beyond 95% of cumulative spend")
t2.metric("Tail spend", charts.money(tail["tail_spend"]))
t3.metric("Consolidation savings", charts.money(tail["estimated_savings"]),
          help="Assumes 6% saving when tail volume moves to strategic suppliers")

# ---------------------------------------------------------------------------
st.subheader("4 · Suspected duplicate payments")
dupes = analytics.duplicate_payment_flags(filtered)
if dupes.empty:
    st.success("No duplicate invoice lines detected in the current selection.")
else:
    st.caption(
        f"{len(dupes)} invoice lines match another line from the same supplier "
        "for the same item, quantity and amount within 3 days — worth "
        f"{charts.money(dupes['line_total'].sum())} if all are true duplicates."
    )
    st.dataframe(
        dupes,
        width="stretch",
        hide_index=True,
        column_config={
            "po_id": "PO",
            "prev_po": "Matches PO",
            "order_date": st.column_config.DateColumn("Order date"),
            "supplier_name": "Supplier",
            "item": "Item",
            "quantity": "Qty",
            "line_total": st.column_config.NumberColumn("Amount", format="dollar"),
            "gap_days": st.column_config.NumberColumn("Days apart"),
        },
    )
