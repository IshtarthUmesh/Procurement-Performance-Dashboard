"""Procurement Spend Analysis Dashboard — entry point.

Run with:
    streamlit run app.py
"""

import streamlit as st

from src import charts

charts.setup_page("Procurement Spend Analysis")

pages = [
    st.Page("views/overview.py", title="Overview",
            icon=":material/dashboard:", default=True),
    st.Page("views/supplier_analysis.py", title="Supplier Analysis",
            icon=":material/factory:"),
    st.Page("views/category_spend.py", title="Category Spend",
            icon=":material/category:"),
    st.Page("views/savings_opportunities.py", title="Savings Opportunities",
            icon=":material/savings:"),
    st.Page("views/vendor_risk.py", title="Vendor Risk",
            icon=":material/warning:"),
]

st.navigation(pages).run()
