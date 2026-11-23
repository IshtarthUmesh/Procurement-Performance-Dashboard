"""Vendor Risk — ML-assisted supplier risk scoring."""

import joblib
import streamlit as st

from src import charts
from src.config import MODEL_PATH
from src.data_loader import get_data
from src.risk_features import (
    COMPOSITE_SPEC,
    FEATURE_COLS,
    build_supplier_features,
    score_suppliers,
)

st.title("Vendor Risk")
st.caption(
    "Every supplier scored 0-100 by blending a transparent weighted scorecard "
    "with an Isolation Forest anomaly model trained on the full order history."
)


@st.cache_resource(show_spinner="Loading risk model...")
def load_model():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            "Risk model is missing. Run `python scripts/train_risk_model.py`."
        )
    artifact = joblib.load(MODEL_PATH)
    if not isinstance(artifact, dict) or "pipeline" not in artifact:
        raise ValueError("Risk model artifact is invalid; retrain it with the training script.")
    if artifact.get("feature_cols") != FEATURE_COLS:
        raise ValueError("Risk model features are outdated; retrain it with the training script.")
    return artifact


@st.cache_data(show_spinner="Scoring suppliers...")
def get_scored_suppliers():
    df = get_data()
    artifact = load_model()
    features = build_supplier_features(df)
    return score_suppliers(features, artifact["pipeline"])


df = get_data()
scored = get_scored_suppliers()

st.sidebar.header("Filters")
tier_filter = st.sidebar.multiselect(
    "Risk tier", options=["High", "Medium", "Low"], key="risk_tier"
)
cat_filter = st.sidebar.multiselect(
    "Category", options=sorted(scored["category"].unique()), key="risk_cat"
)
view = scored
if tier_filter:
    view = view[view["risk_tier"].isin(tier_filter)]
if cat_filter:
    view = view[view["category"].isin(cat_filter)]
st.sidebar.caption(f"{len(view)} of {len(scored)} suppliers shown")
st.sidebar.info(
    "Risk is scored on the **full order history** (the model is trained "
    "once on all data), so date filters don't apply here."
)

high = scored[scored["risk_tier"] == "High"]
c1, c2, c3 = st.columns(3)
c1.metric("High-risk suppliers", f"{len(high)}")
c2.metric("Spend with high-risk suppliers", charts.money(high["total_spend"].sum()))
c3.metric("Share of total spend", f"{high['total_spend'].sum() / scored['total_spend'].sum():.1%}")

left, right = st.columns([3, 2])
with left:
    st.plotly_chart(charts.risk_scatter(view if len(view) else scored),
                    width="stretch")
    st.caption(
        "Dotted lines mark median spend and the high-risk threshold — the "
        "top-right quadrant (high spend, high risk) needs attention first."
    )
with right:
    st.plotly_chart(charts.tier_distribution_bar(scored), width="stretch")

st.subheader("Risk leaderboard")
if view.empty:
    st.info("No suppliers match the selected risk filters.")
else:
    st.dataframe(
        view[
            ["supplier_name", "category", "supplier_country", "total_spend",
             "risk_score", "risk_tier", "late_rate", "reject_rate",
             "price_volatility", "off_contract_share"]
        ],
        width="stretch",
        hide_index=True,
        column_config={
            "supplier_name": "Supplier",
            "category": "Category",
            "supplier_country": "Country",
            "total_spend": st.column_config.NumberColumn("Spend", format="dollar"),
            "risk_score": st.column_config.ProgressColumn(
                "Risk score", format="%.0f", min_value=0, max_value=100
            ),
            "risk_tier": "Tier",
            "late_rate": st.column_config.NumberColumn("Late rate", format="percent"),
            "reject_rate": st.column_config.NumberColumn("Reject rate", format="percent"),
            "price_volatility": st.column_config.NumberColumn(
                "Price volatility", format="%.3f"
            ),
            "off_contract_share": st.column_config.NumberColumn(
                "Off-contract", format="percent"
            ),
        },
    )

if not view.empty:
    st.subheader("Why is this supplier risky?")
    pick = st.selectbox("Supplier", options=view["supplier_name"].tolist())
    row = scored[scored["supplier_name"] == pick].iloc[0]

    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Risk score", f"{row['risk_score']:.0f} / 100")
    r2.metric("Tier", f"{charts.TIER_ICONS[str(row['risk_tier'])]} {row['risk_tier']}")
    r3.metric("Scorecard component", f"{row['composite_score']:.0f}")
    r4.metric("Anomaly component", f"{row['anomaly_score']:.0f}",
              help="Isolation Forest anomaly percentile vs all suppliers")

    b1, b2 = st.columns(2)
    with b1:
        st.plotly_chart(charts.risk_factor_bar(row, COMPOSITE_SPEC), width="stretch")
    with b2:
        st.plotly_chart(charts.delay_distribution(df, pick), width="stretch")

with st.expander("Methodology"):
    st.markdown(
        """
        **Feature vector per supplier** — late delivery rate, average delay
        days, quality reject rate, price volatility (std dev of unit price
        relative to the item's market median that month), off-contract share,
        and order-value variability.

        **Composite scorecard (55% of the score)** — each feature is scaled
        against a risk cap and weighted: late rate 30%, rejects 25%, price
        volatility 20%, average delay 15%, off-contract share 10%.

        **Anomaly model (45% of the score)** — a `StandardScaler +
        IsolationForest` pipeline (300 trees) trained on all suppliers'
        feature vectors. Its anomaly ranking catches suppliers whose overall
        pattern is unusual even when no single metric breaches its cap.

        **Tiers** — High ≥ 70, Medium ≥ 45, otherwise Low. Retrain any time
        with `python scripts/train_risk_model.py`.
        """
    )
