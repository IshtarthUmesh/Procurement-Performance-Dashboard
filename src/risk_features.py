"""Per-supplier feature engineering and the vendor risk score.

The risk score blends two signals:

1. A transparent **composite score** — weighted, capped features a category
   manager can audit line by line.
2. An **Isolation Forest anomaly score** — catches suppliers whose overall
   behaviour pattern is unusual even when no single feature crosses a cap.

Final score = 0.55 * composite + 0.45 * anomaly percentile, on a 0-100 scale.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import RISK_HIGH_THRESHOLD, RISK_MEDIUM_THRESHOLD

FEATURE_COLS = [
    "late_rate",
    "avg_delay_days",
    "reject_rate",
    "price_volatility",
    "off_contract_share",
    "order_value_cv",
]

# Composite score weights and the feature value that maps to "maximum risk"
# for that dimension. Values are capped at 1 after dividing by the cap.
COMPOSITE_SPEC = {
    "late_rate": {"weight": 0.30, "cap": 0.60},
    "avg_delay_days": {"weight": 0.15, "cap": 10.0},
    "reject_rate": {"weight": 0.25, "cap": 0.12},
    "price_volatility": {"weight": 0.20, "cap": 0.15},
    "off_contract_share": {"weight": 0.10, "cap": 0.40},
}


def build_supplier_features(df: pd.DataFrame) -> pd.DataFrame:
    """One row per supplier with the risk feature vector plus context columns."""
    base = (
        df.groupby(["supplier_id", "supplier_name", "supplier_country", "category"])
        .agg(
            total_spend=("line_total", "sum"),
            po_count=("po_id", "nunique"),
            late_rate=("on_time", lambda s: 1 - s.mean()),
            avg_delay_days=("delay_days", "mean"),
            reject_rate=("quality_rejected", "mean"),
            off_contract_share=("on_contract", lambda s: 1 - s.mean()),
        )
        .reset_index()
    )

    # Price volatility: how erratically a supplier prices relative to the
    # market. Each line's unit price is normalised by the median price all
    # suppliers paid for that item in that month (removing inflation and
    # item-level price differences), then we take the std dev per supplier.
    market = (
        df.groupby(["item", "month"])["unit_price"].median().rename("market_price")
    )
    normed = df.merge(market, on=["item", "month"])
    normed["rel_price"] = normed["unit_price"] / normed["market_price"]
    vol = (
        normed.groupby("supplier_id")["rel_price"]
        .std()
        .rename("price_volatility")
        .fillna(0.0)
    )

    # Order-size consistency: coefficient of variation of line totals.
    ov = df.groupby("supplier_id")["line_total"].agg(["mean", "std"])
    order_cv = (ov["std"] / ov["mean"]).rename("order_value_cv").fillna(0.0)

    out = base.merge(vol, on="supplier_id").merge(order_cv, on="supplier_id")
    return out


def composite_score(features: pd.DataFrame) -> pd.Series:
    """Transparent weighted risk score, 0-100."""
    score = pd.Series(0.0, index=features.index)
    for col, spec in COMPOSITE_SPEC.items():
        contribution = (features[col] / spec["cap"]).clip(0, 1) * spec["weight"]
        score = score + contribution
    return score * 100


def risk_tier(score: pd.Series) -> pd.Series:
    return pd.cut(
        score,
        bins=[-np.inf, RISK_MEDIUM_THRESHOLD, RISK_HIGH_THRESHOLD, np.inf],
        labels=["Low", "Medium", "High"],
        right=False,
    )


def score_suppliers(features: pd.DataFrame, pipeline) -> pd.DataFrame:
    """Apply the trained pipeline and blend with the composite score."""
    features = features.copy()

    # IsolationForest.score_samples: lower = more anomalous. Convert to a
    # 0-100 "anomaly percentile" where 100 = most anomalous supplier.
    raw = pipeline.score_samples(features[FEATURE_COLS])
    anomaly_pct = pd.Series(raw, index=features.index).rank(pct=True, ascending=False) * 100

    features["composite_score"] = composite_score(features)
    features["anomaly_score"] = anomaly_pct
    features["risk_score"] = (
        0.55 * features["composite_score"] + 0.45 * features["anomaly_score"]
    ).round(1)
    features["risk_tier"] = risk_tier(features["risk_score"])
    return features.sort_values("risk_score", ascending=False).reset_index(drop=True)
