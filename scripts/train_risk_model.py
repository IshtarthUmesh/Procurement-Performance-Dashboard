"""Train the vendor risk anomaly model and save it to models/.

Pipeline: StandardScaler -> IsolationForest over the per-supplier feature
vector built in ``src.risk_features``. The fitted pipeline is stored with
joblib together with its metadata and is loaded by the Vendor Risk page.

Usage:
    python scripts/train_risk_model.py
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import MODEL_PATH
from src.data_loader import load_purchase_orders
from src.risk_features import FEATURE_COLS, build_supplier_features, score_suppliers


def main() -> None:
    df = load_purchase_orders()
    features = build_supplier_features(df)
    print(f"Training on {len(features)} suppliers x {len(FEATURE_COLS)} features")

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "forest",
                IsolationForest(
                    n_estimators=300,
                    contamination=0.08,
                    random_state=42,
                ),
            ),
        ]
    )
    pipeline.fit(features[FEATURE_COLS])

    scored = score_suppliers(features, pipeline)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "pipeline": pipeline,
            "feature_cols": FEATURE_COLS,
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "n_suppliers": len(features),
        },
        MODEL_PATH,
    )
    print(f"Saved model to {MODEL_PATH}\n")

    print("Top 10 suppliers by risk score:")
    cols = ["supplier_name", "category", "risk_score", "risk_tier",
            "late_rate", "reject_rate", "price_volatility"]
    print(scored[cols].head(10).to_string(index=False))

    tiers = scored["risk_tier"].value_counts()
    print(f"\nTier distribution: {tiers.to_dict()}")


if __name__ == "__main__":
    main()
