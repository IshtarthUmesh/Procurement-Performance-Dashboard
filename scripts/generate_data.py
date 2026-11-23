"""Generate the synthetic procurement dataset used by the dashboard.

Produces ``data/purchase_orders.csv`` (~45,000 PO lines, Jan 2023 - Jun 2026)
with a fixed random seed so the output is fully reproducible.

The generator deliberately embeds realistic patterns for the dashboard to
surface:

* Pareto-shaped spend distribution (a small share of suppliers carries most
  of the spend).
* Price inflation drift in Electronics, Chemicals and Raw Materials.
* Unit-price variance for identical items across suppliers (consolidation
  savings opportunity).
* A set of "risky" vendors with chronic late delivery, elevated quality
  rejects and erratic pricing.
* Maverick (off-contract) purchasing concentrated in specific business units.
* Single-source dependency in the Machinery category.
* Q4 demand seasonality and a handful of near-duplicate invoice lines.

Usage:
    python scripts/generate_data.py
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
START_DATE = date(2023, 1, 1)
END_DATE = date(2026, 6, 30)
TARGET_ROWS = 45_000
OUTPUT_PATH = Path(__file__).resolve().parents[1] / "data" / "purchase_orders.csv"

BUSINESS_UNITS = ["Manufacturing", "Corporate", "R&D", "Retail Operations", "Field Services"]

# Business units where off-contract ("maverick") buying is rampant.
MAVERICK_BUSINESS_UNITS = {"Field Services": 0.32, "R&D": 0.24}
BASELINE_MAVERICK_RATE = 0.05

PAYMENT_TERMS = ["Net 30", "Net 45", "Net 60", "Net 15"]
PAYMENT_TERM_WEIGHTS = [0.45, 0.30, 0.15, 0.10]

# Annual price drift per category (compounding, applied continuously).
CATEGORY_INFLATION = {
    "Electronics": 0.13,
    "Chemicals": 0.11,
    "Raw Materials": 0.08,
    "Logistics": 0.05,
    "Packaging": 0.04,
    "MRO": 0.03,
    "IT Services": 0.03,
    "Machinery": 0.03,
    "Professional Services": 0.04,
    "Office Supplies": 0.02,
}

# Q4 buying spike plus a smaller fiscal push in March.
MONTH_SEASONALITY = {
    1: 0.90, 2: 0.92, 3: 1.10, 4: 0.98, 5: 1.00, 6: 1.02,
    7: 0.95, 8: 0.96, 9: 1.05, 10: 1.25, 11: 1.35, 12: 1.30,
}

# category -> list of (item, base unit price, typical order quantity)
CATALOG: dict[str, list[tuple[str, float, int]]] = {
    "Raw Materials": [
        ("Steel Coil (per ton)", 780.0, 25),
        ("Aluminium Sheet (per ton)", 2350.0, 8),
        ("Copper Wire Rod (per ton)", 8900.0, 3),
        ("Polypropylene Resin (per ton)", 1150.0, 15),
        ("Rubber Compound (per ton)", 1620.0, 10),
    ],
    "Packaging": [
        ("Corrugated Boxes (per 1000)", 620.0, 12),
        ("Stretch Film Rolls", 38.0, 120),
        ("Wooden Pallets", 14.5, 250),
        ("Printed Labels (per 1000)", 42.0, 60),
        ("Bubble Wrap Rolls", 26.0, 90),
    ],
    "Electronics": [
        ("PCB Assembly A-11", 46.0, 400),
        ("Microcontroller MCU-9", 6.8, 2500),
        ("Power Supply Unit 450W", 58.0, 150),
        ("Sensor Module SM-3", 21.5, 600),
        ("Connector Kit CK-8", 3.9, 3000),
    ],
    "MRO": [
        ("Industrial Bearings", 34.0, 80),
        ("Hydraulic Fluid (drum)", 210.0, 12),
        ("Safety Gloves (per 100)", 95.0, 30),
        ("Conveyor Belt Section", 480.0, 6),
        ("Machine Lubricant (drum)", 165.0, 15),
    ],
    "Logistics": [
        ("FTL Freight (per load)", 1850.0, 4),
        ("LTL Freight (per shipment)", 420.0, 10),
        ("Ocean Container (40ft)", 3200.0, 2),
        ("Air Freight (per 100kg)", 610.0, 5),
        ("Last-Mile Delivery Batch", 260.0, 12),
    ],
    "IT Services": [
        ("Cloud Hosting (monthly block)", 4200.0, 1),
        ("SaaS Licences (per seat, annual)", 380.0, 25),
        ("Managed IT Support (monthly)", 5600.0, 1),
        ("Network Equipment Bundle", 2900.0, 2),
        ("Cybersecurity Audit", 8500.0, 1),
    ],
    "Office Supplies": [
        ("Copy Paper (per case)", 32.0, 40),
        ("Toner Cartridges", 78.0, 15),
        ("Office Chairs", 145.0, 8),
        ("Stationery Kit", 18.0, 50),
        ("Whiteboard & Supplies", 60.0, 6),
    ],
    "Chemicals": [
        ("Industrial Solvent (drum)", 340.0, 18),
        ("Adhesive Compound (drum)", 520.0, 10),
        ("Cleaning Agent (drum)", 190.0, 22),
        ("Coating Resin (drum)", 610.0, 8),
        ("Catalyst Powder (per 25kg)", 880.0, 5),
    ],
    "Machinery": [
        ("CNC Spare Parts Kit", 3400.0, 2),
        ("Injection Mould Tooling", 12500.0, 1),
        ("Robotic Arm Servicing", 6800.0, 1),
        ("Compressor Unit", 9200.0, 1),
        ("Precision Jig Set", 2100.0, 3),
    ],
    "Professional Services": [
        ("Legal Advisory (retainer)", 7500.0, 1),
        ("Market Research Study", 11000.0, 1),
        ("Recruitment Fee", 5200.0, 1),
        ("Training Programme", 3800.0, 2),
        ("Audit & Compliance Review", 9600.0, 1),
    ],
}

# Supplier name fragments per region for a plausible global supplier base.
SUPPLIER_NAMES = {
    "USA": ["Apex", "Summit", "Liberty", "Pioneer", "Redwood", "Sterling", "Frontier", "Cascade"],
    "Germany": ["Rheinwerk", "Bergmann", "Hanseatic", "Schwarzwald", "Adler"],
    "China": ["Shenzhou", "Golden Dragon", "Pacific Rim", "Jade Harbor", "Great Wall"],
    "India": ["Bharat", "Suryodaya", "Ganga", "Himalaya", "Lotus"],
    "Japan": ["Sakura", "Fujikawa", "Nippon", "Kansai"],
    "UK": ["Albion", "Thames", "Crown", "Windsor"],
    "Mexico": ["Sierra", "Azteca", "Del Norte"],
    "Vietnam": ["Mekong", "Saigon", "Halong"],
    "Poland": ["Vistula", "Baltic", "Krakow"],
    "South Korea": ["Han River", "Busan", "Taegu"],
}

SUPPLIER_SUFFIXES = {
    "Raw Materials": ["Metals", "Materials", "Alloys", "Resources"],
    "Packaging": ["Packaging", "Containers", "Pack Solutions"],
    "Electronics": ["Electronics", "Components", "Circuits", "Semitech"],
    "MRO": ["Industrial Supply", "MRO Services", "Maintenance Co"],
    "Logistics": ["Logistics", "Freight", "Shipping Lines", "Transport"],
    "IT Services": ["Technologies", "IT Solutions", "Digital Systems"],
    "Office Supplies": ["Office Products", "Supplies Co", "Workplace"],
    "Chemicals": ["Chemicals", "Chem Industries", "Specialty Chem"],
    "Machinery": ["Machinery", "Engineering Works", "Precision Tools"],
    "Professional Services": ["Consulting", "Advisory", "Partners"],
}

# Suppliers deliberately generated with poor performance profiles. The risk
# model is later sanity-checked against this list.
RISKY_SUPPLIER_COUNT = 6


def build_suppliers(rng: np.random.Generator) -> pd.DataFrame:
    """Create ~120 suppliers with category, country and behaviour profile."""
    countries = list(SUPPLIER_NAMES.keys())
    country_weights = np.array([0.28, 0.12, 0.14, 0.10, 0.07, 0.07, 0.06, 0.06, 0.05, 0.05])

    rows = []
    used_names: set[str] = set()
    supplier_id = 1
    for category in CATALOG:
        # Machinery is intentionally single-sourced (concentration risk);
        # other categories have a healthy supplier base.
        n_suppliers = 2 if category == "Machinery" else int(rng.integers(10, 16))
        for _ in range(n_suppliers):
            for _attempt in range(50):
                country = rng.choice(countries, p=country_weights / country_weights.sum())
                prefix = rng.choice(SUPPLIER_NAMES[country])
                suffix = rng.choice(SUPPLIER_SUFFIXES[category])
                name = f"{prefix} {suffix}"
                if name not in used_names:
                    used_names.add(name)
                    break
            rows.append(
                {
                    "supplier_id": f"SUP-{supplier_id:03d}",
                    "supplier_name": name,
                    "supplier_country": country,
                    "category": category,
                    # Pareto-shaped spend share within the category.
                    "spend_weight": float(rng.pareto(1.16) + 0.05),
                    # Price positioning vs. catalog base price (drives the
                    # price-variance savings opportunity).
                    "price_factor": float(rng.uniform(0.93, 1.15)),
                    "is_risky": False,
                }
            )
            supplier_id += 1

    suppliers = pd.DataFrame(rows)

    # Machinery single-source: make one supplier overwhelmingly dominant.
    machinery = suppliers[suppliers["category"] == "Machinery"].index
    suppliers.loc[machinery[0], "spend_weight"] = 40.0
    suppliers.loc[machinery[1], "spend_weight"] = 0.8

    # Plant the risky vendors: meaningful spend, spread across categories.
    eligible = suppliers[
        (suppliers["category"] != "Machinery") & (suppliers["spend_weight"] > 0.4)
    ].sample(n=RISKY_SUPPLIER_COUNT, random_state=SEED)
    suppliers.loc[eligible.index, "is_risky"] = True

    return suppliers


def month_range(start: date, end: date) -> list[date]:
    months = []
    current = date(start.year, start.month, 1)
    while current <= end:
        months.append(current)
        current = date(current.year + (current.month == 12), current.month % 12 + 1, 1)
    return months


def generate_orders(suppliers: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Generate PO lines month by month, supplier by supplier."""
    months = month_range(START_DATE, END_DATE)

    # Category monthly volume budget, scaled so total rows ~= TARGET_ROWS.
    category_share = {
        "Raw Materials": 0.14, "Packaging": 0.12, "Electronics": 0.13,
        "MRO": 0.13, "Logistics": 0.12, "IT Services": 0.07,
        "Office Supplies": 0.09, "Chemicals": 0.10, "Machinery": 0.03,
        "Professional Services": 0.07,
    }
    base_lines_per_month = TARGET_ROWS / len(months)

    supplier_groups = {
        cat: grp.copy() for cat, grp in suppliers.groupby("category")
    }
    for grp in supplier_groups.values():
        grp["share"] = grp["spend_weight"] / grp["spend_weight"].sum()

    rows = []
    po_counter = 1
    for month_start in months:
        season = MONTH_SEASONALITY[month_start.month]
        years_elapsed = (month_start - START_DATE).days / 365.25
        days_in_month = (
            date(month_start.year + (month_start.month == 12), month_start.month % 12 + 1, 1)
            - month_start
        ).days

        for category, items in CATALOG.items():
            inflation = (1 + CATEGORY_INFLATION[category]) ** years_elapsed
            grp = supplier_groups[category]
            n_lines = rng.poisson(base_lines_per_month * category_share[category] * season)

            if n_lines == 0:
                continue
            idx = rng.choice(len(grp), size=n_lines, p=grp["share"].to_numpy())
            chosen = grp.iloc[idx]

            for supplier in chosen.itertuples():
                item_name, base_price, base_qty = items[int(rng.integers(len(items)))]
                order_day = int(rng.integers(days_in_month))
                order_date = month_start + timedelta(days=order_day)

                # --- price -------------------------------------------------
                noise_scale = 0.14 if supplier.is_risky else 0.045
                price_noise = float(rng.normal(1.0, noise_scale))
                unit_price = round(
                    base_price * supplier.price_factor * inflation * max(price_noise, 0.55), 2
                )

                # --- quantity ----------------------------------------------
                quantity = max(1, int(rng.normal(base_qty, base_qty * 0.35)))

                # --- delivery ----------------------------------------------
                lead_time = int(rng.integers(5, 35))
                promised = order_date + timedelta(days=lead_time)
                if supplier.is_risky:
                    delay = max(0, int(rng.normal(11, 6)))
                else:
                    # Healthy suppliers deliver on time ~85% of the time.
                    delay = max(0, int(rng.normal(-4, 4)))
                actual = promised + timedelta(days=delay)

                # --- quality -----------------------------------------------
                reject_prob = 0.11 if supplier.is_risky else 0.015
                rejected = bool(rng.random() < reject_prob)

                # --- contract status ---------------------------------------
                business_unit = BUSINESS_UNITS[int(rng.integers(len(BUSINESS_UNITS)))]
                maverick_rate = MAVERICK_BUSINESS_UNITS.get(business_unit, BASELINE_MAVERICK_RATE)
                on_contract = bool(rng.random() > maverick_rate)

                rows.append(
                    {
                        "po_id": f"PO-{order_date.year}-{po_counter:05d}",
                        "order_date": order_date,
                        "promised_delivery_date": promised,
                        "actual_delivery_date": actual,
                        "supplier_id": supplier.supplier_id,
                        "supplier_name": supplier.supplier_name,
                        "supplier_country": supplier.supplier_country,
                        "category": category,
                        "item": item_name,
                        "business_unit": business_unit,
                        "unit_price": unit_price,
                        "quantity": quantity,
                        "line_total": round(unit_price * quantity, 2),
                        "payment_terms": str(
                            rng.choice(PAYMENT_TERMS, p=PAYMENT_TERM_WEIGHTS)
                        ),
                        "on_contract": on_contract,
                        "quality_rejected": rejected,
                    }
                )
                po_counter += 1

    df = pd.DataFrame(rows)

    # Inject ~40 near-duplicate invoice lines (same supplier/item/amount,
    # order date shifted by 1-2 days, fresh PO id).
    dupes = df.sample(n=40, random_state=SEED).copy()
    shift = rng.integers(1, 3, size=len(dupes))
    dupes["order_date"] = [
        d + timedelta(days=int(s))
        for d, s in zip(dupes["order_date"], shift, strict=True)
    ]
    dupes["po_id"] = [
        f"PO-{d.year}-{po_counter + i:05d}"
        for i, d in enumerate(dupes["order_date"])
    ]
    df = pd.concat([df, dupes], ignore_index=True)

    return df.sort_values("order_date").reset_index(drop=True)


def main() -> None:
    rng = np.random.default_rng(SEED)
    suppliers = build_suppliers(rng)
    orders = generate_orders(suppliers, rng)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    orders.to_csv(OUTPUT_PATH, index=False)

    risky = suppliers.loc[suppliers["is_risky"], "supplier_name"].tolist()
    print(f"Wrote {len(orders):,} PO lines to {OUTPUT_PATH}")
    print(f"Suppliers: {suppliers['supplier_name'].nunique()} | "
          f"Categories: {orders['category'].nunique()} | "
          f"Total spend: ${orders['line_total'].sum():,.0f}")
    print(f"Planted risky vendors (for model sanity-check): {', '.join(sorted(risky))}")


if __name__ == "__main__":
    main()
