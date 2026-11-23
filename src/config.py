"""Central configuration: paths, palette, and analysis assumptions."""

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "data" / "purchase_orders.csv"
MODEL_PATH = ROOT_DIR / "models" / "vendor_risk_model.joblib"

# ---------------------------------------------------------------------------
# Visual system (light theme).
# Categorical hues are assigned in this fixed order and never cycled.
# ---------------------------------------------------------------------------
CATEGORICAL = [
    "#2a78d6",  # blue
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#008300",  # green
    "#4a3aa7",  # violet
    "#e34948",  # red
    "#e87ba4",  # magenta
    "#eb6834",  # orange
]

SEQUENTIAL_BLUES = [
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec",
    "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab",
    "#184f95", "#104281", "#0d366b",
]

STATUS = {
    "good": "#0ca30c",
    "warning": "#fab219",
    "serious": "#ec835a",
    "critical": "#d03b3b",
}

SURFACE = "#fcfcfb"       # chart surface
PAGE_BG = "#f9f9f7"       # page plane
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
AXIS_LINE = "#c3c2b7"

FONT_FAMILY = 'system-ui, -apple-system, "Segoe UI", sans-serif'

# ---------------------------------------------------------------------------
# Analysis assumptions (documented in the README methodology section).
# ---------------------------------------------------------------------------
# Benchmark for price-variance savings: the 10th-percentile unit price paid
# for the same item (a realistically negotiable target, not the absolute min).
PRICE_BENCHMARK_QUANTILE = 0.10
# An item needs at least this many suppliers to count as a consolidation play.
MIN_SUPPLIERS_FOR_BENCHMARK = 3

# Suppliers beyond this cumulative-spend share are "tail" suppliers.
PARETO_TAIL_THRESHOLD = 0.95
# Assumed saving from consolidating tail spend into strategic suppliers.
TAIL_CONSOLIDATION_RATE = 0.06

# Assumed recovery from routing off-contract spend back onto contracts.
MAVERICK_RECOVERY_RATE = 0.05

# Two invoice lines matching on supplier/item/quantity/amount within this
# window are flagged as potential duplicates.
DUPLICATE_WINDOW_DAYS = 3

# Vendor risk score tiers (0-100 scale).
RISK_HIGH_THRESHOLD = 70
RISK_MEDIUM_THRESHOLD = 45
