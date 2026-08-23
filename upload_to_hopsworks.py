"""
upload_to_hopsworks.py
======================
Uploads the cleaned AQI feature dataset to the Hopsworks Feature Store.

Falls back gracefully to saving a local CSV if the hopsworks SDK cannot be
installed (its dependency `hsfs>=3.7.0` is not on public PyPI).
"""
import os
import sys
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

# ── Reconfigure stdout to UTF-8 ──────────────────────────────────────────────
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# ── Optional hopsworks import ─────────────────────────────────────────────────
try:
    import hopsworks  # type: ignore[import]
    HOPSWORKS_AVAILABLE = True
except ImportError:
    HOPSWORKS_AVAILABLE = False
    print(
        "[WARNING] hopsworks SDK not installed (its dependency `hsfs>=3.7.0` is\n"
        "          not available on public PyPI). Falling back to CSV export.\n"
        "          See: https://github.com/logicalclocks/hopsworks-api/issues\n"
    )

from src.features.pipeline import _prepare_hopsworks_dataframe
from src.config import (
    FEATURE_GROUP_NAME,
    FEATURE_GROUP_VERSION,
    TARGET_COLUMN,
    FEATURE_COLUMNS,
)

# ── Load environment variables ────────────────────────────────────────────────
load_dotenv()
hopsworks_api_key = os.getenv("HOPSWORKS_API_KEY")

# ── Step 1: Load & preprocess dataset ─────────────────────────────────────────
print("--- Step 1: Loading Dataset ---")
candidate_paths = [
    "data/lahore_aqi_historical_2024_onwards.csv",
    "data/lahore_aqi_historical_backfilled.csv",
    "data/lahore_aqi_historical.csv",
    "lahore_aqi_historical.csv",  # root-level fallback
]
file_path = next((p for p in candidate_paths if os.path.exists(p)), None)
if file_path is None:
    raise FileNotFoundError(
        "No historical CSV found. Run historical_backfill.py first.\n"
        f"Searched: {candidate_paths}"
    )

print(f"Using dataset: {file_path}")
df = pd.read_csv(file_path)

print("Preprocessing dataset and removing NaNs...")
clean = _prepare_hopsworks_dataframe(df)

print(f"Dataset preprocessed: {len(clean)} clean rows.")
print(f"Columns  : {list(clean.columns)}")
print(f"Date range: {clean['timestamp'].min()} → {clean['timestamp'].max()}")

# ── Step 2: Upload (Hopsworks or CSV fallback) ─────────────────────────────────
if HOPSWORKS_AVAILABLE and hopsworks_api_key:
    print("\n--- Step 2: Connecting to Hopsworks ---")
    project = hopsworks.login(api_key_value=hopsworks_api_key)
    fs = project.get_feature_store()

    print("\n--- Step 3: Creating / Updating Feature Group ---")
    print("\n--- Step 3: Creating / Updating Feature Group ---")
    aqi_fg = fs.get_or_create_feature_group(
        name=FEATURE_GROUP_NAME,
        version=FEATURE_GROUP_VERSION,
        primary_key=["timestamp"],
        description=(
            "Lahore AQI dataset with 17 clean features "
            "(weather, pollutants, lags) and us_aqi target."
        ),
        event_time="timestamp",
        time_travel_format="DELTA",
    )

    print(f"Uploading {len(clean)} rows to Hopsworks… (this may take a few minutes)")
    aqi_fg.insert(clean)
    print("\n✅ Success! Historical dataset uploaded to Hopsworks Feature Store!")

else:
    # ── CSV fallback ──────────────────────────────────────────────────────────
    reason = (
        "hopsworks SDK not installed" if not HOPSWORKS_AVAILABLE
        else "HOPSWORKS_API_KEY not set in .env"
    )
    print(f"\n--- Step 2: Hopsworks unavailable ({reason}) ---")
    print("Saving cleaned feature dataset locally instead...")

    out_dir = Path("data")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "lahore_aqi_features_clean.csv"
    clean.to_csv(out_path, index=False)

    print(f"\n✅ Saved {len(clean)} clean rows → {out_path}")
    print(
        "\nTo upload to Hopsworks later:\n"
        "  1. Install a compatible Python (3.8–3.12) environment.\n"
        "  2. pip install hopsworks\n"
        "  3. Set HOPSWORKS_API_KEY in your .env file.\n"
        "  4. Re-run this script."
    )
