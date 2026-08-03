"""Feature pipeline: fetch raw data, engineer features, persist locally and to Hopsworks."""
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from src.config import (
    FEATURE_GROUP_NAME,
    FEATURE_GROUP_VERSION,
    FEATURES_CSV,
    HISTORICAL_CSV,
)
from src.features.engineering import compute_features
from src.features.fetch_raw import (
    current_reading_to_row,
    fetch_aqicn_current,
    fetch_openmeteo_current_row,
)

load_dotenv()


def run_feature_pipeline(upload_to_hopsworks: bool = False) -> pd.DataFrame:
    """
    Run the hourly feature pipeline:
    1. Fetch current Open-Meteo weather + AQ (same provider as training)
    2. Optionally enrich with AQICN ground-truth PM2.5
    3. Append to historical dataset and compute features
    """
    Path("data").mkdir(exist_ok=True)

    aqicn_data = None
    try:
        aqicn_data = fetch_aqicn_current()
    except Exception:
        pass

    meteo_row = fetch_openmeteo_current_row()
    new_row = current_reading_to_row(aqicn_data, meteo_row)

    if HISTORICAL_CSV.exists():
        history = pd.read_csv(HISTORICAL_CSV, parse_dates=["timestamp"])
        combined = pd.concat([history, new_row], ignore_index=True)
        combined = combined.drop_duplicates(subset=["timestamp"], keep="last")
    else:
        combined = new_row

    featured = compute_features(combined)
    featured.to_csv(FEATURES_CSV, index=False)
    featured.to_csv(HISTORICAL_CSV, index=False)

    if upload_to_hopsworks:
        _upload_to_hopsworks(featured)

    return featured


def _upload_to_hopsworks(df: pd.DataFrame) -> None:
    """Insert feature dataframe into Hopsworks Feature Store."""
    import hopsworks

    api_key = os.getenv("HOPSWORKS_API_KEY")
    if not api_key:
        raise ValueError("HOPSWORKS_API_KEY not found in .env")

    clean = df.dropna(subset=["ground_pm25"]).copy()
    clean["timestamp"] = pd.to_datetime(clean["timestamp"])

    project = hopsworks.login(api_key_value=api_key)
    fs = project.get_feature_store()
    fg = fs.get_or_create_feature_group(
        name=FEATURE_GROUP_NAME,
        version=FEATURE_GROUP_VERSION,
        primary_key=["timestamp"],
        description="Lahore AQI features with weather and derived fields.",
        event_time="timestamp",
    )
    fg.insert(clean)
