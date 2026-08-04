"""Feature engineering: time-based and derived features for direct AQI prediction.

Target: "aqi" (Open-Meteo us_aqi) — fetched from API, never computed manually.
All PM-based formulas are removed from target construction.
Lag features are computed from AQI directly (not PM2.5).
"""
import pandas as pd
import numpy as np


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add time-based and derived features to a raw AQI/weather dataframe.

    IMPORTANT — data leakage notes:
    - All lag/change features use shift(1) so only *past* information leaks.
    - "aqi" is the pre-computed Open-Meteo us_aqi target — it stays as-is.
    - "modeled_pm25" / "modeled_pm10" are input features.
    - "aqi_lag_1" = previous hour's AQI (the "Current AQI" feature, safely shifted).
    """
    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"])
    out = out.sort_values("timestamp").reset_index(drop=True)

    # List of continuous input pollutant/weather columns to clean NaNs for
    continuous_cols = [
        "temperature", "humidity", "pressure", "modeled_pm10",
        "modeled_pm25", "ozone", "no2", "no", "so2", "co"
    ]
    for col in continuous_cols:
        if col not in out.columns:
            out[col] = np.nan
        # Interpolate / fill NaNs in continuous series
        out[col] = pd.to_numeric(out[col], errors="coerce")
        out[col] = out[col].ffill().bfill()
        if out[col].isna().any():
            out[col] = out[col].fillna(0.0)

    # Time-based features
    out["hour"] = out["timestamp"].dt.hour
    out["day"] = out["timestamp"].dt.day
    out["day_of_week"] = out["timestamp"].dt.dayofweek
    out["week_of_year"] = out["timestamp"].dt.isocalendar().week.astype(int)
    out["month"] = out["timestamp"].dt.month

    # AQI change rate: hour-over-hour delta of target (shifted back 1 to avoid leakage)
    if "aqi" in out.columns:
        out["aqi"] = pd.to_numeric(out["aqi"], errors="coerce")
        out["aqi_change_rate"] = out["aqi"].diff().shift(1).fillna(0.0)
        out["aqi_lag_1"] = out["aqi"].shift(1).bfill().fillna(out["aqi"])
    else:
        out["aqi_change_rate"] = 0.0
        out["aqi_lag_1"] = np.nan

    return out


def clean_features_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess and clean dataframe: compute features, remove NaNs, and ensure target is valid.
    """
    from src.config import FEATURE_COLUMNS, TARGET_COLUMN
    featured = compute_features(df)
    
    # Drop rows without target if target is expected
    if TARGET_COLUMN in featured.columns:
        featured = featured.dropna(subset=[TARGET_COLUMN]).reset_index(drop=True)
    
    # Ensure all required feature columns exist and drop any remaining NaNs
    available_cols = [c for c in FEATURE_COLUMNS if c in featured.columns]
    featured = featured.dropna(subset=available_cols).reset_index(drop=True)
    return featured



def aqi_category(aqi: float) -> str:
    """Map an AQI value to the standard EPA category label."""
    if aqi is None or pd.isna(aqi):
        return "Unknown"
    if aqi <= 50:
        return "Good"
    if aqi <= 100:
        return "Moderate"
    if aqi <= 150:
        return "Unhealthy for Sensitive Groups"
    if aqi <= 200:
        return "Unhealthy"
    if aqi <= 300:
        return "Very Unhealthy"
    return "Hazardous"


def pm25_to_aqi_category(pm25: float) -> str:
    """Backward-compatible alias for AQI category labels."""
    return aqi_category(pm25)
