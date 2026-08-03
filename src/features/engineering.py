"""Feature engineering: time-based and derived features."""
import pandas as pd


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add time-based and derived features to a raw AQI/weather dataframe.

    IMPORTANT — data leakage notes:
    - `aqi_change_rate` uses shift(1) so it only contains *past* information.
    - `modeled_pm25` (from Open-Meteo) is highly correlated with `ground_pm25`
      and can inflate model scores. Remove it from FEATURE_COLUMNS in config.py
      if you want a fully honest evaluation without API-provided PM2.5 estimates.
    """
    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"])
    out = out.sort_values("timestamp").reset_index(drop=True)

    out["hour"] = out["timestamp"].dt.hour
    out["day_of_week"] = out["timestamp"].dt.dayofweek
    out["month"] = out["timestamp"].dt.month

    pm25_col = "ground_pm25" if "ground_pm25" in out.columns else "modeled_pm25"
    if pm25_col in out.columns:
        # shift(1) ensures we only use *previous* hour's value — no leakage
        out["aqi_change_rate"] = out[pm25_col].diff().shift(1).fillna(0)
        out["pm25_lag_1"] = out[pm25_col].shift(1)
        out["pm25_lag_24"] = out[pm25_col].shift(24)
    else:
        out["aqi_change_rate"] = 0.0
        out["pm25_lag_1"] = None
        out["pm25_lag_24"] = None

    return out


def pm25_to_aqi_category(pm25: float) -> str:
    """Map PM2.5 concentration (µg/m³) to EPA AQI category label."""
    if pm25 <= 12.0:
        return "Good"
    if pm25 <= 35.4:
        return "Moderate"
    if pm25 <= 55.4:
        return "Unhealthy for Sensitive Groups"
    if pm25 <= 150.4:
        return "Unhealthy"
    if pm25 <= 250.4:
        return "Very Unhealthy"
    return "Hazardous"


def pm25_to_epa_aqi(pm25: float) -> float:
    """Convert PM2.5 (µg/m³) to EPA AQI index value."""
    breakpoints = [
        (0.0, 12.0, 0, 50),
        (12.1, 35.4, 51, 100),
        (35.5, 55.4, 101, 150),
        (55.5, 150.4, 151, 200),
        (150.5, 250.4, 201, 300),
        (250.5, 500.4, 301, 500),
    ]
    for c_low, c_high, aqi_low, aqi_high in breakpoints:
        if c_low <= pm25 <= c_high:
            return ((aqi_high - aqi_low) / (c_high - c_low)) * (pm25 - c_low) + aqi_low
    return 500.0
