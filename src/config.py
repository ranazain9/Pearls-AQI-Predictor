"""Shared configuration for the AQI predictor pipeline."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

# Lahore coordinates (backfill location)
LATITUDE = 31.5204
LONGITUDE = 74.3587
CITY_NAME = "Lahore"
AQICN_STATION_ID = "A471607"

# Data paths
HISTORICAL_CSV = DATA_DIR / "lahore_aqi_historical_2024_onwards.csv"
LEGACY_HISTORICAL_CSV = DATA_DIR / "lahore_aqi_historical.csv"
FEATURES_CSV = DATA_DIR / "lahore_aqi_features.csv"

# Hopsworks
FEATURE_GROUP_NAME = "lahore_aqi_features"
FEATURE_GROUP_VERSION = 2
MODEL_REGISTRY_NAME = "lahore_aqi_predictor"

# Feature columns used for training (excluding target and timestamp).
# NOTE: "aqi" from Open-Meteo us_aqi field is the TARGET — it must NOT appear here.
# "aqi_lag_1" = previous hour's AQI (shifted), used as "Current AQI" feature.
FEATURE_COLUMNS = [
    "no",           # Nitric Oxide (NO)
    "temperature",  # Temperature
    "humidity",     # Humidity
    "aqi_change_rate",  # AQI Change Rate
    "so2",          # Sulfur Dioxide (SO₂)
    "co",           # Carbon Monoxide (CO)
    "modeled_pm10", # PM10
    "day_of_week",  # Day of Week
    "month",        # Month
    "pressure",     # Pressure
    "no2",          # Nitrogen Dioxide (NO₂)
    "ozone",        # Ozone (O₃)
    "hour",         # Hour of Day
    "week_of_year", # Week of Year
    "modeled_pm25", # PM2.5
    "day",          # Day
    "aqi_lag_1",    # Current AQI (previous hour, shifted to avoid leakage)
]

TARGET_COLUMN = "aqi"  # Open-Meteo us_aqi — fetched directly, never computed by us

# Human-readable display names for each feature (used in dashboard/SHAP charts)
FEATURE_LABELS = {
    "no":            "Nitric Oxide (NO)",
    "temperature":   "Temperature",
    "humidity":      "Humidity",
    "aqi_change_rate": "AQI Change Rate",
    "so2":           "Sulfur Dioxide (SO₂)",
    "co":            "Carbon Monoxide (CO)",
    "modeled_pm10":  "PM10",
    "day_of_week":   "Day of Week",
    "month":         "Month",
    "pressure":      "Pressure",
    "no2":           "Nitrogen Dioxide (NO₂)",
    "ozone":         "Ozone (O₃)",
    "hour":          "Hour of Day",
    "week_of_year":  "Week of Year",
    "modeled_pm25":  "PM2.5",
    "day":           "Day",
    "aqi_lag_1":     "Current AQI",
}

# AQI thresholds used for alerting and categorization.
AQI_THRESHOLDS = {
    "good": 50,
    "moderate": 100,
    "unhealthy_sensitive": 150,
    "unhealthy": 200,
    "very_unhealthy": 300,
    "hazardous": 500,
}

FORECAST_HOURS = 72  # 3 days
TIMEZONE = "UTC"
