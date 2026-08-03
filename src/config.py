"""Shared configuration for the AQI predictor pipeline."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

# Lahore coordinates
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
FEATURE_GROUP_VERSION = 1
MODEL_REGISTRY_NAME = "lahore_aqi_predictor"

# Feature columns used for training (excluding target and timestamp)
FEATURE_COLUMNS = [
    "temperature",
    "humidity",
    "wind_speed",
    "pressure",
    "modeled_pm25",
    "modeled_pm10",
    "hour",
    "day_of_week",
    "month",
    "aqi_change_rate",
    "pm25_lag_1",
    "pm25_lag_24",
]

TARGET_COLUMN = "ground_pm25"

# AQI thresholds (US EPA PM2.5-based categories, µg/m³)
AQI_THRESHOLDS = {
    "good": 12.0,
    "moderate": 35.4,
    "unhealthy_sensitive": 55.4,
    "unhealthy": 150.4,
    "very_unhealthy": 250.4,
    "hazardous": 500.4,
}

FORECAST_HOURS = 72  # 3 days
TIMEZONE = "UTC"
