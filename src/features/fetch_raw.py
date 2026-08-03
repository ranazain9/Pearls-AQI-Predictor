"""Fetch raw weather and pollutant data — standardized on Open-Meteo (UTC)."""
import os

import pandas as pd
import requests
from dotenv import load_dotenv

from src.config import AQICN_STATION_ID, FORECAST_HOURS, LATITUDE, LONGITUDE, TIMEZONE

load_dotenv()


def fetch_aqicn_current() -> dict:
    """Fetch current AQI and pollutant data from AQICN (ground-truth only)."""
    token = os.getenv("AQICN_API_TOKEN")
    if not token:
        raise ValueError("AQICN_API_TOKEN not found in .env")

    url = f"https://api.waqi.info/feed/{AQICN_STATION_ID}/?token={token}"
    response = requests.get(url, timeout=15)
    response.raise_for_status()
    result = response.json()
    if result.get("status") != "ok":
        raise RuntimeError(f"AQICN API error: {result.get('data')}")
    return result["data"]


def fetch_openmeteo_forecast(hours: int = FORECAST_HOURS) -> pd.DataFrame:
    """
    Fetch hourly weather + air-quality forecast from Open-Meteo (UTC).
    Same provider as historical_backfill.py — avoids training-serving skew.
    """
    forecast_days = min(16, max(1, (hours + 23) // 24))
    common = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "timezone": TIMEZONE,
        "forecast_days": forecast_days,
    }

    weather_res = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            **common,
            "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,surface_pressure",
        },
        timeout=30,
    )
    weather_res.raise_for_status()
    weather_hourly = weather_res.json().get("hourly", {})

    aq_res = requests.get(
        "https://air-quality-api.open-meteo.com/v1/air-quality",
        params={**common, "hourly": "pm2_5,pm10"},
        timeout=30,
    )
    aq_res.raise_for_status()
    aq_hourly = aq_res.json().get("hourly", {})

    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(weather_hourly["time"], utc=True),
            "temperature": weather_hourly["temperature_2m"],
            "humidity": weather_hourly["relative_humidity_2m"],
            "wind_speed": weather_hourly["wind_speed_10m"],
            "pressure": weather_hourly["surface_pressure"],
            "modeled_pm25": aq_hourly["pm2_5"],
            "modeled_pm10": aq_hourly["pm10"],
        }
    )

    now_utc = pd.Timestamp.now(tz="UTC")
    df = df[df["timestamp"] > now_utc].head(hours).reset_index(drop=True)
    df["timestamp"] = df["timestamp"].dt.tz_localize(None)
    return df


def fetch_openmeteo_current_row() -> pd.DataFrame:
    """Fetch the current UTC hour row from Open-Meteo (weather + modeled AQ)."""
    res = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": LATITUDE,
            "longitude": LONGITUDE,
            "timezone": TIMEZONE,
            "forecast_days": 1,
            "past_days": 1,
            "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,surface_pressure",
        },
        timeout=30,
    )
    res.raise_for_status()
    hourly = res.json().get("hourly", {})

    aq_res = requests.get(
        "https://air-quality-api.open-meteo.com/v1/air-quality",
        params={
            "latitude": LATITUDE,
            "longitude": LONGITUDE,
            "timezone": TIMEZONE,
            "forecast_days": 1,
            "past_days": 1,
            "hourly": "pm2_5,pm10,ozone,nitrogen_dioxide,sulphur_dioxide,carbon_monoxide",
        },
        timeout=30,
    )
    aq_res.raise_for_status()
    aq_hourly = aq_res.json().get("hourly", {})

    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(hourly["time"], utc=True),
            "temperature": hourly["temperature_2m"],
            "humidity": hourly["relative_humidity_2m"],
            "wind_speed": hourly["wind_speed_10m"],
            "pressure": hourly["surface_pressure"],
            "modeled_pm25": aq_hourly.get("pm2_5", [None] * len(hourly["time"])),
            "modeled_pm10": aq_hourly.get("pm10", [None] * len(hourly["time"])),
            "ozone": aq_hourly.get("ozone", [None] * len(hourly["time"])),
            "no2": aq_hourly.get("nitrogen_dioxide", [None] * len(hourly["time"])),
            "so2": aq_hourly.get("sulphur_dioxide", [None] * len(hourly["time"])),
            "co": aq_hourly.get("carbon_monoxide", [None] * len(hourly["time"])),
        }
    )

    now_utc = pd.Timestamp.now(tz="UTC").floor("h")
    row = df[df["timestamp"] == now_utc]
    if row.empty:
        row = df.iloc[[-1]]
    out = row.copy()
    out["timestamp"] = out["timestamp"].dt.tz_localize(None)
    return out.reset_index(drop=True)


def current_reading_to_row(aqicn_data: dict | None, meteo_row: pd.DataFrame) -> pd.DataFrame:
    """Combine Open-Meteo features with optional AQICN ground-truth PM2.5."""
    row = meteo_row.iloc[0].to_dict()
    pm25 = None
    if aqicn_data:
        iaqi = aqicn_data.get("iaqi", {})
        pm25 = iaqi.get("pm25", {}).get("v")
    if pm25 is None:
        pm25 = row.get("modeled_pm25")
    row["ground_pm25"] = pm25
    return pd.DataFrame([row])
