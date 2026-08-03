import os
import sys
import datetime
import requests
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

# Reconfigure stdout to UTF-8 to prevent encoding errors on Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Load environment variables
load_dotenv()
openaq_key = os.getenv("OPENAQ_API_KEY")

if not openaq_key:
    raise ValueError("OPENAQ_API_KEY not found in .env. Please add it to start fetching.")

LATITUDE = 31.5204
LONGITUDE = 74.3587
RADIUS_METERS = 25000  # 25km search radius
START_DATE = "2024-01-01"
# Set end date as yesterday to get full daily records
END_DATE = (datetime.date.today() - datetime.timedelta(days=1)).strftime("%Y-%m-%d")

# Create data directory if not exists
Path("data").mkdir(exist_ok=True)

headers = {
    "X-API-Key": openaq_key
}

# ----------------------------------------------------
# AQI CALCULATION HELPERS (US EPA breakpoint method)
# ----------------------------------------------------
PM25_BREAKPOINTS = [
    (0.0, 12.0, 0, 50),
    (12.1, 35.4, 51, 100),
    (35.5, 55.4, 101, 150),
    (55.5, 150.4, 151, 200),
    (150.5, 250.4, 201, 300),
    (250.5, 350.4, 301, 400),
    (350.5, 500.4, 401, 500),
]

PM10_BREAKPOINTS = [
    (0, 54, 0, 50),
    (55, 154, 51, 100),
    (155, 254, 101, 150),
    (255, 354, 151, 200),
    (355, 424, 201, 300),
    (425, 504, 301, 400),
    (505, 604, 401, 500),
]


def calculate_sub_index(concentration, breakpoints):
    """Convert a pollutant concentration into an AQI sub-index using
    the standard piecewise-linear EPA breakpoint formula."""
    if pd.isna(concentration) or concentration < 0:
        return None

    for c_lo, c_hi, i_lo, i_hi in breakpoints:
        if c_lo <= concentration <= c_hi:
            return round(((i_hi - i_lo) / (c_hi - c_lo)) * (concentration - c_lo) + i_lo)

    # Concentration exceeds the highest defined breakpoint -> cap at max AQI band
    if concentration > breakpoints[-1][1]:
        return 500

    return None


def calculate_aqi(pm25, pm10):
    """Overall AQI = max of the individual pollutant sub-indices."""
    aqi_pm25 = calculate_sub_index(pm25, PM25_BREAKPOINTS)
    aqi_pm10 = calculate_sub_index(pm10, PM10_BREAKPOINTS)

    sub_indices = [v for v in (aqi_pm25, aqi_pm10) if v is not None]
    if not sub_indices:
        return None
    return max(sub_indices)


# ----------------------------------------------------
# STEP 1: Find all PM2.5 Sensors near Lahore on OpenAQ
# ----------------------------------------------------
print("--- STEP 1: Finding Lahore PM2.5 Sensors ---")
locations_url = "https://api.openaq.org/v3/locations"
locations_params = {
    "coordinates": f"{LATITUDE},{LONGITUDE}",
    "radius": RADIUS_METERS,
    "limit": 10
}

sensor_ids = []
try:
    response = requests.get(locations_url, headers=headers, params=locations_params)
    response.raise_for_status()
    locations = response.json().get("results", [])
    print(f"Found {len(locations)} locations within 25km of Lahore.")

    for loc in locations:
        loc_name = loc.get("name", "Unknown Location")
        for s in loc.get("sensors", []):
            if s.get("parameter", {}).get("name") == "pm25":
                sensor_id = s.get("id")
                sensor_ids.append((sensor_id, loc_name))
                print(f"  - Registered Sensor ID: {sensor_id} | Location: {loc_name}")
except Exception as e:
    print(f"Failed to find sensors: {e}")
    sys.exit(1)

if not sensor_ids:
    print("No active PM2.5 sensors found near Lahore. Exiting.")
    sys.exit(1)

# ----------------------------------------------------
# STEP 2: Fetch Historical Measurements from OpenAQ Sensors
# ----------------------------------------------------
print("\n--- STEP 2: Fetching Ground-Truth PM2.5 Measurements ---")
openaq_measurements = []

for sensor_id, loc_name in sensor_ids:
    print(f"Fetching data for Sensor {sensor_id} ({loc_name})...")
    url = f"https://api.openaq.org/v3/sensors/{sensor_id}/measurements"

    # We query from START_DATE to END_DATE using ISO-8601 formatting
    params = {
        "date_from": f"{START_DATE}T00:00:00Z",
        "date_to": f"{END_DATE}T23:59:59Z",
        "limit": 1000,
        "page": 1
    }

    sensor_records = 0
    while True:
        try:
            res = requests.get(url, headers=headers, params=params)
            res.raise_for_status()
            res_json = res.json()
            results = res_json.get("results", [])

            if not results:
                break

            for r in results:
                # We pull timestamp and measurement value
                datetime_from = r.get("period", {}).get("datetimeFrom", {})
                timestamp = datetime_from.get("utc")
                val = r.get("value")
                if timestamp and val is not None:
                    openaq_measurements.append({
                        "timestamp": timestamp,
                        "sensor_id": sensor_id,
                        "location_name": loc_name,
                        "ground_pm25": val
                    })
                    sensor_records += 1

            # Check pagination
            if not results:
                break

            # If this page has fewer than 'limit' records,
            # it means we've reached the last page.
            if len(results) < params["limit"]:
                break

            params["page"] += 1

        except Exception as e:
            print(f"  Error fetching page {params['page']} for sensor {sensor_id}: {e}")
            break

    print(f"  -> Retrieved {sensor_records} records.")

if not openaq_measurements:
    print("No historical ground-truth measurements found. Exiting.")
    sys.exit(1)

# Convert measurements to DataFrame
df_openaq = pd.DataFrame(openaq_measurements)
df_openaq["timestamp"] = pd.to_datetime(df_openaq["timestamp"]).dt.tz_convert(None)  # Remove timezone for uniform join
df_openaq = df_openaq.groupby(pd.Grouper(key="timestamp", freq="h"))["ground_pm25"].mean().reset_index()
print(f"Aggregated ground-truth hourly records count: {len(df_openaq)}")

# ----------------------------------------------------
# STEP 3: Fetch Historical Weather Data from Open-Meteo
# ----------------------------------------------------
print("\n--- STEP 3: Fetching Weather Data from Open-Meteo ---")
weather_url = "https://archive-api.open-meteo.com/v1/archive"
weather_params = {
    "latitude": LATITUDE,
    "longitude": LONGITUDE,
    "start_date": START_DATE,
    "end_date": END_DATE,
    "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,surface_pressure",
    "timezone": "UTC"
}

try:
    weather_res = requests.get(weather_url, params=weather_params)
    weather_res.raise_for_status()
    weather_data = weather_res.json().get("hourly", {})

    df_weather = pd.DataFrame({
        "timestamp": pd.to_datetime(weather_data["time"]),
        "temperature": weather_data["temperature_2m"],
        "humidity": weather_data["relative_humidity_2m"],
        "wind_speed": weather_data["wind_speed_10m"],
        "pressure": weather_data["surface_pressure"]
    })
    print(f"Retrieved {len(df_weather)} hourly weather records.")
except Exception as e:
    print(f"Failed to fetch weather data: {e}")
    sys.exit(1)

# ----------------------------------------------------
# STEP 4: Fetch Historical Air Quality Estimates from Open-Meteo
# ----------------------------------------------------
print("\n--- STEP 4: Fetching Modeled Air Quality Estimates from Open-Meteo ---")
aq_url = "https://air-quality-api.open-meteo.com/v1/air-quality"
aq_params = {
    "latitude": LATITUDE,
    "longitude": LONGITUDE,
    "start_date": START_DATE,
    "end_date": END_DATE,
    "hourly": "pm2_5,pm10",
    "timezone": "UTC"
}

try:
    aq_res = requests.get(aq_url, params=aq_params)
    aq_res.raise_for_status()
    aq_data = aq_res.json().get("hourly", {})

    df_aq_modeled = pd.DataFrame({
        "timestamp": pd.to_datetime(aq_data["time"]),
        "modeled_pm25": aq_data["pm2_5"],
        "modeled_pm10": aq_data["pm10"]
    })
    print(f"Retrieved {len(df_aq_modeled)} hourly air quality model estimates.")
except Exception as e:
    print(f"Failed to fetch air quality model data: {e}")
    sys.exit(1)

# ----------------------------------------------------
# STEP 5: Merge and Save Dataset
# ----------------------------------------------------
print("\n--- STEP 5: Merging Datasets ---")
# Merge weather and modeled air quality (both are complete hourly series from Open-Meteo)
df_merged = pd.merge(df_weather, df_aq_modeled, on="timestamp", how="outer")

# Merge in OpenAQ ground-truth target values
df_final = pd.merge(df_merged, df_openaq, on="timestamp", how="left")

# Sort chronologically
df_final = df_final.sort_values(by="timestamp").reset_index(drop=True)

# Generate time-based and derived features
df_final["hour"] = df_final["timestamp"].dt.hour
df_final["day_of_week"] = df_final["timestamp"].dt.dayofweek
df_final["month"] = df_final["timestamp"].dt.month
df_final["aqi_change_rate"] = df_final["ground_pm25"].diff().fillna(0)
df_final["pm25_lag_1"] = df_final["ground_pm25"].shift(1)
df_final["pm25_lag_24"] = df_final["ground_pm25"].shift(24)

# ----------------------------------------------------
# STEP 6: Calculate AQI target column
# ----------------------------------------------------
print("\n--- STEP 6: Calculating AQI (US EPA breakpoint method) ---")

# Ground-truth AQI: computed from real sensor PM2.5 + modeled PM10
# (modeled_pm10 is used as a stand-in since ground PM10 isn't collected)
df_final["calculated_aqi"] = df_final.apply(
    lambda row: calculate_aqi(row["ground_pm25"], row["modeled_pm10"]),
    axis=1
)

# Fully-modeled AQI: computed only from modeled pollutants (useful where
# ground_pm25 is missing, or to sanity-check the model source vs ground truth)
df_final["calculated_aqi_modeled"] = df_final.apply(
    lambda row: calculate_aqi(row["modeled_pm25"], row["modeled_pm10"]),
    axis=1
)

missing_calculated_aqi = df_final["calculated_aqi"].isna().sum()
print(f"Rows with calculated_aqi: {len(df_final) - missing_calculated_aqi} / {len(df_final)}")

# Check how many missing values exist in the target column
missing_ground_truth = df_final["ground_pm25"].isna().sum()
total_records = len(df_final)
missing_percentage = (missing_ground_truth / total_records) * 100

print(f"\nMerging complete.")
print(f"Total historical data points: {total_records} hours")
print(f"Missing ground-truth readings: {missing_ground_truth} ({missing_percentage:.2f}%)")

# Save final CSV
output_path = "data/lahore_aqi_historical_2024_onwards1.csv"
df_final.to_csv(output_path, index=False)
print(f"Success! Historical backfill dataset saved to: {output_path}")