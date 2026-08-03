import json
import warnings
import numpy as np
import pandas as pd
import requests
import shap
warnings.filterwarnings("ignore", message="X does not have valid feature names")
warnings.filterwarnings("ignore", message=".*sklearn.utils.parallel.delayed.*")

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

try:
    import tensorflow as tf
    TF_AVAILABLE = True
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
except ImportError:
    TF_AVAILABLE = False

# Akora Khattak, Nowshera, Pakistan
LATITUDE = 34.0011
LONGITUDE = 71.9887
FORECAST_DAYS = 3
LOCATION_NAME = "Akora Khattak"
LOCATION_REGION = "Nowshera"
LOCATION_COUNTRY = "Pakistan"

# ----------------------------------------------------
# AQI CALCULATION HELPERS
# ----------------------------------------------------
PM25_BREAKPOINTS = [
    (0.0, 12.0, 0, 50), (12.1, 35.4, 51, 100), (35.5, 55.4, 101, 150),
    (55.5, 150.4, 151, 200), (150.5, 250.4, 201, 300),
    (250.5, 350.4, 301, 400), (350.5, 500.4, 401, 500),
]
PM10_BREAKPOINTS = [
    (0, 54, 0, 50), (55, 154, 51, 100), (155, 254, 101, 150),
    (255, 354, 151, 200), (355, 424, 201, 300),
    (425, 504, 301, 400), (505, 604, 401, 500),
]

def calculate_sub_index(concentration, breakpoints):
    if pd.isna(concentration) or concentration < 0:
        return None
    for c_lo, c_hi, i_lo, i_hi in breakpoints:
        if c_lo <= concentration <= c_hi:
            return round(((i_hi - i_lo) / (c_hi - c_lo)) * (concentration - c_lo) + i_lo)
    if concentration > breakpoints[-1][1]:
        return 500
    return None

def calculate_aqi(pm25, pm10):
    vals = [v for v in (calculate_sub_index(pm25, PM25_BREAKPOINTS),
                         calculate_sub_index(pm10, PM10_BREAKPOINTS)) if v is not None]
    return max(vals) if vals else None

def aqi_category(aqi):
    if aqi is None: return "Unknown"
    if aqi <= 50: return "Good"
    if aqi <= 100: return "Moderate"
    if aqi <= 150: return "Unhealthy for Sensitive Groups"
    if aqi <= 200: return "Unhealthy"
    if aqi <= 300: return "Very Unhealthy"
    return "Hazardous"

FEATURE_LABELS = {
    "temperature": "Temperature", "humidity": "Humidity", "wind_speed": "Wind Speed",
    "pressure": "Pressure", "hour_sin": "Hour of Day", "hour_cos": "Hour of Day (cos)",
    "month_sin": "Month", "month_cos": "Month (cos)", "day_of_week": "Day of Week",
    "pm25_lag_1": "Current AQI", "pm25_lag_24": "Same Hour Yesterday",
    "pm25_roll_mean_3": "3-Hour Trend", "pm25_roll_mean_6": "6-Hour Trend",
}

# ----------------------------------------------------
# STEP 1: Load data & build features
# ----------------------------------------------------
print("--- STEP 1: Loading historical data ---")
df = pd.read_csv("data/lahore_aqi_historical_2024_onwards.csv", parse_dates=["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)
print(f"Loaded {len(df)} rows: {df['timestamp'].min()} to {df['timestamp'].max()}")

df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
df["pm25_roll_mean_3"] = df["ground_pm25"].rolling(3).mean().shift(1)
df["pm25_roll_mean_6"] = df["ground_pm25"].rolling(6).mean().shift(1)

feature_cols = [
    "temperature", "humidity", "wind_speed", "pressure",
    "hour_sin", "hour_cos", "month_sin", "month_cos", "day_of_week",
    "pm25_lag_1", "pm25_lag_24", "pm25_roll_mean_3", "pm25_roll_mean_6",
]
target_col = "ground_pm25"
df_model = df.dropna(subset=feature_cols + [target_col]).reset_index(drop=True)
print(f"Usable training rows after dropping NaNs: {len(df_model)}")

# ----------------------------------------------------
# STEP 2: Backtest for real per-horizon RMSE & Model Selection
# ----------------------------------------------------
print("\n--- STEP 2: Backtesting for per-horizon RMSE and Model Selection ---")

feature_idx = {name: i for i, name in enumerate(feature_cols)}

def build_feature_vector(row, lag_1, lag_24, roll3, roll6):
    v = np.empty(len(feature_cols))
    v[feature_idx["temperature"]] = row["temperature"]
    v[feature_idx["humidity"]] = row["humidity"]
    v[feature_idx["wind_speed"]] = row["wind_speed"]
    v[feature_idx["pressure"]] = row["pressure"]
    v[feature_idx["hour_sin"]] = row["hour_sin"]
    v[feature_idx["hour_cos"]] = row["hour_cos"]
    v[feature_idx["month_sin"]] = row["month_sin"]
    v[feature_idx["month_cos"]] = row["month_cos"]
    v[feature_idx["day_of_week"]] = row["day_of_week"]
    v[feature_idx["pm25_lag_1"]] = lag_1
    v[feature_idx["pm25_lag_24"]] = lag_24
    v[feature_idx["pm25_roll_mean_3"]] = roll3
    v[feature_idx["pm25_roll_mean_6"]] = roll6
    return v

def rollout_72h(df_full_records, model, start_idx, history_dict, max_horizon=72):
    """Single recursive rollout, returns dict {horizon: predicted_value} for 24/48/72."""
    sim_history = dict(history_dict)
    checkpoints = {}
    n = len(df_full_records)
    for step in range(1, max_horizon + 1):
        idx = start_idx + step
        if idx >= n:
            break
        row = df_full_records[idx]
        ts = row["timestamp"]
        lag_1 = sim_history.get(ts - pd.Timedelta(hours=1), np.nan)
        lag_24 = sim_history.get(ts - pd.Timedelta(hours=24), np.nan)
        if pd.isna(lag_1) or pd.isna(lag_24):
            break
        roll3 = np.nanmean([sim_history.get(ts - pd.Timedelta(hours=h), np.nan) for h in range(1, 4)])
        roll6 = np.nanmean([sim_history.get(ts - pd.Timedelta(hours=h), np.nan) for h in range(1, 7)])
        vec = build_feature_vector(row, lag_1, lag_24, roll3, roll6)
        
        # Fast prediction for TF if it's our TFModelWrapper
        if hasattr(model, 'fast_predict'):
            pred = model.fast_predict(vec.reshape(1, -1))[0]
        else:
            pred = model.predict(vec.reshape(1, -1))[0]
            
        sim_history[ts] = pred
        if step in (24, 48, 72):
            checkpoints[step] = pred
    return checkpoints

class TFModelWrapper:
    def __init__(self, input_dim):
        self.scaler = StandardScaler()
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(32, activation='relu', input_shape=(input_dim,)),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        self.model = model
        
    def fit(self, X, y):
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y, epochs=15, batch_size=32, verbose=0)
        return self
        
    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled, verbose=0).flatten()
        
    def fast_predict(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model(X_scaled, training=False).numpy().flatten()

# Split for backtesting
split_idx = int(len(df_model) * 0.85)
train_df = df_model.iloc[:split_idx]
test_df = df_model.iloc[split_idx:].reset_index(drop=True)

models_to_evaluate = {
    "Random Forest": RandomForestRegressor(n_estimators=120, max_depth=14, random_state=42, n_jobs=1),
    "Ridge Regression": Pipeline([('scaler', StandardScaler()), ('ridge', Ridge(alpha=1.0))]),
}

if TF_AVAILABLE:
    models_to_evaluate["Deep Learning (TF)"] = TFModelWrapper(input_dim=len(feature_cols))

full_history = df_model.set_index("timestamp")[target_col].to_dict()
df_model_records = df_model.to_dict("records")
ts_to_idx = {r["timestamp"]: i for i, r in enumerate(df_model_records)}

stride = max(len(test_df) // 40, 12)
sample_points = list(range(0, max(len(test_df) - 73, 0), stride))
print(f"Evaluating {len(sample_points)} backtest rollout points per model...")

best_model_name = None
best_model_rmse = float('inf')
best_aqi_rmse_by_horizon = {}
best_rmse_by_horizon = {}

def pm25_rmse_to_aqi_rmse(pm25_rmse, typical_pm25):
    if pm25_rmse is None:
        return None
    for c_lo, c_hi, i_lo, i_hi in PM25_BREAKPOINTS:
        if c_lo <= typical_pm25 <= c_hi:
            slope = (i_hi - i_lo) / (c_hi - c_lo)
            return round(pm25_rmse * slope, 2)
    return round(pm25_rmse, 2)

typical_pm25 = df_model[target_col].tail(500).mean()

for name, model in models_to_evaluate.items():
    print(f"\nTraining {name} on backtest split...")
    model.fit(train_df[feature_cols].values, train_df[target_col].values)
    
    horizon_errors = {24: [], 48: [], 72: []}
    for idx_count, i in enumerate(sample_points):
        if idx_count > 0 and idx_count % 15 == 0:
            print(f"  ...processed {idx_count}/{len(sample_points)} rollouts")
            
        base_ts = test_df.iloc[i]["timestamp"]
        base_idx = ts_to_idx.get(base_ts)
        if base_idx is None:
            continue
        seed_history = {ts: v for ts, v in full_history.items() if ts <= base_ts}
        checkpoints = rollout_72h(df_model_records, model, base_idx, seed_history)
        for h, pred in checkpoints.items():
            if base_idx + h < len(df_model_records):
                actual_ts = df_model_records[base_idx + h]["timestamp"]
                actual = full_history.get(actual_ts)
                if actual is not None:
                    horizon_errors[h].append((pred, actual))
                    
    rmse_by_horizon = {}
    for h, pairs in horizon_errors.items():
        if pairs:
            preds, actuals = zip(*pairs)
            rmse_by_horizon[h] = round(float(np.sqrt(mean_squared_error(actuals, preds))), 2)
        else:
            rmse_by_horizon[h] = None
    
    aqi_rmse_by_horizon = {h: pm25_rmse_to_aqi_rmse(v, typical_pm25) for h, v in rmse_by_horizon.items()}
    avg_rmse = np.mean([v for v in rmse_by_horizon.values() if v is not None])
    
    print(f"  {name} 72h avg RMSE (PM2.5): {avg_rmse:.2f}")
    
    if avg_rmse < best_model_rmse:
        best_model_rmse = avg_rmse
        best_model_name = name
        best_aqi_rmse_by_horizon = aqi_rmse_by_horizon
        best_rmse_by_horizon = rmse_by_horizon

print(f"\n---> Selected Best Model: {best_model_name} (RMSE: {best_model_rmse:.2f})")

# ----------------------------------------------------
# STEP 3: Train final model on full dataset
# ----------------------------------------------------
print(f"\n--- STEP 3: Training final {best_model_name} on full dataset ---")

if best_model_name == "Random Forest":
    final_model = RandomForestRegressor(n_estimators=150, max_depth=16, random_state=42, n_jobs=-1)
elif best_model_name == "Ridge Regression":
    final_model = Pipeline([('scaler', StandardScaler()), ('ridge', Ridge(alpha=1.0))])
else:
    final_model = TFModelWrapper(input_dim=len(feature_cols))

final_model.fit(df_model[feature_cols].values, df_model[target_col].values)

if isinstance(final_model, RandomForestRegressor):
    final_model.n_jobs = 1

print("Done.")

# ----------------------------------------------------
# STEP 4: Fetch 3-day weather + pollutant forecasts (LIVE Open-Meteo API)
# ----------------------------------------------------
print("\n--- STEP 4: Fetching live forecast data from Open-Meteo ---")

weather_res = requests.get("https://api.open-meteo.com/v1/forecast", params={
    "latitude": LATITUDE, "longitude": LONGITUDE,
    "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,surface_pressure",
    "forecast_days": FORECAST_DAYS, "timezone": "UTC"
})
weather_res.raise_for_status()
wd = weather_res.json()["hourly"]
df_weather_fc = pd.DataFrame({
    "timestamp": pd.to_datetime(wd["time"]), "temperature": wd["temperature_2m"],
    "humidity": wd["relative_humidity_2m"], "wind_speed": wd["wind_speed_10m"],
    "pressure": wd["surface_pressure"],
})

aq_res = requests.get("https://air-quality-api.open-meteo.com/v1/air-quality", params={
    "latitude": LATITUDE, "longitude": LONGITUDE,
    "hourly": "pm2_5,pm10,ozone,nitrogen_dioxide,sulphur_dioxide,carbon_monoxide",
    "forecast_days": FORECAST_DAYS, "timezone": "UTC"
})
aq_res.raise_for_status()
aqd = aq_res.json()["hourly"]
df_aq_fc = pd.DataFrame({
    "timestamp": pd.to_datetime(aqd["time"]), "modeled_pm25": aqd["pm2_5"],
    "modeled_pm10": aqd["pm10"], "ozone": aqd.get("ozone"),
    "no2": aqd.get("nitrogen_dioxide"), "so2": aqd.get("sulphur_dioxide"),
    "co": aqd.get("carbon_monoxide"),
})

df_forecast = pd.merge(df_weather_fc, df_aq_fc, on="timestamp", how="inner")
df_forecast = df_forecast[df_forecast["timestamp"] > df["timestamp"].max()].reset_index(drop=True)
print(f"Live forecast horizon: {len(df_forecast)} hours "
      f"({df_forecast['timestamp'].min()} to {df_forecast['timestamp'].max()})")

# ----------------------------------------------------
# STEP 5: Recursive forecast + SHAP at 24h/48h/72h checkpoints
# ----------------------------------------------------
print("\n--- STEP 5: Running recursive forecast with SHAP ---")

# Setup SHAP explainer dynamically based on the best model
if isinstance(final_model, RandomForestRegressor):
    explainer = shap.TreeExplainer(final_model)
    def get_shap_vals(X):
        # TreeExplainer expects DataFrame/2D array, returns list for multi-output or array for single
        vals = explainer.shap_values(X)
        if isinstance(vals, list): return vals[0][0]
        return vals[0]
else:
    # KernelExplainer for any other model (Ridge or TF)
    print("Initializing KernelExplainer (this may take a moment)...")
    background_data = shap.kmeans(df_model[feature_cols].values, 10)
    
    # We need a wrapper function for KernelExplainer that predicts a 2D array
    if hasattr(final_model, 'fast_predict'):
        def predict_fn(X):
            return final_model.model(final_model.scaler.transform(X), training=False).numpy().flatten()
    else:
        def predict_fn(X):
            return final_model.predict(X)
            
    explainer = shap.KernelExplainer(predict_fn, background_data)
    
    def get_shap_vals(X):
        # KernelExplainer requires 2D input
        return explainer.shap_values(X, silent=True)[0]

history = df[["timestamp", target_col]].dropna().set_index("timestamp")[target_col].to_dict()

results = []
checkpoint_shap = {}
CHECKPOINTS = {24: "24h", 48: "48h", 72: "72h"}

for step, (_, row) in enumerate(df_forecast.iterrows(), start=1):
    ts = row["timestamp"]
    lag_1 = history.get(ts - pd.Timedelta(hours=1), np.nan)
    lag_24 = history.get(ts - pd.Timedelta(hours=24), np.nan)
    roll3 = np.nanmean([history.get(ts - pd.Timedelta(hours=h), np.nan) for h in range(1, 4)])
    roll6 = np.nanmean([history.get(ts - pd.Timedelta(hours=h), np.nan) for h in range(1, 7)])

    features = pd.DataFrame([{
        "temperature": row["temperature"], "humidity": row["humidity"],
        "wind_speed": row["wind_speed"], "pressure": row["pressure"],
        "hour_sin": np.sin(2 * np.pi * ts.hour / 24), "hour_cos": np.cos(2 * np.pi * ts.hour / 24),
        "month_sin": np.sin(2 * np.pi * ts.month / 12), "month_cos": np.cos(2 * np.pi * ts.month / 12),
        "day_of_week": ts.dayofweek, "pm25_lag_1": lag_1, "pm25_lag_24": lag_24,
        "pm25_roll_mean_3": roll3, "pm25_roll_mean_6": roll6,
    }])[feature_cols]

    if hasattr(final_model, 'fast_predict'):
        pred_pm25 = final_model.fast_predict(features.values)[0]
    else:
        pred_pm25 = final_model.predict(features.values)[0]
        
    history[ts] = pred_pm25
    pred_aqi = calculate_aqi(pred_pm25, row["modeled_pm10"])

    results.append({
        "timestamp": ts.isoformat(), "predicted_pm25": round(float(pred_pm25), 1),
        "predicted_aqi": pred_aqi, "category": aqi_category(pred_aqi),
    })

    if step in CHECKPOINTS:
        shap_vals = get_shap_vals(features.values)
        feats_sorted = sorted(zip(feature_cols, shap_vals), key=lambda x: abs(x[1]), reverse=True)
        checkpoint_shap[CHECKPOINTS[step]] = {
            "predicted_aqi": pred_aqi, "predicted_pm25": round(float(pred_pm25), 1),
            "rmse_aqi": best_aqi_rmse_by_horizon.get(step),
            "features": [{"name": FEATURE_LABELS.get(f, f), "value": round(float(v), 2)} for f, v in feats_sorted],
        }

df_result = pd.DataFrame(results)
df_result["timestamp"] = pd.to_datetime(df_result["timestamp"])
df_result["date"] = df_result["timestamp"].dt.date
daily_summary = df_result.groupby("date").agg(
    avg_aqi=("predicted_aqi", "mean"), max_aqi=("predicted_aqi", "max"), min_aqi=("predicted_aqi", "min")
).round(1)
print("\n--- 3-Day Daily AQI Summary ---")
print(daily_summary)

# ----------------------------------------------------
# STEP 6: Export dashboard JSON
# ----------------------------------------------------
print("\n--- STEP 6: Exporting dashboard JSON ---")

latest_row = df.dropna(subset=["ground_pm25", "modeled_pm10"]).iloc[-1]
last_24h = df[df["timestamp"] > latest_row["timestamp"] - pd.Timedelta(hours=24)]

current_aqi_val = calculate_aqi(latest_row["ground_pm25"], latest_row["modeled_pm10"])

dashboard_data = {
    "location": {"name": LOCATION_NAME, "region": LOCATION_REGION, "country": LOCATION_COUNTRY},
    "updated_at": pd.Timestamp.now().isoformat(),
    "data_note": f"Powered by Best Model: {best_model_name}",
    "current": {
        "aqi": current_aqi_val,
        "category": aqi_category(current_aqi_val),
        "timestamp": latest_row["timestamp"].isoformat(),
        "pollutants": {
            "pm25": round(float(latest_row["ground_pm25"]), 1),
            "pm10": round(float(latest_row["modeled_pm10"]), 1),
        },
        "weather": {
            "temperature": round(float(latest_row["temperature"]), 1),
            "humidity": round(float(latest_row["humidity"]), 1),
            "pressure": round(float(latest_row["pressure"]), 1),
        },
    },
    "trend_24h": [
        {"timestamp": r["timestamp"].isoformat(), "aqi": calculate_aqi(r["ground_pm25"], r["modeled_pm10"])}
        for _, r in last_24h.dropna(subset=["ground_pm25", "modeled_pm10"]).iterrows()
    ],
    "forecast": [
        {"horizon": f"{h}h", "day": f"Day {i+1}", **checkpoint_shap[label]}
        for i, (h, label) in enumerate(CHECKPOINTS.items()) if label in checkpoint_shap
    ],
    "forecast_hourly": results,
    "rmse_by_horizon_pm25": best_rmse_by_horizon,
    "rmse_by_horizon_aqi": best_aqi_rmse_by_horizon,
}

with open("data/aqi_dashboard_data.json", "w") as f:
    json.dump(dashboard_data, f, indent=2, default=str)

df_result.to_csv("data/lahore_aqi_3day_forecast.csv", index=False)
print("Saved: data/aqi_dashboard_data.json")
print("Saved: data/lahore_aqi_3day_forecast.csv")
