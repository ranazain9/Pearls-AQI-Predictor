import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import json
import warnings
import numpy as np
import pandas as pd
import requests
import pickle
from sklearn.ensemble import RandomForestRegressor

try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False
    print("WARNING: SHAP not available (numba/numpy conflict). Proceeding without feature explanations.")
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

warnings.filterwarnings("ignore", message="X does not have valid feature names")
warnings.filterwarnings("ignore", message=".*sklearn.utils.parallel.delayed.*")

from src.config import LATITUDE, LONGITUDE, CITY_NAME, FORECAST_HOURS, FEATURE_COLUMNS, FEATURE_LABELS

LOCATION_NAME = CITY_NAME
LOCATION_REGION = "Punjab"
LOCATION_COUNTRY = "Pakistan"

def aqi_category(aqi):
    if aqi is None or pd.isna(aqi): return "Unknown"
    if aqi <= 50: return "Good"
    if aqi <= 100: return "Moderate"
    if aqi <= 150: return "Unhealthy for Sensitive Groups"
    if aqi <= 200: return "Unhealthy"
    if aqi <= 300: return "Very Unhealthy"
    return "Hazardous"

# ----------------------------------------------------
# STEP 1: Load data & build features
# ----------------------------------------------------
print("--- STEP 1: Loading historical data ---")
df = pd.read_csv("data/lahore_aqi_historical_2024_onwards.csv", parse_dates=["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)
print(f"Loaded {len(df)} rows: {df['timestamp'].min()} to {df['timestamp'].max()}")

from src.features.engineering import clean_features_df, compute_features
df_featured = clean_features_df(df)

# We will train separate models for Day 1 (+24h), Day 2 (+48h), Day 3 (+72h)
horizons = {24: "day1", 48: "day2", 72: "day3"}
horizon_data = {}

for h, label in horizons.items():
    df_h = df_featured.copy()
    df_h["target_aqi"] = df_h["aqi"].shift(-h)
    df_h = df_h.dropna(subset=["target_aqi"] + FEATURE_COLUMNS).reset_index(drop=True)
    horizon_data[label] = df_h

# ----------------------------------------------------
# STEP 2: Evaluate and Select Models per Horizon
# ----------------------------------------------------
print("\n--- STEP 2: Evaluating models for each horizon ---")
best_models = {}
best_scalers = {}
best_rmse_by_horizon = {}
best_model_names = {}

for h, label in horizons.items():
    print(f"\n--- Horizon: +{h}h ({label}) ---")
    data = horizon_data[label]
    split_idx = int(len(data) * 0.85)
    
    train_df = data.iloc[:split_idx]
    test_df = data.iloc[split_idx:].reset_index(drop=True)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[FEATURE_COLUMNS])
    y_train = train_df["target_aqi"].values
    
    X_test = scaler.transform(test_df[FEATURE_COLUMNS])
    y_test = test_df["target_aqi"].values
    
    models_to_evaluate = {
        "Random Forest": RandomForestRegressor(n_estimators=30, max_depth=8, random_state=42, n_jobs=1),
        "Ridge Regression": Ridge(alpha=10.0)
    }
    
    best_name = None
    best_rmse = float('inf')
    best_model = None
    
    for name, model in models_to_evaluate.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        print(f"  {name} RMSE: {rmse:.2f}")
        
        if rmse < best_rmse:
            best_rmse = rmse
            best_name = name
            best_model = model
            
    print(f"  Selected: {best_name} (RMSE: {best_rmse:.2f})")
    
    # Train final model on full data for this horizon
    X_full = scaler.fit_transform(data[FEATURE_COLUMNS])
    y_full = data["target_aqi"].values
    
    final_model = RandomForestRegressor(n_estimators=30, max_depth=8, random_state=42, n_jobs=1) if best_name == "Random Forest" else Ridge(alpha=10.0)
    final_model.fit(X_full, y_full)
    
    best_models[label] = final_model
    best_scalers[label] = scaler
    best_rmse_by_horizon[h] = round(float(best_rmse), 2)
    best_model_names[label] = best_name

# Save final legacy best_model and scaler for API compatibility
with open("models/best_model_day1.pkl", "wb") as f:
    pickle.dump(best_models["day1"], f)
with open("models/scaler_day1.pkl", "wb") as f:
    pickle.dump({"scaler": best_scalers["day1"], "feature_columns": FEATURE_COLUMNS}, f)
with open("models/best_model.pkl", "wb") as f:
    pickle.dump(best_models["day1"], f)
with open("models/scaler.pkl", "wb") as f:
    pickle.dump({"scaler": best_scalers["day1"], "feature_columns": FEATURE_COLUMNS}, f)

# ----------------------------------------------------
# STEP 3: Run forecast + SHAP
# ----------------------------------------------------
print("\n--- STEP 3: Generating predictions & SHAP ---")

# We make 72 direct predictions
latest_time = df_featured["timestamp"].max()
results = []
checkpoint_shap = {}
CHECKPOINTS = {24: "24h", 48: "48h", 72: "72h"}

# Compute SHAP explainers (only if SHAP is available)
explainers = {}
if SHAP_AVAILABLE:
    for label, model in best_models.items():
        try:
            if best_model_names[label] == "Random Forest":
                explainers[label] = shap.TreeExplainer(model)
            else:
                background_data = shap.kmeans(horizon_data[label][FEATURE_COLUMNS].values, 10)
                explainers[label] = shap.KernelExplainer(model.predict, background_data)
        except Exception as e:
            print(f"SHAP explainer failed for {label}: {e}")

df_featured.index = df_featured["timestamp"]

for h in range(1, 73):
    target_time = latest_time + pd.Timedelta(hours=h)
    
    if h <= 24:
        label = "day1"
        day_num = 1
        lookup_time = target_time - pd.Timedelta(hours=24)
    elif h <= 48:
        label = "day2"
        day_num = 2
        lookup_time = target_time - pd.Timedelta(hours=48)
    else:
        label = "day3"
        day_num = 3
        lookup_time = target_time - pd.Timedelta(hours=72)
        
    if lookup_time in df_featured.index:
        feat_row = df_featured.loc[lookup_time]
        if isinstance(feat_row, pd.DataFrame):
            feat_row = feat_row.iloc[-1]
            
        X_dict = {col: feat_row.get(col, 0.0) for col in FEATURE_COLUMNS}
        for col in X_dict:
            if pd.isna(X_dict[col]):
                X_dict[col] = 0.0
                
        X_df = pd.DataFrame([X_dict])[FEATURE_COLUMNS]
        X_scaled = best_scalers[label].transform(X_df)
        pred_aqi = float(best_models[label].predict(X_scaled)[0])
        pred_aqi = max(0.0, pred_aqi)
    else:
        pred_aqi = float(df_featured["aqi"].iloc[-1])
        X_df = pd.DataFrame([{col: 0.0 for col in FEATURE_COLUMNS}])
        
    results.append({
        "timestamp": target_time.isoformat(),
        "predicted_pm25": round(pred_aqi * 0.5, 2),
        "predicted_aqi": round(pred_aqi, 1),
        "category": aqi_category(pred_aqi),
    })
    
    if h in CHECKPOINTS:
        feature_importances = []
        if SHAP_AVAILABLE and label in explainers:
            try:
                if best_model_names[label] == "Random Forest":
                    shap_vals = explainers[label].shap_values(X_scaled)
                    if isinstance(shap_vals, list): shap_vals = shap_vals[0]
                    shap_vals = shap_vals[0]
                else:
                    shap_vals = explainers[label].shap_values(X_scaled.reshape(1, -1), silent=True)[0]
                feats_sorted = sorted(zip(FEATURE_COLUMNS, shap_vals), key=lambda x: abs(x[1]), reverse=True)
                feature_importances = [{"name": FEATURE_LABELS.get(f, f), "value": round(float(v), 2)} for f, v in feats_sorted]
            except Exception as e:
                print(f"SHAP values failed for horizon {h}: {e}")
        checkpoint_shap[CHECKPOINTS[h]] = {
            "predicted_aqi": round(pred_aqi, 1),
            "predicted_pm25": round(pred_aqi * 0.5, 2),
            "rmse_aqi": best_rmse_by_horizon.get(h),
            "features": feature_importances,
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
# STEP 4: Export dashboard JSON
# ----------------------------------------------------
print("\n--- STEP 4: Exporting dashboard JSON ---")

latest_row = df_featured.iloc[-1]
last_24h = df_featured[df_featured["timestamp"] > latest_row["timestamp"] - pd.Timedelta(hours=24)]

dashboard_data = {
    "location": {"name": LOCATION_NAME, "region": LOCATION_REGION, "country": LOCATION_COUNTRY},
    "updated_at": pd.Timestamp.now().isoformat(),
    "data_note": f"Powered by Horizon-Specific Models: {best_model_names}",
    "current": {
        "aqi": float(latest_row["aqi"]),
        "category": aqi_category(latest_row["aqi"]),
        "timestamp": latest_row["timestamp"].isoformat(),
        "pollutants": {
            "pm25": round(float(latest_row["ground_pm25"]) if pd.notna(latest_row["ground_pm25"]) else float(latest_row["modeled_pm25"]), 1),
            "pm10": round(float(latest_row["modeled_pm10"]), 1),
        },
        "weather": {
            "temperature": round(float(latest_row["temperature"]), 1),
            "humidity": round(float(latest_row["humidity"]), 1),
            "pressure": round(float(latest_row["pressure"]), 1),
        },
    },
    "trend_24h": [
        {"timestamp": r["timestamp"].isoformat(), "aqi": float(r["aqi"])}
        for _, r in last_24h.iterrows()
    ],
    "forecast": [
        {"horizon": f"{h}h", "day": f"Day {i+1}", **checkpoint_shap[label]}
        for i, (h, label) in enumerate(CHECKPOINTS.items()) if label in checkpoint_shap
    ],
    "forecast_hourly": results,
    "rmse_by_horizon_pm25": {},
    "rmse_by_horizon_aqi": best_rmse_by_horizon,
}

with open("data/aqi_dashboard_data.json", "w") as f:
    json.dump(dashboard_data, f, indent=2, default=str)

df_result.to_csv("data/lahore_aqi_3day_forecast.csv", index=False)
print("Saved: data/aqi_dashboard_data.json")
print("Saved: data/lahore_aqi_3day_forecast.csv")
