"""Pearls AQI Predictor — Flask app serving the dashboard UI + REST API.

Uses the src/ inference pipeline to generate live predictions.

Run:
    python app/flask_api.py
Then open: http://127.0.0.1:5000
"""
import json
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
# pyrefly: ignore [missing-import]
from flask import Flask, jsonify, request, send_file

from src.alerts import check_aqi_alert, check_forecast_alerts
from src.config import CITY_NAME, HISTORICAL_CSV, LATITUDE, LONGITUDE, MODELS_DIR
from src.features.engineering import compute_features
from src.features.fetch_raw import fetch_openmeteo_current_row
from src.inference.predict import predict_next_3_days, predict_next_3_days_with_shap

app = Flask(__name__)
FRONTEND = ROOT / "app" / "frontend.html"
DATA_JSON = ROOT / "data" / "aqi_dashboard_data.json"

def _sanitize_nan(obj):
    """Recursively replace NaN/inf with None so the JSON is valid."""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: _sanitize_nan(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_nan(v) for v in obj]
    return obj
# ─────────────────────────────────────────────
# AQI helpers
# ─────────────────────────────────────────────

def aqi_category(aqi):
    if aqi is None: return "Unknown"
    if aqi <= 50: return "Good"
    if aqi <= 100: return "Moderate"
    if aqi <= 150: return "Unhealthy for Sensitive Groups"
    if aqi <= 200: return "Unhealthy"
    if aqi <= 300: return "Very Unhealthy"
    return "Hazardous"

# ─────────────────────────────────────────────
# Data helpers
# ─────────────────────────────────────────────
try:
    import pandas as pd
    from src.features.fetch_raw import fetch_openmeteo_current_row, fetch_aqicn_current, current_reading_to_row
except ImportError:
    pd = None

def _load_history() -> pd.DataFrame | None:
    if not HISTORICAL_CSV.exists():
        return None
    df = pd.read_csv(HISTORICAL_CSV, parse_dates=["timestamp"])
    
    # FETCH IN BACKEND: dynamically fetch the latest real-time row
    try:
        meteo_row = fetch_openmeteo_current_row()
        try:
            aqicn_data = fetch_aqicn_current()
        except Exception:
            aqicn_data = None
        current_row = current_reading_to_row(aqicn_data, meteo_row)
        
        # Append the current row if it's newer than the last row in history
        last_ts = df["timestamp"].max()
        current_ts = current_row["timestamp"].iloc[0]
        if current_ts > last_ts:
            df = pd.concat([df, current_row], ignore_index=True)
        elif current_ts == last_ts:
            # Update the latest row with the fresh data
            for col in current_row.columns:
                if col in df.columns:
                    df.loc[df.index[-1], col] = current_row[col].iloc[0]
    except Exception as e:
        # If live fetch fails, just fall back to historical CSV data
        print(f"Warning: Failed to fetch live data: {e}")

    return compute_features(df)



def _get_rmse_for_horizon(training_meta: dict, day_key: str) -> float | None:
    rmse_by_horizon = training_meta.get("rmse_by_horizon_aqi", {})
    horizon_num = day_key.replace("day", "")
    if horizon_num in rmse_by_horizon:
        return rmse_by_horizon[horizon_num]
    
    if "metrics" in training_meta and day_key in training_meta.get("metrics", {}):
        best_algo = training_meta.get("best_models", {}).get(day_key)
        if best_algo and best_algo in training_meta["metrics"][day_key]:
            rmse_val = training_meta["metrics"][day_key][best_algo].get("rmse")
            if rmse_val is not None:
                return rmse_val
    return None

def _build_dashboard_json(history: pd.DataFrame, predictions: pd.DataFrame, training_meta: dict, checkpoint_shap: dict = None) -> dict:
    valid = history.dropna(subset=["aqi"])
    latest = valid.iloc[-1]
    pm25_now = float(latest.get("ground_pm25", 0) or latest.get("modeled_pm25", 0))
    pm10_raw = float(latest.get("ground_pm10", 0) or latest.get("modeled_pm10", 0))
    pm10_now = max(pm10_raw, pm25_now)
    aqi_now = float(latest["aqi"])

    cutoff = latest["timestamp"] - pd.Timedelta(hours=24)
    trend_rows = history[history["timestamp"] >= cutoff].dropna(subset=["aqi"])
    trend_24h = [
        {"timestamp": str(r["timestamp"]), "aqi": round(float(r["aqi"]), 1)}
        for _, r in trend_rows.iterrows()
    ]

    checkpoints = {}
    for h, label in [(24, "24h"), (48, "48h"), (72, "72h")]:
        if len(predictions) >= h:
            row = predictions.iloc[h - 1]
            aqi_pred = float(row.get("predicted_aqi", 0))
            pm25_pred = float(row.get("predicted_pm25", 0))
            
            day_key = f"day{h // 24}"
            rmse_val = _get_rmse_for_horizon(training_meta, day_key)
            shap_feats = checkpoint_shap.get(label, []) if checkpoint_shap else []

            checkpoints[label] = {
                "predicted_aqi": round(aqi_pred, 1),
                "predicted_pm25": round(pm25_pred, 2),
                "rmse_aqi": round(rmse_val, 1) if rmse_val is not None else None,
                "features": shap_feats,
            }

    import os
    
    # Get file modification times
    feature_time = "Unknown"
    try:
        from src.config import HISTORICAL_CSV
        if HISTORICAL_CSV.exists():
            mtime = os.path.getmtime(HISTORICAL_CSV)
            feature_time = pd.Timestamp(mtime, unit='s').isoformat()
    except Exception:
        pass
        
    model_time = "Unknown"
    try:
        from src.config import MODELS_DIR
        train_json = MODELS_DIR / "training_results.json"
        if train_json.exists():
            mtime = os.path.getmtime(train_json)
            model_time = pd.Timestamp(mtime, unit='s').isoformat()
    except Exception:
        pass

    best_models = training_meta.get("best_models", {})
    best_model = ", ".join([f"D{i}: {name.replace('_', ' ').title()}" for i, (d, name) in enumerate(best_models.items(), 1)]) if best_models else training_meta.get("best_model", "random_forest").replace("_", " ").title()

    updated_at_ts = pd.Timestamp.now().isoformat()

    return {
        "location": {
            "name": CITY_NAME,
            "region": "Punjab",
            "country": "Pakistan",
        },
        "updated_at": updated_at_ts,
        "pipeline_metrics": {
            "feature_pipeline_last_run": feature_time,
            "model_pipeline_last_run": model_time,
            "prediction_last_run": updated_at_ts
        },
        "data_note": f"Powered by Best Model: {best_model} · via src pipeline",
        "current": {
            "aqi": aqi_now,
            "category": aqi_category(aqi_now),
            "timestamp": str(latest["timestamp"]),
            "pollutants": {
                "pm25": round(pm25_now, 1),
                "pm10": round(pm10_now, 1),
                "ozone": round(float(latest["ozone"]), 1) if "ozone" in latest and latest["ozone"] is not None else None,
                "no2": round(float(latest["no2"]), 1) if "no2" in latest and latest["no2"] is not None else None,
                "so2": round(float(latest["so2"]), 1) if "so2" in latest and latest["so2"] is not None else None,
                "co": round(float(latest["co"]), 1) if "co" in latest and latest["co"] is not None else None,
            },
            "weather": {
                "temperature": round(float(latest.get("temperature", 0) or 0), 1),
                "humidity": round(float(latest.get("humidity", 0) or 0), 1),
                "pressure": round(float(latest.get("pressure", 0) or 0), 1),
            },
        },
        "trend_24h": trend_24h,
        "forecast": [
            {
                "horizon": label,
                "day": f"Day {i + 1}",
                **checkpoints[label],
            }
            for i, label in enumerate(["24h", "48h", "72h"]) if label in checkpoints
        ],
        "forecast_hourly": predictions.to_dict(orient="records"),
        "rmse_by_horizon_aqi": {k.replace("h", ""): v["rmse_aqi"] for k, v in checkpoints.items() if v.get("rmse_aqi") is not None},
    }

@app.route("/")
def index():
    """Serve the AQI dashboard frontend."""
    return send_file(FRONTEND)

@app.route("/api/data")
def dashboard_data():
    """
    Generate and return the full dashboard JSON consumed by frontend.html.
    Falls back to pre-generated file if the model hasn't been trained yet.
    """
    history = _load_history()
    scaler_exists = (MODELS_DIR / "scaler_day1.pkl").exists()
    training_json = MODELS_DIR / "training_results.json"

    if history is None or history.empty or not scaler_exists:
        if DATA_JSON.exists():
            with open(DATA_JSON) as f:
                return jsonify(json.load(f))
        return jsonify({"error": "No data available. Run aqi/pipeline.py or src/training/train.py first."}), 503

    try:
        predictions, checkpoint_shap = predict_next_3_days_with_shap(history)
        meta = json.loads(training_json.read_text()) if training_json.exists() else {}
        data = _build_dashboard_json(history, predictions, meta, checkpoint_shap)
        return jsonify(_sanitize_nan(data))
    except Exception as e:
        import traceback
        traceback.print_exc()
        if DATA_JSON.exists():
            with open(DATA_JSON) as f:
                fallback_data = json.load(f)
                fallback_data["api_error"] = str(e)
                return jsonify(fallback_data)
        return jsonify({"error": str(e)}), 500

@app.route("/api/forecast")
def forecast_api():
    """Return raw 72-hour PM2.5 + AQI predictions as JSON."""
    history = _load_history()
    if history is None or not (MODELS_DIR / "scaler_day1.pkl").exists():
        return jsonify({"error": "Model not trained. Run aqi/pipeline.py first."}), 503
    try:
        days = int(request.args.get("days", 3))
        hours = min(days * 24, 72)
        predictions = predict_next_3_days(history).head(hours)
        alerts = check_forecast_alerts(predictions)
        return jsonify({"forecast": predictions.to_dict(orient="records"), "alerts": alerts})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/current")
def current_api():
    """Return the current AQI reading with alert status."""
    history = _load_history()
    if history is None or history.empty:
        return jsonify({"error": "No historical data available"}), 404
    latest = history.dropna(subset=["aqi"]).iloc[-1]
    pm25 = float(latest.get("ground_pm25", 0) or latest.get("modeled_pm25", 0))
    aqi_now = float(latest["aqi"])
    return jsonify({
        "timestamp": str(latest["timestamp"]),
        "pm25": pm25,
        "aqi": aqi_now,
        "category": aqi_category(aqi_now),
        "alert": check_aqi_alert(aqi_now),
    })

@app.route("/api/metrics")
def metrics_api():
    """Return training metrics for all evaluated models."""
    results_path = MODELS_DIR / "training_results.json"
    if not results_path.exists():
        return jsonify({"error": "No training results. Run src/training/train.py first."}), 404
    with open(results_path) as f:
        return jsonify(json.load(f))

@app.route("/health")
def health():
    return jsonify({"status": "ok", "service": "pearls-aqi-predictor"})

if __name__ == "__main__":
    debug = os.getenv("FLASK_DEBUG", "false").lower() in ("1", "true", "yes")
    port = int(os.getenv("PORT", "5000"))
    print(f"Starting Pearls AQI Dashboard (src pipeline) on http://127.0.0.1:{port}")
    app.run(host="0.0.0.0", port=port, debug=debug)