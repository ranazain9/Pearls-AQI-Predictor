"""Pearls AQI Predictor — Flask app serving the dashboard UI + REST API.

Uses the src/ inference pipeline to generate live predictions.

Run:
    python app/flask_api.py
Then open: http://127.0.0.1:5000
"""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, send_file

from src.alerts import check_aqi_alert, check_forecast_alerts
from src.config import CITY_NAME, HISTORICAL_CSV, LATITUDE, LONGITUDE, MODELS_DIR
from src.features.engineering import compute_features, pm25_to_aqi_category, pm25_to_epa_aqi
from src.features.fetch_raw import fetch_openmeteo_current_row
from src.inference.predict import predict_next_3_days

app = Flask(__name__)
FRONTEND = ROOT / "app" / "frontend.html"
DATA_JSON = ROOT / "data" / "aqi_dashboard_data.json"

# ─────────────────────────────────────────────
# AQI helpers
# ─────────────────────────────────────────────

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

def _sub_index(conc, bps):
    if conc is None or (isinstance(conc, float) and np.isnan(conc)):
        return None
    for c_lo, c_hi, i_lo, i_hi in bps:
        if c_lo <= conc <= c_hi:
            return round(((i_hi - i_lo) / (c_hi - c_lo)) * (conc - c_lo) + i_lo)
    return 500 if conc > bps[-1][1] else None

def calc_aqi(pm25, pm10=None):
    vals = [v for v in [_sub_index(pm25, PM25_BREAKPOINTS), _sub_index(pm10, PM10_BREAKPOINTS)] if v is not None]
    return max(vals) if vals else None

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

def _load_history() -> pd.DataFrame | None:
    if not HISTORICAL_CSV.exists():
        return None
    df = pd.read_csv(HISTORICAL_CSV, parse_dates=["timestamp"])
    return compute_features(df)

def _build_dashboard_json(history: pd.DataFrame, predictions: pd.DataFrame, training_meta: dict) -> dict:
    """Assemble the exact JSON structure the frontend expects."""

    # Current reading from latest ground truth
    valid = history.dropna(subset=["ground_pm25"])
    latest = valid.iloc[-1]
    pm25_now = float(latest["ground_pm25"])
    pm10_now = float(latest.get("modeled_pm10", 0) or 0)
    aqi_now = calc_aqi(pm25_now, pm10_now)

    # 24h trend
    cutoff = latest["timestamp"] - pd.Timedelta(hours=24)
    trend_rows = history[history["timestamp"] >= cutoff].dropna(subset=["ground_pm25"])
    trend_24h = [
        {"timestamp": str(r["timestamp"]), "aqi": calc_aqi(float(r["ground_pm25"]), float(r.get("modeled_pm10") or 0))}
        for _, r in trend_rows.iterrows()
    ]

    # Build 24h / 48h / 72h checkpoint cards from predictions
    checkpoints = {}
    for h, label in [(24, "24h"), (48, "48h"), (72, "72h")]:
        if len(predictions) >= h:
            row = predictions.iloc[h - 1]
            aqi_pred = float(row.get("predicted_aqi", 0))
            pm25_pred = float(row.get("predicted_pm25", 0))
            checkpoints[label] = {
                "predicted_aqi": round(aqi_pred, 1),
                "predicted_pm25": round(pm25_pred, 2),
                "rmse_aqi": None,  # src pipeline doesn't run backtest — shown as N/A in UI
                "features": [],    # SHAP not run in live API mode for speed
            }

    # Best model from training_results.json
    best_model = training_meta.get("best_model", "random_forest").replace("_", " ").title()

    return {
        "location": {
            "name": CITY_NAME,
            "region": "Punjab",
            "country": "Pakistan",
        },
        "updated_at": pd.Timestamp.now().isoformat(),
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
        "rmse_by_horizon_pm25": {},
        "rmse_by_horizon_aqi": {},
    }

# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────

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
    scaler_exists = (MODELS_DIR / "scaler.pkl").exists()
    training_json = MODELS_DIR / "training_results.json"

    if history is None or history.empty or not scaler_exists:
        # Fall back to pre-generated file from aqi/pipeline.py
        if DATA_JSON.exists():
            with open(DATA_JSON) as f:
                return jsonify(json.load(f))
        return jsonify({"error": "No data available. Run aqi/pipeline.py or src/training/train.py first."}), 503

    try:
        predictions = predict_next_3_days(history)
        meta = json.loads(training_json.read_text()) if training_json.exists() else {}
        data = _build_dashboard_json(history, predictions, meta)
        return jsonify(data)
    except Exception as e:
        # Fall back to pre-generated file
        if DATA_JSON.exists():
            with open(DATA_JSON) as f:
                return jsonify(json.load(f))
        return jsonify({"error": str(e)}), 500

@app.route("/api/forecast")
def forecast_api():
    """Return raw 72-hour PM2.5 + AQI predictions as JSON."""
    history = _load_history()
    if history is None or not (MODELS_DIR / "scaler.pkl").exists():
        return jsonify({"error": "Model not trained. Run src/training/train.py first."}), 503
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
    latest = history.dropna(subset=["ground_pm25"]).iloc[-1]
    pm25 = float(latest["ground_pm25"])
    return jsonify({
        "timestamp": str(latest["timestamp"]),
        "pm25": pm25,
        "aqi": calc_aqi(pm25, float(latest.get("modeled_pm10") or 0)),
        "category": pm25_to_aqi_category(pm25),
        "alert": check_aqi_alert(pm25),
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
