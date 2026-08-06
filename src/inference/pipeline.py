import json
import logging
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent

from src.config import MODELS_DIR, CITY_NAME
from src.features.engineering import compute_features
from src.inference.predict import predict_next_3_days_with_shap

def aqi_category(aqi):
    if aqi is None: return "Unknown"
    if aqi <= 50: return "Good"
    if aqi <= 100: return "Moderate"
    if aqi <= 150: return "Unhealthy for Sensitive Groups"
    if aqi <= 200: return "Unhealthy"
    if aqi <= 300: return "Very Unhealthy"
    return "Hazardous"

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

    best_models = training_meta.get("best_models", {})
    best_model = ", ".join([f"D{i}: {name.replace('_', ' ').title()}" for i, (d, name) in enumerate(best_models.items(), 1)]) if best_models else training_meta.get("best_model", "random_forest").replace("_", " ").title()

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
        "rmse_by_horizon_aqi": {k.replace("h", ""): v["rmse_aqi"] for k, v in checkpoints.items() if v.get("rmse_aqi") is not None},
    }

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

HISTORICAL_CSV = ROOT / "data" / "lahore_aqi_historical_2024_onwards.csv"
DATA_JSON = ROOT / "data" / "aqi_dashboard_data.json"
TRAINING_JSON = MODELS_DIR / "training_results.json"

def _load_history() -> pd.DataFrame:
    if not HISTORICAL_CSV.exists():
        raise FileNotFoundError(f"Historical data not found at {HISTORICAL_CSV}")
    df = pd.read_csv(HISTORICAL_CSV, parse_dates=["timestamp"])
    return compute_features(df)

def run_inference_pipeline():
    """Run the inference pipeline and generate the static dashboard JSON."""
    logger.info("Starting inference pipeline...")
    
    # 1. Load History
    logger.info("Loading history...")
    history = _load_history()
    
    # 2. Run Predictions
    logger.info("Generating predictions and SHAP values...")
    predictions, checkpoint_shap = predict_next_3_days_with_shap(history)
    
    # 3. Load Training Meta
    meta = {}
    if TRAINING_JSON.exists():
        meta = json.loads(TRAINING_JSON.read_text(encoding='utf-8'))
    
    # 4. Build JSON Payload
    logger.info("Building dashboard JSON payload...")
    data = _build_dashboard_json(history, predictions, meta, checkpoint_shap)
    
    # Sanitize payload (replace NaN with None, convert Timestamps)
    import math
    def _sanitize_nan(obj):
        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        if isinstance(obj, float):
            if math.isnan(obj) or math.isinf(obj):
                return None
            return obj
        if isinstance(obj, dict):
            return {k: _sanitize_nan(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_sanitize_nan(v) for v in obj]
        return obj
        
    sanitized_data = _sanitize_nan(data)
    
    # 5. Save to File
    DATA_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(DATA_JSON, "w", encoding="utf-8") as f:
        json.dump(sanitized_data, f, indent=2)
        
    logger.info(f"Successfully generated {DATA_JSON.name}")
    return sanitized_data

if __name__ == "__main__":
    run_inference_pipeline()
