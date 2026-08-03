"""Inference: hourly Open-Meteo forecast with aligned lag features."""
import json
import pickle

import pandas as pd

from src.config import FORECAST_HOURS, MODELS_DIR, TARGET_COLUMN
from src.features.engineering import compute_features, pm25_to_aqi_category, pm25_to_epa_aqi
from src.features.fetch_raw import fetch_openmeteo_forecast


def load_model_artifacts() -> tuple[object, dict]:
    """Load saved scaler and Best Model from disk (Sklearn or TensorFlow)."""
    with open(MODELS_DIR / "scaler.pkl", "rb") as f:
        artifact = pickle.load(f)

    with open(MODELS_DIR / "training_results.json") as f:
        metadata = json.load(f)

    deploy_model = metadata.get("deploy_model", "random_forest")
    
    if deploy_model == "tensorflow_nn":
        import tensorflow as tf
        model = tf.keras.models.load_model(MODELS_DIR / "best_model.keras")
        artifact["model_type"] = "tensorflow"
    else:
        pkl_path = MODELS_DIR / "best_model.pkl"
        if not pkl_path.exists():
            raise FileNotFoundError("No trained model found. Run training_pipeline.py first.")
        with open(pkl_path, "rb") as f:
            model = pickle.load(f)
        artifact["model_type"] = "sklearn"

    artifact["model_name"] = deploy_model
    return model, artifact


def predict_next_3_days(history: pd.DataFrame) -> pd.DataFrame:
    """
    Produce 72 hourly PM2.5 forecasts using Open-Meteo weather + AQ forecast.
    Lag features align with training (1 row = 1 hour).
    """
    model, artifact = load_model_artifacts()
    scaler = artifact["scaler"]
    feature_cols = artifact["feature_columns"]
    model_type = artifact["model_type"]

    featured = compute_features(history)
    featured = featured.dropna(subset=[TARGET_COLUMN])

    forecast_rows = fetch_openmeteo_forecast(hours=FORECAST_HOURS)
    if forecast_rows.empty:
        raise RuntimeError("Open-Meteo forecast unavailable.")

    # Hourly PM2.5 series seeded from history (matches training shift semantics)
    pm25_series: list[float] = list(featured[TARGET_COLUMN].values)

    predictions = []

    for _, row in forecast_rows.iterrows():
        lag_1 = pm25_series[-1]
        lag_24 = pm25_series[-24] if len(pm25_series) >= 24 else pm25_series[0]
        change_rate = lag_1 - pm25_series[-2] if len(pm25_series) >= 2 else 0.0

        feat_row = {
            "temperature": row["temperature"],
            "humidity": row["humidity"],
            "wind_speed": row["wind_speed"],
            "pressure": row["pressure"],
            "modeled_pm25": row["modeled_pm25"],
            "modeled_pm10": row["modeled_pm10"],
            "hour": row["timestamp"].hour,
            "day_of_week": row["timestamp"].dayofweek,
            "month": row["timestamp"].month,
            "aqi_change_rate": change_rate,
            "pm25_lag_1": lag_1,
            "pm25_lag_24": lag_24,
        }

        X = pd.DataFrame([feat_row])[feature_cols]
        X_scaled = scaler.transform(X)
        
        if model_type == "tensorflow":
            pred = float(model.predict(X_scaled, verbose=0).flatten()[0])
        else:
            pred = float(model.predict(X_scaled)[0])
            
        pred = max(0.0, pred)

        pm25_series.append(pred)

        predictions.append(
            {
                "timestamp": row["timestamp"],
                "predicted_pm25": round(pred, 2),
                "predicted_aqi": round(pm25_to_epa_aqi(pred), 1),
                "category": pm25_to_aqi_category(pred),
            }
        )

    return pd.DataFrame(predictions)
