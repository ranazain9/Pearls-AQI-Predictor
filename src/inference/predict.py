"""Inference: Predict 3-day AQI forecast using separate horizon models."""
import json
import pickle
import pandas as pd
import numpy as np
from src.config import FORECAST_HOURS, MODELS_DIR, TARGET_COLUMN, FEATURE_COLUMNS, FEATURE_LABELS
from src.features.engineering import aqi_category, aqi_to_pm25, clean_features_df, compute_features


def load_horizon_artifacts(day: int) -> tuple[object, dict]:
    """Load scaler and model for a specific horizon day (1, 2, or 3) from local or Hopsworks."""
    import os

    pkl_path = MODELS_DIR / f"best_model_day{day}.pkl"
    scaler_path = MODELS_DIR / f"scaler_day{day}.pkl"

    # If local model artifacts exist, load them immediately (fastest, 0 ms latency, no 280MB cloud download)
    if pkl_path.exists() and scaler_path.exists():
        with open(scaler_path, "rb") as f:
            artifact = pickle.load(f)
        with open(pkl_path, "rb") as f:
            model = pickle.load(f)
        artifact["model"] = model
        return model, artifact

    api_key = os.getenv("HOPSWORKS_API_KEY")
    if api_key:
        try:
            import hopsworks
            from src.config import MODEL_REGISTRY_NAME
            project = hopsworks.login(api_key_value=api_key)
            mr = project.get_model_registry()
            # Fetch version=1 as enforced by training pipeline
            hw_model = mr.get_model(MODEL_REGISTRY_NAME, version=1)
            model_dir = hw_model.download()

            with open(f"{model_dir}/scaler_day{day}.pkl", "rb") as f:
                artifact = pickle.load(f)
            with open(f"{model_dir}/best_model_day{day}.pkl", "rb") as f:
                model = pickle.load(f)

            artifact["model"] = model
            return model, artifact
        except ImportError:
            pass
        except Exception as e:
            print(f"Failed to load from Hopsworks, falling back to local: {e}")

    raise FileNotFoundError(f"Model for Day {day} not found. Run training first.")

    artifact["model"] = model
    return model, artifact


def compute_shap_explanations(model: object, X_scaled: np.ndarray, feature_cols: list[str]) -> list[dict]:
    """
    Compute SHAP values (or feature contribution fallback) for input vector X_scaled.
    Returns list of dicts [{"name": label, "val": float}, ...] sorted by absolute contribution.
    """
    shap_vals = None
    try:
        import shap
        model_type = type(model).__name__.lower()
        if any(t in model_type for t in ["forest", "xgb", "lgbm", "tree", "gbm"]):
            explainer = shap.TreeExplainer(model)
            sv = explainer.shap_values(X_scaled)
            if isinstance(sv, list):
                sv = sv[0]
            shap_vals = sv[0]
        elif "ridge" in model_type or "linear" in model_type:
            if hasattr(model, "coef_"):
                shap_vals = X_scaled[0] * model.coef_
            else:
                explainer = shap.Explainer(model, X_scaled)
                shap_vals = explainer(X_scaled).values[0]
    except Exception:
        shap_vals = None

    if shap_vals is None:
        if hasattr(model, "feature_importances_"):
            shap_vals = X_scaled[0] * model.feature_importances_
        elif hasattr(model, "coef_"):
            shap_vals = X_scaled[0] * model.coef_
        else:
            shap_vals = np.zeros(len(feature_cols))

    feats_sorted = []
    for col, val in zip(feature_cols, shap_vals):
        label = FEATURE_LABELS.get(col, col)
        # FIXED: Use "val" instead of "value" for consistency
        feats_sorted.append({"name": label, "val": round(float(val), 2)})

    feats_sorted.sort(key=lambda x: abs(x["val"]), reverse=True)
    return feats_sorted


def predict_next_3_days_with_shap(history: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Produce 72 hourly AQI forecasts using three separate models for each horizon,
    and return SHAP feature explanations for the 24h, 48h, and 72h forecast checkpoints.
    """
    models = {}
    scalers = {}
    feature_cols = {}
    
    for d in [1, 2, 3]:
        model, art = load_horizon_artifacts(d)
        models[d] = model
        scalers[d] = art["scaler"]
        feature_cols[d] = art["feature_columns"]

    featured = clean_features_df(history)
    featured = featured.sort_values("timestamp").reset_index(drop=True)
    featured.index = featured["timestamp"]

    latest_time = featured["timestamp"].max()
    predictions = []
    checkpoint_shap = {}
    checkpoint_hours = {24: "24h", 48: "48h", 72: "72h"}

    for h in range(1, FORECAST_HOURS + 1):
        target_time = latest_time + pd.Timedelta(hours=h)
        
        # FIXED: Use latest available data for feature extraction
        # Each model looks at the current state to predict future
        lookup_time = featured["timestamp"].max()
        
        if h <= 24:
            day = 1
        elif h <= 48:
            day = 2
        else:
            day = 3
            
        if lookup_time in featured.index:
            feat_row = featured.loc[lookup_time]
            if isinstance(feat_row, pd.DataFrame):
                feat_row = feat_row.iloc[-1]
                
            X_dict = {col: feat_row.get(col, 0.0) for col in feature_cols[day]}
            for col in X_dict:
                if pd.isna(X_dict[col]):
                    X_dict[col] = 0.0
            
            X_df = pd.DataFrame([X_dict])[feature_cols[day]]
            X_scaled = scalers[day].transform(X_df)
            
            pred = float(models[day].predict(X_scaled)[0])
            pred = max(0.0, pred)

            # FIXED: Compute SHAP for checkpoints (this ensures features are populated)
            if h in checkpoint_hours:
                checkpoint_shap[checkpoint_hours[h]] = compute_shap_explanations(
                    models[day], X_scaled, feature_cols[day]
                )
        else:
            # Fallback if lookup fails
            pred = float(featured["aqi"].iloc[-1] if not featured["aqi"].empty else 100.0)
            if h in checkpoint_hours:
                # CRITICAL FIX: Still try to compute SHAP with latest available data
                try:
                    feat_row = featured.iloc[-1]
                    X_dict = {col: feat_row.get(col, 0.0) for col in feature_cols[day]}
                    for col in X_dict:
                        if pd.isna(X_dict[col]):
                            X_dict[col] = 0.0
                    X_df = pd.DataFrame([X_dict])[feature_cols[day]]
                    X_scaled = scalers[day].transform(X_df)
                    checkpoint_shap[checkpoint_hours[h]] = compute_shap_explanations(
                        models[day], X_scaled, feature_cols[day]
                    )
                except Exception:
                    checkpoint_shap[checkpoint_hours[h]] = []

        predictions.append(
            {
                "timestamp": target_time,
                "predicted_pm25": aqi_to_pm25(pred),
                "predicted_aqi": round(pred, 1),
                "category": aqi_category(pred),
            }
        )

    return pd.DataFrame(predictions), checkpoint_shap


def predict_next_3_days(history: pd.DataFrame) -> pd.DataFrame:
    """Wrapper returning dataframe predictions for backward compatibility."""
    preds_df, _ = predict_next_3_days_with_shap(history)
    return preds_df