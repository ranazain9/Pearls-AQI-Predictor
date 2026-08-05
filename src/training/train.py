"""Training pipeline: Train separate models per forecast horizon (Day 1, 2, 3) and save them."""
import json
import os
import pickle
import shutil
import sys
from pathlib import Path

# --- Make sure the project root is on sys.path so 'src' is importable ---
ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.config import (
    FEATURE_COLUMNS,
    HISTORICAL_CSV,
    MODEL_REGISTRY_NAME,
    MODELS_DIR,
    TARGET_COLUMN,
)

load_dotenv()

def load_training_data() -> pd.DataFrame:
    """Load features from local CSV or Hopsworks Feature Store."""
    hopsworks_key = os.getenv("HOPSWORKS_API_KEY")
    if hopsworks_key:
        try:
            import hopsworks
            project = hopsworks.login(api_key_value=hopsworks_key)
            fs = project.get_feature_store()
            fg = fs.get_feature_group("lahore_aqi_features", version=2)
            df = fg.read()
            if len(df) > 0:
                print("Loaded dataset from Hopsworks Feature Store.")
                return df
        except Exception as e:
            print(f"Could not load from Hopsworks: {e}. Falling back to local CSV.")

    if not HISTORICAL_CSV.exists():
        raise FileNotFoundError(
            f"{HISTORICAL_CSV} not found. Run historical_backfill.py first."
        )
    print(f"Loading local dataset from: {HISTORICAL_CSV}")
    return pd.read_csv(HISTORICAL_CSV, parse_dates=["timestamp"])

def prepare_xy_for_horizon(df: pd.DataFrame, horizon_hours: int) -> tuple[pd.DataFrame, pd.Series, StandardScaler]:
    """Prepare feature matrix X, target y (shifted by horizon_hours), and fitted scaler."""
    from src.features.engineering import clean_features_df

    # Compute clean features (lags, etc.) and drop NaNs
    data = clean_features_df(df)
    
    # Target is direct AQI at the future horizon
    data["target_aqi"] = data["aqi"].shift(-horizon_hours)
    
    # Drop rows without target or missing feature columns
    available = [c for c in FEATURE_COLUMNS if c in data.columns]
    data = data.dropna(subset=["target_aqi"] + available).reset_index(drop=True)

    X = data[available]
    y = data["target_aqi"]

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=available, index=X.index)
    return X_scaled, y, scaler


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute RMSE, MAE, and R²."""
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    return {"mae": mae, "rmse": rmse, "r2": r2}

def train_random_forest(X_train, y_train) -> RandomForestRegressor:
    model = RandomForestRegressor(n_estimators=30, max_depth=8, random_state=42, n_jobs=1)
    model.fit(X_train, y_train)
    return model

def train_ridge_regression(X_train, y_train) -> Ridge:
    model = Ridge(alpha=10.0)
    model.fit(X_train, y_train)
    return model

def train_xgboost(X_train, y_train):
    try:
        from xgboost import XGBRegressor
        model = XGBRegressor(n_estimators=150, max_depth=6, random_state=42, n_jobs=-1, learning_rate=0.05)
        model.fit(X_train, y_train)
        return model
    except ImportError:
        return None

def train_lightgbm(X_train, y_train):
    try:
        from lightgbm import LGBMRegressor
        model = LGBMRegressor(n_estimators=150, max_depth=6, random_state=42, n_jobs=-1, learning_rate=0.05, verbose=-1)
        model.fit(X_train, y_train)
        return model
    except ImportError:
        return None

def run_training_pipeline(register_to_hopsworks: bool = False) -> dict:
    """Train separate models per forecast horizon (24h, 48h, 72h)."""
    MODELS_DIR.mkdir(exist_ok=True)

    df = load_training_data()

    horizons = {
        1: 24,
        2: 48,
        3: 72
    }

    metadata = {
        "best_models": {},
        "metrics": {},
        "feature_columns": FEATURE_COLUMNS
    }

    for day, hours in horizons.items():
        print(f"\n--- Training Models for Day {day} (Horizon: +{hours}h) ---")
        X, y, scaler = prepare_xy_for_horizon(df, hours)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        candidate_models = {
            "random_forest": train_random_forest(X_train, y_train),
            "ridge_regression": train_ridge_regression(X_train, y_train)
        }

        xgb = train_xgboost(X_train, y_train)
        if xgb is not None:
            candidate_models["xgboost"] = xgb

        lgbm = train_lightgbm(X_train, y_train)
        if lgbm is not None:
            candidate_models["lightgbm"] = lgbm

        best_model_name = None
        best_rmse = float('inf')
        horizon_metrics = {}

        for name, model in candidate_models.items():
            preds = model.predict(X_test)
            metrics = evaluate_model(y_test, preds)
            horizon_metrics[name] = metrics
            print(f"  {name} RMSE: {metrics['rmse']:.2f}")

            if metrics["rmse"] < best_rmse:
                best_rmse = metrics["rmse"]
                best_model_name = name

        print(f"Selected Best Model for Day {day}: {best_model_name} (RMSE: {best_rmse:.2f})")
        
        # Fit winner on all data
        winner_model = candidate_models[best_model_name]
        
        # Save Scaler and Winner Model for this horizon
        with open(MODELS_DIR / f"scaler_day{day}.pkl", "wb") as f:
            pickle.dump({"scaler": scaler, "feature_columns": list(X.columns)}, f)

        with open(MODELS_DIR / f"best_model_day{day}.pkl", "wb") as f:
            pickle.dump(winner_model, f)

        metadata["best_models"][f"day{day}"] = best_model_name
        metadata["metrics"][f"day{day}"] = horizon_metrics

    # Save a legacy best_model.pkl for compatibility with any dashboard loading single model
    # We will copy the Day 1 model as the default best_model.pkl
    shutil.copyfile(MODELS_DIR / "best_model_day1.pkl", MODELS_DIR / "best_model.pkl")
    shutil.copyfile(MODELS_DIR / "scaler_day1.pkl", MODELS_DIR / "scaler.pkl")
    
    # Store legacy fields for fallback compatibility
    metadata["deploy_model"] = metadata["best_models"]["day1"]
    metadata["best_model"] = metadata["best_models"]["day1"]
    metadata["best_metrics"] = metadata["metrics"]["day1"][metadata["best_models"]["day1"]]

    with open(MODELS_DIR / "training_results.json", "w") as f:
        json.dump(metadata, f, indent=2)

    if register_to_hopsworks:
        try:
            _register_models_hopsworks(MODELS_DIR, metadata)
        except Exception as e:
            print(f"Failed to register models to Hopsworks: {e}")

    return metadata

def _register_models_hopsworks(model_dir, metadata: dict) -> None:
    """Save model directory to Hopsworks Model Registry."""
    import hopsworks

    api_key = os.getenv("HOPSWORKS_API_KEY")
    if not api_key:
        return

    project = hopsworks.login(api_key_value=api_key)
    mr = project.get_model_registry()
    registered = mr.python.create_model(
        name=MODEL_REGISTRY_NAME,
        metrics=metadata["best_metrics"],
        description="Direct multi-horizon AQI forecasting models (Day 1, 2, 3).",
    )
    # Save the entire MODELS_DIR so it includes all horizon models and scalers
    registered.save(str(model_dir))

if __name__ == "__main__":
    run_training_pipeline(register_to_hopsworks=True)
