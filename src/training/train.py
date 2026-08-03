"""Training pipeline: Train multiple models (RF, Ridge, TF), evaluate, select best, register."""
import json
import os
import pickle
import shutil
import sys
from pathlib import Path

# --- Make sure the project root is on sys.path so 'src' is importable ---
# This allows: python src/training/train.py  OR  python -m src.training.train
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
            fg = fs.get_feature_group("lahore_aqi_features", version=1)
            df = fg.read()
            if len(df) > 0:
                return df
        except Exception:
            pass

    if not HISTORICAL_CSV.exists():
        raise FileNotFoundError(
            f"{HISTORICAL_CSV} not found. Run historical_backfill.py first."
        )
    return pd.read_csv(HISTORICAL_CSV, parse_dates=["timestamp"])

def prepare_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, StandardScaler]:
    """Prepare feature matrix, target, and fitted scaler."""
    from src.features.engineering import compute_features

    data = compute_features(df)
    data = data.dropna(subset=[TARGET_COLUMN])

    available = [c for c in FEATURE_COLUMNS if c in data.columns]
    data = data.dropna(subset=available)

    X = data[available]
    y = data[TARGET_COLUMN]

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
    model = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    return model

def train_ridge_regression(X_train, y_train) -> Ridge:
    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)
    return model

def train_xgboost(X_train, y_train):
    try:
        from xgboost import XGBRegressor
        model = XGBRegressor(n_estimators=300, random_state=42, n_jobs=-1, learning_rate=0.05)
        model.fit(X_train, y_train)
        return model
    except ImportError:
        return None

def train_lightgbm(X_train, y_train):
    try:
        from lightgbm import LGBMRegressor
        model = LGBMRegressor(n_estimators=300, random_state=42, n_jobs=-1, learning_rate=0.05)
        model.fit(X_train, y_train)
        return model
    except ImportError:
        return None

def run_training_pipeline(register_to_hopsworks: bool = False) -> dict:
    """Train multiple models, save best artifacts, optionally register to Hopsworks."""
    MODELS_DIR.mkdir(exist_ok=True)

    df = load_training_data()
    X, y, scaler = prepare_xy(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {
        "random_forest": train_random_forest(X_train, y_train),
        "ridge_regression": train_ridge_regression(X_train, y_train)
    }

    xgb = train_xgboost(X_train, y_train)
    if xgb is not None:
        models["xgboost"] = xgb
    else:
        print("XGBoost not installed. Skipping.")

    lgbm = train_lightgbm(X_train, y_train)
    if lgbm is not None:
        models["lightgbm"] = lgbm
    else:
        print("LightGBM not installed. Skipping.")

    best_model_name = None
    best_rmse = float('inf')
    all_metrics = {}

    for name, model in models.items():
        print(f"Evaluating {name}...")
        preds = model.predict(X_test)
            
        metrics = evaluate_model(y_test, preds)
        all_metrics[name] = metrics
        
        print(f"  {name} RMSE: {metrics['rmse']:.2f}")
        
        if metrics["rmse"] < best_rmse:
            best_rmse = metrics["rmse"]
            best_model_name = name

    print(f"\nSelected Best Model: {best_model_name} (RMSE: {best_rmse:.2f})")
    best_model = models[best_model_name]

    # Save Scaler
    with open(MODELS_DIR / "scaler.pkl", "wb") as f:
        pickle.dump({"scaler": scaler, "feature_columns": list(X.columns)}, f)

    for f in MODELS_DIR.glob("*_model.*"):
        if f.is_dir():
            shutil.rmtree(f, ignore_errors=True)
        else:
            try:
                f.unlink()
            except PermissionError:
                print(f"Warning: Could not delete {f.name} (likely in use by another process).")

    # Save ALL trained models so they all get uploaded to Hopsworks
    for name, model in models.items():
        with open(MODELS_DIR / f"{name}_model.pkl", "wb") as f:
            pickle.dump(model, f)

    # Also save the winner explicitly as 'best_model' so our dashboard knows which one to load
    with open(MODELS_DIR / "best_model.pkl", "wb") as f:
        pickle.dump(best_model, f)

    metadata = {
        "deploy_model": best_model_name,
        "best_model": best_model_name,
        "metrics": all_metrics,
        "best_metrics": all_metrics[best_model_name],
        "feature_columns": list(X.columns),
    }
    with open(MODELS_DIR / "training_results.json", "w") as f:
        json.dump(metadata, f, indent=2)

    if register_to_hopsworks:
        try:
            _register_model_hopsworks(MODELS_DIR, all_metrics[best_model_name])
        except Exception as e:
            print(f"Failed to register model to Hopsworks: {e}")

    return metadata

def _register_model_hopsworks(model_dir, metrics: dict) -> None:
    """Save model directory to Hopsworks Model Registry."""
    import hopsworks

    api_key = os.getenv("HOPSWORKS_API_KEY")
    if not api_key:
        return

    project = hopsworks.login(api_key_value=api_key)
    mr = project.get_model_registry()
    registered = mr.python.create_model(
        name=MODEL_REGISTRY_NAME,
        metrics=metrics,
        description="Best AQI predictor model.",
    )
    # Save the entire MODELS_DIR so it includes scaler and best_model.*
    registered.save(str(model_dir))

if __name__ == "__main__":
    run_training_pipeline()
