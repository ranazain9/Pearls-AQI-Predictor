"""Quick sanity checks for project review."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import HISTORICAL_CSV
from src.features.engineering import compute_features, pm25_to_aqi_category, pm25_to_epa_aqi
import pandas as pd

# Bug: passing AQI to category function
cat_wrong = pm25_to_aqi_category(pm25_to_epa_aqi(40))
cat_right = pm25_to_aqi_category(40)
print(f"Category bug: wrong={cat_wrong!r}, correct={cat_right!r}")
assert cat_wrong != cat_right, "Expected category bug to differ"

if HISTORICAL_CSV.exists():
    df = pd.read_csv(HISTORICAL_CSV, parse_dates=["timestamp"])
    featured = compute_features(df)
    print(f"Historical rows: {len(df)}")

models_dir = Path("models")
if (models_dir / "scaler.pkl").exists():
    from src.inference.predict import load_model_artifacts, predict_next_3_days

    model, art = load_model_artifacts()
    print(f"Model: {art.get('model_name')}")
    if HISTORICAL_CSV.exists():
        fc = predict_next_3_days(featured)
        print(f"Forecast: {len(fc)} rows, first AQI={fc.iloc[0]['predicted_aqi']}")

print("All checks passed")
