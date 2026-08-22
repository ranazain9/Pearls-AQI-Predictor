import sys
import traceback
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.features.engineering import clean_features_df
from src.inference.predict import predict_next_3_days_with_shap

try:
    df = pd.read_csv('data/lahore_aqi_historical_2024_onwards.csv', parse_dates=['timestamp'])
    df = clean_features_df(df)
    preds, shap_dict = predict_next_3_days_with_shap(df)
    print("Success! Generated", len(preds), "predictions.")
    print("\nCheckpoint SHAP Feature Contributions:")
    for horizon, feats in shap_dict.items():
        print(f"\n--- {horizon} ---")
        for f in feats[:5]:
            print(f"  {f['name']}: {f['value']}")
except Exception as e:
    traceback.print_exc()

