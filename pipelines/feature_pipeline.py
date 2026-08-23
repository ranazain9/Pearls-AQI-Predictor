#!/usr/bin/env python3
"""CLI entry point for the hourly feature pipeline."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.features.pipeline import run_feature_pipeline
from src.inference.pipeline import run_inference_pipeline

if __name__ == "__main__":
    upload = "--upload" in sys.argv
    df = run_feature_pipeline(upload_to_hopsworks=upload)
    print(f"Feature pipeline complete. {len(df)} rows in dataset.")

    print("Running inference pipeline to update dashboard JSON...")
    run_inference_pipeline()
    print("Inference pipeline complete. aqi_dashboard_data.json updated.")
