#!/usr/bin/env python3
"""CLI entry point for the daily training pipeline."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.training.train import run_training_pipeline

if __name__ == "__main__":
    register = "--register" in sys.argv
    results = run_training_pipeline(register_to_hopsworks=register)
    print(json.dumps(results, indent=2))
