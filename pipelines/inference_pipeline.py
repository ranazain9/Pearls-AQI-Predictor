#!/usr/bin/env python3
"""CLI entry point for the hourly inference pipeline."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.inference.pipeline import run_inference_pipeline

if __name__ == "__main__":
    run_inference_pipeline()
    print("Inference pipeline complete. Static JSON generated.")
