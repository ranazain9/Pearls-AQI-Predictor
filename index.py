import json
import os
from pathlib import Path
from flask import Flask, jsonify, send_file, request

app = Flask(__name__)
ROOT = Path(__file__).resolve().parent

def _find_data_json():
    candidates = [
        ROOT / "data" / "aqi_dashboard_data.json",
        Path.cwd() / "data" / "aqi_dashboard_data.json",
        Path(__file__).parent / "data" / "aqi_dashboard_data.json",
        Path("/var/task/data/aqi_dashboard_data.json")
    ]
    for p in candidates:
        if p.exists():
            return p
    return None

def _find_frontend_html():
    candidates = [
        ROOT / "index.html",
        ROOT / "app" / "frontend.html",
        Path.cwd() / "index.html",
        Path.cwd() / "app" / "frontend.html",
        Path(__file__).parent / "index.html",
        Path(__file__).parent / "app" / "frontend.html",
        Path("/var/task/index.html"),
        Path("/var/task/app/frontend.html")
    ]
    for p in candidates:
        if p.exists():
            return p
    return None

@app.route("/")
@app.route("/index.py")
@app.route("/index.html")
def home():
    """Serve the frontend dashboard."""
    html_path = _find_frontend_html()
    if html_path and html_path.exists():
        return send_file(html_path)
    return "<h1>Pearls AQI Predictor Dashboard</h1>", 200

@app.route("/api/data")
@app.route("/data/aqi_dashboard_data.json")
def get_data():
    """Serve the pre-computed AQI dashboard JSON."""
    json_path = _find_data_json()
    if json_path and json_path.exists():
        with open(json_path, "r", encoding="utf-8") as f:
            return jsonify(json.load(f))
    return jsonify({"error": "Dashboard data not found"}), 404

@app.route("/health")
def health():
    return jsonify({"status": "ok", "service": "pearls-aqi-predictor"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)