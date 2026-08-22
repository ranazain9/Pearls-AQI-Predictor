import json
from pathlib import Path
from flask import Flask, jsonify, send_file

app = Flask(__name__)
ROOT = Path(__file__).resolve().parent

@app.route("/")
def home():
    """Serve the frontend dashboard."""
    frontend_path = ROOT / "app" / "frontend.html"
    return send_file(frontend_path)

@app.route("/api/data")
def get_data():
    """Serve the pre-computed AQI dashboard JSON."""
    data_path = ROOT / "data" / "aqi_dashboard_data.json"
    if data_path.exists():
        with open(data_path, "r", encoding="utf-8") as f:
            return jsonify(json.load(f))
    return jsonify({"error": "Dashboard data not found"}), 404

@app.route("/health")
def health():
    return jsonify({"status": "ok", "service": "pearls-aqi-predictor"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)