"""
Streamlit dashboard has been replaced by the HTML/Flask dashboard.

To run the dashboard, use one of these instead:

  python aqi/app.py         →  http://127.0.0.1:5000   (uses pre-generated data)
  python app/flask_api.py   →  http://127.0.0.1:5000   (uses live src inference)

The full interactive dashboard is at:
  aqi/frontend.html   (served by either Flask app above)
"""
raise SystemExit(
    "\n\n🛑  Streamlit has been replaced by the HTML dashboard.\n"
    "    Run:  python aqi/app.py\n"
    "    Open: http://127.0.0.1:5000\n"
)
