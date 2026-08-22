import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import numpy as np
import requests
import json
from typing import Dict, Any, List, Tuple
from enum import Enum

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="Pearls AQI Predictor",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==================== DESIGN SYSTEM ====================
class AQICategory(Enum):
    GOOD = {'label': 'Good', 'color': '#00D9A3', 'bg': 'rgba(0, 217, 163, 0.08)', 'accent': '#00D9A3', 'max': 50}
    MODERATE = {'label': 'Moderate', 'color': '#FFB84D', 'bg': 'rgba(255, 184, 77, 0.08)', 'accent': '#FFB84D', 'max': 100}
    USG = {'label': 'Unhealthy for Sensitive Groups', 'color': '#FF9500', 'bg': 'rgba(255, 149, 0, 0.08)', 'accent': '#FF9500', 'max': 150}
    UNHEALTHY = {'label': 'Unhealthy', 'color': '#FF5252', 'bg': 'rgba(255, 82, 82, 0.08)', 'accent': '#FF5252', 'max': 200}
    VERY_UNHEALTHY = {'label': 'Very Unhealthy', 'color': '#D946EF', 'bg': 'rgba(217, 70, 239, 0.08)', 'accent': '#D946EF', 'max': 300}
    HAZARDOUS = {'label': 'Hazardous', 'color': '#8B1538', 'bg': 'rgba(139, 21, 56, 0.12)', 'accent': '#FF4757', 'max': 500}

# Color palette - refined and distinctive
COLORS = {
    'bg_primary': '#0A0E17',      # Deep navy
    'bg_secondary': '#141B28',    # Slightly lighter navy
    'bg_tertiary': '#1A232F',     # Card backgrounds
    'surface': '#202938',          # Surface hover state
    'border_light': '#2A3847',     # Subtle borders
    'border_mid': '#3A4A5C',       # Stronger borders
    'text_primary': '#F5F7FA',     # Almost white
    'text_secondary': '#B0BAC9',   # Muted
    'text_tertiary': '#8A95A8',    # Faint
    'accent_primary': '#00D9A3',   # Teal
    'accent_secondary': '#FFB84D', # Gold
    'accent_tertiary': '#D946EF',  # Purple
    'success': '#00D9A3',
    'warning': '#FFB84D',
    'error': '#FF5252',
}

# Typography system
TYPOGRAPHY = {
    'display_xl': 'font-size: 2.5rem; font-weight: 700; letter-spacing: -0.02em;',
    'display_lg': 'font-size: 2rem; font-weight: 700; letter-spacing: -0.01em;',
    'heading_lg': 'font-size: 1.5rem; font-weight: 600; letter-spacing: -0.01em;',
    'heading_md': 'font-size: 1.25rem; font-weight: 600;',
    'heading_sm': 'font-size: 1rem; font-weight: 600;',
    'body_lg': 'font-size: 1rem; font-weight: 400; line-height: 1.6;',
    'body_md': 'font-size: 0.95rem; font-weight: 400; line-height: 1.6;',
    'body_sm': 'font-size: 0.875rem; font-weight: 400; line-height: 1.5;',
    'label': 'font-size: 0.75rem; font-weight: 500; letter-spacing: 0.05em; text-transform: uppercase;',
    'mono': 'font-family: "IBM Plex Mono", monospace; font-size: 0.9rem;',
}

# ==================== ADVANCED CUSTOM CSS ====================
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

:root {{
--bg-primary: {COLORS['bg_primary']};
--bg-secondary: {COLORS['bg_secondary']};
--bg-tertiary: {COLORS['bg_tertiary']};
--surface: {COLORS['surface']};
--border-light: {COLORS['border_light']};
--border-mid: {COLORS['border_mid']};
--text-primary: {COLORS['text_primary']};
--text-secondary: {COLORS['text_secondary']};
--text-tertiary: {COLORS['text_tertiary']};
--accent-primary: {COLORS['accent_primary']};
--accent-secondary: {COLORS['accent_secondary']};
}}

* {{
margin: 0;
padding: 0;
box-sizing: border-box;
}}

html, body, [class*="css"] {{
font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}}

html, body {{
background: linear-gradient(135deg, {COLORS['bg_primary']} 0%, {COLORS['bg_secondary']} 100%);
background-attachment: fixed;
}}

[data-testid="stAppViewContainer"] {{
background:
radial-gradient(ellipse 1200px 600px at 20% -5%, rgba(0, 217, 163, 0.05), transparent 70%),
radial-gradient(ellipse 1000px 700px at 80% 20%, rgba(217, 70, 239, 0.035), transparent 70%),
radial-gradient(ellipse 900px 500px at 50% 100%, rgba(255, 184, 77, 0.02), transparent 70%),
linear-gradient(135deg, {COLORS['bg_primary']} 0%, {COLORS['bg_secondary']} 100%);
background-attachment: fixed;
color: {COLORS['text_primary']};
}}

[data-testid="stSidebar"] {{
background-color: {COLORS['bg_secondary']};
border-right: 1px solid {COLORS['border_light']};
}}

/* Hide Streamlit default header to let our custom navbar shine */
[data-testid="stHeader"] {{
display: none;
}}

/* ========== SCROLLBAR ========== */
::-webkit-scrollbar {{ width: 8px; height: 8px; }}
::-webkit-scrollbar-track {{ background: transparent; }}
::-webkit-scrollbar-thumb {{
background: linear-gradient(180deg, {COLORS['accent_primary']}, {COLORS['accent_tertiary']});
border-radius: 10px;
transition: background 0.3s ease;
}}
::-webkit-scrollbar-thumb:hover {{
background: linear-gradient(180deg, {COLORS['accent_secondary']}, {COLORS['accent_primary']});
}}

/* ========== TYPOGRAPHY ========== */
h1, h2, h3, h4, h5, h6 {{
color: {COLORS['text_primary']};
font-weight: 700;
letter-spacing: -0.01em;
}}
p {{ color: {COLORS['text_secondary']}; line-height: 1.6; }}

/* ========== BUTTONS ========== */
button[kind="primary"] {{
    background: linear-gradient(135deg, {COLORS['accent_primary']} 0%, #00B386 100%) !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 700 !important;
}}
button[kind="primary"] * {{
    color: #000000 !important;
}}
button[kind="primary"]:hover {{
    background: linear-gradient(135deg, #00E6B0 0%, #00C696 100%) !important;
    transform: translateY(-1px);
}}
button[kind="secondary"] {{
    border-radius: 8px !important;
    font-weight: 600 !important;
}}

/* ========== TABS ========== */
.stTabs [data-baseweb="tab-list"] {{ border-bottom: 1px solid {COLORS['border_light']}; gap: 8px; }}
.stTabs [data-baseweb="tab"] {{
border-radius: 6px 6px 0 0;
background-color: transparent !important;
border: none !important;
color: {COLORS['text_tertiary']} !important;
padding: 12px 16px !important;
transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
font-weight: 500 !important;
}}
.stTabs [aria-selected="true"] {{
!important;
color: {COLORS['bg_primary']} !important;
border: none !important;
}}
.stTabs [data-baseweb="tab"]:hover {{
background-color: {COLORS['surface']} !important;
color: {COLORS['text_primary']} !important;
}}

/* ========== METRICS ========== */
[data-testid="stMetricValue"] {{
font-size: 2.5rem !important;
font-weight: 700 !important;
letter-spacing: -0.02em !important;
background: linear-gradient(135deg, {COLORS['accent_primary']}, {COLORS['accent_secondary']});
-webkit-background-clip: text;
-webkit-text-fill-color: transparent;
background-clip: text;
}}
[data-testid="stMetricLabel"] {{
font-size: 0.75rem !important;
color: {COLORS['text_tertiary']} !important;
font-weight: 600 !important;
letter-spacing: 0.05em !important;
}}
[data-testid="stMetricDelta"] {{ color: {COLORS['accent_secondary']} !important; font-weight: 500 !important; }}

/* ========== CONTAINERS ========== */
.premium-card {{
background: linear-gradient(145deg, rgba(30, 36, 48, 0.7), rgba(20, 27, 40, 0.9));
border: 1px solid rgba(255, 255, 255, 0.06);
backdrop-filter: blur(16px);
-webkit-backdrop-filter: blur(16px);
border-radius: 16px;
padding: 24px;
transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
position: relative;
overflow: hidden;
animation: slideInUp 0.5s cubic-bezier(0.4, 0, 0.2, 1) both;
}}
.premium-card::before {{
content: '';
position: absolute;
top: 0;
left: -100%;
width: 50%;
height: 100%;
background: linear-gradient(to right, transparent, rgba(255, 255, 255, 0.03), transparent);
transform: skewX(-25deg);
transition: all 0.75s ease;
pointer-events: none;
}}
.premium-card:hover {{
border-color: rgba(0, 217, 163, 0.35);
box-shadow: 0 15px 35px rgba(0, 217, 163, 0.14), 0 5px 15px rgba(0, 0, 0, 0.4);
transform: translateY(-4px);
}}
.premium-card:hover::before {{ left: 200%; }}

.glass-effect {{
background: rgba(20, 27, 40, 0.65);
backdrop-filter: blur(20px);
-webkit-backdrop-filter: blur(20px);
border: 1px solid rgba(255, 255, 255, 0.08);
border-radius: 16px;
box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
}}

/* ========== SECTION HEADER ========== */
.section-header {{
display: flex;
align-items: center;
justify-content: space-between;
gap: 12px;
margin: 8px 0 18px 0;
}}
.section-header-left {{ display: flex; align-items: center; gap: 12px; }}
.section-icon {{
width: 38px;
height: 38px;
border-radius: 10px;
display: flex;
align-items: center;
justify-content: center;
font-size: 18px;
background: linear-gradient(135deg, rgba(0,217,163,0.18), rgba(217,70,239,0.10));
border: 1px solid rgba(255,255,255,0.08);
flex-shrink: 0;
}}
.section-title {{ font-size: 1.35rem; font-weight: 700; color: {COLORS['text_primary']}; letter-spacing: -0.01em; }}
.section-subtitle {{ font-size: 0.82rem; color: {COLORS['text_tertiary']}; margin-top: 2px; }}

/* ========== LIVE PULSE ========== */
.live-pulse {{
display: inline-flex;
align-items: center;
gap: 6px;
font-size: 0.72rem;
color: {COLORS['accent_primary']};
font-weight: 600;
letter-spacing: 0.05em;
text-transform: uppercase;
background: rgba(0, 217, 163, 0.08);
border: 1px solid rgba(0, 217, 163, 0.25);
padding: 5px 10px 5px 8px;
border-radius: 999px;
}}
.live-dot {{
width: 7px;
height: 7px;
border-radius: 50%;
background: {COLORS['accent_primary']};
box-shadow: 0 0 8px {COLORS['accent_primary']};
animation: pulseDot 1.8s ease-in-out infinite;
}}

/* ========== CUSTOM ALERT (theme-matched, replaces st.info) ========== */
.guidance-banner {{
display: flex;
align-items: flex-start;
gap: 12px;
background: linear-gradient(135deg, rgba(255,255,255,0.05), rgba(255,255,255,0.02));
border: 1px solid rgba(255,255,255,0.08);
border-left: 3px solid var(--banner-accent, {COLORS['accent_primary']});
border-radius: 12px;
padding: 14px 18px;
margin: 4px 0 20px 0;
color: {COLORS['text_secondary']};
font-size: 0.92rem;
line-height: 1.55;
}}
.guidance-banner strong {{ color: {COLORS['text_primary']}; }}
.guidance-icon {{ font-size: 1.1rem; line-height: 1.55; flex-shrink: 0; }}

/* ========== BUTTONS ========== */
.stButton > button {{
background: linear-gradient(135deg, {COLORS['accent_primary']}, {COLORS['accent_secondary']});
color: {COLORS['bg_primary']} !important;
border: none !important;
border-radius: 8px;
font-weight: 600;
font-size: 0.95rem;
padding: 10px 16px !important;
transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
box-shadow: 0 4px 12px rgba(0, 217, 163, 0.2);
position: relative;
overflow: hidden;
}}
.stButton > button::before {{
content: '';
position: absolute;
top: 0;
left: -100%;
width: 100%;
height: 100%;
background: rgba(255, 255, 255, 0.1);
transition: left 0.3s ease;
}}
.stButton > button:hover {{
box-shadow: 0 8px 20px rgba(0, 217, 163, 0.3);
transform: translateY(-2px);
}}
.stButton > button:hover::before {{ left: 100%; }}

.btn-secondary {{
background-color: {COLORS['bg_tertiary']} !important;
color: {COLORS['text_primary']} !important;
border: 1px solid {COLORS['border_mid']} !important;
}}
.btn-secondary:hover {{
background-color: {COLORS['surface']} !important;
border-color: {COLORS['accent_primary']} !important;
}}

/* ========== INPUTS (native + BaseWeb widgets) ========== */
[data-testid="stTextInput"] input,
[data-testid="stNumberInput"] input,
textarea {{
background-color: {COLORS['bg_tertiary']} !important;
color: {COLORS['text_primary']} !important;
border: 1px solid {COLORS['border_light']} !important;
border-radius: 8px;
padding: 10px 12px !important;
font-family: "IBM Plex Mono", monospace;
transition: all 0.3s ease !important;
}}

/* Streamlit selectbox / multiselect render via BaseWeb, not a native <select> */
[data-testid="stSelectbox"] div[data-baseweb="select"] > div,
[data-testid="stMultiSelect"] div[data-baseweb="select"] > div {{
background-color: {COLORS['bg_tertiary']} !important;
color: {COLORS['text_primary']} !important;
border: 1px solid {COLORS['border_light']} !important;
border-radius: 8px !important;
transition: all 0.3s ease !important;
}}
[data-testid="stSelectbox"] div[data-baseweb="select"] > div:hover,
[data-testid="stMultiSelect"] div[data-baseweb="select"] > div:hover,
[data-testid="stTextInput"] input:hover,
[data-testid="stNumberInput"] input:hover {{
border-color: {COLORS['accent_primary']} !important;
box-shadow: 0 0 0 3px rgba(0, 217, 163, 0.1) !important;
}}
[data-testid="stTextInput"] input:focus,
[data-testid="stNumberInput"] input:focus {{
border-color: {COLORS['accent_primary']} !important;
box-shadow: 0 0 0 3px rgba(0, 217, 163, 0.2) !important;
outline: none !important;
}}

/* ========== DIVIDERS ========== */
hr {{
border: none;
height: 1px;
background: linear-gradient(90deg, transparent, {COLORS['border_light']}, transparent);
margin: 2rem 0;
}}

/* ========== CHARTS ========== */
[data-testid="stPlotlyChart"] {{
border-radius: 14px;
overflow: hidden;
border: 1px solid {COLORS['border_light']};
background: linear-gradient(145deg, rgba(30, 36, 48, 0.45), rgba(20, 27, 40, 0.65));
transition: all 0.3s ease;
padding: 6px;
}}
[data-testid="stPlotlyChart"]:hover {{
border-color: {COLORS['border_mid']};
box-shadow: 0 8px 24px rgba(0, 217, 163, 0.08);
}}

/* ========== SPINNERS ========== */
.stSpinner {{ text-align: center; }}
.stSpinner > div {{ border-color: {COLORS['accent_primary']} !important; border-right-color: transparent !important; }}

/* ========== STATUS BADGES ========== */
.status-badge {{
display: inline-block;
padding: 6px 14px;
border-radius: 999px;
font-size: 0.75rem;
font-weight: 700;
letter-spacing: 0.06em;
text-transform: uppercase;
backdrop-filter: blur(8px);
border: 1px solid;
transition: all 0.3s ease;
}}

/* ========== ANIMATIONS ========== */
@keyframes slideInUp {{
from {{ opacity: 0; transform: translateY(16px); }}
to {{ opacity: 1; transform: translateY(0); }}
}}
@keyframes fadeIn {{ from {{ opacity: 0; }} to {{ opacity: 1; }} }}
@keyframes pulseDot {{
0%, 100% {{ opacity: 1; transform: scale(1); }}
50% {{ opacity: 0.4; transform: scale(0.75); }}
}}
@keyframes pulse {{ 0%, 100% {{ opacity: 1; }} 50% {{ opacity: 0.5; }} }}

.animate-in {{ animation: slideInUp 0.6s cubic-bezier(0.4, 0, 0.2, 1) forwards; }}
.fade-in {{ animation: fadeIn 0.4s ease-out; }}

/* ========== LAYOUT ========== */
.block-container {{
padding-top: 1.75rem;
padding-bottom: 2rem;
padding-left: 2rem;
padding-right: 2rem;
max-width: 1600px;
}}

/* ========== FOOTER ========== */
.app-footer {{
background: linear-gradient(180deg, transparent, rgba(0, 0, 0, 0.25));
border-top: 1px solid {COLORS['border_light']};
margin-top: 3rem;
padding: 1.75rem 0 0.5rem 0;
text-align: center;
}}

/* ========== RESPONSIVE ========== */
@media (max-width: 768px) {{
.block-container {{ padding-left: 1rem; padding-right: 1rem; }}
[data-testid="stMetricValue"] {{ font-size: 2rem !important; }}
h1 {{ font-size: 1.75rem !important; }}
.premium-card {{ padding: 16px; }}
.section-title {{ font-size: 1.15rem; }}
}}
</style>
""", unsafe_allow_html=True)

# ==================== HELPER FUNCTIONS ====================
def get_aqi_category(aqi: int) -> AQICategory:
    """Get AQI category enum"""
    if aqi <= 50:
        return AQICategory.GOOD
    elif aqi <= 100:
        return AQICategory.MODERATE
    elif aqi <= 150:
        return AQICategory.USG
    elif aqi <= 200:
        return AQICategory.UNHEALTHY
    elif aqi <= 300:
        return AQICategory.VERY_UNHEALTHY
    else:
        return AQICategory.HAZARDOUS

def get_health_guidance(aqi: int) -> str:
    """Get health guidance based on AQI"""
    guidance = {
        'good': "Air quality is satisfactory. Air pollution poses little or no risk.",
        'moderate': "Air quality is acceptable. Sensitive individuals may experience minor symptoms.",
        'usg': "**Sensitive groups:** Reduce prolonged or heavy exertion outdoors.",
        'unhealthy': "**Everyone:** Reduce prolonged exertion outdoors. Sensitive groups should avoid outdoor activity.",
        'very_unhealthy': "**Everyone:** Avoid outdoor activities. Keep windows closed.",
        'hazardous': "**Health warning:** Emergency conditions. Remain indoors and keep activity minimal.",
    }

    category = get_aqi_category(aqi)
    key = category.name.lower()
    return guidance.get(key, guidance['good'])

# ==================== DATA FUNCTIONS ====================
@st.cache_data(ttl=15)
def fetch_dashboard_data() -> Dict[str, Any]:
    """Fetch real air quality data and map it exactly like frontend.html's adaptApiResponse."""
    import requests
    import json
    import os

    from pathlib import Path
    local_json_path = Path(__file__).resolve().parent.parent / "data" / "aqi_dashboard_data.json"
    api_url = os.environ.get("API_URL", "http://127.0.0.1:5000/api/data")

    raw_data = None

    # 1. First priority: Load fresh static JSON directly (instant, 0 ms latency, no 500 errors)
    if local_json_path.exists():
        try:
            with open(local_json_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
                if loaded and "current" in loaded and loaded.get("current"):
                    raw_data = loaded
        except Exception:
            pass

    # 2. Second priority: If local JSON not found, fetch from live API
    if not raw_data:
        try:
            res = requests.get(api_url, timeout=5)
            if res.ok:
                raw_data = res.json()
            else:
                raw_data = {"error": f"HTTP error! status: {res.status_code}"}
        except Exception as e:
            raw_data = {"error": str(e)}

    if not raw_data or "error" in raw_data or not raw_data.get("current"):
        error_msg = raw_data.get("error", "Unknown error") if raw_data else "Failed to load dashboard data"
        st.error(f"❌ Failed to load dashboard data: {error_msg}")
        st.stop()

    curr = raw_data.get("current", {})
    loc = raw_data.get("location", {})
    weather = curr.get("weather", {})
    poll = curr.get("pollutants", {})

    trend_raw = raw_data.get("trend_24h", [])
    trend_times = []
    trend_vals = []
    if trend_raw:
        trend_times = [pd.to_datetime(t["timestamp"]).strftime('%H:%M') for t in trend_raw]
        trend_vals = [int(round(t.get("aqi", 0))) for t in trend_raw]

    trend_avg = sum(trend_vals) / len(trend_vals) if trend_vals else 0
    trend_min = min(trend_vals) if trend_vals else 0
    trend_max = max(trend_vals) if trend_vals else 0

    trend_text = "—"
    if len(trend_vals) >= 2:
        first = trend_vals[0]
        last = trend_vals[-1]
        pct = (((last - first) / (first if first else 1)) * 100)
        trend_text = f"{'+' if pct > 0 else ''}{pct:.1f}% over last 24h"

    # Forecast and SHAP Map
    forecast_data = []
    shap_map = {}
    raw_fc = raw_data.get("forecast", [])

    for index, item in enumerate(raw_fc):
        rawH = str(item.get('horizon', '')).lower().strip()
        horizonKey = '24h'
        if '24' in rawH or index == 0: horizonKey = '24h'
        elif '48' in rawH or index == 1: horizonKey = '48h'
        elif '72' in rawH or index == 2: horizonKey = '72h'

        aqi = int(round(item.get('predicted_aqi', item.get('aqi', 0))))
        rmse = item.get('rmse_aqi')
        rmse_val = rmse if rmse is not None else "N/A"

        forecast_data.append({
            'day': item.get('day', f"Day {index + 1}"),
            'horizon': horizonKey,
            'predicted_aqi': aqi,
            'predicted_pm25': item.get('predicted_pm25', 0),
            'rmse': rmse_val
        })

        raw_features = item.get('features') or item.get('shap_values') or item.get('shap') or []

        topInc = "None"
        topDec = "None"
        maxIncVal = 0.0
        maxDecVal = 0.0
        formatted_features = []

        for f in raw_features:
            name = f.get('feature') or f.get('name') or f.get('label') or "Feature"
            val_raw = f.get('shap_value') or f.get('value') or f.get('val') or f.get('contribution') or 0
            val = round(float(val_raw), 4) if val_raw else 0.0

            if val > maxIncVal:
                maxIncVal = val
                topInc = f"{name} (+{val})"
            if val < maxDecVal:
                maxDecVal = val
                topDec = f"{name} ({val})"

            formatted_features.append({'name': name, 'val': val})

        shap_map[horizonKey] = {
            'pred': aqi,
            'topInc': topInc if maxIncVal > 0 else "None",
            'topDec': topDec if maxDecVal < 0 else "None",
            'features': formatted_features
        }

    return {
        'location': loc.get('name', 'Lahore'),
        'region': f"{loc.get('region', '')}, {loc.get('country', '')}",
        'updated_full': raw_data.get('updated_at', ''),
        'current_aqi': int(round(curr.get("aqi", 0))),
        'pollutants': [
            {'name': 'PM2.5', 'val': poll.get('pm25', '—'), 'unit': 'µg/m³', 'trend': '—'},
            {'name': 'PM10', 'val': poll.get('pm10', '—'), 'unit': 'µg/m³', 'trend': '—'},
            {'name': 'O₃', 'val': poll.get('ozone', '—'), 'unit': 'µg/m³', 'trend': '—'},
            {'name': 'NO₂', 'val': poll.get('no2', '—'), 'unit': 'µg/m³', 'trend': '—'},
            {'name': 'SO₂', 'val': poll.get('so2', '—'), 'unit': 'µg/m³', 'trend': '—'},
            {'name': 'CO', 'val': poll.get('co', '—'), 'unit': 'µg/m³', 'trend': '—'},
        ],
        'weather': [
            {'label': 'Temperature', 'val': f"{weather.get('temperature', '—')}°C" if weather.get('temperature') is not None else "—", 'icon': '🌡️', 'detail': ''},
            {'label': 'Humidity', 'val': f"{weather.get('humidity', '—')}%" if weather.get('humidity') is not None else "—", 'icon': '💧', 'detail': ''},
            {'label': 'Pressure', 'val': f"{weather.get('pressure', '—')} hPa" if weather.get('pressure') is not None else "—", 'icon': '🔽', 'detail': ''},
        ],
        'trend_labels': trend_times,
        'trend_values': trend_vals,
        'trend_avg': int(trend_avg),
        'trend_min': trend_min,
        'trend_max': trend_max,
        'trend_text': trend_text,
        'forecast': forecast_data,
        'forecast_hourly': raw_data.get("forecast_hourly", []),
        'shap_map': shap_map
    }

# ==================== VISUALIZATION FUNCTIONS ====================
def create_advanced_gauge(aqi: int) -> go.Figure:
    """Create sophisticated AQI gauge"""
    category = get_aqi_category(aqi).value

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=aqi,
        title={'text': "Current AQI", 'font': {'size': 16, 'color': COLORS['text_tertiary']}},
        number={'font': {'size': 46, 'color': category['color']}},
        delta={'reference': 60, 'suffix': '', 'position': 'bottom', 'font': {'size': 13}},
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={
            'axis': {'range': [0, 500], 'tickfont': {'size': 10, 'color': COLORS['text_tertiary']}},
            'bar': {'color': category['color'], 'thickness': 0.25},
            'bgcolor': 'rgba(255,255,255,0.02)',
            'borderwidth': 0,
            'steps': [
                {'range': [0, 50], 'color': 'rgba(0, 217, 163, 0.15)'},
                {'range': [50, 100], 'color': 'rgba(255, 184, 77, 0.15)'},
                {'range': [100, 150], 'color': 'rgba(255, 149, 0, 0.15)'},
                {'range': [150, 200], 'color': 'rgba(255, 82, 82, 0.15)'},
                {'range': [200, 300], 'color': 'rgba(217, 70, 239, 0.15)'},
                {'range': [300, 500], 'color': 'rgba(139, 21, 56, 0.15)'},
            ],
            'threshold': {
                'line': {'color': category['color'], 'width': 2},
                'thickness': 0.75,
                'value': aqi
            }
        }
    ))

    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font={'color': COLORS['text_secondary'], 'family': 'Inter'},
        height=280,
        margin={'t': 70, 'b': 10, 'l': 20, 'r': 20},
        showlegend=False,
    )

    return fig

def create_trend_chart(labels: list, values: list) -> go.Figure:
    """Create 24-hour trend with advanced styling"""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=labels,
        y=values,
        mode='lines+markers',
        name='AQI',
        line=dict(
            color=COLORS['accent_primary'],
            width=3,
            shape='spline'
        ),
        marker=dict(
            size=6,
            color=COLORS['accent_primary'],
            line=dict(width=2, color=COLORS['bg_tertiary']),
            opacity=0.9
        ),
        fill='tozeroy',
        fillcolor='rgba(0, 217, 163, 0.1)',
        hovertemplate='<b style="color:#00D9A3">%{y}</b><br><sub style="color:#8A95A8">%{x}</sub><extra></extra>',
    ))

    # Add average line
    avg = np.mean(values) if len(values) else 0
    fig.add_hline(
        y=avg,
        line_dash="dash",
        line_color=COLORS['accent_secondary'],
        annotation_text=f"Avg: {int(avg)}",
        annotation_position="right",
        opacity=0.5
    )

    fig.update_layout(
        xaxis_title="Time",
        yaxis_title="AQI Value",
        hovermode='x unified',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font={'color': COLORS['text_secondary'], 'family': 'Inter'},
        xaxis={
            'showgrid': False,
            'zeroline': False,
            'fixedrange': True,
        },
        yaxis={
            'showgrid': False,
            'zeroline': False,
            'fixedrange': True,
        },
        height=340,
        margin={'t': 20, 'b': 50, 'l': 60, 'r': 80},
        hoverlabel={'bgcolor': COLORS['bg_tertiary'], 'bordercolor': COLORS['border_mid']},
    )

    return fig

def create_forecast_chart(data: Dict) -> go.Figure:
    """Create 72-hour forecast prediction chart with formatted timestamps and PM2.5 tooltips."""
    forecast_hourly = data.get('forecast_hourly', [])
    if not forecast_hourly:
        return go.Figure()

    x_labels = []
    pm25_vals = []
    values = []

    for i, f in enumerate(forecast_hourly):
        values.append(f.get('predicted_aqi', f.get('aqi', 0)))
        pm25_vals.append(f.get('predicted_pm25', 0))
        ts = f.get('timestamp')
        if ts:
            try:
                dt = pd.to_datetime(ts)
                x_labels.append(dt.strftime('%a %H:%M'))
            except Exception:
                x_labels.append(f"+{i+1}h")
        else:
            x_labels.append(f"+{i+1}h")

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=x_labels,
        y=values,
        customdata=pm25_vals,
        mode='lines+markers',
        name='Predicted AQI',
        line=dict(
            color=COLORS['accent_tertiary'],
            width=3,
            shape='spline'
        ),
        marker=dict(
            size=4,
            color=COLORS['accent_tertiary'],
            opacity=0.85
        ),
        fill='tozeroy',
        fillcolor='rgba(217, 70, 239, 0.12)',
        hovertemplate='<b>%{x}</b><br>Predicted AQI: <b>%{y:.1f}</b><br>PM2.5: <b>%{customdata:.1f} µg/m³</b><extra></extra>',
    ))

    fig.update_layout(
        xaxis_title="Forecast Time",
        yaxis_title="Predicted AQI",
        hovermode='x unified',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font={'color': COLORS['text_secondary'], 'family': 'Inter'},
        xaxis={
            'showgrid': False,
            'zeroline': False,
            'nticks': 12,
        },
        yaxis={
            'showgrid': True,
            'gridcolor': 'rgba(255,255,255,0.05)',
            'zeroline': False,
        },
        height=340,
        margin={'t': 20, 'b': 50, 'l': 60, 'r': 80},
        hoverlabel={'bgcolor': COLORS['bg_tertiary'], 'bordercolor': COLORS['border_mid']},
    )

    return fig

def create_shap_chart(features: List[Dict]) -> go.Figure:
    """Create a horizontal bar chart for SHAP values"""
    if not features:
        return go.Figure()
        
    # Sort by absolute value ascending for horizontal bar chart (so biggest is at top)
    sorted_features = sorted(features, key=lambda x: abs(x['val']))
    
    names = [f['name'] for f in sorted_features]
    vals = [f['val'] for f in sorted_features]
    
    # Colors: Error (red) for positive impact (increases AQI), Success (green) for negative
    colors = [COLORS['error'] if v > 0 else COLORS['success'] for v in vals]
    text_labels = [f"+{v:.2f}" if v > 0 else f"{v:.2f}" for v in vals]
    
    fig = go.Figure(go.Bar(
        x=vals,
        y=names,
        orientation='h',
        marker_color=colors,
        text=text_labels,
        textposition='outside',
        textfont={'color': COLORS['text_primary'], 'family': 'Inter'},
        hovertemplate='<b>%{y}</b><br>Impact: %{text}<extra></extra>'
    ))
    
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font={'color': COLORS['text_secondary'], 'family': 'Inter'},
        margin={'t': 20, 'b': 20, 'l': 120, 'r': 40},
        xaxis={
            'showgrid': True, 
            'gridcolor': COLORS['border_light'], 
            'zeroline': True, 
            'zerolinecolor': COLORS['border_mid'],
            'title': 'Impact on Predicted AQI',
            'fixedrange': True
        },
        yaxis={
            'showgrid': False,
            'fixedrange': True
        },
        height=380,
        showlegend=False,
        hoverlabel={'bgcolor': COLORS['bg_tertiary'], 'bordercolor': COLORS['border_mid']}
    )
    return fig

# ==================== RENDER HELPERS ====================
def render_section_header(icon: str, title: str, subtitle: str = None, right_html: str = ""):
    """Consistent section header with icon chip + title, used across all sections."""
    subtitle_html = f'<div class="section-subtitle">{subtitle}</div>' if subtitle else ""
    html = (
        '<div class="section-header">'
        '<div class="section-header-left">'
        f'<div class="section-icon">{icon}</div>'
        '<div>'
        f'<div class="section-title">{title}</div>'
        f'{subtitle_html}'
        '</div>'
        '</div>'
        f'{right_html}'
        '</div>'
    )
    st.markdown(html, unsafe_allow_html=True)

def render_stat_card(label: str, value: str, metric: str, change: str = None, accent: str = None):
    """Render premium stat card with glowing elements"""
    accent_color = accent or COLORS['accent_primary']
    change_html = (
        f'<span style="background: {accent_color}22; padding: 4px 8px; border-radius: 6px; '
        f'font-size: 11px; color: {accent_color}; font-weight: 700; border: 1px solid {accent_color}44;">{change}</span>'
        if change else ''
    )
    html_content = (
        '<div class="premium-card" style="padding: 24px; min-height: 120px; display: flex; flex-direction: column; justify-content: space-between;">'
        '<div style="display: flex; justify-content: space-between; align-items: flex-start;">'
        f'<span style="{TYPOGRAPHY["label"]}; color: {COLORS["text_tertiary"]}; text-transform: uppercase; letter-spacing: 0.1em; font-size: 0.75rem;">{label}</span>'
        f'{change_html}'
        '</div>'
        '<div style="margin-top: 12px;">'
        f'<div style="font-family: \'Inter\', sans-serif; font-size: 2.2rem; font-weight: 800; color: #FFFFFF; text-shadow: 0 0 20px {accent_color}44; line-height: 1.1;">{value}</div>'
        f'<div style="{TYPOGRAPHY["body_sm"]}; color: {COLORS["text_secondary"]}; margin-top: 4px; font-weight: 500;">{metric}</div>'
        '</div>'
        f'<div style="position: absolute; bottom: 0; left: 0; width: 100%; height: 4px; background: linear-gradient(90deg, {accent_color}, transparent); opacity: 0.85;"></div>'
        '</div>'
    )
    st.markdown(html_content, unsafe_allow_html=True)

def render_pollutant_grid(pollutants: List[Dict]):
    """Render pollutants with a sleek modern grid"""
    cols = st.columns(6)
    for idx, pollutant in enumerate(pollutants):
        with cols[idx]:
            html = (
                f'<div class="premium-card" style="text-align: center; padding: 20px 12px; '
                f'border-top: 3px solid {COLORS["accent_primary"]}; animation-delay: {idx * 0.05}s;">'
                f'<div style="{TYPOGRAPHY["label"]}; color: {COLORS["text_tertiary"]}; margin-bottom: 12px; letter-spacing: 1px;">{pollutant["name"]}</div>'
                f'<div style="font-family: \'Inter\', sans-serif; font-size: 1.8rem; font-weight: 800; color: #FFFFFF; margin-bottom: 2px;">{pollutant["val"]}</div>'
                f'<div style="{TYPOGRAPHY["body_sm"]}; color: {COLORS["text_secondary"]}; opacity: 0.7;">{pollutant["unit"]}</div>'
                '</div>'
            )
            st.markdown(html, unsafe_allow_html=True)

def render_forecast_card(forecast: Dict, index: int):
    """Render individual forecast card with confidence"""
    aqi = int(forecast['predicted_aqi'])
    category = get_aqi_category(aqi).value
    confidence = forecast.get('confidence', 85)

    html = (
        f'<div class="premium-card" style="border-top: 4px solid {category["color"]}; position: relative; '
        f'overflow: hidden; padding: 24px; display: flex; flex-direction: column; gap: 16px; animation-delay: {index * 0.08}s;">'
        '<div style="display: flex; justify-content: space-between; align-items: center;">'
        '<div>'
        f'<div style="{TYPOGRAPHY["label"]}; color: {COLORS["text_tertiary"]}; letter-spacing: 1px; text-transform: uppercase;">{forecast["day"]}</div>'
        f'<div style="{TYPOGRAPHY["heading_md"]}; color: #FFFFFF; margin-top: 4px; font-weight: 700;">{forecast["horizon"]}</div>'
        '</div>'
        '<div style="text-align: right;">'
        f'<div style="font-size: 10px; color: {COLORS["text_tertiary"]}; margin-bottom: 2px; text-transform: uppercase; letter-spacing: 0.5px;">Confidence</div>'
        f'<div style="font-size: 1.1rem; font-weight: 800; color: {COLORS["accent_secondary"]}; text-shadow: 0 0 10px {COLORS["accent_secondary"]}33;">{confidence}%</div>'
        '</div>'
        '</div>'
        f'<div style="background: linear-gradient(135deg, rgba(255, 255, 255, 0.03), rgba(255, 255, 255, 0.01)); '
        f'border-radius: 12px; padding: 20px; text-align: center; border: 1px solid {category["color"]}44; '
        'box-shadow: inset 0 0 20px rgba(0, 0, 0, 0.2);">'
        f'<div style="color: {COLORS["text_tertiary"]}; font-size: 10px; margin-bottom: 8px; text-transform: uppercase; letter-spacing: 0.1em;">Predicted AQI</div>'
        f'<div style="font-family: \'Inter\', sans-serif; font-size: 3rem; font-weight: 800; color: {category["color"]}; '
        f'text-shadow: 0 0 25px {category["color"]}66; line-height: 1;">{aqi}</div>'
        f'<div style="color: {category["color"]}; font-size: 13px; font-weight: 700; margin-top: 8px; letter-spacing: 0.5px;">{category["label"]}</div>'
        '</div>'
        '<div style="border-top: 1px solid rgba(255, 255, 255, 0.05); padding-top: 12px; display: flex; justify-content: space-between; font-size: 11px;">'
        f'<span style="color: {COLORS["text_tertiary"]};">Model RMSE</span>'
        f'<span style="color: {COLORS["text_primary"]}; font-weight: 700; font-family: \'IBM Plex Mono\', monospace;">±{forecast.get("rmse", "N/A")}</span>'
        '</div>'
        '</div>'
    )
    st.markdown(html, unsafe_allow_html=True)

def render_weather_grid(weather: List[Dict]):
    """Render weather info in advanced grid"""
    cols = st.columns(len(weather))
    for idx, item in enumerate(weather):
        with cols[idx]:
            html = (
                '<div class="premium-card" style="text-align: center; padding: 24px; display: flex; '
                f'flex-direction: column; align-items: center; gap: 8px; animation-delay: {idx * 0.06}s;">'
                f'<div style="font-size: 28px; background: {COLORS["accent_secondary"]}11; padding: 12px; '
                f'border-radius: 50%; border: 1px solid {COLORS["accent_secondary"]}33;">{item.get("icon", "•")}</div>'
                f'<div style="{TYPOGRAPHY["label"]}; color: {COLORS["text_tertiary"]}; text-transform: uppercase; letter-spacing: 1px; margin-top: 8px;">{item["label"]}</div>'
                f'<div style="font-family: \'Inter\', sans-serif; font-size: 1.5rem; font-weight: 700; color: #FFFFFF;">{item["val"]}</div>'
                '</div>'
            )
            st.markdown(html, unsafe_allow_html=True)

def render_header():
    """Render a clean navigation bar mimicking the frontend."""
    # We use columns to layout the logo, links, and refresh button horizontally
    col1, col2, col3, col4, col5, col6 = st.columns([3, 1.5, 1.5, 1.5, 1, 2])
    
    with col1:
        st.markdown('<div style="font-size: 1.5rem; font-weight: 800; color: #FFFFFF; padding-top: 4px; display: flex; align-items: center; gap: 8px;"><span>🔮</span> Pearls AQI</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div style="padding-top: 10px; text-align: center;"><a href="#dashboard" style="color: #00D9A3; text-decoration: none; font-weight: 600; border-bottom: 2px solid #00D9A3; padding-bottom: 4px;">Dashboard</a></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div style="padding-top: 10px; text-align: center;"><a href="#forecast" style="color: #8A95A8; text-decoration: none; font-weight: 600; transition: color 0.2s;">Forecast</a></div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div style="padding-top: 10px; text-align: center;"><a href="#analysis" style="color: #8A95A8; text-decoration: none; font-weight: 600; transition: color 0.2s;">Analysis</a></div>', unsafe_allow_html=True)
    with col6:
        if st.button("⟳ Refresh", type="primary", use_container_width=True):
            fetch_dashboard_data.clear()
            st.session_state.data = None
            st.rerun()

def render_guidance_banner(text: str, accent: str):
    """Theme-matched replacement for st.info() so the banner doesn't clash with the dark UI."""
    html = (
        f'<div class="guidance-banner" style="--banner-accent: {accent};">'
        '<span class="guidance-icon">💡</span>'
        f'<span>{text}</span>'
        '</div>'
    )
    st.markdown(html, unsafe_allow_html=True)

# ==================== MAIN APP ====================
def main():
    # Session state
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'active_forecast' not in st.session_state:
        st.session_state.active_forecast = 0

    # Header
    render_header()
    st.divider()

    # Load data
    if st.session_state.data is None:
        with st.spinner("Analyzing air quality data..."):
            st.session_state.data = fetch_dashboard_data()

    data = st.session_state.data
    category = get_aqi_category(data['current_aqi']).value

    # ==================== HERO SECTION ====================
    st.markdown('<div id="dashboard"></div>', unsafe_allow_html=True)
    # Using vertical_alignment='center' nicely vertically aligns the text metrics and the gauge
    col1, col2 = st.columns([1.5, 1], gap="large", vertical_alignment="center")

    with col1:
        st.markdown(f'<div style="font-size: 1.15rem; color: {COLORS["text_secondary"]}; font-weight: 500; margin-bottom: 8px;">Real-time Air Quality</div>', unsafe_allow_html=True)
        st.markdown(f'<div style="font-size: 5rem; font-weight: 800; font-family: \'Inter\', sans-serif; color: {category["color"]}; line-height: 1; letter-spacing: -2px;">{data["current_aqi"]}</div>', unsafe_allow_html=True)

        html = (
            '<div style="margin-bottom: 16px;">'
            f'<span class="status-badge" style="background: {category["bg"]}; color: {category["color"]}; border-color: {category["color"]}33;">'
            f'{category["label"].upper()}'
            '</span>'
            '</div>'
        )
        st.markdown(html, unsafe_allow_html=True)

        html = f'<div style="{TYPOGRAPHY["display_lg"]}; color: {COLORS["text_primary"]}; margin-bottom: 8px;">Air Quality is {category["label"]}</div>'
        st.markdown(html, unsafe_allow_html=True)

        html = f'<div style="{TYPOGRAPHY["body_md"]}; color: {COLORS["accent_secondary"]}; margin-bottom: 16px; font-weight: 500;">📈 {data["trend_text"]}</div>'
        st.markdown(html, unsafe_allow_html=True)

        render_guidance_banner(get_health_guidance(data['current_aqi']), category['color'])

        # Stats grid
        stat_cols = st.columns(4, gap="small")
        with stat_cols[0]:
            render_stat_card("Current", str(data['current_aqi']), "AQI", accent=category['color'])
        with stat_cols[1]:
            render_stat_card("24h Avg", str(data['trend_avg']), "AQI")
        with stat_cols[2]:
            render_stat_card("24h Min", str(data['trend_min']), "AQI")
        with stat_cols[3]:
            render_stat_card("24h Max", str(data['trend_max']), "AQI")

    with col2:
        st.plotly_chart(create_advanced_gauge(data['current_aqi']), width="content", config={'displayModeBar': False})

    st.divider()

    # ==================== POLLUTANTS SECTION ====================
    render_section_header("🧪", "Pollutant Levels", "Concentrations across primary tracked pollutants")
    render_pollutant_grid(data['pollutants'])

    st.divider()

    # ==================== WEATHER SECTION ====================
    render_section_header("🌤️", "Weather Conditions", "Meteorological inputs feeding the forecast model")
    render_weather_grid(data['weather'])

    st.divider()

    # ==================== TRENDS SECTION ====================
    render_section_header("📊", "24-Hour AQI Trend", "Observed AQI over the past day")
    st.plotly_chart(
        create_trend_chart(data['trend_labels'], data['trend_values']),
        width="stretch",
        config={'displayModeBar': False}
    )

    st.divider()

    # ==================== FORECAST SECTION ====================
    st.markdown('<div id="forecast"></div>', unsafe_allow_html=True)
    render_section_header("🔮", "72-Hour Forecast", "Model predictions for the next three horizons")

    forecast_cols = st.columns(3, gap="medium")
    for idx, forecast in enumerate(data['forecast']):
        with forecast_cols[idx]:
            render_forecast_card(forecast, idx)

    st.divider()

    # ==================== PREDICTIONS CHART ====================
    render_section_header("📈", "72-Hour Hourly AQI Projection", "Hour-by-hour projected AQI & PM2.5 levels")
    st.plotly_chart(
        create_forecast_chart(data),
        width="stretch",
        config={'displayModeBar': False}
    )

    st.divider()

    # ==================== SHAP ANALYSIS ====================
    st.markdown('<div id="analysis"></div>', unsafe_allow_html=True)
    render_section_header("🧠", "Feature Impact Analysis", "What is driving the predictions")

    shap_map = data.get('shap_map', {})
    
    if shap_map:
        # Add CSS for tabs to look like outlined buttons
        st.markdown("""
        <style>
        .stTabs [data-baseweb="tab-list"] {
            gap: 12px;
            background-color: transparent;
        }
        .stTabs [data-baseweb="tab"] {
            height: 42px;
            background-color: rgba(255, 255, 255, 0.03);
            border: 1px solid rgba(255, 255, 255, 0.1) !important;
            border-radius: 8px !important;
            color: #8A95A8;
            font-family: 'Inter', sans-serif;
            font-weight: 600;
            padding: 0 20px;
            transition: all 0.2s;
        }
       
        .stTabs [aria-selected="true"] {
            color: #00D9A3 !important;
        }
        </style>""", unsafe_allow_html=True)
        
        tab1, tab2, tab3 = st.tabs(["24h Horizon", "48h Horizon", "72h Horizon"])
        
        with tab1:
            features = shap_map.get('24h', {}).get('features', [])
            if features:
                st.plotly_chart(create_shap_chart(features), width="stretch", config={'displayModeBar': False})
            else:
                render_guidance_banner("Feature impact data is not available for this horizon.", COLORS['text_tertiary'])
        
        with tab2:
            features = shap_map.get('48h', {}).get('features', [])
            if features:
                st.plotly_chart(create_shap_chart(features), width="stretch", config={'displayModeBar': False})
            else:
                render_guidance_banner("Feature impact data is not available for this horizon.", COLORS['text_tertiary'])
                
        with tab3:
            features = shap_map.get('72h', {}).get('features', [])
            if features:
                st.plotly_chart(create_shap_chart(features), width="stretch", config={'displayModeBar': False})
            else:
                render_guidance_banner("Feature impact data is not available for this horizon.", COLORS['text_tertiary'])
    else:
        render_guidance_banner("Feature impact data is not currently available.", COLORS['text_tertiary'])

    st.divider()

    # ==================== FOOTER ====================
    html = (
        '<div class="app-footer">'
        f'<p style="margin: 0 0 4px 0; color: {COLORS["text_tertiary"]}; font-size: 13px;">🔮 Pearls AQI Predictor · AI-Powered Air Quality Intelligence</p>'
        f'<p style="margin: 0; font-size: 11px; color: {COLORS["text_tertiary"]};">Last updated: {data["updated_full"]} · Lahore, Punjab, Pakistan</p>'
        '</div>'
    )
    st.markdown(html, unsafe_allow_html=True)

if __name__ == "__main__":
    main()