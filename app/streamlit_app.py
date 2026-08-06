"""Streamlit native dashboard that mimics the HTML frontend UI/UX."""
import streamlit as st
import requests
import json
import pandas as pd
from pathlib import Path

st.set_page_config(
    page_title="Pearls AQI Predictor",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Exact UI/UX Custom CSS (Glassmorphism & Colors) ---
st.markdown("""
<style>
    /* Main Background */
    .stApp {
        background-color: #12161C;
        background-image: 
            radial-gradient(ellipse 900px 500px at 15% -10%, rgba(232, 163, 61, 0.07), transparent 60%),
            radial-gradient(ellipse 700px 500px at 90% 10%, rgba(161, 92, 196, 0.05), transparent 60%);
        color: #EDEFF2;
        font-family: 'Inter', sans-serif;
    }
    
    /* Hide Streamlit top bar and footer */
    header {visibility: hidden;}
    footer {visibility: hidden;}

    /* Card Styling */
    div[data-testid="stVerticalBlock"] > div > div[data-testid="stVerticalBlock"] {
        background: #1B212B;
        border: 1px solid #2A3140;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.15);
    }
    
    /* Titles */
    h1, h2, h3 {
        font-family: 'Space Grotesk', sans-serif;
        color: #EDEFF2;
    }
    
    /* Metrics */
    div[data-testid="stMetricValue"] {
        font-family: 'IBM Plex Mono', monospace;
        color: #EDEFF2;
    }
    div[data-testid="stMetricLabel"] {
        color: #8B93A1;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        font-size: 0.8rem;
    }
    
    /* Hero Card Specific Injection */
    .hero-card {
        text-align: center;
        padding: 40px 20px;
        border-radius: 16px;
        margin-bottom: 24px;
        border: 1px solid rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
    }
    .hero-aqi {
        font-size: 5.5rem;
        font-weight: 800;
        margin: 15px 0;
        font-family: 'IBM Plex Mono', monospace;
    }
    .hero-category {
        font-size: 1.8rem;
        font-weight: 600;
        font-family: 'Space Grotesk', sans-serif;
    }
    .hero-guidance {
        font-size: 1.1rem;
        opacity: 0.85;
        max-width: 600px;
        margin: 15px auto 0;
        line-height: 1.5;
    }
    .trend-chip {
        display: inline-block;
        padding: 6px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# --- Helpers ---
def get_category_color(aqi):
    if aqi <= 50: return "#4CAF7D"  # Good
    if aqi <= 100: return "#D9C24C" # Moderate
    if aqi <= 150: return "#E8A33D" # USG
    if aqi <= 200: return "#E2673B" # Unhealthy
    if aqi <= 300: return "#A15CC4" # Very Unhealthy
    return "#8B2E3E" # Hazardous

def get_health_guidance(aqi):
    if aqi <= 50: return "Air quality is considered satisfactory, and air pollution poses little or no risk."
    if aqi <= 100: return "Air quality is acceptable; however, for some pollutants there may be a moderate health concern for a very small number of people who are unusually sensitive to air pollution."
    if aqi <= 150: return "Members of sensitive groups may experience health effects. The general public is not likely to be affected."
    if aqi <= 200: return "Everyone may begin to experience health effects; members of sensitive groups may experience more serious health effects."
    if aqi <= 300: return "Health warnings of emergency conditions. The entire population is more likely to be affected."
    return "Health alert: everyone may experience more serious health effects."

@st.cache_data(ttl=60)
def load_data():
    """Fetch from local API, fallback to local JSON file."""
    try:
        # Try local Flask server
        res = requests.get("http://127.0.0.1:5000/api/data", timeout=2)
        if res.ok:
            return res.json()
    except Exception:
        pass
    
    # Fallback to local JSON
    json_path = Path(__file__).parent.parent / "data" / "aqi_dashboard_data.json"
    if json_path.exists():
        with open(json_path) as f:
            return json.load(f)
    return None

# --- Main App ---
data = load_data()

if not data:
    st.error("❌ No data available. Please ensure `app/flask_api.py` is running or `data/aqi_dashboard_data.json` exists.")
    st.stop()

if "error" in data:
    st.error(f"❌ Backend Error: {data['error']}")
    st.stop()

curr = data.get("current", {})
aqi_now = curr.get("aqi", 0)
cat_label = curr.get("category", "Unknown")
cat_color = get_category_color(aqi_now)
guidance = get_health_guidance(aqi_now)
updated = data.get("updated_at", "")
note = data.get("data_note", "")

# Calculate Trend Chip
trend_24 = data.get("trend_24h", [])
trend_chip_html = ""
if trend_24 and len(trend_24) > 0:
    past_24_aqi = trend_24[0].get("aqi", aqi_now)
    pct_change = ((aqi_now - past_24_aqi) / past_24_aqi) * 100 if past_24_aqi > 0 else 0
    if pct_change > 0:
        trend_chip_html = f'<div class="trend-chip" style="background: rgba(226, 103, 59, 0.15); color: #E2673B;">+{pct_change:.1f}% over last 24h</div>'
    else:
        trend_chip_html = f'<div class="trend-chip" style="background: rgba(76, 175, 125, 0.15); color: #4CAF7D;">{pct_change:.1f}% over last 24h</div>'

# --- Header ---
st.markdown(f"""
<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px;">
    <div style="display: flex; align-items: center; gap: 12px;">
        <div style="width: 38px; height: 38px; border-radius: 10px; background: linear-gradient(135deg, #E8A33D, #C97A2B); display: flex; align-items: center; justify-content: center; font-weight: bold; color: #12161C;">P</div>
        <div>
            <h2 style="margin: 0; font-size: 1.2rem;">Pearls AQI</h2>
            <div style="font-size: 0.8rem; color: #8B93A1;">{note}</div>
        </div>
    </div>
    <div style="text-align: right;">
        <div style="font-weight: 500;">{data.get('location', {}).get('name', 'Lahore, Pakistan')}</div>
        <div style="font-size: 0.75rem; color: #8B93A1;">Updated: {pd.to_datetime(updated).strftime('%Y-%m-%d %H:%M')}</div>
    </div>
</div>
""", unsafe_allow_html=True)

# --- Hero Section ---
st.markdown(f"""
<div class="hero-card" style="background: linear-gradient(145deg, {cat_color}1A, {cat_color}05);">
    {trend_chip_html}
    <div class="hero-category" style="color: {cat_color};">Air Quality is {cat_label}</div>
    <div class="hero-aqi" style="color: {cat_color};">{int(aqi_now)}</div>
    <div class="hero-guidance">{guidance}</div>
</div>
""", unsafe_allow_html=True)

# --- Pollutants & Weather ---
col1, col2 = st.columns([7, 5])

with col1:
    st.subheader("Pollutants")
    p1, p2, p3 = st.columns(3)
    pol = curr.get("pollutants", {})
    p1.metric("PM2.5", f"{pol.get('pm25', '—')} µg/m³")
    p2.metric("PM10", f"{pol.get('pm10', '—')} µg/m³")
    p3.metric("O3", f"{pol.get('ozone', '—')} µg/m³")
    
    st.write("") # Spacer
    p4, p5, p6 = st.columns(3)
    p4.metric("NO2", f"{pol.get('no2', '—')} µg/m³")
    p5.metric("SO2", f"{pol.get('so2', '—')} µg/m³")
    p6.metric("CO", f"{pol.get('co', '—')} µg/m³")

with col2:
    st.subheader("Weather")
    w1, w2 = st.columns(2)
    wea = curr.get("weather", {})
    w1.metric("Temperature", f"{wea.get('temperature', '—')}°C")
    w2.metric("Humidity", f"{wea.get('humidity', '—')}%")
    
    st.write("") # Spacer
    w3, w4 = st.columns(2)
    w3.metric("Pressure", f"{wea.get('pressure', '—')} hPa")
    w4.metric("Status", "Clear") # Placeholder

# --- Trends and Forecast Summaries ---
st.markdown("<br>", unsafe_allow_html=True)
c1, c2 = st.columns([5, 7])

with c1:
    st.subheader("24-Hour Trend")
    trend = data.get("trend_24h", [])
    if trend:
        vals = [t["aqi"] for t in trend]
        avg_t = sum(vals)/len(vals)
        t1, t2, t3 = st.columns(3)
        t1.metric("24h Avg", int(avg_t))
        t2.metric("24h Min", int(min(vals)))
        t3.metric("24h Max", int(max(vals)))
    else:
        st.info("No trend data available.")

with c2:
    st.subheader("72-Hour Forecast Checkpoints")
    fc = data.get("forecast", [])
    f1, f2, f3 = st.columns(3)
    for i, (col, lbl) in enumerate(zip([f1, f2, f3], ["Day 1", "Day 2", "Day 3"])):
        if i < len(fc):
            f_item = fc[i]
            col.metric(f_item.get("horizon", lbl) + f" ({lbl.replace(' ', '')})", 
                       int(f_item.get("predicted_aqi", 0)), 
                       delta=f"±{f_item.get('rmse_aqi', 0)} RMSE", delta_color="off")

# --- Line Chart ---
st.markdown("<br>", unsafe_allow_html=True)
st.subheader("Predicted AQI Trend (Hourly)")
hourly = data.get("forecast_hourly", [])
if hourly:
    df_fc = pd.DataFrame(hourly)
    if "timestamp" in df_fc.columns and "predicted_aqi" in df_fc.columns:
        df_fc["timestamp"] = pd.to_datetime(df_fc["timestamp"])
        df_fc = df_fc.set_index("timestamp")
        st.line_chart(df_fc[["predicted_aqi"]], height=300, color="#E8A33D")
    else:
        st.info("Hourly forecast format unrecognized.")
else:
    st.info("No hourly forecast available.")

# --- SHAP ---
st.markdown("<br>", unsafe_allow_html=True)
st.subheader("SHAP Feature Importance")
if fc:
    tabs = st.tabs([item.get("horizon", f"Day {i+1}") for i, item in enumerate(fc)])
    for i, tab in enumerate(tabs):
        with tab:
            feats = fc[i].get("features", [])
            if feats:
                df_shap = pd.DataFrame(feats)
                if not df_shap.empty:
                    df_shap["val_abs"] = df_shap["val"].abs()
                    df_shap = df_shap.sort_values("val_abs", ascending=False).head(8)
                    st.bar_chart(df_shap.set_index("name")["val"], height=350)
            else:
                st.info("No SHAP values available for this horizon.")
else:
    st.info("No SHAP data available.")
