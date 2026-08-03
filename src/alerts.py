"""Hazardous AQI level alerts."""
from src.config import AQI_THRESHOLDS
from src.features.engineering import pm25_to_aqi_category


def check_aqi_alert(pm25: float) -> dict | None:
    """
    Return an alert dict if PM2.5 exceeds unhealthy thresholds, else None.
    """
    category = pm25_to_aqi_category(pm25)

    if pm25 <= AQI_THRESHOLDS["moderate"]:
        return None

    severity = "warning"
    if pm25 > AQI_THRESHOLDS["hazardous"]:
        severity = "critical"
    elif pm25 > AQI_THRESHOLDS["very_unhealthy"]:
        severity = "severe"
    elif pm25 > AQI_THRESHOLDS["unhealthy"]:
        severity = "high"

    return {
        "severity": severity,
        "category": category,
        "pm25": pm25,
        "message": f"AQI alert: {category} conditions detected (PM2.5 = {pm25:.1f} µg/m³).",
    }


def check_forecast_alerts(forecast_df) -> list[dict]:
    """Scan a forecast dataframe for hazardous periods."""
    alerts = []
    for _, row in forecast_df.iterrows():
        alert = check_aqi_alert(row["predicted_pm25"])
        if alert:
            alert["timestamp"] = str(row["timestamp"])
            alerts.append(alert)
    return alerts
