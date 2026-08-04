"""Hazardous AQI level alerts."""
from src.config import AQI_THRESHOLDS
from src.features.engineering import aqi_category


def check_aqi_alert(aqi: float) -> dict | None:
    """
    Return an alert dict if AQI exceeds unhealthy thresholds, else None.
    """
    category = aqi_category(aqi)

    if aqi <= AQI_THRESHOLDS["moderate"]:
        return None

    severity = "warning"
    if aqi > AQI_THRESHOLDS["hazardous"]:
        severity = "critical"
    elif aqi > AQI_THRESHOLDS["very_unhealthy"]:
        severity = "severe"
    elif aqi > AQI_THRESHOLDS["unhealthy"]:
        severity = "high"

    return {
        "severity": severity,
        "category": category,
        "pm25": aqi,
        "message": f"AQI alert: {category} conditions detected (AQI = {aqi:.1f}).",
    }


def check_forecast_alerts(forecast_df) -> list[dict]:
    """Scan a forecast dataframe for hazardous periods."""
    alerts = []
    for _, row in forecast_df.iterrows():
        alert = check_aqi_alert(row["predicted_aqi"])
        if alert:
            alert["timestamp"] = str(row["timestamp"])
            alerts.append(alert)
    return alerts
