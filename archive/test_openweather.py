import os
import sys
import requests
from dotenv import load_dotenv

# Enable UTF-8 output on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Load environment variables
load_dotenv()

# Read API Key
API_KEY = os.getenv("OPEN_WEATHER_API")

if not API_KEY:
    raise ValueError(
        "OPEN_WEATHER_API was not found.\n"
        "Make sure your .env file contains:\n\n"
        "OPEN_WEATHER_API=YOUR_API_KEY"
    )

# Lahore Coordinates
LATITUDE = 31.5204
LONGITUDE = 74.3587

# API URLs
WEATHER_URL = "https://api.openweathermap.org/data/2.5/weather"
AIR_URL = "https://api.openweathermap.org/data/2.5/air_pollution"

# AQI Descriptions
AQI_LEVELS = {
    1: "Good",
    2: "Fair",
    3: "Moderate",
    4: "Poor",
    5: "Very Poor",
}

# Common parameters
params = {
    "lat": LATITUDE,
    "lon": LONGITUDE,
    "appid": API_KEY,
    "units": "metric",
}

print("Fetching weather data for Lahore...\n")

session = requests.Session()

try:
    weather_response = session.get(WEATHER_URL, params=params, timeout=15)
    if weather_response.status_code == 401:
        print("ERROR: Invalid OpenWeather API Key.")
        print("Please check your API key in the .env file.")
        sys.exit()

    weather_response.raise_for_status()
    weather = weather_response.json()

    air_response = session.get(AIR_URL, params=params, timeout=15)
    air_response.raise_for_status()
    pollution = air_response.json()

    print("=" * 50)
    print("CURRENT WEATHER")
    print("=" * 50)
    print(f"Location      : {weather['name']}, {weather['sys']['country']}")
    print(f"Temperature   : {weather['main']['temp']}°C")
    print(f"Feels Like    : {weather['main']['feels_like']}°C")
    print(f"Humidity      : {weather['main']['humidity']}%")
    print(f"Pressure      : {weather['main']['pressure']} hPa")
    print(f"Wind Speed    : {weather['wind']['speed']} m/s")
    print(f"Condition     : {weather['weather'][0]['description'].title()}")

    print("\n" + "=" * 50)
    print("AIR QUALITY")
    print("=" * 50)

    air = pollution["list"][0]
    aqi = air["main"]["aqi"]
    components = air["components"]

    print(f"OpenWeather AQI : {AQI_LEVELS.get(aqi, 'Unknown')} ({aqi}/5)\n")
    print("Pollutant Concentrations (µg/m³)")
    print("-" * 35)
    print(f"PM2.5 : {components['pm2_5']}")
    print(f"PM10  : {components['pm10']}")
    print(f"CO    : {components['co']}")
    print(f"NO₂   : {components['no2']}")
    print(f"O₃    : {components['o3']}")
    print(f"SO₂   : {components['so2']}")
    print(f"NH₃   : {components['nh3']}")

except Exception as e:
    print(f"Error: {e}")
