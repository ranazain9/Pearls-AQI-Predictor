import os
import sys
import requests
from dotenv import load_dotenv

# Reconfigure stdout to use utf-8 to prevent encoding errors on Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Load environment variables from .env file
load_dotenv()

# Retrieve OpenWeather API Key
api_key = os.getenv("OPEN_WEATHER_API")

if not api_key:
    raise ValueError("OPEN_WEATHER_API key not found in .env file.")

# Lahore Coordinates
lat = 31.5204
lon = 74.3587

# 1. Fetch Current Weather Data
weather_url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}&units=metric"
# 2. Fetch Current Air Pollution Data
pollution_url = f"https://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={api_key}"

print("Requesting data from OpenWeather API for Lahore...")

# Helper mapping for OpenWeather AQI levels
aqi_levels = {
    1: "Good (1/5)",
    2: "Fair (2/5)",
    3: "Moderate (3/5)",
    4: "Poor (4/5)",
    5: "Very Poor (5/5)"
}

try:
    # Get weather
    w_response = requests.get(weather_url)
    w_response.raise_for_status()
    weather_data = w_response.json()
    
    # Get pollution
    p_response = requests.get(pollution_url)
    p_response.raise_for_status()
    pollution_data = p_response.json()
    
    print("\n--- Current Weather (OpenWeather) ---")
    print(f"Location    : {weather_data.get('name', 'Lahore')}, {weather_data.get('sys', {}).get('country')}")
    print(f"Temperature : {weather_data['main']['temp']} °C (Feels like: {weather_data['main']['feels_like']} °C)")
    print(f"Humidity    : {weather_data['main']['humidity']}%")
    print(f"Pressure    : {weather_data['main']['pressure']} hPa")
    print(f"Wind Speed  : {weather_data['wind']['speed']} m/s")
    print(f"Conditions  : {weather_data['weather'][0]['description'].capitalize()}")
    
    print("\n--- Current Air Pollution (OpenWeather) ---")
    if "list" in pollution_data and len(pollution_data["list"]) > 0:
        p_item = pollution_data["list"][0]
        aqi_val = p_item["main"]["aqi"]
        components = p_item["components"]
        
        print(f"OpenWeather AQI : {aqi_levels.get(aqi_val, f'{aqi_val}/5')}")
        print("Pollutant Concentrations (µg/m³):")
        print(f" - PM2.5 : {components.get('pm2_5')}")
        print(f" - PM10  : {components.get('pm10')}")
        print(f" - CO    : {components.get('co')}")
        print(f" - NO2   : {components.get('no2')}")
        print(f" - O3    : {components.get('o3')}")
        print(f" - SO2   : {components.get('so2')}")
    else:
        print("No air pollution data returned.")
        
except Exception as e:
    print(f"Request failed: {e}")
