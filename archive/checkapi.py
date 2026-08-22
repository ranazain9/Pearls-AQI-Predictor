import os
from pathlib import Path

import requests
from dotenv import load_dotenv

# Load .env from the same folder as this script
env_path = Path(__file__).parent / ".env"
print(f"Loading .env from: {env_path}")

load_dotenv(env_path)

# Read API key
api_key = os.getenv("OPEN_WEATHER_API")

print("\n========== API KEY ==========")

if api_key:
    print(f"API Key       : {api_key}")
    print(f"Key Length    : {len(api_key)}")
else:
    print("API Key       : NOT FOUND")
    exit()

print("\n========== REQUEST ==========")

url = "https://api.openweathermap.org/data/2.5/weather"

params = {
    "lat": 31.5204,
    "lon": 74.3587,
    "appid": api_key,
    "units": "metric",
}

try:
    response = requests.get(url, params=params, timeout=15)

    print(f"Status Code   : {response.status_code}")
    print("\nResponse:")
    print(response.text)

except Exception as e:
    print(f"Error: {e}")