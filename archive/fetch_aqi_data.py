import os
import sys
import requests
from dotenv import load_dotenv

# Reconfigure stdout to use utf-8 to prevent encoding errors on Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Load environment variables from the .env file
load_dotenv()

# Retrieve the token
token = os.getenv("AQICN_API_TOKEN")

if not token:
    raise ValueError("AQICN_API_TOKEN not found. Please make sure it is defined in the .env file.")

# We use the station ID 'A471607' which corresponds to the active Lahore monitoring station.
# The default string "Lahore" resolves to the inactive "Lahore US Embassy" feed.
city = "A471607"
url = f"https://api.waqi.info/feed/{city}/?token={token}"

print(f"Requesting updated AQI data for Lahore (Station: {city})...")

try:
    # Send GET request to AQICN API
    response = requests.get(url)
    response.raise_for_status()
    
    result = response.json()
    
    if result.get("status") == "ok":
        data = result["data"]
        print("\n--- Raw Data Successfully Fetched! ---")
        print(f"City      : {data['city']['name']}")
        print(f"AQI       : {data['aqi']}")
        print(f"Dominant  : {data.get('dominentpol', 'N/A')}")
        print(f"Time (Local): {data['time']['s']}")
        print(f"Geo Coord : {data['city']['geo']}")
        
        # Check if individual pollutant details exist
        iaqi = data.get("iaqi", {})
        print("\nPollutants Info:")
        for pollutant, val in iaqi.items():
            print(f" - {pollutant.upper()}: {val.get('v')}")
            
    else:
        print(f"API Error: {result.get('data')}")
except Exception as e:
    print(f"Request failed: {e}")
