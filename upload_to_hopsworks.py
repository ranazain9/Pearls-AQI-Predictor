import os
import sys
import pandas as pd
# pyrefly: ignore [missing-import]
import hopsworks
from dotenv import load_dotenv

# Reconfigure stdout to UTF-8
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Load environment variables
load_dotenv()
hopsworks_api_key = os.getenv("HOPSWORKS_API_KEY")

if not hopsworks_api_key:
    raise ValueError("HOPSWORKS_API_KEY not found in .env. Please add it to connect.")

print("--- Step 1: Loading Dataset ---")
file_path = "data/lahore_aqi_historical.csv"
if not os.path.exists(file_path):
    raise FileNotFoundError(f"{file_path} does not exist. Run historical_backfill.py first.")

df = pd.read_csv(file_path)

# Drop missing target values (we only want rows with actual ground-truth PM2.5 for model training)
missing_before = df["ground_pm25"].isna().sum()
df = df.dropna(subset=["ground_pm25"]).reset_index(drop=True)
print(f"Dropped {missing_before} rows with missing 'ground_pm25'. Dataset now has {len(df)} clean rows.")

# Convert timestamp string to datetime
df["timestamp"] = pd.to_datetime(df["timestamp"])

print("\n--- Step 2: Connecting to Hopsworks ---")
# Log in to Hopsworks (automatically uses the HOPSWORKS_API_KEY env variable)
project = hopsworks.login(api_key_value=hopsworks_api_key)

# Get the feature store
fs = project.get_feature_store()

print("\n--- Step 3: Creating and Inserting Data into Feature Group ---")
# Create or get the feature group
aqi_fg = fs.get_or_create_feature_group(
    name="lahore_aqi_features",
    version=1,
    primary_key=["hour", "month", "day_of_week"],
    description="Lahore PM2.5 Air Quality historical dataset with weather features.",
    event_time="timestamp"
)

# Insert the data into the feature group
print("Uploading dataset to Hopsworks... This might take a few minutes for 10,000+ rows.")
aqi_fg.insert(df)
print("\nSuccess! Historical dataset uploaded to Hopsworks Feature Store!")
