import joblib
import pandas as pd
import numpy as np
import requests
from datetime import datetime, UTC
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from global_land_mask import globe
import json

# =====================================================================
#                         CONFIGURATION
# =====================================================================

with open("../config.json") as file:
    config = json.load(file)

# =====================================================================
# ⚙️ GRID CONCENTRATION CONFIGURATION
# =====================================================================
STEP = config["server"]["prediction"]["grid_step"] 

MAX_WORKERS = config["server"]["prediction"]["max_workers"]
BATCH_SIZE = config["server"]["prediction"]["batch_size"]
# =====================================================================

# Load the trained model
try:
    model = joblib.load(config["server"]["prediction"]["model_path"])
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Alberta boundaries
LAT_MIN, LAT_MAX = 49.0, 60.0
LON_MIN, LON_MAX = -120.0, -110.0
MAX_RETRIES = 3

today = datetime.now(UTC)

# -----------------------
# 🌍 ALBERTA BORDER FILTER
# -----------------------
def is_inside_alberta(lat, lon):
    # 1. Base land check
    if not globe.is_land(lat, lon):
        return False
    
    # 2. General rectangular bounds check
    if not (LAT_MIN <= lat <= LAT_MAX) or not (LON_MIN <= lon <= LON_MAX):
        return False
    
    # 3. Diagonal Rocky Mountains boundary check below 53.8° N
    if lat < 53.8:
        min_alberta_lon = -120.0 + (53.8 - lat) * 1.25
        if lon < min_alberta_lon:
            return False
            
    return True

# -----------------------
# GRID GENERATION
# -----------------------
lats = np.arange(LAT_MIN, LAT_MAX + STEP, STEP)
lons = np.arange(LON_MIN, LON_MAX + STEP, STEP)

raw_points_count = len(lats) * len(lons)
print(f"\n--- Grid Analysis ---")
print(f"Chosen STEP size: {STEP}")
print(f"Raw mathematical grid intersections: {raw_points_count}")

points = [
    (round(lat, 2), round(lon, 2))
    for lat in lats
    for lon in lons
    if is_inside_alberta(lat, lon)
]

print(f"Actual points in Alberta landmass: {len(points)}")
print(f"Total batches to request: {len(points) // BATCH_SIZE + 1}")
print(f"---------------------\n")

if len(points) == 0:
    print("Error: No grid points were generated. Try using a smaller STEP value or verify the boundary limits.")
    exit()

# -----------------------
# WEATHER & ECOLOGICAL FETCH (FIXED TO HOURLY BATCHING)
# -----------------------
def get_weather_batch(batch_points):
    lats_list = [p[0] for p in batch_points]
    lons_list = [p[1] for p in batch_points]

    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": ",".join(map(str, lats_list)),
        "longitude": ",".join(map(str, lons_list)),
        # Fetch hourly blocks for today to bypass daily limitations natively
        "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,precipitation,soil_moisture_0_to_1cm",
        "forecast_days": 1,
        "timezone": "UTC"
    }
    
    for attempt in range(MAX_RETRIES):
        try:
            res = requests.get(url, params=params, timeout=15)

            if res.status_code == 429:
                wait = 2 ** attempt
                print(f"Rate limited. Waiting {wait}s...")
                time.sleep(wait)
                continue

            if res.status_code != 200:
                print(f"API Error {res.status_code}: {res.text}")
                continue

            return res.json()

        except requests.RequestException as e:
            print(f"Connection attempt {attempt + 1} failed: {e}")
            time.sleep(1.5)

    return None

# -----------------------
# PARSE AND AGGREGATE RESPONSE (FIXED)
# -----------------------
def parse_batch_response(data, batch_points):
    rows = []
    
    if isinstance(data, dict):
        data = [data]

    for j, entry in enumerate(data):
        hourly = entry.get("hourly", {})
        if not hourly:
            continue

        lat, lon = batch_points[j]
        
        # Open-Meteo returns lists of 24 elements (one for each hour of today)
        h_temps = hourly.get("temperature_2m", [])
        h_humids = hourly.get("relative_humidity_2m", [])
        h_winds = hourly.get("wind_speed_10m", [])
        h_rains = hourly.get("precipitation", [])
        h_soils = hourly.get("soil_moisture_0_to_1cm", [])

        # Prevent parsing errors if any list comes back empty
        if not h_temps:
            continue

        # Generate custom summary aggregates to perfectly match training sets
        temp_mean = np.mean(h_temps)
        humidity_mean = np.mean(h_humids)
        wind_max = np.max(h_winds)
        rain_sum = np.sum(h_rains)
        soil_mean = np.mean(h_soils)

        rows.append({
            "latitude": lat,
            "longitude": lon,
            "temp": temp_mean,
            "humidity": humidity_mean,
            "wind": wind_max,
            "rain": rain_sum,
            "day_of_year": today.timetuple().tm_yday,
            "soil_moisture": soil_mean
        })

    return rows

# -----------------------
# BATCH PROCESSING
# -----------------------
batches = [points[i:i + BATCH_SIZE] for i in range(0, len(points), BATCH_SIZE)]

all_rows = []
completed = 0

print("Starting weather data fetch...")
with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures = {executor.submit(get_weather_batch, batch): batch for batch in batches}

    for future in as_completed(futures):
        batch = futures[future]
        try:
            data = future.result()
            if data:
                rows = parse_batch_response(data, batch)
                all_rows.extend(rows)
            else:
                print(f"Warning: Failed to retrieve data for a batch of {len(batch)} points.")
        except Exception as e:
            print(f"Error during execution of batch: {e}")

        completed += len(batch)
        print(f"Progress: {completed}/{len(points)} points processed. Rows collected: {len(all_rows)}")
        
        time.sleep(0.5)

print(f"\nTotal rows successfully collected: {len(all_rows)}")

# -----------------------
# MODEL PREDICTION
# -----------------------
if all_rows and model is not None:
    df_features = pd.DataFrame(all_rows)

    FEATURES = [
        "latitude",
        "longitude",
        "temp",
        "humidity",
        "wind",
        "rain",
        "day_of_year",
        "soil_moisture"
    ]

    df_features = df_features[FEATURES].dropna()

    proba = model.predict_proba(df_features)
    
    if proba.shape[1] == 1:
        only_class = model.classes_[0]
        if only_class == 1:
            fire_probs = proba[:, 0]
        else:
            fire_probs = 1 - proba[:, 0]
    else:
        fire_probs = proba[:, 1]

    df_results = pd.DataFrame({
        "lat": df_features["latitude"],
        "lon": df_features["longitude"],
        "fire_risk": fire_probs
    })

    df_results.to_csv(config["server"]["prediction"]["output_path"], index=False)
    print(f"Success! Saved {len(df_results)} mapped risk values to path indicated in 'config.json'.")

else:
    if model is None:
        print("Execution finished, but no predictions were made because the model was missing.")
    else:
        print("No weather data was collected. Try running the script again with a STEP value of 1.0.")