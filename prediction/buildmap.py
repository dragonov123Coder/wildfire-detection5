import joblib
import pandas as pd
import numpy as np
import requests
from datetime import datetime, UTC
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from global_land_mask import globe

# =====================================================================
# ⚙️ GRID CONCENTRATION CONFIGURATION
# =====================================================================
# Change the STEP value below to control the density of your grid points:
# - STEP = 1.0  -> Coarse/Fast (~70 points)  <-- RECOMMENDED FOR TESTING
# - STEP = 0.5  -> Medium Density (~250 points)
# - STEP = 0.25 -> High Density (~1,000 points)
# - STEP = 0.15 -> Very High Density (~2,500 points)
STEP = 1.0  

# Adjust to prevent overwhelming the API with too many parallel requests
MAX_WORKERS = 3  # Lowered from 10 to reduce rate-limit/timeout issues
BATCH_SIZE = 50  # Lowered from 100 for smaller, safer request payloads
# =====================================================================

# Load the trained model
try:
    model = joblib.load("wildfire_model.pkl")
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
# 🌲 Vegetation Proxy (MATCH TRAINING)
# -----------------------
def vegetation_proxy(lat, lon):
    if lat > 65:
        return np.random.uniform(0.1, 0.3)
    elif lat > 55:
        return np.random.uniform(0.3, 0.6)
    else:
        return np.random.uniform(0.6, 0.9)

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
# WEATHER FETCH
# -----------------------
def get_weather_batch(batch_points):
    lats_list = [p[0] for p in batch_points]
    lons_list = [p[1] for p in batch_points]

    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": ",".join(map(str, lats_list)),
        "longitude": ",".join(map(str, lons_list)),
        "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,precipitation"
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
# PARSE RESPONSE
# -----------------------
def parse_batch_response(data, batch_points):
    rows = []

    if isinstance(data, list):
        for j, entry in enumerate(data):
            cw = entry.get("current")
            if not cw:
                continue

            lat, lon = batch_points[j]
            rows.append({
                "latitude": lat,
                "longitude": lon,
                "temp": cw.get("temperature_2m"),
                "humidity": cw.get("relative_humidity_2m"),
                "wind": cw.get("wind_speed_10m"),
                "rain": cw.get("precipitation", 0),
                "day_of_year": today.timetuple().tm_yday,
                "vegetation_index": vegetation_proxy(lat, lon)
            })

    elif isinstance(data, dict):
        current = data.get("current", {})
        temp = current.get("temperature_2m")
        humidity = current.get("relative_humidity_2m")
        wind = current.get("wind_speed_10m")
        rain = current.get("precipitation")

        for j, (lat, lon) in enumerate(batch_points):
            rows.append({
                "latitude": lat,
                "longitude": lon,
                "temp": temp[j] if isinstance(temp, list) else temp,
                "humidity": humidity[j] if isinstance(humidity, list) else humidity,
                "wind": wind[j] if isinstance(wind, list) else wind,
                "rain": rain[j] if isinstance(rain, list) else rain,
                "day_of_year": today.timetuple().tm_yday,
                "vegetation_index": vegetation_proxy(lat, lon)
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
        
        # Slight delay to avoid hammering the server
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
        "vegetation_index"
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

    df_results.to_csv("alberta_fire_risk_map.csv", index=False)
    print(f"Success! Saved {len(df_results)} mapped risk values to 'alberta_fire_risk_map.csv'.")

else:
    if model is None:
        print("Execution finished, but no predictions were made because the model was missing.")
    else:
        print("No weather data was collected. Try running the script again with a STEP value of 1.0.")