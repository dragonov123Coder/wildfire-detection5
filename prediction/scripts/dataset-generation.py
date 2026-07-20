import os
import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime
from global_land_mask import globe
import json

# -----------------------
# CONFIGURATION
# -----------------------
with open("../config.json") as file:
    config = json.load(file)
    dataset_generation_config = config["server"]["prediction"]["dataset_generation"] 
    
OUTPUT_PATH = config["server"]["prediction"]["wildfire_dataset_path"]
SAMPLES_PER_YEAR = dataset_generation_config["samples_per_year"]  # Positive samples to pick per year (keeps API usage reasonable)

# Canada Bounding Box
bounds = config["server"]["prediction"]["dataset_bounding_box"]
LAT_MIN, LAT_MAX = bounds[0]
LON_MIN, LON_MAX = bounds[1]

# -----------------------
# USER INPUT VALIDATION
# -----------------------
def get_year_range():
    print("--- FIRMS Dataset Generation Setup ---")
    return dataset_generation_config["start_year"], dataset_generation_config["end_year"]
# -----------------------
# DYNAMIC FILE LOADER
# -----------------------
def load_firms_data(start_year, end_year):
    """
    Attempts to load files for each year. 
    Looks for yearly files first (e.g., data/firms_2018.csv),
    then falls back to filtering a shared file (data/firms_active_fires.csv).
    """
    dfs = []
    for year in range(start_year, end_year + 1):
        paths_to_try = [
            f"data/input/firms/viirs-jpss1_{year}_Canada.csv",
            f"data/firms_active_fires_{year}.csv",
            "data/firms_active_fires.csv"  # Consolidated file containing multiple years
        ]
        
        loaded = False
        for path in paths_to_try:
            if os.path.exists(path):
                try:
                    df = pd.read_csv(path)
                    df.columns = [col.strip().lower() for col in df.columns]
                    
                    # Parse dates to ensure accurate year filtering
                    df["acq_datetime"] = pd.to_datetime(df["acq_date"])
                    df_year = df[df["acq_datetime"].dt.year == year].copy()
                    
                    if len(df_year) > 0:
                        dfs.append(df_year)
                        print(f"Loaded {len(df_year)} records for year {year} from '{path}'.")
                        loaded = True
                        break
                except Exception as e:
                    print(f"Error reading {path}: {e}")
                    
        if not loaded:
            print(f"Warning: Could not locate or process data for year {year}.")
            
    if not dfs:
        return pd.DataFrame()
        
    return pd.concat(dfs, ignore_index=True)

# -----------------------
# VEGETATION PROXY
# -----------------------
def vegetation_proxy(lat, lon):
    if lat > 65:
        return np.random.uniform(0.1, 0.3)
    elif lat > 55:
        return np.random.uniform(0.3, 0.6)
    else:
        return np.random.uniform(0.6, 0.9)
# -----------------------
# GENERATE NEGATIVE CASES (UPDATED)
# -----------------------
def generate_negatives(target_count, df_firms, start_year, end_year):
    """
    Generates negative (non-fire) cases by picking completely random dates and 
    locations inside Canada, ensuring they are on land and far away from any 
    recorded active fires on that day.
    """
    negatives = []
    attempts = 0
    max_attempts = target_count * 10  # Prevent infinite loop if constraints are too tight
    
    print(f"Generating {target_count} safe, independent negative cases...")

    # Pre-parse dates in the master dataset for lightning-fast lookups
    df_firms['acq_date_str'] = df_firms['acq_datetime'].dt.strftime("%Y-%m-%d")

    while len(negatives) < target_count and attempts < max_attempts:
        attempts += 1
        print(f"\033c{round((len(negatives)/target_count)*100, 2)}% complete", flush=True)
        
        # 1. Generate a random date STRICTLY within fire season (April to Sept)
        year = np.random.randint(start_year, end_year + 1)
        
        # Pick a random day of the year between day 92 (April 2) and day 273 (Sept 30)
        random_day = np.random.randint(92, 274)
        random_date = datetime.strptime(f"{year}-{random_day}", "%Y-%j")
        random_date_str = random_date.strftime("%Y-%m-%d")

        # 2. Generate a random location in Canada
        neg_lat = round(np.random.uniform(LAT_MIN, LAT_MAX), 4)
        neg_lon = round(np.random.uniform(LON_MIN, LON_MAX), 4)

        # 3. Validation checks
        # Check A: Must be on land
        if not globe.is_land(neg_lat, neg_lon):
            continue

        # Check B: Make sure it is safe (no active fires nearby on this date)
        # Filter master data to just this day
        day_fires = df_firms[df_firms["acq_date_str"] == random_date_str]
        
        if not day_fires.empty:
            # Check if any fire coordinates on this day are within a 0.5 degree (~55km) buffer zone
            too_close = any(
                abs(neg_lat - f_lat) < 0.5 and abs(neg_lon - f_lon) < 0.5 
                for f_lat, f_lon in zip(day_fires["latitude"], day_fires["longitude"])
            )
            if too_close:
                continue  # Skip and try again, too risky!

        # If it passes all checks, it's a safe negative
        negatives.append({
            "latitude": neg_lat,
            "longitude": neg_lon,
            "acq_date": random_date_str,
            "fire": 0
        })

    if len(negatives) < target_count:
        print(f"Warning: Could only generate {len(negatives)} negatives out of {target_count} requested.")

    return pd.DataFrame(negatives)

# -----------------------
# HISTORICAL WEATHER FETCH
# -----------------------
def fetch_historical_weather_with_vegetation(date_str, coords):
    """
    Fetches historical weather AND real land/vegetation metrics (LAI and Soil Moisture)
    for a given date and coordinate set in a single API request.
    """
    lats = [c[0] for c in coords]
    lons = [c[1] for c in coords]
    
    url = "https://archive-api.open-meteo.com/v1/archive"
    
    # We append leaf_area_index and soil_moisture_0_to_7cm directly to daily aggregations
    params = {
        "latitude": ",".join(map(str, lats)),
        "longitude": ",".join(map(str, lons)),
        "start_date": date_str,
        "end_date": date_str,
        "daily": (
            "temperature_2m_mean,"
            "relative_humidity_2m_mean,"
            "wind_speed_10m_max,"
            "precipitation_sum,"
            "soil_moisture_0_to_7cm_mean"  # <-- Fixed: suffix instead of prefix
        ),
        "timezone": "UTC"
    }

    try:
        response = requests.get(url, params=params, timeout=30)
        if response.status_code == 200:
            data = response.json()
            if isinstance(data, dict):
                data = [data]
            return data
        else:
            print(f"API Error {response.status_code} for date {date_str}")
            return None
    except Exception as e:
        print(f"Request failed for date {date_str}: {e}")
        return None
    
# -----------------------
# MAIN RUNNER
# -----------------------
def main():
    start_year, end_year = get_year_range()
    
    print("\nProcessing loaded year files...")
    df_firms = load_firms_data(start_year, end_year)
    
    if df_firms.empty:
        print("No training data could be loaded. Ensure the files exist in the data/ folder.")
        return
        
    # Apply spatial bounding box filter
    df_firms = df_firms[
        (df_firms["latitude"] >= LAT_MIN) & (df_firms["latitude"] <= LAT_MAX) &
        (df_firms["longitude"] >= LON_MIN) & (df_firms["longitude"] <= LON_MAX)
    ]
    
    if len(df_firms) == 0:
        print("No coordinates matched the regional boundaries.")
        return

    # Sample equally per year to maintain historical balance
    sampled_dfs = []
    for year, group in df_firms.groupby(df_firms["acq_datetime"].dt.year):
        sample_n = min(SAMPLES_PER_YEAR, len(group))
        sampled_dfs.append(group.sample(n=sample_n, random_state=42))
        
    df_positives = pd.concat(sampled_dfs, ignore_index=True)
    df_positives["fire"] = 1
    
    # Cast dates back to clean string representation
    df_positives["acq_date"] = df_positives["acq_datetime"].dt.strftime("%Y-%m-%d")
    
    print(f"\nSampled {len(df_positives)} active fire cases across the selected period.")
    
    print("Generating non-fire (negative) cases...")
    # Pass the required target count, the full firms dataframe, and year ranges
    df_negatives = generate_negatives(len(df_positives), df_firms, start_year, end_year)
    print(f"Generated {len(df_negatives)} non-fire cases.")

    # Combine positive and negative cases
    df_combined = pd.concat([
        df_positives[["latitude", "longitude", "acq_date", "fire"]],
        df_negatives
    ], ignore_index=True)

    print("\nFetching weather parameters from Open-Meteo...")
    weather_records = []
    grouped_by_date = df_combined.groupby("acq_date")
    total_dates = len(grouped_by_date)
    
    for i, (date_str, group) in enumerate(grouped_by_date, 1):
        coords = list(zip(group["latitude"], group["longitude"]))
        print(f"[{i}/{total_dates}] Weather query for {date_str} ({len(coords)} coordinates)...")
        
        # ... inside your weather parsing loop where you extract entries:
        weather_data = fetch_historical_weather_with_vegetation(date_str, coords)
        
        if weather_data:
            for idx, entry in enumerate(weather_data):
                daily = entry.get("daily", {})
                
                # Standard weather features
                temp = daily.get("temperature_2m_mean", [None])[0]
                humidity = daily.get("relative_humidity_2m_mean", [None])[0]
                wind = daily.get("wind_speed_10m_max", [None])[0]
                rain = daily.get("precipitation_sum", [None])[0]
                
                # Extract the corrected soil moisture string
                soil_moisture = daily.get("soil_moisture_0_to_7cm_mean", [None])[0]
                
                row = group.iloc[idx]
                dt = datetime.strptime(date_str, "%Y-%m-%d")
                day_of_year = dt.timetuple().tm_yday
                
                weather_records.append({
                    "latitude": row["latitude"],
                    "longitude": row["longitude"],
                    "temp": temp,
                    "humidity": humidity,
                    "wind": wind,
                    "rain": rain,
                    "day_of_year": day_of_year,
                    "soil_moisture": soil_moisture, # Real fuel dryness tracker
                    "fire": row["fire"]
                })
                
        # Respect Open-Meteo API limits
        time.sleep(0.05)

    # Convert to DataFrame
    final_df = pd.DataFrame(weather_records).dropna()
    
    print(f"\nFinal training dataset class breakdown:")
    print(final_df["fire"].value_counts())
    
    final_df.to_csv(OUTPUT_PATH, index=False)
    print(f"Successfully saved compiled dataset to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
    import train