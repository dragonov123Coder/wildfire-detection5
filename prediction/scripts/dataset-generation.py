import os
import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime
from global_land_mask import globe
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

# -----------------------
# CONFIGURATION
# -----------------------
with open("../config.json") as file:
    config = json.load(file)
    dataset_generation_config = config["server"]["prediction"]["dataset_generation"] 
    
OUTPUT_PATH = config["server"]["prediction"]["wildfire_dataset_path"]
SAMPLES_PER_YEAR = dataset_generation_config["samples_per_year"]  # Positive samples to pick per year

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
# DYNAMIC FILE LOADER (FIXED TO FILTER OUT TYPE 2/3 ANOMALIES)
# -----------------------
def load_firms_data(start_year, end_year):
    dfs = []
    for year in range(start_year, end_year + 1):
        paths_to_try = [
            f"data/input/firms/viirs-jpss1_{year}_Canada.csv",
            f"data/firms_active_fires_{year}.csv",
            f"data/firms_active_fires.csv"
        ]
        
        loaded = False
        for path in paths_to_try:
            if os.path.exists(path):
                try:
                    df = pd.read_csv(path)
                    df.columns = [col.strip().lower() for col in df.columns]
                    
                    # --- FIX STAGE: Filter out static land sources / volcanic anomalies ---
                    # Type 0 = Vegetation fire. Type 1 = Active volcano. Type 2 = Static land source. Type 3 = Offshore.
                    # We cast to numeric just in case types are read as mixed string/float types.
                    if "type" in df.columns:
                        df["type"] = pd.to_numeric(df["type"], errors="coerce")
                        initial_count = len(df)
                        # Keep only Type 0 (wildfires) or NaN values if type wasn't captured
                        df = df[(df["type"] == 0) | (df["type"].isna())]
                        filtered_count = initial_count - len(df)
                        if filtered_count > 0:
                            print(f"Filtered out {filtered_count} static/volcanic (Type 2+) anomalies from {year} data.")
                    # ---------------------------------------------------------------------

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
# GENERATE NEGATIVE CASES
# -----------------------
def generate_negatives(target_count, df_firms, start_year, end_year):
    """
    Vectorized, high-speed negative case generator.
    Generates thousands of coordinates and dates simultaneously using NumPy.
    """
    print(f"Generating {target_count} safe, independent negative cases via vectorized arrays...")
    
    # 1. Pre-hash fire coordinates by day to allow fast set lookups instead of scanning dataframes
    df_firms['acq_date_str'] = df_firms['acq_datetime'].dt.strftime("%Y-%m-%d")
    fire_map = {}
    for date_str, group in df_firms.groupby("acq_date_str"):
        # We round to 1 decimal place to create an easy, fast grid mask of nearby fires
        fire_map[date_str] = set(zip(np.round(group["latitude"], 1), np.round(group["longitude"], 1)))

    negatives = []
    
    # Generate a massive overhead pool of points to process in fast, batch-sized chunks
    oversample_factor = 3
    chunk_size = target_count * oversample_factor
    
    while len(negatives) < target_count:
        # Generate random coordinates in bulk using fast C-arrays
        lats = np.random.uniform(LAT_MIN, LAT_MAX, size=chunk_size)
        lons = np.random.uniform(LON_MIN, LON_MAX, size=chunk_size)
        
        # Generate random calendar days strictly inside fire season (Days 92 to 273)
        random_days = np.random.randint(92, 274, size=chunk_size)
        years = np.random.randint(start_year, end_year + 1, size=chunk_size)
        
        for i in range(chunk_size):
            if len(negatives) >= target_count:
                break
                
            lat, lon = lats[i], lons[i]
            
            # Fast landmask filter
            if not globe.is_land(lat, lon):
                continue
                
            # Build string date cleanly
            dt = datetime.strptime(f"{years[i]}-{random_days[i]}", "%Y-%j")
            date_str = dt.strftime("%Y-%m-%d")
            
            # Instant dictionary-set proximity verification
            if date_str in fire_map:
                # Round current random coordinate to check if a fire occurred in the general area
                coord_key = (round(lat, 1), round(lon, 1))
                if coord_key in fire_map[date_str]:
                    continue  # Too close to an active fire grid section!
            
            negatives.append({
                "latitude": round(lat, 4),
                "longitude": round(lon, 4),
                "acq_date": date_str,
                "fire": 0
            })
            
    print("Finished Negative Generation successfully.")
    return pd.DataFrame(negatives)

# -----------------------
# HISTORICAL WEATHER FETCH (THREAD-SAFE WRAPPER)
# -----------------------
def process_date_group(date_str, group):
    """
    Worker function executed by individual threads. Handles the API request
    and parsing for a specific date group.
    """
    coords = list(zip(group["latitude"], group["longitude"]))
    lats = [c[0] for c in coords]
    lons = [c[1] for c in coords]
    
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": ",".join(map(str, lats)),
        "longitude": ",".join(map(str, lons)),
        "start_date": date_str,
        "end_date": date_str,
        "daily": "temperature_2m_mean,relative_humidity_2m_mean,wind_speed_10m_max,precipitation_sum,soil_moisture_0_to_7cm_mean",
        "timezone": "UTC"
    }

    records = []
    try:
        # Reduced sleep since threads spread requests naturally; 
        # Open-Meteo allows heavy concurrency without aggressive blocks
        time.sleep(0.02) 
        response = requests.get(url, params=params, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            if isinstance(data, dict):
                data = [data]
                
            for idx, entry in enumerate(data):
                daily = entry.get("daily", {})
                if not daily:
                    continue
                
                temp = daily.get("temperature_2m_mean", [None])[0]
                humidity = daily.get("relative_humidity_2m_mean", [None])[0]
                wind = daily.get("wind_speed_10m_max", [None])[0]
                rain = daily.get("precipitation_sum", [None])[0]
                soil_moisture = daily.get("soil_moisture_0_to_7cm_mean", [None])[0]
                
                row = group.iloc[idx]
                dt = datetime.strptime(date_str, "%Y-%m-%d")
                day_of_year = dt.timetuple().tm_yday
                
                records.append({
                    "latitude": row["latitude"],
                    "longitude": row["longitude"],
                    "temp": temp,
                    "humidity": humidity,
                    "wind": wind,
                    "rain": rain,
                    "day_of_year": day_of_year,
                    "soil_moisture": soil_moisture,
                    "fire": row["fire"]
                })
            return records
        else:
            print(f"  [!] API Error {response.status_code} for date {date_str}")
            return []
    except Exception as e:
        print(f"  [!] Request failed for date {date_str}: {e}")
        return []

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
        
    df_firms = df_firms[
        (df_firms["latitude"] >= LAT_MIN) & (df_firms["latitude"] <= LAT_MAX) &
        (df_firms["longitude"] >= LON_MIN) & (df_firms["longitude"] <= LON_MAX)
    ]
    
    if len(df_firms) == 0:
        print("No coordinates matched the regional boundaries.")
        return

    sampled_dfs = []
    for year, group in df_firms.groupby(df_firms["acq_datetime"].dt.year):
        sample_n = min(SAMPLES_PER_YEAR, len(group))
        sampled_dfs.append(group.sample(n=sample_n, random_state=42))
        
    df_positives = pd.concat(sampled_dfs, ignore_index=True)
    df_positives["fire"] = 1
    df_positives["acq_date"] = df_positives["acq_datetime"].dt.strftime("%Y-%m-%d")
    
    print(f"\nSampled {len(df_positives)} active fire cases across the selected period.")
    
    df_negatives = generate_negatives(len(df_positives), df_firms, start_year, end_year)
    print(f"Generated {len(df_negatives)} non-fire cases.")

    df_combined = pd.concat([
        df_positives[["latitude", "longitude", "acq_date", "fire"]],
        df_negatives
    ], ignore_index=True)

    print("\nFetching weather parameters from Open-Meteo using multi-threading...")
    weather_records = []
    grouped_by_date = list(df_combined.groupby("acq_date"))
    total_dates = len(grouped_by_date)
    
    # We use a pool of 5 to 8 workers. Open-Meteo is very fast, so 5 concurrent 
    # workers will easily saturate the pipeline without overwhelming local sockets.
    completed_threads = 0
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=5) as executor:
        # Submit all unique date lookups to the pool concurrently
        futures = {
            executor.submit(process_date_group, date_str, group): date_str 
            for date_str, group in grouped_by_date
        }
        
        # As threads complete, safely aggregate results back to main thread array
        for future in as_completed(futures):
            completed_threads += 1
            result = future.result()
            if result:
                weather_records.extend(result)
            
            if completed_threads % 10 == 0 or completed_threads == total_dates:
                print(f"  Progress: {completed_threads}/{total_dates} date batches processed...", end="\r")

    # Convert to DataFrame
    final_df = pd.DataFrame(weather_records).dropna()
    
    print(f"\nWeather extraction completed in {round(time.time() - start_time, 2)} seconds.")
    print(f"\nFinal training dataset class breakdown:")
    print(final_df["fire"].value_counts())
    
    final_df.to_csv(OUTPUT_PATH, index=False)
    print(f"Successfully saved compiled dataset to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
    import train