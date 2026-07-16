import os
import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime
from global_land_mask import globe

# -----------------------
# CONFIGURATION
# -----------------------
OUTPUT_PATH = "data/wildfire_dataset.csv"
SAMPLES_PER_YEAR = 100  # Positive samples to pick per year (keeps API usage reasonable)

# Canada Bounding Box
LAT_MIN, LAT_MAX = 42.0, 83.0
LON_MIN, LON_MAX = -141.0, -52.0

# -----------------------
# USER INPUT VALIDATION
# -----------------------
def get_year_range():
    print("--- FIRMS Dataset Generation Setup ---")
    while True:
        try:
            start_year = int(input("Enter start year (2018-2024): ").strip())
            if 2018 <= start_year <= 2024:
                break
            print("Invalid year. Must be between 2018 and 2024.")
        except ValueError:
            print("Please enter a valid integer.")
            
    while True:
        try:
            end_year = int(input(f"Enter end year ({start_year}-2024): ").strip())
            if start_year <= end_year <= 2024:
                break
            print(f"Invalid year. Must be between {start_year} and 2024.")
        except ValueError:
            print("Please enter a valid integer.")
            
    return start_year, end_year

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
            f"data/firms/viirs-jpss1_{year}_Canada.csv",
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
# GENERATE NEGATIVE CASES
# -----------------------
def generate_negatives(positives_df):
    negatives = []
    
    for _, group in positives_df.iterrows():
        lat = group["latitude"]
        lon = group["longitude"]
        acq_date = group["acq_date"]
        
        # Pull other fires on the same day to prevent close overlaps
        day_fires = positives_df[positives_df["acq_date"] == acq_date]
        fire_coords = list(zip(day_fires["latitude"], day_fires["longitude"]))

        attempts = 0
        while attempts < 20:
            lat_offset = np.random.uniform(-3.0, 3.0)
            lon_offset = np.random.uniform(-3.0, 3.0)
            
            neg_lat = round(lat + lat_offset, 4)
            neg_lon = round(lon + lon_offset, 4)

            if (LAT_MIN <= neg_lat <= LAT_MAX) and (LON_MIN <= neg_lon <= LON_MAX):
                if globe.is_land(neg_lat, neg_lon):
                    too_close = any(
                        abs(neg_lat - f_lat) < 0.5 and abs(neg_lon - f_lon) < 0.5 
                        for f_lat, f_lon in fire_coords
                    )
                    if not too_close:
                        negatives.append({
                            "latitude": neg_lat,
                            "longitude": neg_lon,
                            "acq_date": acq_date,
                            "fire": 0
                        })
                        break
            attempts += 1
            
    return pd.DataFrame(negatives)

# -----------------------
# HISTORICAL WEATHER FETCH
# -----------------------
def fetch_historical_weather(date_str, coords):
    lats = [c[0] for c in coords]
    lons = [c[1] for c in coords]
    
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": ",".join(map(str, lats)),
        "longitude": ",".join(map(str, lons)),
        "start_date": date_str,
        "end_date": date_str,
        "daily": "temperature_2m_mean,relative_humidity_2m_mean,wind_speed_10m_max,precipitation_sum",
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
    df_negatives = generate_negatives(df_positives)
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
        
        weather_data = fetch_historical_weather(date_str, coords)
        
        if weather_data:
            for idx, entry in enumerate(weather_data):
                daily = entry.get("daily", {})
                
                temp = daily.get("temperature_2m_mean", [None])[0]
                humidity = daily.get("relative_humidity_2m_mean", [None])[0]
                wind = daily.get("wind_speed_10m_max", [None])[0]
                rain = daily.get("precipitation_sum", [None])[0]
                
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
                    "vegetation_index": vegetation_proxy(row["latitude"], row["longitude"]),
                    "fire": row["fire"]
                })
        
        # Respect Open-Meteo API limits
        time.sleep(1.0)

    # Convert to DataFrame
    final_df = pd.DataFrame(weather_records).dropna()
    
    print(f"\nFinal training dataset class breakdown:")
    print(final_df["fire"].value_counts())
    
    final_df.to_csv(OUTPUT_PATH, index=False)
    print(f"Successfully saved compiled dataset to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
    import train