import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime
from global_land_mask import globe

# -----------------------
# CONFIGURATION
# -----------------------
FIRMS_CSV_PATH = "data/firms/viirs-jpss1_2024_Canada.csv"  # Path to your downloaded FIRMS CSV
OUTPUT_PATH = "data/wildfire_dataset.csv"
SAMPLE_SIZE = 300  # Number of positive cases to sample (keeps API requests manageable)

# Canada Bounding Box (Adjust as needed)
LAT_MIN, LAT_MAX = 42.0, 83.0
LON_MIN, LON_MAX = -141.0, -52.0

# -----------------------
# VEGETATION PROXY (MATCHING TRAINING)
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
    """
    Generates a negative (no-fire) sample for each positive fire sample.
    Negative points are located on land, within the bounding box, 
    and offset from known fire locations on the same day.
    """
    negatives = []
    positive_coords_by_date = positives_df.groupby("acq_date")

    for date, group in positives_df.iterrows():
        lat = group["latitude"]
        lon = group["longitude"]
        acq_date = group["acq_date"]
        
        # Get all known fire locations for this day to avoid overlapping
        day_fires = positives_df[positives_df["acq_date"] == acq_date]
        fire_coords = list(zip(day_fires["latitude"], day_fires["longitude"]))

        attempts = 0
        while attempts < 20:
            # Shift coordinates by a random offset (approx 1 to 3 degrees)
            lat_offset = np.random.uniform(-3.0, 3.0)
            lon_offset = np.random.uniform(-3.0, 3.0)
            
            neg_lat = round(lat + lat_offset, 4)
            neg_lon = round(lon + lon_offset, 4)

            # Ensure the negative point is within limits, on land, and not near known fires
            if (LAT_MIN <= neg_lat <= LAT_MAX) and (LON_MIN <= neg_lon <= LON_MAX):
                if globe.is_land(neg_lat, neg_lon):
                    # Check distance from any fire on that day (approx > 0.5 degrees)
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
    """
    Fetches historical weather for multiple coordinates on a specific date 
    using the Open-Meteo Archive API.
    """
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
            # If only 1 coordinate was queried, Open-Meteo returns a dict instead of a list
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
    print("Loading FIRMS data...")
    df_firms = pd.read_csv(FIRMS_CSV_PATH)
    
    # Clean up column names and types
    df_firms.columns = [col.strip().lower() for col in df_firms.columns]
    
    # Filter for geographic boundaries
    df_firms = df_firms[
        (df_firms["latitude"] >= LAT_MIN) & (df_firms["latitude"] <= LAT_MAX) &
        (df_firms["longitude"] >= LON_MIN) & (df_firms["longitude"] <= LON_MAX)
    ]
    
    if len(df_firms) == 0:
        print("No active fires found within the specified bounding box.")
        return

    # Sample a manageable subset of fires to avoid API rate-limiting
    sample_size = min(SAMPLE_SIZE, len(df_firms))
    df_positives = df_firms.sample(n=sample_size, random_state=42).copy()
    df_positives["fire"] = 1
    
    print(f"Sampled {len(df_positives)} active fire cases.")
    
    print("Generating non-fire (negative) cases...")
    df_negatives = generate_negatives(df_positives)
    print(f"Generated {len(df_negatives)} non-fire cases.")

    # Combine positive and negative cases
    df_combined = pd.concat([
        df_positives[["latitude", "longitude", "acq_date", "fire"]],
        df_negatives
    ], ignore_index=True)

    print("\nFetching historical weather from Open-Meteo...")
    weather_records = []
    
    # Group coordinates by date to batch requests efficiently
    grouped_by_date = df_combined.groupby("acq_date")
    total_dates = len(grouped_by_date)
    
    for i, (date_str, group) in enumerate(grouped_by_date, 1):
        coords = list(zip(group["latitude"], group["longitude"]))
        print(f"[{i}/{total_dates}] Fetching weather for {date_str} ({len(coords)} points)...")
        
        weather_data = fetch_historical_weather(date_str, coords)
        
        if weather_data:
            for idx, entry in enumerate(weather_data):
                daily = entry.get("daily", {})
                
                # Fetching the first element from the lists
                temp = daily.get("temperature_2m_mean", [None])[0]
                humidity = daily.get("relative_humidity_2m_mean", [None])[0]
                wind = daily.get("wind_speed_10m_max", [None])[0]
                rain = daily.get("precipitation_sum", [None])[0]
                
                row = group.iloc[idx]
                
                # Parse date to extract Day of Year
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
        
        # Brief pause to respect API rate limits
        time.sleep(1.0)

    # Create final dataset
    final_df = pd.DataFrame(weather_records)
    
    # Drop rows that could not fetch weather successfully
    final_df = final_df.dropna()
    
    print(f"\nFinal dataset class distribution:")
    print(final_df["fire"].value_counts())
    
    final_df.to_csv(OUTPUT_PATH, index=False)
    print(f"Successfully saved compiled dataset to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()