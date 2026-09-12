"""Soil moisture ingestion from NASA POWER, with CSV caching.

NASA POWER's point API only serves one lat/lon per request, but its
underlying MERRA-2 data is on a coarse ~0.5 deg (lat) x 0.625 deg (lon) grid
-- much coarser than our 10km analysis grid. So instead of querying once per
grid cell, we snap every grid cell to its nearest native NASA POWER grid
point and only query the small set of unique points that cover Alberta
(a few hundred, not ~7800), then join each cell to its nearest point's
values. This keeps the number of API calls small and each call is a cheap,
reliable long single-point time series (NASA POWER's strength, unlike its
slower regional/bounding-box endpoint).
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent

# Native MERRA-2 grid spacing underlying NASA POWER.
POWER_LAT_STEP = 0.5
POWER_LON_STEP = 0.625
PARAMETER = "GWETROOT"  # root-zone soil wetness, fraction 0-1


def load_config():
    with open(ROOT / "config.json") as f:
        return json.load(f)


def load_grid(config):
    return pd.read_csv(ROOT / config["paths"]["grid_csv"])


def snap_to_power_grid(lat, lon):
    power_lat = np.round(lat / POWER_LAT_STEP) * POWER_LAT_STEP
    power_lon = np.round(lon / POWER_LON_STEP) * POWER_LON_STEP
    return power_lat, power_lon


def _fetch_point(power_lat, power_lon, start_date, end_date, config, max_retries=8):
    url = config["apis"]["nasa_power_url"]
    params = {
        "parameters": PARAMETER,
        "community": "AG",
        "longitude": power_lon,
        "latitude": power_lat,
        "start": start_date.replace("-", ""),
        "end": end_date.replace("-", ""),
        "format": "JSON",
    }
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, params=params, timeout=60)
            resp.raise_for_status()
            payload = resp.json()
            series = payload["properties"]["parameter"][PARAMETER]
            df = pd.DataFrame(
                {
                    "power_lat": power_lat,
                    "power_lon": power_lon,
                    "date": [f"{d[:4]}-{d[4:6]}-{d[6:8]}" for d in series.keys()],
                    "soil_moisture": list(series.values()),
                }
            )
            # NASA POWER uses -999 as a missing-data sentinel.
            df["soil_moisture"] = df["soil_moisture"].replace(-999, np.nan)
            return df
        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response is not None else None
            wait = min(20 * (2**attempt), 300) if status == 429 else min(2**attempt, 60)
            print(f"  point ({power_lat},{power_lon}): HTTP {status} ({e}), retrying in {wait}s...")
            time.sleep(wait)
        except (requests.RequestException, ValueError, KeyError) as e:
            wait = min(2**attempt, 60)
            print(f"  point ({power_lat},{power_lon}): request failed ({e}), retrying in {wait}s...")
            time.sleep(wait)
    raise RuntimeError(f"Failed to fetch NASA POWER data for point ({power_lat},{power_lon})")


def load_cache(config):
    path = ROOT / config["paths"]["soil_moisture_cache_csv"]
    if path.exists():
        return pd.read_csv(path, dtype={"date": str})
    return pd.DataFrame(columns=["power_lat", "power_lon", "date", "soil_moisture"])


def save_cache(df, config):
    path = ROOT / config["paths"]["soil_moisture_cache_csv"]
    path.parent.mkdir(parents=True, exist_ok=True)
    df.sort_values(["power_lat", "power_lon", "date"]).to_csv(path, index=False)


def get_soil_moisture(grid_df, start_date, end_date, config):
    """Return soil moisture per grid cell over [start_date, end_date], joined
    from the nearest native NASA POWER grid point, using and updating a
    local CSV cache keyed by (power_lat, power_lon, date)."""
    grid_df = grid_df.copy()
    grid_df["power_lat"], grid_df["power_lon"] = snap_to_power_grid(
        grid_df["center_lat"].values, grid_df["center_lon"].values
    )
    unique_points = grid_df[["power_lat", "power_lon"]].drop_duplicates().reset_index(drop=True)

    cache = load_cache(config)
    date_range = pd.date_range(start_date, end_date).strftime("%Y-%m-%d")

    if not cache.empty:
        needed = pd.MultiIndex.from_product(
            [zip(unique_points["power_lat"], unique_points["power_lon"]), date_range]
        )
        have_points = set(zip(cache["power_lat"], cache["power_lon"], cache["date"]))
        missing_points = [
            (lat, lon)
            for lat, lon in zip(unique_points["power_lat"], unique_points["power_lon"])
            if not all((lat, lon, d) in have_points for d in date_range)
        ]
    else:
        missing_points = list(zip(unique_points["power_lat"], unique_points["power_lon"]))

    if not missing_points:
        print("All requested soil moisture data already cached.")
    else:
        print(f"Fetching soil moisture for {len(missing_points)} unique NASA POWER points...")
        # Save after every point (not just at the end) so a crash or
        # transient failure partway through only loses the in-progress
        # point -- rerun to pick up where it left off.
        for i, (lat, lon) in enumerate(missing_points):
            try:
                fetched = _fetch_point(lat, lon, start_date, end_date, config)
            except RuntimeError as e:
                print(f"  WARNING: point ({lat},{lon}) failed permanently ({e}). Skipping -- rerun later to retry.")
                continue
            cache = pd.concat([cache, fetched], ignore_index=True).drop_duplicates(
                ["power_lat", "power_lon", "date"], keep="last"
            )
            save_cache(cache, config)
            print(f"  fetched and saved point {i + 1}/{len(missing_points)} ({lat}, {lon})")
            time.sleep(0.3)

    merged = grid_df[["cell_id", "power_lat", "power_lon"]].merge(cache, on=["power_lat", "power_lon"], how="left")
    mask = (merged["date"] >= start_date) & (merged["date"] <= end_date)
    return merged.loc[mask, ["cell_id", "date", "soil_moisture"]].reset_index(drop=True)


def main():
    config = load_config()
    grid_df = load_grid(config)
    start = config["dates"]["train_start"]
    end = config["dates"]["train_end"]
    df = get_soil_moisture(grid_df, start, end, config)
    print(f"Soil moisture data: {len(df)} rows for {df['cell_id'].nunique()} cells")
    print(f"Missing values: {df['soil_moisture'].isna().sum()}")


if __name__ == "__main__":
    main()
