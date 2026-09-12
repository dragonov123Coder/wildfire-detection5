"""Weather ingestion from Open-Meteo, with CSV caching.

The Canadian FWI System's noon-observation convention calls for temperature/
RH/wind at local noon. Open-Meteo's `daily` block doesn't offer a relative
humidity aggregate, but it does offer daily max temperature and max wind
speed -- both good proxies for peak early-afternoon burning conditions (the
standard adaptation used when only daily data is available). So only
relative humidity is fetched hourly (and its local-noon reading extracted);
temperature and wind come from the much smaller daily block. This cuts the
hourly payload to a third of fetching all three variables hourly, which
meaningfully reduces how often Open-Meteo's rate limits get hit.
Results are cached to a local CSV so repeated pipeline runs don't re-fetch
already-downloaded data.
"""
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
TIMEZONE = "America/Edmonton"


def load_config():
    with open(ROOT / "config.json") as f:
        return json.load(f)


def load_grid(config):
    return pd.read_csv(ROOT / config["paths"]["grid_csv"])


class DailyQuotaExceeded(Exception):
    """Open-Meteo's daily API call quota is exhausted. Unlike a transient 429
    burst limit, this doesn't clear for hours -- retrying is pointless until
    it resets, so this should abort the whole pull rather than back off."""


def _fetch_batch(cells, start_date, end_date, config, forecast=False):
    """Fetch hourly RH + daily max temp/max wind/precip for a batch of cells."""
    url = config["apis"]["open_meteo_forecast_url"] if forecast else config["apis"]["open_meteo_archive_url"]
    params = {
        "latitude": ",".join(f"{v:.4f}" for v in cells["center_lat"]),
        "longitude": ",".join(f"{v:.4f}" for v in cells["center_lon"]),
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "relative_humidity_2m",
        "daily": "temperature_2m_max,wind_speed_10m_max,precipitation_sum",
        "timezone": TIMEZONE,
        "wind_speed_unit": "kmh",
    }
    resp = requests.get(url, params=params, timeout=60)
    if resp.status_code == 429:
        try:
            reason = resp.json().get("reason", "")
        except ValueError:
            reason = ""
        if "daily" in reason.lower():
            raise DailyQuotaExceeded(reason or "Daily API request limit exceeded")
    resp.raise_for_status()
    data = resp.json()
    # Open-Meteo returns a bare object for a single location and a list of
    # objects (one per location) when multiple lat/lon pairs are requested.
    if isinstance(data, dict):
        data = [data]
    return data


def _parse_batch(data, cell_ids):
    rows = []
    for cell_id, loc in zip(cell_ids, data):
        hourly = loc["hourly"]
        daily = loc["daily"]
        temp_by_date = dict(zip(daily["time"], daily["temperature_2m_max"]))
        wind_by_date = dict(zip(daily["time"], daily["wind_speed_10m_max"]))
        precip_by_date = dict(zip(daily["time"], daily["precipitation_sum"]))
        for t, rh in zip(hourly["time"], hourly["relative_humidity_2m"]):
            if t.endswith("T12:00"):
                date = t[:10]
                rows.append(
                    {
                        "cell_id": cell_id,
                        "date": date,
                        "temp": temp_by_date.get(date),
                        "rh": rh,
                        "wind": wind_by_date.get(date),
                        "precip": precip_by_date.get(date) or 0.0,
                    }
                )
    return pd.DataFrame(rows)


def _year_chunks(start_date, end_date, max_days=366):
    """Split a date range into <=max_days chunks, to keep individual
    requests (which are hourly-resolution x many cells) a bounded size."""
    chunks = []
    cur = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    while cur <= end:
        chunk_end = min(cur + pd.Timedelta(days=max_days - 1), end)
        chunks.append((cur.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d")))
        cur = chunk_end + pd.Timedelta(days=1)
    return chunks


def _fetch_with_retry(batch, chunk_start, chunk_end, config, forecast, max_retries=8):
    """Fetch one batch/date-chunk, backing off patiently on HTTP 429 (rate
    limit is transient and typically clears within a minute or two) and more
    briskly on other transient errors."""
    for attempt in range(max_retries):
        try:
            return _fetch_batch(batch, chunk_start, chunk_end, config, forecast=forecast)
        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response is not None else None
            wait = min(20 * (2**attempt), 300) if status == 429 else min(2**attempt, 60)
            print(f"    {chunk_start}..{chunk_end}: HTTP {status} ({e}), retrying in {wait}s (attempt {attempt + 1}/{max_retries})...")
            time.sleep(wait)
        except (requests.RequestException, ValueError) as e:
            wait = min(2**attempt, 60)
            print(f"    {chunk_start}..{chunk_end}: request failed ({e}), retrying in {wait}s...")
            time.sleep(wait)
    raise RuntimeError(f"Failed to fetch {chunk_start}..{chunk_end} after {max_retries} attempts")


def load_cache(config):
    path = ROOT / config["paths"]["weather_cache_csv"]
    if path.exists():
        return pd.read_csv(path, dtype={"cell_id": int, "date": str})
    return pd.DataFrame(columns=["cell_id", "date", "temp", "rh", "wind", "precip"])


def save_cache(df, config):
    path = ROOT / config["paths"]["weather_cache_csv"]
    path.parent.mkdir(parents=True, exist_ok=True)
    df.sort_values(["cell_id", "date"]).to_csv(path, index=False)


def _fetch_batch_job(batch, chunk_start, chunk_end, config, forecast, quota_hit):
    """Worker-thread entry point: fetch one batch, or skip immediately if
    another thread has already signalled the daily quota is exhausted."""
    if quota_hit.is_set():
        return ("skipped", batch, None)
    try:
        data = _fetch_with_retry(batch, chunk_start, chunk_end, config, forecast)
        time.sleep(1.5)
        return ("ok", batch, data)
    except DailyQuotaExceeded as e:
        quota_hit.set()
        return ("quota", batch, e)
    except RuntimeError as e:
        return ("failed", batch, e)


def get_weather(grid_df, start_date, end_date, config, forecast=False):
    """Return weather for all grid cells over [start_date, end_date], reading
    from and updating the local CSV cache so repeated runs don't re-fetch
    data that's already been downloaded.

    Iterates date-chunk-by-date-chunk (outer) and cell-batch-by-cell-batch
    (inner), saving after every (chunk, batch) pair -- not just once a batch
    clears every year, or once the whole pull finishes. That way a failure
    on, say, year 7 of 7 doesn't discard years 1-6 that already succeeded
    for that batch; rerunning only has to fill in the specific gap left.

    Batches within a chunk are fetched concurrently
    (apis.max_concurrent_requests workers, default 1 = sequential)."""
    cache = load_cache(config)
    date_chunks = _year_chunks(start_date, end_date)
    batch_size = config["apis"]["max_locations_per_batch"]
    max_workers = config["apis"].get("max_concurrent_requests", 1)

    for chunk_start, chunk_end in date_chunks:
        chunk_dates = pd.date_range(chunk_start, chunk_end).strftime("%Y-%m-%d")
        needed = pd.MultiIndex.from_product([grid_df["cell_id"], chunk_dates], names=["cell_id", "date"])
        if not cache.empty:
            have = pd.MultiIndex.from_frame(cache[["cell_id", "date"]])
            missing_mask = ~needed.isin(have)
        else:
            missing_mask = np.ones(len(needed), dtype=bool)
        missing = needed[missing_mask].to_frame(index=False)
        if missing.empty:
            continue

        missing_cells = sorted(missing["cell_id"].unique())
        cells_df = grid_df[grid_df["cell_id"].isin(missing_cells)].sort_values("cell_id").reset_index(drop=True)
        batches = [cells_df.iloc[i : i + batch_size] for i in range(0, len(cells_df), batch_size)]
        print(
            f"Fetching weather for {len(cells_df)} cells, {chunk_start} to {chunk_end} "
            f"({len(batches)} batch(es) of {batch_size}, {max_workers} concurrent)..."
        )

        quota_hit = threading.Event()
        done_cells = 0
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_fetch_batch_job, b, chunk_start, chunk_end, config, forecast, quota_hit) for b in batches]
            for future in as_completed(futures):
                status, batch, payload = future.result()
                if status == "skipped":
                    continue
                if status == "quota":
                    print(
                        f"  Open-Meteo daily quota exhausted ({payload}). Progress so far is saved -- rerun this "
                        "script later (quota typically resets within a day) to continue from where it left off."
                    )
                    continue
                if status == "failed":
                    print(f"  WARNING: a batch failed permanently ({payload}). Skipping -- rerun later to retry.")
                    continue

                fetched = _parse_batch(payload, batch["cell_id"].tolist())
                cache = pd.concat([cache, fetched], ignore_index=True).drop_duplicates(["cell_id", "date"], keep="last")
                save_cache(cache, config)
                done_cells += len(batch)
                print(f"  saved {done_cells}/{len(cells_df)} cells for {chunk_start}..{chunk_end}")

        if quota_hit.is_set():
            result = cache.merge(grid_df[["cell_id", "center_lat"]], on="cell_id").rename(columns={"center_lat": "lat"})
            mask = (result["date"] >= start_date) & (result["date"] <= end_date) & result["cell_id"].isin(grid_df["cell_id"])
            return result.loc[mask].reset_index(drop=True)

    result = cache.merge(grid_df[["cell_id", "center_lat"]], on="cell_id").rename(columns={"center_lat": "lat"})
    mask = (result["date"] >= start_date) & (result["date"] <= end_date) & result["cell_id"].isin(grid_df["cell_id"])
    return result.loc[mask].reset_index(drop=True)


def main():
    config = load_config()
    grid_df = load_grid(config)
    start = config["dates"]["train_start"]
    end = config["dates"]["train_end"]
    df = get_weather(grid_df, start, end, config)
    print(f"Weather data: {len(df)} rows for {df['cell_id'].nunique()} cells")


if __name__ == "__main__":
    main()
