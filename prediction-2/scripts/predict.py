"""Run inference for a target date, producing a daily fire-risk map.

FWI is a running recursive state, so predicting for any date requires
rolling the calculation forward from the start of that fire season. Recent
dates (within the last few days, including "today") are served from
Open-Meteo's forecast endpoint (which also returns actuals for the very
recent past); older dates use the more reliable historical archive.
"""
import argparse
import json
import sys
from pathlib import Path

import lightgbm as lgb
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

import fwi as fwi_mod
import soil_moisture_nasapower as sm
import weather_openmeteo as wo
from train import ALL_COLUMNS


def load_config():
    with open(ROOT / "config.json") as f:
        return json.load(f)


def load_grid(config):
    return pd.read_csv(ROOT / config["paths"]["grid_csv"])


def load_static_features(config):
    path = ROOT / config["paths"]["static_features_csv"]
    if not path.exists():
        raise FileNotFoundError(
            f"Static features not found at {path}. Run build_dataset.py (or fuel_type.py/terrain.py) first."
        )
    return pd.read_csv(path)


def get_target_weather(grid_df, target_date, config):
    target_ts = pd.Timestamp(target_date)
    season_start = pd.Timestamp(f"{target_ts.year}-{config['dates']['fire_season_start_mmdd']}")
    if target_ts < season_start:
        raise ValueError(f"target_date {target_date} is before the fire season start {season_start.date()}")

    # Open-Meteo's historical archive typically lags a few days behind
    # real time; anything more recent than that is fetched via the
    # forecast endpoint instead (which also serves actuals for very
    # recent past dates, not just future predictions).
    archive_cutoff = min(target_ts, pd.Timestamp.today().normalize() - pd.Timedelta(days=6))

    frames = []
    if season_start <= archive_cutoff:
        frames.append(
            wo.get_weather(
                grid_df, season_start.strftime("%Y-%m-%d"), archive_cutoff.strftime("%Y-%m-%d"), config, forecast=False
            )
        )
        recent_start = archive_cutoff + pd.Timedelta(days=1)
    else:
        recent_start = season_start

    if recent_start <= target_ts:
        frames.append(
            wo.get_weather(grid_df, recent_start.strftime("%Y-%m-%d"), target_ts.strftime("%Y-%m-%d"), config, forecast=True)
        )

    return pd.concat(frames, ignore_index=True).drop_duplicates(["cell_id", "date"])


def get_target_soil_moisture(grid_df, target_date, config, lookback_days=10):
    """NASA POWER has reporting latency, so fall back to the most recent
    available value within a lookback window rather than requiring an exact
    match on target_date."""
    target_ts = pd.Timestamp(target_date)
    start = (target_ts - pd.Timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    soil = sm.get_soil_moisture(grid_df, start, target_ts.strftime("%Y-%m-%d"), config)
    soil = soil.dropna(subset=["soil_moisture"]).sort_values("date")
    return soil.groupby("cell_id").tail(1)[["cell_id", "soil_moisture"]]


def correct_probability(p_sampled, negative_sampling_fraction):
    """Undo the bias introduced by negative-class downsampling at training
    time (King & Zeng 2001 prior correction), so risk_score approximates a
    true probability rather than one inflated by the downsampling ratio."""
    s = negative_sampling_fraction
    return (s * p_sampled) / (s * p_sampled + (1 - p_sampled))


def predict(target_date, config=None):
    config = config or load_config()
    grid_df = load_grid(config)

    print(f"Fetching weather through {target_date}...")
    weather = get_target_weather(grid_df, target_date, config)

    print("Computing FWI series...")
    fwi_df = fwi_mod.compute_fwi_series(
        weather[["cell_id", "date", "lat", "temp", "rh", "wind", "precip"]],
        startup=config["fwi_startup"],
        fire_season_start_mmdd=config["dates"]["fire_season_start_mmdd"],
        strict=False,
    )
    fwi_df["date"] = pd.to_datetime(fwi_df["date"]).dt.strftime("%Y-%m-%d")
    target_fwi = fwi_df[fwi_df["date"] == target_date]
    if target_fwi.empty:
        raise RuntimeError(f"No FWI output for target date {target_date}; check weather coverage.")

    print("Fetching soil moisture...")
    soil = get_target_soil_moisture(grid_df, target_date, config)

    static = load_static_features(config)

    df = target_fwi.merge(soil, on="cell_id", how="left")
    df = df.merge(static, on="cell_id", how="left")
    df = df.merge(grid_df[["cell_id", "center_lat", "center_lon"]], on="cell_id", how="left")

    target_ts = pd.Timestamp(target_date)
    df["month"] = target_ts.month
    df["day_of_year"] = target_ts.dayofyear
    df["fuel_type_code"] = df["fuel_type_code"].astype("category")

    model_path = ROOT / config["paths"]["model_path"]
    model = lgb.Booster(model_file=str(model_path))
    with open(model_path.parent / "model_meta.json") as f:
        meta = json.load(f)

    missing = df[ALL_COLUMNS].isna().any(axis=1)
    if missing.any():
        print(f"Warning: {int(missing.sum())} cells have missing features and will get unreliable risk scores")

    raw_pred = model.predict(df[ALL_COLUMNS])
    df["risk_score_raw"] = raw_pred
    df["risk_score"] = correct_probability(raw_pred, meta["negative_sampling_fraction"])

    out = df[["cell_id", "center_lat", "center_lon", "risk_score", "risk_score_raw", "FWI"]].rename(
        columns={"center_lat": "lat", "center_lon": "lon"}
    )
    out["date"] = target_date
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=pd.Timestamp.today().strftime("%Y-%m-%d"))
    parser.add_argument("--out", default=None, help="Output CSV path (defaults to config paths.risk_map_csv)")
    args = parser.parse_args()

    config = load_config()
    result = predict(args.date, config)

    out_path = Path(args.out) if args.out else ROOT / config["paths"]["risk_map_csv"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_path, index=False)
    print(f"Wrote risk map for {args.date} to {out_path} ({len(result)} cells)")
    print(result["risk_score"].describe())


if __name__ == "__main__":
    main()
