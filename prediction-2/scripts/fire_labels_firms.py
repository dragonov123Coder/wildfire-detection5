"""Fire occurrence labels from FIRMS active fire detections (VIIRS 375m).

Reads one or more FIRMS CSV exports (e.g. yearly Canada-wide VIIRS NOAA-20/
JPSS-1 or S-NPP extracts) from config["paths"]["firms_dir"], filters down to
the Alberta bbox + training date range + confidence, and snaps each surviving
detection to its nearest grid cell + date to produce a binary fire label.

If that directory is empty, FIRMS archive data for a custom region + date
range must be requested manually (there is no stable direct-download URL):
  1. https://firms.modaps.eosdis.nasa.gov/download/ -> "Create New Request"
  2. Region: bounding box covering Alberta (lat 49-60, lon -120 to -110)
  3. Source: VIIRS (S-NPP or NOAA-20/JPSS-1), 375m
  4. Date range: config.json dates.train_start to dates.train_end
  5. Format: CSV -> save into config["paths"]["firms_dir"]
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parent.parent


def load_config():
    with open(ROOT / "config.json") as f:
        return json.load(f)


def load_grid(config):
    return pd.read_csv(ROOT / config["paths"]["grid_csv"])


def load_firms(config, min_confidence=("n", "h")):
    firms_dir = ROOT / config["paths"]["firms_dir"]
    csv_files = sorted(firms_dir.glob("*.csv")) if firms_dir.exists() else []
    if not csv_files:
        bbox = config["region"]["bbox"]
        raise FileNotFoundError(
            f"No FIRMS CSV files found in {firms_dir}.\n"
            "Download them manually from https://firms.modaps.eosdis.nasa.gov/download/ :\n"
            "  1. Choose 'Create New Request'\n"
            f"  2. Region: bounding box lat {bbox['lat_min']}-{bbox['lat_max']}, "
            f"lon {bbox['lon_min']}-{bbox['lon_max']}\n"
            "  3. Source: VIIRS (S-NPP or NOAA-20/JPSS-1), 375m\n"
            f"  4. Date range: {config['dates']['train_start']} to {config['dates']['train_end']}\n"
            "  5. Format: CSV\n"
            f"  6. Save the resulting file(s) into {firms_dir}"
        )

    bbox = config["region"]["bbox"]
    train_start = pd.Timestamp(config["dates"]["train_start"])
    train_end = pd.Timestamp(config["dates"]["train_end"])

    frames = []
    for path in csv_files:
        df = pd.read_csv(path, usecols=["latitude", "longitude", "acq_date", "confidence"])
        # Pre-filter to the Alberta bbox before anything else -- these
        # exports are Canada-wide, so most rows are irrelevant.
        df = df[
            df["latitude"].between(bbox["lat_min"], bbox["lat_max"])
            & df["longitude"].between(bbox["lon_min"], bbox["lon_max"])
        ]
        if min_confidence:
            df = df[df["confidence"].isin(min_confidence)]
        df["date"] = pd.to_datetime(df["acq_date"])
        df = df[(df["date"] >= train_start) & (df["date"] <= train_end)]
        if not df.empty:
            frames.append(df[["latitude", "longitude", "date"]])

    if not frames:
        return pd.DataFrame(columns=["latitude", "longitude", "date"])

    result = pd.concat(frames, ignore_index=True)
    result["date"] = result["date"].dt.strftime("%Y-%m-%d")
    return result


def snap_to_grid(firms_df, grid_df, grid_crs):
    """Assign each FIRMS detection to its nearest grid cell centroid."""
    to_proj = Transformer.from_crs("EPSG:4326", grid_crs, always_xy=True)
    x, y = to_proj.transform(firms_df["longitude"].values, firms_df["latitude"].values)

    tree = cKDTree(grid_df[["x_center", "y_center"]].values)
    resolution_m = grid_df["x_max"].iloc[0] - grid_df["x_min"].iloc[0]
    dist, idx = tree.query(np.column_stack([x, y]))

    out = firms_df.copy()
    out["cell_id"] = grid_df["cell_id"].values[idx]
    # Drop detections that fall outside the grid entirely (nearest cell is
    # implausibly far away -- more than one full cell width off).
    out = out[dist <= resolution_m].drop(columns=["latitude", "longitude"])
    return out


def build_fire_labels(config, grid_df=None):
    grid_df = grid_df if grid_df is not None else load_grid(config)
    firms_df = load_firms(config)
    snapped = snap_to_grid(firms_df, grid_df, config["region"]["projected_crs"])
    labels = snapped.groupby(["cell_id", "date"]).size().reset_index(name="detection_count")
    labels["fire"] = 1
    return labels[["cell_id", "date", "fire", "detection_count"]]


def main():
    config = load_config()
    labels = build_fire_labels(config)
    out_path = ROOT / "data/processed/fire_labels.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    labels.to_csv(out_path, index=False)
    print(f"Wrote {len(labels)} fire cell-days to {out_path}")
    print(f"Unique cells with fire: {labels['cell_id'].nunique()}")
    print(f"Date range: {labels['date'].min()} to {labels['date'].max()}")


if __name__ == "__main__":
    main()
