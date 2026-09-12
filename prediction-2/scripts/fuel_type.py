"""Fuel type ingestion from the CFFDRS FBP Fuel Types 2024 national raster (NRCan).

The national raster (~3GB, EPSG:3978 Canada Atlas Lambert) is downloaded once
to data/input/fuel_type_canada_30m.tif. This script reads only the windowed
subset covering the Alberta grid's bbox (not the whole file into memory) and
computes the majority (mode) fuel type class per grid cell via zonal stats.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from rasterio.windows import from_bounds
from rasterstats import zonal_stats
from shapely.geometry import box

ROOT = Path(__file__).resolve().parent.parent


def load_config():
    with open(ROOT / "config.json") as f:
        return json.load(f)


def load_grid(config):
    return pd.read_csv(ROOT / config["paths"]["grid_csv"])


def read_alberta_window(raster_path, grid_df, grid_crs, pad_m=5000):
    with rasterio.open(raster_path) as src:
        to_raster_crs = Transformer.from_crs(grid_crs, src.crs, always_xy=True)
        xs, ys = to_raster_crs.transform(
            [grid_df["x_min"].min() - pad_m, grid_df["x_max"].max() + pad_m],
            [grid_df["y_min"].min() - pad_m, grid_df["y_max"].max() + pad_m],
        )
        left, right = min(xs), max(xs)
        bottom, top = min(ys), max(ys)
        window = from_bounds(left, bottom, right, top, transform=src.transform)
        data = src.read(1, window=window)
        transform = src.window_transform(window)
        nodata = src.nodata
        raster_crs = src.crs
        category_names = None
        try:
            tags = src.tags(1)
            category_names = tags if tags else None
        except Exception:
            pass
    return data, transform, raster_crs, nodata, category_names


def majority_fuel_type(data, transform, raster_crs, grid_df, grid_crs, nodata):
    to_raster_crs = Transformer.from_crs(grid_crs, raster_crs, always_xy=True)
    geoms = []
    for r in grid_df.itertuples():
        xs, ys = to_raster_crs.transform([r.x_min, r.x_max], [r.y_min, r.y_max])
        geoms.append(box(min(xs), min(ys), max(xs), max(ys)))

    stats = zonal_stats(geoms, data, affine=transform, categorical=True, nodata=nodata)
    majority = []
    for s in stats:
        s = {k: v for k, v in s.items() if not (nodata is not None and k == nodata)}
        if not s:
            majority.append(np.nan)
            continue
        majority.append(max(s.items(), key=lambda kv: kv[1])[0])
    return majority


def main():
    config = load_config()
    grid_df = load_grid(config)
    grid_crs = config["region"]["projected_crs"]
    raster_path = ROOT / config["paths"]["fuel_type_raster"]

    print(f"Reading Alberta window from {raster_path}...")
    data, transform, raster_crs, nodata, category_names = read_alberta_window(raster_path, grid_df, grid_crs)
    print(f"Window shape: {data.shape}, raster CRS: {raster_crs}, nodata: {nodata}")
    if category_names:
        print(f"Raster tags (may include legend info): {category_names}")

    print("Computing majority fuel type per grid cell...")
    grid_df["fuel_type_code"] = majority_fuel_type(data, transform, raster_crs, grid_df, grid_crs, nodata)

    out_path = ROOT / "data/processed/fuel_type_features.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    grid_df[["cell_id", "fuel_type_code"]].to_csv(out_path, index=False)
    print(f"Wrote fuel type for {len(grid_df)} cells to {out_path}")
    print("Fuel type code distribution:")
    print(grid_df["fuel_type_code"].value_counts(dropna=False).head(20))


if __name__ == "__main__":
    main()
