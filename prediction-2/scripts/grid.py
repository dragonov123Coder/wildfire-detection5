"""Build a uniform analysis grid over Alberta in a projected CRS.

Alberta's borders are straight lines of latitude/longitude, so a lat/lon
bounding box is an accurate approximation of the province (no polygon
clipping needed). Cells are built in EPSG:3400 (Alberta 10-TM) so each cell
is a true square of `grid.resolution_m` meters, rather than a lat/lon square
that would be distorted by longitude convergence at these latitudes.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer

ROOT = Path(__file__).resolve().parent.parent


def load_config():
    with open(ROOT / "config.json") as f:
        return json.load(f)


def build_grid(config):
    bbox = config["region"]["bbox"]
    crs = config["region"]["projected_crs"]
    resolution_m = config["grid"]["resolution_m"]

    to_proj = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    to_latlon = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)

    # Project all four corners and take the enclosing box, since a
    # lat/lon bbox is not a rectangle once projected.
    corner_lons = [bbox["lon_min"], bbox["lon_min"], bbox["lon_max"], bbox["lon_max"]]
    corner_lats = [bbox["lat_min"], bbox["lat_max"], bbox["lat_min"], bbox["lat_max"]]
    corner_x, corner_y = to_proj.transform(corner_lons, corner_lats)
    x_min, x_max = min(corner_x), max(corner_x)
    y_min, y_max = min(corner_y), max(corner_y)

    x_edges = np.arange(x_min, x_max + resolution_m, resolution_m)
    y_edges = np.arange(y_min, y_max + resolution_m, resolution_m)

    rows = []
    cell_id = 0
    for yi in range(len(y_edges) - 1):
        cy_min, cy_max = y_edges[yi], y_edges[yi + 1]
        cy_center = (cy_min + cy_max) / 2
        for xi in range(len(x_edges) - 1):
            cx_min, cx_max = x_edges[xi], x_edges[xi + 1]
            cx_center = (cx_min + cx_max) / 2
            rows.append((cell_id, cx_center, cy_center, cx_min, cy_min, cx_max, cy_max))
            cell_id += 1

    df = pd.DataFrame(
        rows,
        columns=["cell_id", "x_center", "y_center", "x_min", "y_min", "x_max", "y_max"],
    )

    lon_center, lat_center = to_latlon.transform(df["x_center"].values, df["y_center"].values)
    df["center_lon"] = lon_center
    df["center_lat"] = lat_center

    # Keep only cells whose centroid falls back within the requested lat/lon
    # bbox (the projected enclosing rectangle is slightly larger than the
    # true bbox near the corners).
    mask = (
        (df["center_lat"] >= bbox["lat_min"])
        & (df["center_lat"] <= bbox["lat_max"])
        & (df["center_lon"] >= bbox["lon_min"])
        & (df["center_lon"] <= bbox["lon_max"])
    )
    df = df.loc[mask].reset_index(drop=True)
    df["cell_id"] = df.index

    return df


def main():
    config = load_config()
    df = build_grid(config)
    out_path = ROOT / config["paths"]["grid_csv"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} grid cells to {out_path}")
    print(
        f"Lat range: {df['center_lat'].min():.3f} - {df['center_lat'].max():.3f}, "
        f"Lon range: {df['center_lon'].min():.3f} - {df['center_lon'].max():.3f}"
    )


if __name__ == "__main__":
    main()
