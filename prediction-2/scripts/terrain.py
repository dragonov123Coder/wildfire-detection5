"""Terrain ingestion from Copernicus GLO-30 DEM.

Downloads the 1-degree DEM tiles covering the analysis bbox from the public
`copernicus-dem-30m` S3 bucket (no credentials needed), builds a decimated
mosaic (native 30m is far finer than needed for averaging over 10km grid
cells), reprojects it into the grid's projected CRS to compute slope/aspect
in meters, then zonal-stats mean elevation/slope/aspect per grid cell.
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import requests
from rasterio.io import MemoryFile
from rasterio.merge import merge as rio_merge
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterstats import zonal_stats
from shapely.geometry import box

ROOT = Path(__file__).resolve().parent.parent
DEM_BASE_URL = "https://copernicus-dem-30m.s3.amazonaws.com"
TILE_DECIMATED_SIZE = 400  # pixels per side per 1deg tile after decimation (~278m/px)


def load_config():
    with open(ROOT / "config.json") as f:
        return json.load(f)


def load_grid(config):
    return pd.read_csv(ROOT / config["paths"]["grid_csv"])


def tile_list_for_bbox(bbox):
    lat_start = int(np.floor(bbox["lat_min"]))
    lat_end = int(np.floor(bbox["lat_max"] - 1e-9))
    lon_start = int(np.floor(bbox["lon_min"]))
    lon_end = int(np.floor(bbox["lon_max"] - 1e-9))
    return [(lat, lon) for lat in range(lat_start, lat_end + 1) for lon in range(lon_start, lon_end + 1)]


def tile_name(lat, lon):
    lat_str = f"N{lat:02d}" if lat >= 0 else f"S{-lat:02d}"
    lon_str = f"W{-lon:03d}" if lon < 0 else f"E{lon:03d}"
    return f"Copernicus_DSM_COG_10_{lat_str}_00_{lon_str}_00_DEM"


def download_tiles(tiles, dest_dir, max_retries=4):
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, (lat, lon) in enumerate(tiles):
        name = tile_name(lat, lon)
        out_path = dest_dir / f"{name}.tif"
        if out_path.exists() and out_path.stat().st_size > 0:
            paths.append(out_path)
            print(f"  [{i + 1}/{len(tiles)}] {name} (cached)")
            continue
        url = f"{DEM_BASE_URL}/{name}/{name}.tif"
        for attempt in range(max_retries):
            try:
                resp = requests.get(url, timeout=120)
                if resp.status_code == 404:
                    print(f"  [{i + 1}/{len(tiles)}] {name}: not available (404), skipping")
                    break
                resp.raise_for_status()
                out_path.write_bytes(resp.content)
                paths.append(out_path)
                print(f"  [{i + 1}/{len(tiles)}] {name}: downloaded ({len(resp.content) / 1e6:.1f} MB)")
                break
            except requests.RequestException as e:
                wait = 2**attempt
                print(f"  [{i + 1}/{len(tiles)}] {name}: failed ({e}), retrying in {wait}s...")
                time.sleep(wait)
        else:
            print(f"  WARNING: could not download {name} after {max_retries} attempts, skipping")
    return paths


def _read_decimated(path, out_size=TILE_DECIMATED_SIZE):
    with rasterio.open(path) as src:
        data = src.read(
            1, out_shape=(1, out_size, out_size), resampling=Resampling.average, masked=True
        ).astype("float32")
        data = data.filled(np.nan)
        transform = src.transform * src.transform.scale(src.width / out_size, src.height / out_size)
        profile = src.profile.copy()
        profile.update(height=out_size, width=out_size, transform=transform, dtype="float32", nodata=np.nan)
    memfile = MemoryFile()
    with memfile.open(**profile) as dst:
        dst.write(data, 1)
    return memfile


def build_mosaic(tile_paths):
    memfiles = [_read_decimated(p) for p in tile_paths]
    datasets = [mf.open() for mf in memfiles]
    src_crs = datasets[0].crs
    mosaic, transform = rio_merge(datasets, resampling=Resampling.average, nodata=np.nan)
    for ds in datasets:
        ds.close()
    for mf in memfiles:
        mf.close()
    return mosaic[0], transform, src_crs


def reproject_to_projected(mosaic, transform, src_crs, dst_crs, dst_resolution_m):
    height, width = mosaic.shape
    bounds = rasterio.transform.array_bounds(height, width, transform)
    dst_transform, dst_width, dst_height = calculate_default_transform(
        src_crs, dst_crs, width, height, *bounds, resolution=dst_resolution_m
    )
    dst = np.full((dst_height, dst_width), np.nan, dtype="float32")
    reproject(
        source=mosaic,
        destination=dst,
        src_transform=transform,
        src_crs=src_crs,
        src_nodata=np.nan,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        dst_nodata=np.nan,
        resampling=Resampling.bilinear,
    )
    return dst, dst_transform


def compute_slope_aspect(elevation, resolution_m):
    """Approximate slope (degrees from horizontal) and aspect (compass
    degrees, 0=north) from a projected elevation grid via finite differences.
    Precision is not critical here -- terrain is averaged over 10km cells."""
    gy, gx = np.gradient(elevation, resolution_m)
    slope = np.degrees(np.arctan(np.sqrt(gx**2 + gy**2)))
    aspect = (np.degrees(np.arctan2(-gx, -gy)) + 360) % 360
    return slope, aspect


def zonal_mean(array, transform, grid_df):
    geoms = [box(r.x_min, r.y_min, r.x_max, r.y_max) for r in grid_df.itertuples()]
    stats = zonal_stats(geoms, array, affine=transform, stats=["mean"], nodata=np.nan)
    return [s["mean"] for s in stats]


def main():
    config = load_config()
    grid_df = load_grid(config)
    bbox = config["region"]["bbox"]
    dem_dir = ROOT / config["paths"]["dem_tiles_dir"]

    tiles = tile_list_for_bbox(bbox)
    print(f"Need {len(tiles)} DEM tiles covering the bbox")
    tile_paths = download_tiles(tiles, dem_dir)
    print(f"Have {len(tile_paths)} tiles, building decimated mosaic...")

    mosaic, transform, src_crs = build_mosaic(tile_paths)
    dst_crs = config["region"]["projected_crs"]
    resolution_m = 200
    elevation, dst_transform = reproject_to_projected(mosaic, transform, src_crs, dst_crs, resolution_m)
    slope, aspect = compute_slope_aspect(elevation, resolution_m)

    grid_df["elevation_m"] = zonal_mean(elevation, dst_transform, grid_df)
    grid_df["slope_deg"] = zonal_mean(slope, dst_transform, grid_df)
    grid_df["aspect_deg"] = zonal_mean(aspect, dst_transform, grid_df)

    out_path = ROOT / "data/processed/terrain_features.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    grid_df[["cell_id", "elevation_m", "slope_deg", "aspect_deg"]].to_csv(out_path, index=False)
    print(f"Wrote terrain features for {len(grid_df)} cells to {out_path}")


if __name__ == "__main__":
    main()
