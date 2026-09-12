"""Assemble the final training dataset.

Joins the daily FWI System outputs, soil moisture, static fuel/terrain
features, and fire occurrence labels into one (cell_id, date) table, then
downsamples the (overwhelming) negative class since fire is a rare event.
"""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

import fire_labels_firms as fl
import fwi as fwi_mod
import soil_moisture_nasapower as sm
import weather_openmeteo as wo


def load_config():
    with open(ROOT / "config.json") as f:
        return json.load(f)


def load_grid(config):
    return pd.read_csv(ROOT / config["paths"]["grid_csv"])


def load_static_features(config, grid_df):
    fuel_path = ROOT / "data/processed/fuel_type_features.csv"
    terrain_path = ROOT / "data/processed/terrain_features.csv"
    if not fuel_path.exists() or not terrain_path.exists():
        raise FileNotFoundError("Static features not found. Run scripts/fuel_type.py and scripts/terrain.py first.")
    fuel = pd.read_csv(fuel_path)
    terrain = pd.read_csv(terrain_path)
    static = grid_df[["cell_id"]].merge(fuel, on="cell_id", how="left").merge(terrain, on="cell_id", how="left")

    out_path = ROOT / config["paths"]["static_features_csv"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    static.to_csv(out_path, index=False)
    return static


def join_dataset(fwi_df, soil_df, static_df, labels_df, grid_df):
    """Join FWI + soil moisture + static features + labels on (cell_id, date)."""
    df = fwi_df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")

    df = df.merge(soil_df[["cell_id", "date", "soil_moisture"]], on=["cell_id", "date"], how="left")
    df = df.merge(static_df, on="cell_id", how="left")
    df = df.merge(labels_df[["cell_id", "date", "fire"]], on=["cell_id", "date"], how="left")
    df["fire"] = df["fire"].fillna(0).astype(int)

    dt = pd.to_datetime(df["date"])
    df["month"] = dt.dt.month
    df["day_of_year"] = dt.dt.dayofyear

    df = df.merge(grid_df[["cell_id", "center_lat", "center_lon"]], on="cell_id", how="left")
    return df


def downsample_negatives(df, ratio, seed):
    """Keep all fire-positive rows; randomly sample negatives at up to
    `ratio` times the positive count so the training table stays a
    manageable size despite fire being an extremely rare event.

    Returns (sampled_df, negative_sampling_fraction). The fraction (kept /
    total negatives) is needed later to correct predicted probabilities back
    to the true (un-undersampled) scale -- see predict.py.
    """
    positives = df[df["fire"] == 1]
    negatives = df[df["fire"] == 0]
    n_neg = min(len(negatives), max(len(positives), 1) * ratio)
    negatives_sampled = negatives.sample(n=n_neg, random_state=seed) if n_neg < len(negatives) else negatives
    out = pd.concat([positives, negatives_sampled], ignore_index=True)
    out = out.sample(frac=1, random_state=seed).reset_index(drop=True)
    fraction = len(negatives_sampled) / len(negatives) if len(negatives) > 0 else 1.0
    return out, fraction


def fetch_weather_and_fwi_for_cell_years(cell_years, grid_df, config):
    """Fetch weather and compute FWI only for the specific (cell, year) pairs
    needed -- one full calendar year per year needed, not the full grid or
    the full multi-year history. This is the key cost-saver versus
    build_full_dataset: FWI needs a cell's full-year history to compute
    correctly, but it does NOT need every cell in the grid, and it does not
    need years that aren't actually being sampled from.

    Uses the full calendar year rather than just the configured fire season
    (Mar-Nov) because real fire detections occur in every month (~9% of
    FIRMS detections fall in Dec/Jan/Feb) -- restricting to the nominal
    season would silently drop those positives from the sample."""
    season_start_mmdd = config["dates"]["fire_season_start_mmdd"]

    fwi_frames = []
    for year, group in cell_years.groupby("year"):
        cells_needed = grid_df[grid_df["cell_id"].isin(group["cell_id"])].reset_index(drop=True)
        start_date = f"{year}-01-01"
        end_date = f"{year}-12-31"
        print(f"  year {year}: fetching weather for {len(cells_needed)} cells, {start_date} to {end_date}...")
        weather = wo.get_weather(cells_needed, start_date, end_date, config)
        fwi_df = fwi_mod.compute_fwi_series(
            weather[["cell_id", "date", "lat", "temp", "rh", "wind", "precip"]],
            startup=config["fwi_startup"],
            fire_season_start_mmdd=season_start_mmdd,
            strict=False,
        )
        fwi_frames.append(fwi_df)
    return pd.concat(fwi_frames, ignore_index=True)


def build_sampled_dataset(config, grid_df, n_positive, neg_pos_ratio, seed):
    """Build a small training table by sampling a fixed number of positive
    fire cell-days and fetching weather ONLY for the cell-years those
    positives touch, drawing negatives from other days in that same
    already-fetched cell-year pool (zero extra API cost). This trades
    representativeness of the full Alberta grid for a drastically smaller
    API footprint -- appropriate for fast local iteration/testing under a
    tight request budget, not as the final production dataset (see the
    caveat printed at the end about what the resulting risk scores mean).
    """
    print("Loading fire labels...")
    labels = fl.build_fire_labels(config, grid_df)

    n = min(n_positive, len(labels))
    print(f"Sampling {n} of {len(labels)} positive fire cell-days...")
    positive_labels = labels.sample(n=n, random_state=seed).reset_index(drop=True)
    positive_labels["year"] = pd.to_datetime(positive_labels["date"]).dt.year
    cell_years = positive_labels[["cell_id", "year"]].drop_duplicates().reset_index(drop=True)
    print(f"  -> {len(cell_years)} unique (cell, year) pairs to fetch weather for")

    print("Fetching weather + computing FWI for sampled cell-years only...")
    fwi_df = fetch_weather_and_fwi_for_cell_years(cell_years, grid_df, config)
    fwi_df["date"] = pd.to_datetime(fwi_df["date"]).dt.strftime("%Y-%m-%d")

    n_negative_target = max(int(round(n * neg_pos_ratio)), 0)
    fwi_idx = pd.MultiIndex.from_frame(fwi_df[["cell_id", "date"]])
    pos_idx = pd.MultiIndex.from_frame(positive_labels[["cell_id", "date"]])
    negative_candidates = fwi_df[~fwi_idx.isin(pos_idx)]
    n_negative = min(n_negative_target, len(negative_candidates))
    print(f"Sampling {n_negative} negative (non-fire) examples from the same cell-years (no extra API calls)...")
    negative_rows = negative_candidates.sample(n=n_negative, random_state=seed).reset_index(drop=True)
    negative_rows["fire"] = 0

    positive_fwi = fwi_df.merge(positive_labels[["cell_id", "date"]], on=["cell_id", "date"], how="inner")
    positive_fwi["fire"] = 1

    combined = pd.concat([positive_fwi, negative_rows], ignore_index=True)

    print("Fetching soil moisture for the sampled cells (already cached if the earlier full pull covers them)...")
    sample_cells = grid_df[grid_df["cell_id"].isin(combined["cell_id"].unique())].reset_index(drop=True)
    soil = sm.get_soil_moisture(sample_cells, combined["date"].min(), combined["date"].max(), config)

    print("Loading static features...")
    static = load_static_features(config, grid_df)

    df = combined.merge(soil[["cell_id", "date", "soil_moisture"]], on=["cell_id", "date"], how="left")
    df = df.merge(static, on="cell_id", how="left")
    dt = pd.to_datetime(df["date"])
    df["month"] = dt.dt.month
    df["day_of_year"] = dt.dt.dayofyear
    df = df.merge(grid_df[["cell_id", "center_lat", "center_lon"]], on="cell_id", how="left")
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    fraction = n_negative / len(negative_candidates) if len(negative_candidates) > 0 else 1.0
    print(
        "\nNOTE: this dataset only draws from cell-years that had at least one recorded fire "
        "(a fire-experienced-geography sample, not a uniform random sample of all Alberta cells/days). "
        "Treat risk_score as a relative ranking within fire-prone conditions, not a calibrated absolute "
        "probability across the whole province -- for that, rerun with build_full_dataset() once the full "
        "grid weather pull completes."
    )
    return df, fraction


def build_full_dataset(config, grid_df, start_date, end_date):
    print("Fetching weather...")
    weather = wo.get_weather(grid_df, start_date, end_date, config)

    print("Computing FWI series...")
    fwi_df = fwi_mod.compute_fwi_series(
        weather[["cell_id", "date", "lat", "temp", "rh", "wind", "precip"]],
        startup=config["fwi_startup"],
        strict=False,
        fire_season_start_mmdd=config["dates"]["fire_season_start_mmdd"],
    )

    print("Fetching soil moisture...")
    soil = sm.get_soil_moisture(grid_df, start_date, end_date, config)

    print("Loading static features...")
    static = load_static_features(config, grid_df)

    print("Loading fire labels...")
    labels = fl.build_fire_labels(config, grid_df)

    print("Joining...")
    return join_dataset(fwi_df, soil, static, labels, grid_df)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--full",
        action="store_true",
        help="Build the full-grid dataset (build_full_dataset) instead of the API-budget-friendly sampled one.",
    )
    args = parser.parse_args()

    config = load_config()
    grid_df = load_grid(config)

    if args.full:
        start = config["dates"]["train_start"]
        end = config["dates"]["train_end"]
        df = build_full_dataset(config, grid_df, start, end)
        n_fire = df["fire"].sum()
        print(f"Full assembled table: {len(df)} rows, {n_fire} fire-positive cell-days ({100 * df['fire'].mean():.4f}%)")
        sampled, neg_sampling_fraction = downsample_negatives(
            df, config["training"]["negative_downsample_ratio"], config["training"]["random_seed"]
        )
    else:
        n_positive = config["training"]["n_positive_samples"]
        ratio = config["training"]["positive_negative_ratio"]
        seed = config["training"]["random_seed"]
        sampled, neg_sampling_fraction = build_sampled_dataset(config, grid_df, n_positive, ratio, seed)

    print(f"Final dataset: {len(sampled)} rows, {sampled['fire'].sum()} positive")
    print(f"Negative sampling fraction: {neg_sampling_fraction:.6f}")

    out_path = ROOT / config["paths"]["training_dataset_csv"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sampled.to_csv(out_path, index=False)
    print(f"Wrote training dataset to {out_path}")

    meta_path = out_path.parent / "dataset_meta.json"
    with open(meta_path, "w") as f:
        json.dump(
            {
                "negative_sampling_fraction": neg_sampling_fraction,
                "n_positive": int(sampled["fire"].sum()),
                "n_rows": int(len(sampled)),
                "mode": "full" if args.full else "sampled",
            },
            f,
            indent=2,
        )
    print(f"Wrote dataset metadata to {meta_path}")


if __name__ == "__main__":
    main()
