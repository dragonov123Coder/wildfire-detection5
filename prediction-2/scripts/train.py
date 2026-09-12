"""Train a LightGBM binary classifier for daily fire-occurrence risk per grid
cell, using a temporal train/val/test split (never train on the future), and
evaluate against a simple FWI-only logistic regression baseline.
"""
import json
from pathlib import Path

import lightgbm as lgb
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent

FEATURE_COLUMNS = [
    "FFMC", "DMC", "DC", "ISI", "BUI", "FWI",
    "soil_moisture", "elevation_m", "slope_deg", "aspect_deg",
    "month", "day_of_year", "center_lat", "center_lon",
]
CATEGORICAL_COLUMNS = ["fuel_type_code"]
ALL_COLUMNS = FEATURE_COLUMNS + CATEGORICAL_COLUMNS


def load_config():
    with open(ROOT / "config.json") as f:
        return json.load(f)


def load_dataset(config):
    df = pd.read_csv(ROOT / config["paths"]["training_dataset_csv"])
    df["date"] = pd.to_datetime(df["date"])
    df["fuel_type_code"] = df["fuel_type_code"].astype("category")
    return df


def temporal_split(df, split_config):
    train_end = pd.Timestamp(split_config["train_end"])
    val_end = pd.Timestamp(split_config["val_end"])
    test_start = pd.Timestamp(split_config["test_start"])
    train = df[df["date"] <= train_end]
    val = df[(df["date"] > train_end) & (df["date"] <= val_end)]
    test = df[df["date"] >= test_start]
    return train, val, test


def evaluate(y_true, y_pred, label):
    if len(y_true) == 0 or y_true.sum() == 0:
        print(f"  {label}: no positive examples, skipping metrics")
        return
    roc = roc_auc_score(y_true, y_pred)
    pr = average_precision_score(y_true, y_pred)
    print(f"  {label}: ROC-AUC={roc:.4f}  PR-AUC={pr:.4f}  n={len(y_true)}  positives={int(y_true.sum())}")


def train_lightgbm(train, val):
    """With only ~1-2K training rows, the previous defaults (num_leaves=31,
    no regularization) overfit within a single tree: validation AUC peaked
    at round 1 and got steadily worse afterward, so early stopping locked in
    a 1-tree model with almost no discriminative power (predictions barely
    left a 0.47-0.52 band). These settings cap tree complexity and add
    regularization/subsampling so boosting can actually make progress past
    round 1 -- empirically verified to reach ~90 trees with a materially
    wider, better-discriminating prediction distribution on held-out data."""
    train_set = lgb.Dataset(train[ALL_COLUMNS], label=train["fire"], categorical_feature=CATEGORICAL_COLUMNS)
    val_set = lgb.Dataset(
        val[ALL_COLUMNS], label=val["fire"], categorical_feature=CATEGORICAL_COLUMNS, reference=train_set
    )
    params = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": 0.03,
        "num_leaves": 15,
        "min_child_samples": 20,
        "lambda_l2": 0.5,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "feature_pre_filter": False,
        "verbose": -1,
    }
    model = lgb.train(
        params,
        train_set,
        num_boost_round=500,
        valid_sets=[val_set],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )
    return model


def train_fwi_baseline(train):
    model = LogisticRegression(class_weight="balanced", max_iter=1000)
    model.fit(train[["FWI"]], train["fire"])
    return model


def main():
    config = load_config()
    df = load_dataset(config)
    train, val, test = temporal_split(df, config["dates"]["temporal_split"])
    print(f"Train: {len(train)} rows ({train['fire'].sum()} positive)")
    print(f"Val:   {len(val)} rows ({val['fire'].sum()} positive)")
    print(f"Test:  {len(test)} rows ({test['fire'].sum()} positive)")

    print("\nFWI-only baseline (logistic regression):")
    baseline = train_fwi_baseline(train)
    evaluate(val["fire"], baseline.predict_proba(val[["FWI"]])[:, 1], "val")
    evaluate(test["fire"], baseline.predict_proba(test[["FWI"]])[:, 1], "test")

    print("\nLightGBM model:")
    model = train_lightgbm(train, val)
    evaluate(val["fire"], model.predict(val[ALL_COLUMNS]), "val")
    evaluate(test["fire"], model.predict(test[ALL_COLUMNS]), "test")

    model_path = ROOT / config["paths"]["model_path"]
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(model_path))
    print(f"\nSaved model to {model_path}")

    dataset_meta_path = (ROOT / config["paths"]["training_dataset_csv"]).parent / "dataset_meta.json"
    neg_sampling_fraction = 1.0
    if dataset_meta_path.exists():
        with open(dataset_meta_path) as f:
            neg_sampling_fraction = json.load(f)["negative_sampling_fraction"]

    meta_path = model_path.parent / "model_meta.json"
    with open(meta_path, "w") as f:
        json.dump(
            {
                "feature_columns": FEATURE_COLUMNS,
                "categorical_columns": CATEGORICAL_COLUMNS,
                "negative_sampling_fraction": neg_sampling_fraction,
            },
            f,
            indent=2,
        )
    print(f"Saved model metadata to {meta_path}")


if __name__ == "__main__":
    main()
