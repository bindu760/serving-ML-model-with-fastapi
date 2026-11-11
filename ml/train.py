"""Train and evaluate the retail demand model."""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from app.features import FEATURE_COLUMNS, build_features
from ml.evaluate import regression_metrics


def train(data_path: Path, output_dir: Path) -> dict[str, object]:
    """Time-split data, compare baselines, fit model, and save artifacts."""
    data = pd.read_csv(data_path, parse_dates=["date"])
    required = {"date", "product_id", "units_sold", "price", "is_holiday"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Input CSV missing columns: {', '.join(sorted(missing))}")
    data["product_id"] = data["product_id"].astype(str)
    data = data.sort_values(["product_id", "date"]).reset_index(drop=True)
    codes = {product: index for index, product in enumerate(sorted(data["product_id"].unique()))}
    features = build_features(data, codes)
    cutoff = data["date"].max() - pd.Timedelta(days=59)
    train_mask = features["date"] < cutoff
    valid_mask = ~train_mask
    X_train, y_train = (
        features.loc[train_mask, FEATURE_COLUMNS],
        features.loc[train_mask, "units_sold"].astype(float),
    )
    X_valid, y_valid = (
        features.loc[valid_mask, FEATURE_COLUMNS],
        features.loc[valid_mask, "units_sold"].astype(float),
    )
    if X_valid.empty or X_train.empty:
        raise ValueError("Need more than 60 days of data for the requested time-based split.")
    estimator = HistGradientBoostingRegressor(
        max_iter=160, learning_rate=0.08, max_leaf_nodes=15, l2_regularization=1.0, random_state=42
    )
    estimator.fit(X_train, y_train)
    predicted = np.maximum(0, estimator.predict(X_valid))
    naive = features.loc[valid_mask, "lag_7"].to_numpy(float)
    rolling = features.loc[valid_mask, "rolling_mean_7"].to_numpy(float)
    metrics = {
        "model": regression_metrics(y_valid.to_numpy(), predicted),
        "same_day_last_week": regression_metrics(y_valid.to_numpy(), naive),
        "moving_average_7": regression_metrics(y_valid.to_numpy(), rolling),
    }
    print(f"{'Model':<22} {'MAE':>9} {'RMSE':>9} {'MAPE':>9} {'sMAPE':>9}")
    for name, values in metrics.items():
        print(
            f"{name:<22} {values['mae']:9.2f} {values['rmse']:9.2f} {values['mape']:8.2f}% {values['smape']:8.2f}%"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"model": estimator, "feature_columns": FEATURE_COLUMNS, "product_codes": codes},
        output_dir / "model.joblib",
    )
    metadata = {
        "version": "1.0.0",
        "trained_at": datetime.now(UTC).isoformat(),
        "metrics": {**metrics["model"], "validation": {k: v for k, v in metrics.items() if k != "model"}},
        "training_date_range": {"start": str(data.date.min().date()), "end": str(data.date.max().date())},
        "products": sorted(codes),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def main() -> None:
    """Train from command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/sales.csv"))
    parser.add_argument("--out", type=Path, default=Path("models"))
    args = parser.parse_args()
    train(args.data, args.out)


if __name__ == "__main__":
    main()
