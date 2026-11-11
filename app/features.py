"""Shared, leakage-safe feature engineering for training and serving."""

from __future__ import annotations

import pandas as pd

FEATURE_COLUMNS = [
    "day_of_week",
    "day_of_month",
    "month",
    "week_of_year",
    "is_weekend",
    "is_holiday",
    "lag_1",
    "lag_7",
    "lag_14",
    "lag_28",
    "rolling_mean_7",
    "rolling_std_7",
    "rolling_mean_28",
    "rolling_std_28",
    "price",
    "product_code",
]


def build_features(history: pd.DataFrame, product_codes: dict[str, int]) -> pd.DataFrame:
    """Build one row per observation from past-only product history."""
    frame = history.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.sort_values(["product_id", "date"]).reset_index(drop=True)
    if "is_holiday" not in frame:
        frame["is_holiday"] = 0
    frame["is_holiday"] = frame["is_holiday"].fillna(0).astype(int)
    dates = frame["date"]
    frame["day_of_week"] = dates.dt.dayofweek
    frame["day_of_month"] = dates.dt.day
    frame["month"] = dates.dt.month
    frame["week_of_year"] = dates.dt.isocalendar().week.astype(int)
    frame["is_weekend"] = dates.dt.dayofweek.isin([5, 6]).astype(int)
    grouped = frame.groupby("product_id", sort=False)["units_sold"]
    for lag in (1, 7, 14, 28):
        frame[f"lag_{lag}"] = grouped.shift(lag)
    prior = grouped.shift(1)
    for window in (7, 28):
        roll = prior.groupby(frame["product_id"], sort=False).rolling(window, min_periods=1)
        frame[f"rolling_mean_{window}"] = roll.mean().reset_index(level=0, drop=True).sort_index()
        frame[f"rolling_std_{window}"] = roll.std().reset_index(level=0, drop=True).sort_index().fillna(0)
    frame["product_code"] = frame["product_id"].map(product_codes).fillna(-1).astype(int)
    frame[FEATURE_COLUMNS] = (
        frame[FEATURE_COLUMNS].replace([float("inf"), float("-inf")], float("nan")).fillna(0)
    )
    return frame


def make_future_feature(
    history: pd.DataFrame,
    date: pd.Timestamp,
    product_id: str,
    price: float,
    is_holiday: int,
    product_codes: dict[str, int],
) -> pd.DataFrame:
    """Build one feature row for a future date."""
    future = pd.DataFrame(
        [
            {
                "date": date,
                "product_id": product_id,
                "units_sold": 0.0,
                "price": price,
                "is_holiday": is_holiday,
            }
        ]
    )
    return build_features(pd.concat([history, future], ignore_index=True), product_codes).tail(1)[
        FEATURE_COLUMNS
    ]
