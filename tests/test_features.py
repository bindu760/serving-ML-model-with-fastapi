"""Tests for leakage-safe shared features."""

import pandas as pd

from app.features import FEATURE_COLUMNS, build_features


def test_lags_and_rollings_use_only_prior_days() -> None:
    """Changing a row's target cannot change that row's lag features."""
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=35),
            "product_id": "rice",
            "units_sold": range(35),
            "price": 10.0,
            "is_holiday": 0,
        }
    )
    codes = {"rice": 0}
    features = build_features(frame, codes)
    assert features.loc[28, "lag_1"] == 27
    assert features.loc[28, "lag_7"] == 21
    assert features.loc[28, "lag_14"] == 14
    assert features.loc[28, "lag_28"] == 0
    changed = frame.copy()
    changed.loc[28, "units_sold"] = 9999
    assert build_features(changed, codes).loc[28, "lag_1"] == 27
    assert features.loc[28, "rolling_mean_7"] == 24
    assert list(features[FEATURE_COLUMNS]) == FEATURE_COLUMNS


def test_features_handle_short_history_and_match_future_builder() -> None:
    """Serving future features reuse the same columns and engineered values."""
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=30),
            "product_id": "rice",
            "units_sold": range(30),
            "price": 10.0,
            "is_holiday": 0,
        }
    )
    codes = {"rice": 0}
    train_style = build_features(frame, codes).tail(1)[FEATURE_COLUMNS]
    future_history = pd.concat(
        [
            frame,
            pd.DataFrame(
                [
                    {
                        "date": pd.Timestamp("2025-01-31"),
                        "product_id": "rice",
                        "units_sold": 0,
                        "price": 10,
                        "is_holiday": 0,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    inference_style = build_features(future_history, codes).tail(1)[FEATURE_COLUMNS]
    assert list(inference_style.columns) == list(train_style.columns)
    assert inference_style.iloc[0]["lag_1"] == 29
