"""Model loading and recursive multi-step forecasting."""

import json
from pathlib import Path

import joblib
import pandas as pd

from app.features import FEATURE_COLUMNS, make_future_feature


class ModelService:
    """Loaded estimator, metadata and optional stored observations."""

    def __init__(self, model_dir: Path, data_path: Path) -> None:
        """Load artifact and metadata once at application startup."""
        artifact = joblib.load(model_dir / "model.joblib")
        self.model = artifact["model"]
        self.feature_columns = artifact.get("feature_columns", FEATURE_COLUMNS)
        self.product_codes: dict[str, int] = artifact["product_codes"]
        self.metadata = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))
        self.data = pd.read_csv(data_path, parse_dates=["date"]) if data_path.exists() else pd.DataFrame()
        self.residual_std = float(self.metadata.get("metrics", {}).get("residual_std", 0.0))

    @property
    def products(self) -> list[str]:
        """Return known product identifiers."""
        return list(self.product_codes)

    def forecast(
        self, product_id: str, horizon_days: int, history: pd.DataFrame | None = None
    ) -> list[dict[str, object]]:
        """Forecast recursively, feeding each estimate into the next day."""
        if product_id not in self.product_codes:
            raise KeyError(product_id)
        if history is None:
            if self.data.empty:
                raise ValueError("No stored sales data is available; provide at least 28 history records.")
            history = self.data[self.data["product_id"].astype(str) == product_id].copy()
        else:
            history = history.copy()
            history["product_id"] = product_id
        history["date"] = pd.to_datetime(history["date"])
        if history["date"].duplicated().any():
            raise ValueError("History must contain one record per date.")
        history = history.sort_values("date")
        if len(history) < 28:
            raise ValueError(f"At least 28 daily history records are required; received {len(history)}.")
        if not history["date"].diff().dropna().eq(pd.Timedelta(days=1)).all():
            raise ValueError("History dates must be consecutive daily records with one record per date.")
        price = float(history.iloc[-1]["price"])
        result = []
        for _ in range(horizon_days):
            next_date = history["date"].max() + pd.Timedelta(days=1)
            feature = make_future_feature(history, next_date, product_id, price, 0, self.product_codes)
            predicted = max(0.0, float(self.model.predict(feature[self.feature_columns])[0]))
            band = 1.96 * self.residual_std
            result.append(
                {
                    "date": next_date.date(),
                    "predicted_units": round(predicted, 2),
                    "lower": round(max(0.0, predicted - band), 2),
                    "upper": round(predicted + band, 2),
                }
            )
            history = pd.concat(
                [
                    history,
                    pd.DataFrame(
                        [
                            {
                                "date": next_date,
                                "product_id": product_id,
                                "units_sold": predicted,
                                "price": price,
                                "is_holiday": 0,
                            }
                        ]
                    ),
                ],
                ignore_index=True,
            )
        return result
