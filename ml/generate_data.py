"""Generate reproducible synthetic grocery demand data."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

PRODUCTS = {
    "rice": (42, 2.0),
    "dal": (34, 1.7),
    "cooking_oil": (25, 1.2),
    "sugar": (21, 1.1),
    "noodles": (48, 2.5),
    "milk": (30, 2.2),
    "soap": (17, 0.8),
}
FESTIVALS = [("2024-10-12", "2025-10-02"), ("2024-11-01", "2025-10-21")]


def generate_data(output: Path, years: int = 2, seed: int = 42) -> pd.DataFrame:
    """Create daily sales with weekly patterns, trend, holidays and noise."""
    rng = np.random.default_rng(seed)
    end = pd.Timestamp("2025-12-31")
    dates = pd.date_range(end=end, periods=365 * years, freq="D")
    rows = []
    for product, (base, price_base) in PRODUCTS.items():
        for index, day in enumerate(dates):
            weekend = 1.18 if day.dayofweek >= 5 else 1.0
            annual = 1 + 0.10 * np.sin(2 * np.pi * day.dayofyear / 365.25)
            trend = 1 + 0.00015 * index
            holiday = int(any(abs((day - pd.Timestamp(d)).days) <= 3 for pair in FESTIVALS for d in pair))
            festival = 1.75 if holiday else 1.0
            price = price_base * (1 + 0.025 * np.sin(index / 45) + rng.normal(0, 0.008))
            demand = max(
                0,
                base * weekend * annual * trend * festival / (price / price_base)
                + rng.normal(0, base * 0.11),
            )
            units = 0 if rng.random() < 0.012 else int(round(demand))
            rows.append(
                {
                    "date": day.date(),
                    "product_id": product,
                    "units_sold": units,
                    "price": round(price, 2),
                    "is_holiday": holiday,
                }
            )
    frame = pd.DataFrame(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    return frame


def main() -> None:
    """Parse command line arguments and write sample data."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/sales.csv"))
    parser.add_argument("--years", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    generate_data(args.out, args.years, args.seed)
    print(f"Wrote synthetic sales data to {args.out}")


if __name__ == "__main__":
    main()
