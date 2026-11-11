# Demand Forecast API

A small production-style FastAPI service that trains a scikit-learn model and forecasts daily grocery demand recursively for up to 14 days. It is intended as an explainable university project; the included data is synthetic and should not be treated as a business forecast.

## Architecture

```text
sales.csv -> shared feature engineering -> time-based split -> model + metadata
                                                   |
client -> FastAPI -> lifespan loads artifact once -> recursive forecasts
```

The application is split into configuration, schemas, shared feature engineering, model service and routers. Training and inference call the same feature functions, reducing train/serve skew. The model is a CPU-only `HistGradientBoostingRegressor`.

## Setup

Python 3.11 or newer is recommended.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

Generate two years of synthetic daily sales for seven grocery products, then train:

```bash
python -m ml.generate_data --out data/sales.csv
python -m ml.train --data data/sales.csv --out models/
uvicorn app.main:app --reload
```

The training command uses the last 60 calendar days for validation, prints the model and baseline comparison, and writes `models/model.joblib` and `models/metadata.json`. Set `DEMAND_MODEL_DIR` and `DEMAND_DATA_PATH` to use other locations.

## API examples

Swagger UI is at `http://localhost:8000/docs`. Example requests:

```bash
curl http://localhost:8000/health
curl http://localhost:8000/model/info
curl -X POST http://localhost:8000/v1/forecast -H 'Content-Type: application/json' -d '{"product_id":"rice","horizon_days":3}'
curl -X POST http://localhost:8000/v1/forecast/batch -H 'Content-Type: application/json' -d '{"product_ids":["rice","milk"],"horizon_days":7}'
curl -X POST 'http://localhost:8000/v1/upload-and-forecast?product_id=rice' -F 'file=@data/sales.csv'
```

Forecast response shape:

```json
{"product_id":"rice","forecast":[{"date":"2026-01-01","predicted_units":51.2,"lower":36.4,"upper":66.0}]}
```

Explicit history can be included in `/v1/forecast` as `history`: an array of at least 28 consecutive daily objects with `date`, `units_sold`, `price`, and optional `is_holiday`. Without it, the API uses the included CSV for that product. Uploads require `date,product_id,units_sold,price,is_holiday`; the product ID must be present in the trained model. Errors use 422 for invalid/short history, 400 for malformed CSV, and 404 for unknown products.

## Use your own data

Prepare a CSV with exactly the required columns `date,product_id,units_sold,price,is_holiday`. Use one row per product per day, consistent product IDs, nonnegative sales and positive prices. Keep dates daily and consecutive within each product. Train with `python -m ml.train --data path/to/your.csv --out models/`; point `DEMAND_DATA_PATH` to that same CSV when serving. The API learns the product vocabulary from training, so new IDs require retraining.

## Metrics

Training prints a comparison table for MAE, RMSE, MAPE (excluding zero actuals), and sMAPE. The actual values from this repository's reproducible generated-data run are recorded below after training; small differences can arise across library versions.

| Model | MAE | RMSE | MAPE | sMAPE |
|---|---:|---:|---:|---:|
| HistGradientBoostingRegressor | 3.92 | 5.68 | 10.34% | 12.69% |
| Same day last week | 4.40 | 6.88 | 12.25% | 15.06% |
| 7-day moving average | 3.91 | 5.59 | 10.93% | 12.76% |

## Configuration and operations

Environment variables use the `DEMAND_` prefix: `DEMAND_MODEL_DIR`, `DEMAND_DATA_PATH`, `DEMAND_API_KEY`, `DEMAND_CORS_ORIGINS`, and `DEMAND_LOG_LEVEL`. When `DEMAND_API_KEY` is set, versioned forecast routes require `X-API-Key`; health and model info remain public. Requests are logged with request ID and latency. CORS origins are comma-separated.

```bash
pytest
ruff check .
docker compose up --build
```

The container runs as a non-root user. Docker Compose mounts `models/` so artifacts can be replaced without rebuilding the image.

## Limitations and next steps

Synthetic demand cannot represent real promotions, stockouts or local calendars. Forecast intervals use a single validation residual standard deviation and are approximate. Price is held at the most recent value during recursive forecasts; future holidays are not inferred. For production, consider Redis caching, MLflow experiment tracking, drift monitoring, richer promotion/holiday inputs, and comparison with Prophet or LightGBM.

## Viva Q&A

**Why a time-based split?** Future observations must not leak into training when estimating how the model will perform on future dates.

**Why recursive forecasting?** The estimator predicts one day at a time. Feeding each estimate back into history lets the next prediction use the same lag features as a real future day.

**Why share feature code?** Training and serving must calculate calendar, lag and rolling values the same way. One implementation prevents the two paths from silently drifting apart.

**Why HistGradientBoosting?** It is included in scikit-learn, handles nonlinear tabular relationships, runs on CPU, and avoids heavyweight dependencies.
