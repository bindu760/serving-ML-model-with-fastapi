"""Forecast API routes."""

import io

import pandas as pd
from fastapi import APIRouter, File, HTTPException, Request, UploadFile

from app.schemas import BatchForecastRequest, ForecastRequest

router = APIRouter(prefix="/v1", tags=["forecast"])
REQUIRED_COLUMNS = {"date", "product_id", "units_sold", "price", "is_holiday"}


def _forecast(
    request: Request, product_id: str, days: int, history: pd.DataFrame | None = None
) -> dict[str, object]:
    """Call forecasting service and translate domain errors."""
    try:
        rows = request.app.state.model_service.forecast(product_id, days, history)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Unknown product_id: {product_id}") from None
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"product_id": product_id, "forecast": rows}


@router.post("/forecast")
def forecast(payload: ForecastRequest, request: Request) -> dict[str, object]:
    """Forecast one product."""
    history = pd.DataFrame([record.model_dump() for record in payload.history]) if payload.history else None
    return _forecast(request, payload.product_id, payload.horizon_days, history)


@router.post("/forecast/batch")
def forecast_batch(payload: BatchForecastRequest, request: Request) -> dict[str, object]:
    """Forecast each requested product."""
    return {
        "forecasts": [_forecast(request, product, payload.horizon_days) for product in payload.product_ids]
    }


@router.post("/upload-and-forecast")
async def upload_and_forecast(
    request: Request, file: UploadFile = File(...), product_id: str | None = None, horizon_days: int = 7
) -> dict[str, object]:
    """Validate a CSV upload and forecast a selected product."""
    if not 1 <= horizon_days <= 14:
        raise HTTPException(status_code=422, detail="horizon_days must be between 1 and 14")
    try:
        frame = pd.read_csv(io.BytesIO(await file.read()))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {exc}") from exc
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise HTTPException(
            status_code=400, detail=f"CSV is missing required columns: {', '.join(sorted(missing))}"
        )
    if frame.empty:
        raise HTTPException(status_code=400, detail="CSV contains no sales records")
    selected = product_id or str(frame["product_id"].iloc[0])
    selected_frame = frame[frame["product_id"].astype(str) == selected].copy()
    if selected_frame.empty:
        raise HTTPException(status_code=404, detail=f"CSV contains no history for product_id: {selected}")
    return _forecast(request, selected, horizon_days, selected_frame)
