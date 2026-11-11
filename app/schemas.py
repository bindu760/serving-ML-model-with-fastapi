"""Validated API request and response models."""

from datetime import date

from pydantic import BaseModel, ConfigDict, Field


class HistoryRecord(BaseModel):
    """A single day's sales observation."""

    date: date
    units_sold: float = Field(ge=0)
    price: float = Field(gt=0)
    is_holiday: bool = False


class ForecastRequest(BaseModel):
    """Forecast request using explicit or stored history."""

    model_config = ConfigDict(json_schema_extra={"example": {"product_id": "rice", "horizon_days": 7}})
    product_id: str = Field(min_length=1)
    horizon_days: int = Field(default=7, ge=1, le=14)
    history: list[HistoryRecord] | None = Field(default=None, min_length=28)


class BatchForecastRequest(BaseModel):
    """Request forecasts for several products."""

    product_ids: list[str] = Field(min_length=1)
    horizon_days: int = Field(default=7, ge=1, le=14)


class ForecastPoint(BaseModel):
    """Forecast value and residual-based bounds."""

    date: date
    predicted_units: float
    lower: float
    upper: float


class ForecastResponse(BaseModel):
    """Forecast result for one product."""

    product_id: str
    forecast: list[ForecastPoint]
