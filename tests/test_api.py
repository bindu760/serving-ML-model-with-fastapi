"""HTTP contract tests."""

import pandas as pd


def test_health_info_and_forecast(client) -> None:
    """Health, model metadata and single forecast endpoints work."""
    assert client.get("/health").json() == {"status": "ok", "model_loaded": True}
    assert "products" in client.get("/model/info").json()
    result = client.post("/v1/forecast", json={"product_id": "rice"})
    assert result.status_code == 200
    assert len(result.json()["forecast"]) == 7


def test_batch_unknown_product_and_short_history(client) -> None:
    """Batch requests succeed and invalid products/history return useful errors."""
    batch = client.post("/v1/forecast/batch", json={"product_ids": ["rice", "dal"]})
    assert batch.status_code == 200 and len(batch.json()["forecasts"]) == 2
    assert client.post("/v1/forecast", json={"product_id": "mystery"}).status_code == 404
    history = [
        {"date": str(day.date()), "units_sold": 2, "price": 1.5}
        for day in pd.date_range("2025-01-01", periods=10)
    ]
    assert client.post("/v1/forecast", json={"product_id": "rice", "history": history}).status_code == 422


def test_csv_upload_and_auth_on_off(client, monkeypatch) -> None:
    """CSV uploads are validated and optional API key auth is configurable."""
    bad = client.post("/v1/upload-and-forecast", files={"file": ("bad.csv", b"foo\n1\n", "text/csv")})
    assert bad.status_code == 400
    monkeypatch.setenv("DEMAND_API_KEY", "secret")
    from app.config import get_settings

    get_settings.cache_clear()
    from fastapi.testclient import TestClient

    from app.main import create_app

    with TestClient(create_app()) as protected:
        assert protected.post("/v1/forecast", json={"product_id": "rice"}).status_code == 401
        assert (
            protected.post(
                "/v1/forecast", headers={"X-API-Key": "secret"}, json={"product_id": "rice"}
            ).status_code
            == 200
        )
    get_settings.cache_clear()
