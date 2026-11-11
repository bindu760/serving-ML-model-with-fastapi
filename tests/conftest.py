"""Shared API test fixtures."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import get_settings
from app.main import create_app
from ml.generate_data import generate_data
from ml.train import train


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Generate and train a small fixture model, then return a test client."""
    data_path = tmp_path / "sales.csv"
    model_dir = tmp_path / "models"
    generate_data(data_path, years=1)
    train(data_path, model_dir)
    monkeypatch.setenv("DEMAND_MODEL_DIR", str(model_dir))
    monkeypatch.setenv("DEMAND_DATA_PATH", str(data_path))
    get_settings.cache_clear()
    with TestClient(create_app()) as test_client:
        yield test_client
    get_settings.cache_clear()
