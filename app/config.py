"""Application settings loaded from environment variables."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration for the API."""

    app_name: str = "Demand Forecast API"
    model_dir: Path = Path("models")
    data_path: Path = Path("data/sales.csv")
    api_key: str | None = None
    cors_origins: str = "*"
    log_level: str = "INFO"
    model_config = SettingsConfigDict(env_file=".env", env_prefix="DEMAND_", extra="ignore")

    @property
    def cors_origin_list(self) -> list[str]:
        """Return configured CORS origins."""
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    """Return cached process settings."""
    return Settings()
