"""Application settings — override any value via environment variable or a .env file."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Database
    DATABASE_URL: str = "sqlite:///./sm_revenue.db"
    DB_ECHO: bool = False                        # set True for SQL debug logs

    # Model storage
    MODEL_STORE_DIR: str = "model_store"         # directory where .pkl files live

    # App
    APP_TITLE: str = "SM Revenue Forecasting API"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False


settings = Settings()

# Ensure model store directory exists
Path(settings.MODEL_STORE_DIR).mkdir(parents=True, exist_ok=True)
