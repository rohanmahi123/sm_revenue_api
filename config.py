"""Application settings — override any value via environment variable or a .env file."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Database — main (Supabase): auth, users, file_uploads, batches
    DATABASE_URL: str = ""
    # Database — ML (local SQLite): trained_models, datasets
    ML_DATABASE_URL: str = ""
    DB_ECHO: bool = False                        # set True for SQL debug logs

    # Supabase
    SUPABASE_URL: str = ""
    SUPABASE_KEY: str = ""



    # Model storage
    MODEL_STORE_DIR: str = "model_store"         # directory where .pkl files live

    # App
    APP_TITLE: str = "SM Revenue Forecasting API"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False


settings = Settings()

# Ensure model store directory exists
Path(settings.MODEL_STORE_DIR).mkdir(parents=True, exist_ok=True)
