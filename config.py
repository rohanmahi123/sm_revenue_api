"""Application settings — override any value via environment variable or a .env file."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Database — main (Supabase): auth, users, file_uploads, batches
    DATABASE_URL: str = "postgresql://postgres.gvliwqtcoxdatnhzmxjc:X4Wp6inx6bf9MxLL@aws-1-ap-northeast-1.pooler.supabase.com:5432/postgres"
    # Database — ML (local SQLite): trained_models, datasets
    ML_DATABASE_URL: str = "sqlite:///./sm_revenue.db"
    DB_ECHO: bool = False                        # set True for SQL debug logs

    # Supabase
    SUPABASE_URL: str = "https://gvliwqtcoxdatnhzmxjc.supabase.co"
    SUPABASE_KEY: str = "sb_secret_xP8qKpBLrUE2oGw3DCgi8A_Lt3gfgaF"

    # Auth — must match Part 1 SECRET_KEY exactly
    # SECRET_KEY: str = "sb_secret_xP8qKpBLrUE2oGw3DCgi8A_Lt3gfgaF"

    # Model storage
    MODEL_STORE_DIR: str = "model_store"         # directory where .pkl files live

    # App
    APP_TITLE: str = "SM Revenue Forecasting API"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False


settings = Settings()

# Ensure model store directory exists
Path(settings.MODEL_STORE_DIR).mkdir(parents=True, exist_ok=True)
