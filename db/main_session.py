"""
Session for the main Supabase DB — auth, users, file_uploads, ingestion_batches.
Separate from db/session.py which connects to the ML SQLite DB.
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from config import settings

engine = create_engine(
    settings.DATABASE_URL,
    connect_args={"sslmode": "require"} if "postgresql" in settings.DATABASE_URL else {},
    echo=settings.DB_ECHO,
)

MainSessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def get_main_db():
    """FastAPI dependency — yields a Supabase DB session."""
    db = MainSessionLocal()
    try:
        yield db
    finally:
        db.close()
