"""
SQLAlchemy ORM models.
Tables:
  - datasets       → uploaded CSV metadata (hash, columns, row count, date range)
  - trained_models → saved model files + metrics, linked to a dataset
"""

from datetime import datetime

from sqlalchemy import (
    JSON,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    pass


class Dataset(Base):
    """Stores metadata for every unique CSV upload (keyed by SHA-256 hash)."""

    __tablename__ = "datasets"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String(128), index=True, nullable=False)

    # File identity
    original_filename = Column(String(512), nullable=False)
    file_hash = Column(String(64), nullable=False, index=True)   # SHA-256
    file_size_bytes = Column(Integer, nullable=True)

    # Content summary
    row_count = Column(Integer, nullable=True)
    column_names = Column(JSON, nullable=True)          # list[str]
    date_range_start = Column(String(32), nullable=True)
    date_range_end = Column(String(32), nullable=True)
    target_columns = Column(JSON, nullable=True)        # list[str]

    # Audit
    uploaded_at = Column(DateTime, default=datetime.utcnow)
    notes = Column(Text, nullable=True)

    # Relations
    trained_models = relationship(
        "TrainedModel", back_populates="dataset", cascade="all, delete-orphan"
    )


class TrainedModel(Base):
    """
    Stores a serialised sklearn pipeline (.pkl) and its training metadata.
    One dataset can have multiple trained models (different external factor combos).
    """

    __tablename__ = "trained_models"

    id = Column(Integer, primary_key=True, index=True)
    dataset_id = Column(Integer, ForeignKey("datasets.id"), nullable=False, index=True)
    user_id = Column(String(128), index=True, nullable=False)

    # Identity
    model_name = Column(String(256), nullable=False, unique=True, index=True)
    description = Column(Text, nullable=True)

    # Storage
    model_file_path = Column(String(1024), nullable=False)   # path to .pkl on disk

    # Training config
    targets = Column(JSON, nullable=True)                    # ["Total Revenue", ...]
    feature_columns = Column(JSON, nullable=True)            # list[str]
    external_factors_used = Column(JSON, nullable=True)      # {"CCI": True, ...}
    test_size = Column(Float, default=0.25)
    random_state = Column(Integer, default=42)

    # Per-target metrics (nested dict)
    metrics = Column(JSON, nullable=True)
    # {"Total Revenue": {"baseline": {"MAE":..,"RMSE":..,"R2":..,"MAPE":..},
    #                    "ridge":    {...}},
    #  "best_model_per_target": {"Total Revenue": "ridge", ...}}

    # Audit
    trained_at = Column(DateTime, default=datetime.utcnow)
    training_duration_seconds = Column(Float, nullable=True)

    # Relations
    dataset = relationship("Dataset", back_populates="trained_models")
