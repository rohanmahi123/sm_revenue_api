"""Pydantic schemas for request bodies and API responses."""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ──────────────────────────────────────────────
# Dataset schemas
# ──────────────────────────────────────────────

class DatasetBase(BaseModel):
    original_filename: str
    notes: Optional[str] = None


class DatasetCreate(DatasetBase):
    pass


class DatasetResponse(BaseModel):
    id: int
    user_id: str
    original_filename: str
    file_hash: str
    file_size_bytes: Optional[int]
    row_count: Optional[int]
    column_names: Optional[List[str]]
    date_range_start: Optional[str]
    date_range_end: Optional[str]
    target_columns: Optional[List[str]]
    uploaded_at: datetime
    notes: Optional[str]
    model_count: int = 0          # filled in by the router

    model_config = {"from_attributes": True}


# ──────────────────────────────────────────────
# External factors (provided as JSON body)
# ──────────────────────────────────────────────

class ExternalFactorRow(BaseModel):
    """One row of external factor data: a date + all macro values."""
    date: str = Field(..., description="Month date string, e.g. '2023-06-01'")
    CCI: Optional[float] = None
    CPI: Optional[float] = None
    Oil: Optional[float] = None
    GDP: Optional[float] = None
    Unemployment: Optional[float] = None
    ROI: Optional[float] = None


class ExternalFactorsPayload(BaseModel):
    """
    Full time series of external macro factors.
    Each entry covers one calendar month; the API merges them onto daily data.
    """
    rows: List[ExternalFactorRow]


# ──────────────────────────────────────────────
# Training schemas
# ──────────────────────────────────────────────

class TrainRequest(BaseModel):
    """
    Sent as multipart/form-data fields alongside the CSV file.
    The `external_factors` field is a JSON string of ExternalFactorsPayload.
    """
    user_id: str = Field(..., description="Caller's user / tenant identifier")
    model_name: str = Field(..., description="Unique name for this trained model")
    description: Optional[str] = None
    sheet_name: str = Field("10 SL", description="Excel sheet name (ignored for CSV uploads)")
    test_size: float = Field(0.25, ge=0.05, le=0.5)
    random_state: int = 42


class MetricsPerModel(BaseModel):
    MAE: float
    RMSE: float
    R2: float
    MAPE: float


class MetricsPerTarget(BaseModel):
    baseline: MetricsPerModel
    ridge: MetricsPerModel


class TrainResponse(BaseModel):
    model_id: int
    dataset_id: int
    model_name: str
    is_new_dataset: bool          # False → CSV hash matched an existing dataset
    training_duration_seconds: float
    metrics: Dict[str, Any]       # full nested metrics dict
    best_model_per_target: Dict[str, str]
    message: str


# ──────────────────────────────────────────────
# Trained model schemas
# ──────────────────────────────────────────────

class TrainedModelResponse(BaseModel):
    id: int
    dataset_id: int
    user_id: str
    model_name: str
    description: Optional[str]
    targets: Optional[List[str]]
    feature_columns: Optional[List[str]]
    external_factors_used: Optional[Dict[str, bool]]
    test_size: float
    random_state: int
    metrics: Optional[Dict[str, Any]]
    trained_at: datetime
    training_duration_seconds: Optional[float]

    model_config = {"from_attributes": True}


# ──────────────────────────────────────────────
# Prediction schemas
# ──────────────────────────────────────────────

class PredictRow(BaseModel):
    """One row for prediction (daily input)."""
    order_date: str = Field(..., description="Date string e.g. '2026-01-15'")
    Region: Optional[str] = None
    Geo: Optional[str] = None
    Country: Optional[str] = None
    Item_type: Optional[str] = Field(None, alias="Item type")
    Customer: Optional[str] = None
    Raw_Material: Optional[float] = Field(None, alias="Raw Material")
    Direct_Labor: Optional[float] = Field(None, alias="Direct Labor")
    Freight: Optional[float] = None
    Storage: Optional[float] = None
    Packaging: Optional[float] = None
    Indirect_Labor: Optional[float] = Field(None, alias="Indirect Labor")
    Rent_Utility: Optional[float] = Field(None, alias="Rent & Utility")
    Overhead: Optional[float] = None
    CCI: Optional[float] = None
    CPI: Optional[float] = None
    Oil: Optional[float] = None
    GDP: Optional[float] = None
    Unemployment: Optional[float] = None
    ROI: Optional[float] = None

    model_config = {"populate_by_name": True}


class PredictRequest(BaseModel):
    rows: List[PredictRow]


class PredictResponseRow(BaseModel):
    order_date: str
    predicted_total_revenue: Optional[float]
    predicted_COGS: Optional[float]
    predicted_SGA: Optional[float]
    model_used_revenue: Optional[str]
    model_used_COGS: Optional[str]
    model_used_SGA: Optional[str]


class PredictResponse(BaseModel):
    model_id: int
    model_name: str
    predictions: List[PredictResponseRow]
    heatmap_base64: Optional[str] = None   # PNG encoded as base64 — None if < 2 factors given
    heatmap_note: Optional[str] = None     # explains what the heatmap shows
