

from __future__ import annotations

import io
from collections import defaultdict
from typing import List, Optional

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from auth import get_current_user
from db.main_session import get_main_db
from db.models import FileUpload, IngestionBatch, TrainedModel
from db.session import get_db
from ml.predictor import predict
from schemas import PredictRequest, PredictResponseRow

# ── Router ────────────────────────────────────────────────────────────────────

router = APIRouter(prefix="/forecast-map", tags=["Sales Forecast Map"])


# ── Response schema (local — no changes to schemas.py needed) ─────────────────

from pydantic import BaseModel


class RegionForecast(BaseModel):
    region: str
    total_revenue: float
    total_COGS: float
    total_SGA: float
    row_count: int


class ForecastMapResponse(BaseModel):
    model_id: int
    model_name: str
    predictions: List[PredictResponseRow]
    region_summary: List[RegionForecast]
    forecast_map_base64: Optional[str] = None
    map_note: str = ""


# ── Map generator ─────────────────────────────────────────────────────────────

def _generate_forecast_map(
    region_summary: List[RegionForecast],
    model_name: str,
) -> str | None:
    """Returns None — chart generation not used in this endpoint."""
    return None


# ── Endpoint ──────────────────────────────────────────────────────────────────

@router.post(
    "/{model_id}",
    response_model=ForecastMapResponse,
    summary="Predict + generate a regional sales forecast map",
    description=(
        "Same input as POST /predict/{model_id}. "
        "Additionally groups predictions by Region and returns "
        "a visual forecast map (bubble chart + grouped bar) as base64 PNG. "
        "Pass rows for different regions to see each region on the map."
    ),
)
def forecast_map(
    model_id: int,
    payload: PredictRequest,
    db: Session = Depends(get_db),
):
    tm = db.query(TrainedModel).get(model_id)
    if not tm:
        raise HTTPException(status_code=404, detail="Model not found.")

    if not payload.rows:
        raise HTTPException(status_code=422, detail="rows list must not be empty.")

    raw_rows = [r.model_dump(by_alias=True) for r in payload.rows]

    # ── Run predictions ───────────────────────────────────────────────────────
    try:
        results = predict(tm.model_file_path, raw_rows)
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="Model .pkl file not found on disk.")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction error: {exc}")

    # ── Group by Region ───────────────────────────────────────────────────────
    region_totals: dict = defaultdict(lambda: {"revenue": 0.0, "cogs": 0.0, "sga": 0.0, "count": 0})

    for row_input, row_pred in zip(raw_rows, results):
        region = row_input.get("Region") or "Unknown"
        region_totals[region]["revenue"] += row_pred.get("predicted_total_revenue") or 0.0
        region_totals[region]["cogs"]    += row_pred.get("predicted_COGS") or 0.0
        region_totals[region]["sga"]     += row_pred.get("predicted_SGA") or 0.0
        region_totals[region]["count"]   += 1

    region_summary = [
        RegionForecast(
            region=reg,
            total_revenue=round(vals["revenue"], 2),
            total_COGS=round(vals["cogs"], 2),
            total_SGA=round(vals["sga"], 2),
            row_count=vals["count"],
        )
        for reg, vals in sorted(region_totals.items(), key=lambda x: -x[1]["revenue"])
    ]

    # ── Generate map ──────────────────────────────────────────────────────────
    map_b64 = _generate_forecast_map(region_summary, tm.model_name)

    note = (
        f"Map grouped by Region column. {len(region_summary)} region(s) found: "
        f"{', '.join(r.region for r in region_summary)}."
    ) if region_summary else "No Region data found in input rows."

    return ForecastMapResponse(
        model_id=tm.id,
        model_name=tm.model_name,
        predictions=results,
        region_summary=region_summary,
        forecast_map_base64=map_b64,
        map_note=note,
    )


# ── Batch forecast map schemas ────────────────────────────────────────────────

class DailyForecastRow(BaseModel):
    order_date: str
    region: Optional[str]
    geo: Optional[str]
    country: Optional[str]
    item_type: Optional[str]
    customer: Optional[str]
    predicted_total_revenue: Optional[float]
    predicted_COGS: Optional[float]
    predicted_SGA: Optional[float]
    model_used_revenue: Optional[str]
    model_used_COGS: Optional[str]
    model_used_SGA: Optional[str]


class BatchForecastMapResponse(BaseModel):
    model_id: int
    model_name: str
    batch_id: int
    row_count: int
    daily_predictions: List[DailyForecastRow]   # day-by-day full detail
    region_summary: List[RegionForecast]          # aggregated by region
    forecast_map_base64: Optional[str] = None
    map_note: str = ""
    external_factors_info: Optional[str] = None


# ── Shared helpers (reused from prediction.py logic) ──────────────────────────

import io as _io
import os as _os
import requests as _http
from ml.ext_factors import load_ext_factors, merge_ext_factors

_KEEP_AS_STR  = {"Region", "Geo", "Country", "Item type", "Customer",
                 "Order Id", "Order Date", "order_date"}
_SL_RENAME    = {
    "Order Date": "order_date",
    "Item type":  "Item_type",
    "Item Type":  "Item_type",
    "Raw Material":   "Raw_Material",
    "Direct Labor":   "Direct_Labor",
    "Indirect Labor": "Indirect_Labor",
    "Rent & Utility": "Rent_Utility",
}


def _build_url(file_path: str) -> str:
    if file_path.startswith("http://") or file_path.startswith("https://"):
        return file_path
    from config import settings as _settings
    base = _settings.SUPABASE_URL.rstrip("/")
    return f"{base}/storage/v1/object/public/File%20Storage/{file_path}"


def _load_and_clean(file_path: str) -> pd.DataFrame:
    resp = _http.get(_build_url(file_path), timeout=60)
    resp.raise_for_status()
    df = pd.read_csv(_io.BytesIO(resp.content))
    df.columns = [c.strip() for c in df.columns]
    for col in df.columns:
        if col not in _KEEP_AS_STR:
            df[col] = (
                df[col].astype(str)
                .str.replace("$", "", regex=False)
                .str.replace(",", "", regex=False)
                .str.strip()
            )
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.rename(columns=_SL_RENAME)


# ── New endpoint ──────────────────────────────────────────────────────────────

@router.post(
    "/from-batch/{model_id}",
    response_model=BatchForecastMapResponse,
    summary="Predict from SUBLEDGER batch + generate forecast map",
    description=(
        "Reads the SUBLEDGER CSV for the given batch_id, runs prediction "
        "on every row using the trained model, and returns: "
        "(1) day-by-day predictions with full row detail (date, region, customer…), "
        "(2) aggregated region summary, "
        "(3) forecast map chart as base64 PNG. "
        "If the CSV already contains external factor columns (CCI, CPI, Oil, GDP, "
        "Unemployment, ROI) they are used automatically — no manual input needed."
    ),
)
def forecast_map_from_batch(
    model_id: int,
    batch_id: int,
    db: Session = Depends(get_db),
    main_db: Session = Depends(get_main_db),
    current_user=Depends(get_current_user),
):
    # ── Validate model ────────────────────────────────────────────────────────
    tm = db.query(TrainedModel).get(model_id)
    if not tm:
        raise HTTPException(status_code=404, detail="Model not found.")

    # ── Look up SUBLEDGER file ────────────────────────────────────────────────
    batch = main_db.query(IngestionBatch).filter(
        IngestionBatch.id == batch_id,
        IngestionBatch.company_id == current_user.company_id,
    ).first()
    if not batch:
        raise HTTPException(status_code=404, detail=f"Batch {batch_id} not found.")

    sl_upload = main_db.query(FileUpload).filter(
        FileUpload.batch_id == batch_id,
        FileUpload.file_type == "SUBLEDGER",
    ).first()
    if not sl_upload or not sl_upload.file_path:
        raise HTTPException(status_code=404, detail="No SUBLEDGER file found for this batch.")

    # ── Load + clean CSV ──────────────────────────────────────────────────────
    try:
        df = _load_and_clean(sl_upload.file_path)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to read SL file: {exc}")

    if df.empty:
        raise HTTPException(status_code=422, detail="SUBLEDGER CSV is empty.")

    if "order_date" not in df.columns:
        raise HTTPException(status_code=422, detail="SUBLEDGER CSV must have an 'Order Date' column.")

    # ── Merge external factors from stored file (by month) ───────────────────
    from config import settings as _settings
    ext_df = load_ext_factors(_settings.MODEL_STORE_DIR)
    if ext_df is not None:
        df, ext_info = merge_ext_factors(df, ext_df)
    else:
        ext_info = "No external_factors.csv found in model_store — model used training medians."

    # ── Run prediction ────────────────────────────────────────────────────────
    input_rows = df.to_dict(orient="records")
    try:
        results = predict(tm.model_file_path, input_rows)
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="Model .pkl file not found on disk.")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction error: {exc}")

    # ── Build day-by-day output ───────────────────────────────────────────────
    daily_predictions: List[DailyForecastRow] = []
    region_totals: dict = defaultdict(
        lambda: {"revenue": 0.0, "cogs": 0.0, "sga": 0.0, "count": 0}
    )

    for row_input, row_pred in zip(input_rows, results):
        region   = str(row_input.get("Region") or "Unknown")
        rev  = row_pred.get("predicted_total_revenue") or 0.0
        cogs = row_pred.get("predicted_COGS") or 0.0
        sga  = row_pred.get("predicted_SGA") or 0.0

        region_totals[region]["revenue"] += rev
        region_totals[region]["cogs"]    += cogs
        region_totals[region]["sga"]     += sga
        region_totals[region]["count"]   += 1

        daily_predictions.append(DailyForecastRow(
            order_date=row_pred.get("order_date", ""),
            region=str(row_input.get("Region") or ""),
            geo=str(row_input.get("Geo") or ""),
            country=str(row_input.get("Country") or ""),
            item_type=str(row_input.get("Item_type") or ""),
            customer=str(row_input.get("Customer") or ""),
            predicted_total_revenue=row_pred.get("predicted_total_revenue"),
            predicted_COGS=row_pred.get("predicted_COGS"),
            predicted_SGA=row_pred.get("predicted_SGA"),
            model_used_revenue=row_pred.get("model_used_revenue"),
            model_used_COGS=row_pred.get("model_used_COGS"),
            model_used_SGA=row_pred.get("model_used_SGA"),
        ))

    # ── Region summary ────────────────────────────────────────────────────────
    region_summary = [
        RegionForecast(
            region=reg,
            total_revenue=round(vals["revenue"], 2),
            total_COGS=round(vals["cogs"], 2),
            total_SGA=round(vals["sga"], 2),
            row_count=vals["count"],
        )
        for reg, vals in sorted(region_totals.items(), key=lambda x: -x[1]["revenue"])
    ]

    # ── Generate chart ────────────────────────────────────────────────────────
    map_b64 = _generate_forecast_map(region_summary, tm.model_name)
    note = (
        f"{len(region_summary)} region(s): "
        f"{', '.join(r.region for r in region_summary)}."
    )

    return BatchForecastMapResponse(
        model_id=tm.id,
        model_name=tm.model_name,
        batch_id=batch_id,
        row_count=len(daily_predictions),
        daily_predictions=daily_predictions,
        region_summary=region_summary,
        forecast_map_base64=map_b64,
        map_note=note,
        external_factors_info=ext_info,
    )
