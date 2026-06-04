"""
POST /predict/{model_id}
  – Accepts a date range (start_date → end_date) + one snapshot of external
    factor values.  Internally expands the range into monthly rows, applies
    the same external factors to every month, runs the model, and returns
    the full monthly breakdown so the frontend can render a forecast chart /
    map without any extra processing.
"""

import io
import os
from datetime import date, datetime

import pandas as pd
import requests as http_requests
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from auth import get_current_user
from config import settings
from db.main_session import get_main_db
from db.models import FileUpload, IngestionBatch, TrainedModel
from db.session import get_db
from ml.predictor import predict
from schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    BatchPredictRow,
    BatchPredictSummary,
    ForecastRequest,
    ForecastResponse,
    ForecastSummary,
    MonthlyForecast,
)

router = APIRouter(prefix="/predict", tags=["Prediction"])


def _monthly_dates(start: date, end: date) -> list[date]:
    """Return the 1st of every month from start's month to end's month inclusive."""
    dates = []
    current = start.replace(day=1)
    end_anchor = end.replace(day=1)
    while current <= end_anchor:
        dates.append(current)
        month = current.month + 1
        year = current.year
        if month > 12:
            month = 1
            year += 1
        current = current.replace(year=year, month=month)
    return dates


@router.post(
    "/{model_id}",
    response_model=ForecastResponse,
    summary="Date-range forecast: monthly predictions from start to end date",
    description=(
        "Provide a start/end date range plus optional cost-component defaults "
        "and a single external-factor snapshot (CCI, CPI, Oil, GDP, "
        "Unemployment, ROI).  The API generates one input row per month, "
        "applies the same external factors to every month as a constant "
        "baseline, runs the model, and returns the full monthly breakdown "
        "ready for frontend charting or a forecast map."
    ),
)
def run_prediction(
    model_id: int,
    payload: ForecastRequest,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    tm = db.query(TrainedModel).get(model_id)
    if not tm:
        raise HTTPException(status_code=404, detail="Model not found.")

    # ── Parse dates ───────────────────────────────────────────────────────────
    try:
        start = datetime.strptime(payload.start_date, "%Y-%m-%d").date()
        end = datetime.strptime(payload.end_date, "%Y-%m-%d").date()
    except ValueError:
        raise HTTPException(status_code=422, detail="Dates must be in YYYY-MM-DD format.")

    if end < start:
        raise HTTPException(status_code=422, detail="end_date must be >= start_date.")

    months = _monthly_dates(start, end)
    if not months:
        raise HTTPException(status_code=422, detail="Date range produced no months.")

    # ── Build one input row per month ─────────────────────────────────────────
    # The external factors are constant (latest snapshot) across all months.
    # Cost components default to whatever the frontend supplied (or None → NaN).
    base_row = {
        "Region":        payload.Region,
        "Geo":           payload.Geo,
        "Country":       payload.Country,
        "Item type":     payload.Item_type,
        "Customer":      payload.Customer,
        "Raw Material":  payload.Raw_Material,
        "Direct Labor":  payload.Direct_Labor,
        "Freight":       payload.Freight,
        "Storage":       payload.Storage,
        "Packaging":     payload.Packaging,
        "Indirect Labor": payload.Indirect_Labor,
        "Rent & Utility": payload.Rent_Utility,
        "Overhead":      payload.Overhead,
        "CCI":           payload.CCI,
        "CPI":           payload.CPI,
        "Oil":           payload.Oil,
        "GDP":           payload.GDP,
        "Unemployment":  payload.Unemployment,
        "ROI":           payload.ROI,
    }

    input_rows = [
        {**base_row, "order_date": m.isoformat()}
        for m in months
    ]

    # ── Run model ─────────────────────────────────────────────────────────────
    try:
        results = predict(tm.model_file_path, input_rows)
    except FileNotFoundError:
        raise HTTPException(
            status_code=500,
            detail="Model file not found on disk. It may have been deleted manually.",
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction error: {exc}")

    # ── Build monthly forecast list ───────────────────────────────────────────
    monthly_predictions: list[MonthlyForecast] = []
    total_revenue = total_cogs = total_sga = 0.0

    for m, row in zip(months, results):
        rev  = row.get("predicted_total_revenue") or 0.0
        cogs = row.get("predicted_COGS") or 0.0
        sga  = row.get("predicted_SGA") or 0.0

        total_revenue += rev
        total_cogs    += cogs
        total_sga     += sga

        monthly_predictions.append(
            MonthlyForecast(
                month=m.strftime("%Y-%m"),
                date=m.isoformat(),
                predicted_total_revenue=row.get("predicted_total_revenue"),
                predicted_COGS=row.get("predicted_COGS"),
                predicted_SGA=row.get("predicted_SGA"),
                model_used_revenue=row.get("model_used_revenue"),
                model_used_COGS=row.get("model_used_COGS"),
                model_used_SGA=row.get("model_used_SGA"),
            )
        )

    summary = ForecastSummary(
        total_revenue=round(total_revenue, 4),
        total_COGS=round(total_cogs, 4),
        total_SGA=round(total_sga, 4),
        months_count=len(months),
    )

    return ForecastResponse(
        model_id=tm.id,
        model_name=tm.model_name,
        start_date=payload.start_date,
        end_date=payload.end_date,
        monthly_predictions=monthly_predictions,
        summary=summary,
    )


# ── Column mapping: SL CSV column names → model input names ──────────────────
_SL_RENAME = {
    "Order Date": "order_date",
    "order_date": "order_date",
    "Item type": "Item_type",
    "Item Type": "Item_type",
    "Raw Material": "Raw_Material",
    "Direct Labor": "Direct_Labor",
    "Indirect Labor": "Indirect_Labor",
    "Rent & Utility": "Rent_Utility",
}


def _resolve_file_url(file_path: str) -> str:
    """
    Convert a Supabase Storage relative path to a full public URL.
    If it's already a full URL, return as-is.
    """
    if file_path.startswith("http://") or file_path.startswith("https://"):
        return file_path
    supabase_url = settings.SUPABASE_URL.rstrip("/")
    bucket = "File%20Storage"
    return f"{supabase_url}/storage/v1/object/public/{bucket}/{file_path}"


def _clean_sl_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the same cleaning as load_financial_csv() in preprocessor.py:
    - Strip $, commas, whitespace from all columns
    - Convert numeric columns to float
    - Keep categorical and date columns as-is
    """
    KEEP_AS_STR = {"Region", "Geo", "Country", "Item type", "Customer",
                   "Order Date", "order_date"}
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    for col in df.columns:
        if col in KEEP_AS_STR:
            continue
        df[col] = (
            df[col].astype(str)
            .str.replace("$", "", regex=False)
            .str.replace(",", "", regex=False)
            .str.strip()
        )
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _load_sl_csv(file_path: str) -> pd.DataFrame:
    """
    Load the SUBLEDGER CSV from a Supabase Storage path or full URL.
    Returns a DataFrame with original column names intact.
    """
    url = _resolve_file_url(file_path)
    resp = http_requests.get(url, timeout=60)
    resp.raise_for_status()
    return pd.read_csv(io.BytesIO(resp.content))


@router.post(
    "/from-batch/{model_id}",
    response_model=BatchPredictResponse,
    summary="Predict using the SUBLEDGER CSV already uploaded for a batch",
    description=(
        "Looks up the SUBLEDGER file that was uploaded as part of `batch_id` "
        "(scoped to the authenticated user), reads every row from that CSV, "
        "and runs the specified model against all rows. "
        "Optional external-factor overrides (CCI, CPI, Oil, GDP, Unemployment, ROI) "
        "are applied uniformly to every row, supplementing or replacing values "
        "found in the CSV."
    ),
)
def predict_from_batch(
    model_id: int,
    payload: BatchPredictRequest,
    db: Session = Depends(get_db),           # ML SQLite — trained models
    main_db: Session = Depends(get_main_db), # Supabase — batches & file uploads
    current_user=Depends(get_current_user),
):
    # ── Validate model (SQLite) ───────────────────────────────────────────────
    tm = db.query(TrainedModel).get(model_id)
    if not tm:
        raise HTTPException(status_code=404, detail="Model not found.")

    # ── Look up the SUBLEDGER file for this batch (Supabase) ─────────────────
    batch = main_db.query(IngestionBatch).filter(
        IngestionBatch.id == payload.batch_id,
    ).first()
    if not batch:
        raise HTTPException(status_code=404, detail=f"Batch {payload.batch_id} not found.")

    sl_upload = main_db.query(FileUpload).filter(
        FileUpload.batch_id == payload.batch_id,
        FileUpload.file_type == "SUBLEDGER",
    ).first()
    if not sl_upload or not sl_upload.file_path:
        raise HTTPException(
            status_code=404,
            detail="No SUBLEDGER file found for this batch.",
        )

    # ── Load CSV ──────────────────────────────────────────────────────────────
    try:
        df = _load_sl_csv(sl_upload.file_path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to read SL file: {exc}")

    if df.empty:
        raise HTTPException(status_code=422, detail="SUBLEDGER CSV is empty.")

    # ── Clean: strip $, commas — same as training preprocessing ──────────────
    df = _clean_sl_df(df)

    # ── Normalise column names ────────────────────────────────────────────────
    df = df.rename(columns=_SL_RENAME)

    if "order_date" not in df.columns:
        raise HTTPException(
            status_code=422,
            detail="SUBLEDGER CSV must contain an 'Order Date' column.",
        )

    # ── Detect which external factors are already in the CSV ─────────────────
    EXT_FACTORS = ["CCI", "CPI", "Oil", "GDP", "Unemployment", "ROI"]
    factors_from_csv = [f for f in EXT_FACTORS if f in df.columns and df[f].notna().any()]
    factors_missing  = [f for f in EXT_FACTORS if f not in factors_from_csv]

    # User-provided overrides (only applied if explicitly passed)
    ext_overrides = {
        k: v for k, v in {
            "CCI": payload.CCI,
            "CPI": payload.CPI,
            "Oil": payload.Oil,
            "GDP": payload.GDP,
            "Unemployment": payload.Unemployment,
            "ROI": payload.ROI,
        }.items() if v is not None
    }

    input_rows = []
    for _, row in df.iterrows():
        r = row.to_dict()
        # Apply user overrides on top (user value wins over CSV value if provided)
        r.update(ext_overrides)
        input_rows.append(r)

    # ── Run model ─────────────────────────────────────────────────────────────
    try:
        results = predict(tm.model_file_path, input_rows)
    except FileNotFoundError:
        raise HTTPException(
            status_code=500,
            detail="Model file not found on disk. It may have been deleted manually.",
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction error: {exc}")

    # ── Build response ────────────────────────────────────────────────────────
    predictions = []
    total_revenue = total_cogs = total_sga = 0.0

    for res in results:
        rev  = res.get("predicted_total_revenue") or 0.0
        cogs = res.get("predicted_COGS") or 0.0
        sga  = res.get("predicted_SGA") or 0.0
        total_revenue += rev
        total_cogs    += cogs
        total_sga     += sga

        predictions.append(BatchPredictRow(
            order_date=res.get("order_date", ""),
            predicted_total_revenue=res.get("predicted_total_revenue"),
            predicted_COGS=res.get("predicted_COGS"),
            predicted_SGA=res.get("predicted_SGA"),
            model_used_revenue=res.get("model_used_revenue"),
            model_used_COGS=res.get("model_used_COGS"),
            model_used_SGA=res.get("model_used_SGA"),
        ))

    # Build info message about external factor sources
    if factors_from_csv and not factors_missing:
        ext_msg = f"All external factors used from CSV: {factors_from_csv}."
    elif factors_from_csv and factors_missing:
        ext_msg = (f"External factors from CSV: {factors_from_csv}. "
                   f"Missing (not in CSV, set to NaN): {factors_missing}.")
    else:
        ext_msg = "No external factors found in CSV — model used training medians."
    if ext_overrides:
        ext_msg += f" User overrides applied: {list(ext_overrides.keys())}."

    return BatchPredictResponse(
        model_id=tm.id,
        model_name=tm.model_name,
        batch_id=payload.batch_id,
        sl_file_path=sl_upload.file_path,
        predictions=predictions,
        summary=BatchPredictSummary(
            total_revenue=round(total_revenue, 4),
            total_COGS=round(total_cogs, 4),
            total_SGA=round(total_sga, 4),
            row_count=len(predictions),
        ),
        external_factors_info=ext_msg,
    )
