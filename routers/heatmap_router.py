"""
Heatmap endpoints
=================

POST /heatmap/upload
    Upload 1–6 xlsx/csv files (one per external factor).
    Backend auto-detects which factor each file contains,
    merges them on date, stores to disk, and returns the
    first correlation heatmap.

POST /heatmap/from-csv
    Upload the full training / SL CSV (the same file used in
    POST /train or POST /predict/from-batch).
    Backend extracts only the external factor columns (CCI, CPI,
    Oil, GDP, Unemployment, ROI) + Order Date from it, stores
    the result, and returns the heatmap.
    This merges WITH any data already stored from /upload.

POST /heatmap/refresh
    Send fresh values for all 6 external factors.
    Backend appends the new row to the stored historical data
    and returns an updated heatmap base64 PNG.

GET /heatmap/data
    Return the stored factor data as JSON rows (for debugging /
    frontend table display).
"""

import io
from typing import List, Optional

import pandas as pd
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from auth import get_current_user
from config import settings
from db.main_session import get_main_db
from db.models import FileUpload, IngestionBatch
from ml.heatmap import (
    EXTERNAL_FACTORS,
    _parse_date_series,
    generate_heatmap_from_df,
    load_factor_data,
    merge_factor_files,
    parse_factor_file,
    save_factor_data,
)

# Date column candidates in the SL CSV
_SL_DATE_COLS = ["order_date", "order date", "date", "observation_date", "months", "month"]

router = APIRouter(prefix="/heatmap", tags=["Heatmap"])


# ── Schemas ───────────────────────────────────────────────────────────────────

class HeatmapUploadResponse(BaseModel):
    factors_loaded: List[str]
    row_count: int
    date_range_start: Optional[str]
    date_range_end: Optional[str]
    heatmap_base64: Optional[str]
    message: str


class FromCsvResponse(BaseModel):
    factors_found: List[str]
    row_count: int
    date_range_start: Optional[str]
    date_range_end: Optional[str]
    heatmap_base64: Optional[str]
    message: str


class RefreshRequest(BaseModel):
    CCI: Optional[float] = Field(None, description="Consumer Confidence Index")
    CPI: Optional[float] = Field(None, description="Consumer Price Index (%)")
    Oil: Optional[float] = Field(None, description="Crude Oil Price")
    GDP: Optional[float] = Field(None, description="GDP (AUD bn)")
    Unemployment: Optional[float] = Field(None, description="Unemployment Rate (%)")
    ROI: Optional[float] = Field(None, description="Return on Investment (%)")


class HeatmapRefreshResponse(BaseModel):
    row_count: int
    new_row: dict
    heatmap_base64: Optional[str]
    message: str


class FactorDataResponse(BaseModel):
    row_count: int
    columns: List[str]
    rows: List[dict]


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post(
    "/upload",
    response_model=HeatmapUploadResponse,
    summary="Upload external factor files to build the heatmap base dataset",
    description=(
        "Upload 1–6 xlsx or csv files. Each file should contain one external "
        "factor (CCI, CPI, Oil, GDP, Unemployment, ROI) with a date column and "
        "a value column. The factor is auto-detected from the column headers. "
        "Files are merged on date and stored per company. "
        "Re-uploading replaces the existing stored data."
    ),
)
async def upload_factor_files(
    files: List[UploadFile] = File(
        ...,
        description="1 to 6 xlsx/csv files — one per external factor",
    ),
    current_user=Depends(get_current_user),
):
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")
    if len(files) > 6:
        raise HTTPException(status_code=400, detail="Maximum 6 files (one per factor).")

    parsed_dfs: list[pd.DataFrame] = []
    factors_loaded: list[str] = []
    errors: list[str] = []

    for upload in files:
        content = await upload.read()
        if not content:
            errors.append(f"'{upload.filename}' is empty — skipped.")
            continue
        try:
            df = parse_factor_file(content, upload.filename or "upload")
            factor_col = [c for c in df.columns if c != "date"][0]
            factors_loaded.append(factor_col)
            parsed_dfs.append(df)
        except Exception as exc:
            errors.append(f"'{upload.filename}': {exc}")

    if not parsed_dfs:
        raise HTTPException(
            status_code=422,
            detail="No valid factor files could be parsed. " + " | ".join(errors),
        )

    # Check duplicates
    if len(factors_loaded) != len(set(factors_loaded)):
        raise HTTPException(
            status_code=422,
            detail=f"Duplicate factor files detected: {factors_loaded}",
        )

    merged = merge_factor_files(parsed_dfs)
    save_factor_data(merged, settings.MODEL_STORE_DIR, current_user.company_id)

    heatmap_b64 = generate_heatmap_from_df(merged)

    date_min = str(merged["date"].min().date()) if not merged.empty else None
    date_max = str(merged["date"].max().date()) if not merged.empty else None

    msg = f"Loaded {len(factors_loaded)} factor(s): {', '.join(factors_loaded)}."
    if errors:
        msg += " Warnings: " + " | ".join(errors)
    if heatmap_b64 is None:
        msg += " Note: need at least 2 factors with overlapping dates to render heatmap."

    return HeatmapUploadResponse(
        factors_loaded=factors_loaded,
        row_count=int(merged.shape[0]),
        date_range_start=date_min,
        date_range_end=date_max,
        heatmap_base64=heatmap_b64,
        message=msg,
    )


@router.post(
    "/from-csv",
    response_model=FromCsvResponse,
    summary="Extract external factors from the full training/SL CSV and build heatmap",
    description=(
        "Upload the same CSV/xlsx that was used in POST /train or "
        "POST /predict/from-batch. The backend finds the date column and "
        "any external factor columns (CCI, CPI, Oil, GDP, Unemployment, ROI) "
        "inside it, extracts those rows, merges with any data already stored "
        "from POST /heatmap/upload, saves, and returns an updated heatmap."
    ),
)
async def heatmap_from_csv(
    file: UploadFile = File(..., description="Full training or SL CSV/xlsx file"),
    current_user=Depends(get_current_user),
):
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    filename = file.filename or "upload"
    suffix = filename.rsplit(".", 1)[-1].lower()

    if suffix in ("xlsx", "xls"):
        raise HTTPException(
            status_code=422,
            detail="Excel files (.xlsx/.xls) are not supported. Please upload a CSV file.",
        )

    try:
        raw = pd.read_csv(io.BytesIO(content))
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Cannot read file: {exc}")

    if raw.empty:
        raise HTTPException(status_code=422, detail="File has no data rows.")

    # ── Extract external factors ──────────────────────────────────────────────
    try:
        df, factors_found = _extract_factors_from_df(raw)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    if df.empty:
        raise HTTPException(status_code=422, detail="No valid rows after parsing.")

    # ── Merge with existing stored data and save ──────────────────────────────
    had_existing = load_factor_data(settings.MODEL_STORE_DIR, current_user.company_id) is not None
    merged = _merge_and_save(df, settings.MODEL_STORE_DIR, current_user.company_id)

    heatmap_b64 = generate_heatmap_from_df(merged)
    date_min = str(merged["date"].min().date()) if not merged.empty else None
    date_max = str(merged["date"].max().date()) if not merged.empty else None

    msg = f"Extracted {len(factors_found)} factor(s) from CSV: {', '.join(factors_found)}."
    if had_existing:
        msg += " Merged with existing stored data."
    if heatmap_b64 is None:
        msg += " Need ≥2 factors with overlapping dates to render heatmap."

    return FromCsvResponse(
        factors_found=factors_found,
        row_count=int(merged.shape[0]),
        date_range_start=date_min,
        date_range_end=date_max,
        heatmap_base64=heatmap_b64,
        message=msg,
    )


def _extract_factors_from_df(raw: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Shared helper: given any wide DataFrame (SL / training CSV),
    find the date column + external factor columns, return a tidy
    (date | factor...) DataFrame and the list of factor names found.
    """
    cols_lower = {str(c).strip().lower(): c for c in raw.columns}

    # Find date column
    date_col_orig = None
    for candidate in _SL_DATE_COLS:
        if candidate in cols_lower:
            date_col_orig = cols_lower[candidate]
            break
    if date_col_orig is None:
        raise ValueError(
            f"Cannot find a date column. Expected one of: {_SL_DATE_COLS}. "
            f"Found: {list(raw.columns)}"
        )

    # Find external factor columns
    factor_col_map: dict[str, str] = {}
    for col_orig in raw.columns:
        col_lower = str(col_orig).strip().lower()
        for factor in EXTERNAL_FACTORS:
            if col_lower == factor.lower() and factor not in factor_col_map:
                factor_col_map[factor] = col_orig

    if not factor_col_map:
        raise ValueError(
            f"No external factor columns found. "
            f"Expected column names matching: {EXTERNAL_FACTORS}."
        )

    keep = [date_col_orig] + list(factor_col_map.values())
    df = raw[keep].copy()
    df = df.rename(columns={date_col_orig: "date",
                             **{v: k for k, v in factor_col_map.items()}})

    df["date"] = _parse_date_series(df["date"])
    df = df.dropna(subset=["date"])
    df["date"] = df["date"].dt.to_period("M").dt.to_timestamp()

    for factor in factor_col_map:
        df[factor] = pd.to_numeric(
            df[factor].astype(str).str.replace(r"[$,%\s]", "", regex=True),
            errors="coerce",
        )

    df = (df.drop_duplicates(subset=["date"])
            .sort_values("date")
            .reset_index(drop=True))
    df = df.dropna(how="all", subset=list(factor_col_map.keys()))
    return df, list(factor_col_map.keys())


def _merge_and_save(new_df: pd.DataFrame, store_dir: str, company_id) -> pd.DataFrame:
    """Merge new_df with any existing stored data, save, and return merged."""
    existing = load_factor_data(store_dir, company_id)
    if existing is not None and not existing.empty:
        merged = pd.merge(existing, new_df, on="date", how="outer", suffixes=("", "_new"))
        for factor in EXTERNAL_FACTORS:
            new_col = f"{factor}_new"
            if new_col in merged.columns:
                merged[factor] = merged[new_col].combine_first(merged.get(factor))
                merged.drop(columns=[new_col], inplace=True)
        merged = merged.sort_values("date").reset_index(drop=True)
    else:
        merged = new_df
    save_factor_data(merged, store_dir, company_id)
    return merged


@router.post(
    "/from-subledger/{batch_id}",
    response_model=FromCsvResponse,
    summary="Extract external factors from the uploaded SUBLEDGER file of a batch",
    description=(
        "Reads the SUBLEDGER CSV that was already uploaded as part of `batch_id`, "
        "extracts the external factor columns (CCI, CPI, Oil, GDP, Unemployment, ROI) "
        "from it, merges with any data stored from POST /heatmap/upload, saves, "
        "and returns an updated correlation heatmap."
    ),
)
def heatmap_from_subledger(
    batch_id: int,
    main_db: Session = Depends(get_main_db), # Supabase — batches & file uploads
    current_user=Depends(get_current_user),
):
    # ── Look up the SUBLEDGER file (Supabase) ─────────────────────────────────
    batch = main_db.query(IngestionBatch).filter(
        IngestionBatch.id == batch_id,
        IngestionBatch.company_id == current_user.company_id,
    ).first()
    if not batch:
        raise HTTPException(status_code=404, detail="Batch not found.")

    sl_upload = main_db.query(FileUpload).filter(
        FileUpload.batch_id == batch_id,
        FileUpload.file_type == "SUBLEDGER",
    ).first()
    if not sl_upload or not sl_upload.file_path:
        raise HTTPException(
            status_code=404,
            detail="No SUBLEDGER file found for this batch.",
        )

    # ── Load CSV from Supabase Storage ────────────────────────────────────────
    import os, requests as http_requests
    file_path = sl_upload.file_path
    supabase_url = settings.SUPABASE_URL.rstrip("/")
    bucket = "File%20Storage"
    if file_path.startswith("http://") or file_path.startswith("https://"):
        full_url = file_path
    else:
        full_url = f"{supabase_url}/storage/v1/object/public/{bucket}/{file_path}"
    try:
        resp = http_requests.get(full_url, timeout=60)
        resp.raise_for_status()
        content = resp.content
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to read SUBLEDGER file: {exc}")

    suffix = file_path.rsplit(".", 1)[-1].lower()
    if suffix in ("xlsx", "xls"):
        raise HTTPException(
            status_code=422,
            detail="Excel SUBLEDGER files (.xlsx/.xls) are not supported. Use a CSV batch.",
        )
    try:
        raw = pd.read_csv(io.BytesIO(content))
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Cannot parse SUBLEDGER file: {exc}")

    if raw.empty:
        raise HTTPException(status_code=422, detail="SUBLEDGER file has no data rows.")

    # ── Extract external factors ──────────────────────────────────────────────
    try:
        df, factors_found = _extract_factors_from_df(raw)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    if df.empty:
        raise HTTPException(status_code=422, detail="No valid rows after parsing.")

    # ── Merge with existing stored data and save ──────────────────────────────
    merged = _merge_and_save(df, settings.MODEL_STORE_DIR, current_user.company_id)

    heatmap_b64 = generate_heatmap_from_df(merged)
    date_min = str(merged["date"].min().date()) if not merged.empty else None
    date_max = str(merged["date"].max().date()) if not merged.empty else None

    msg = (f"Extracted {len(factors_found)} factor(s) from SUBLEDGER "
           f"(batch {batch_id}): {', '.join(factors_found)}.")
    if heatmap_b64 is None:
        msg += " Need ≥2 factors with overlapping dates to render heatmap."

    return FromCsvResponse(
        factors_found=factors_found,
        row_count=int(merged.shape[0]),
        date_range_start=date_min,
        date_range_end=date_max,
        heatmap_base64=heatmap_b64,
        message=msg,
    )


@router.post(
    "/refresh",
    response_model=HeatmapRefreshResponse,
    summary="Append new external factor values and get an updated heatmap",
    description=(
        "Send the latest values for any/all external factors. "
        "The values are appended as a new row (dated today) to the stored "
        "historical dataset and the correlation heatmap is recomputed. "
        "You must have called POST /heatmap/upload at least once first."
    ),
)
def refresh_heatmap(
    payload: RefreshRequest,
    current_user=Depends(get_current_user),
):
    stored = load_factor_data(settings.MODEL_STORE_DIR, current_user.company_id)
    if stored is None or stored.empty:
        raise HTTPException(
            status_code=404,
            detail="No factor data found for your company. "
                   "Please call POST /heatmap/upload first.",
        )

    # Build new row from provided values
    new_values = {
        k: v for k, v in payload.model_dump().items() if v is not None
    }
    if not new_values:
        raise HTTPException(
            status_code=422,
            detail="Provide at least one external factor value.",
        )

    # Use today as the date for the new row (month-normalised)
    import datetime
    today = pd.Timestamp(datetime.date.today()).to_period("M").to_timestamp()
    new_row = {"date": today, **new_values}

    new_df = pd.DataFrame([new_row])

    # Append — if today's row already exists, replace it
    combined = stored[stored["date"] != today]
    combined = pd.concat([combined, new_df], ignore_index=True).sort_values("date")

    save_factor_data(combined, settings.MODEL_STORE_DIR, current_user.company_id)

    heatmap_b64 = generate_heatmap_from_df(combined)

    display_row = {k: v for k, v in new_row.items() if k != "date"}
    display_row["date"] = str(today.date())

    return HeatmapRefreshResponse(
        row_count=int(combined.shape[0]),
        new_row=display_row,
        heatmap_base64=heatmap_b64,
        message=(
            "Heatmap updated with new values."
            if heatmap_b64
            else "Row appended but heatmap needs ≥2 factors with data."
        ),
    )


@router.get(
    "/data",
    response_model=FactorDataResponse,
    summary="Return the stored external factor dataset as JSON",
)
def get_factor_data(current_user=Depends(get_current_user)):
    stored = load_factor_data(settings.MODEL_STORE_DIR, current_user.company_id)
    if stored is None or stored.empty:
        raise HTTPException(
            status_code=404,
            detail="No factor data found. Call POST /heatmap/upload first.",
        )

    df = stored.copy()
    df["date"] = df["date"].astype(str)

    return FactorDataResponse(
        row_count=int(df.shape[0]),
        columns=list(df.columns),
        rows=df.to_dict(orient="records"),
    )


# ── New: heatmap data from backend stored external_factors.csv ────────────────

class CorrelationMatrix(BaseModel):
    factors: List[str]                    # ["CCI", "CPI", "Oil", ...]
    matrix: List[List[float]]             # 2D correlation values — ready for Plotly


class StorageHeatmapResponse(BaseModel):
    factors_found: List[str]
    row_count: int
    date_range_start: Optional[str]
    date_range_end: Optional[str]
    correlation: CorrelationMatrix        # frontend uses this to draw heatmap
    monthly_data: List[dict]             # raw monthly values for time-series charts
    message: str


@router.get(
    "/from-storage",
    response_model=StorageHeatmapResponse,
    summary="Get external factor data + correlation matrix from backend storage",
    description=(
        "Reads model_store/external_factors.csv, extracts CCI, CPI, Oil, GDP, "
        "Unemployment, ROI columns, computes the correlation matrix, and returns "
        "both the correlation matrix (ready for Plotly heatmap) and the raw "
        "monthly values (ready for time-series charts). "
        "No file upload needed — uses the pre-stored merged dataset."
    ),
)
def heatmap_from_storage(current_user=Depends(get_current_user)):
    from ml.ext_factors import EXT_FACTOR_COLS, EXT_FACTORS_FILENAME
    import os

    path = os.path.join(settings.MODEL_STORE_DIR, EXT_FACTORS_FILENAME)
    if not os.path.exists(path):
        raise HTTPException(
            status_code=404,
            detail="external_factors.csv not found in model_store.",
        )

    try:
        raw = pd.read_csv(path)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Cannot read external_factors.csv: {exc}")

    if raw.empty:
        raise HTTPException(status_code=422, detail="external_factors.csv is empty.")

    # ── Find date column ──────────────────────────────────────────────────────
    date_col = None
    for candidate in ["Order Date", "order_date", "date", "Date"]:
        if candidate in raw.columns:
            date_col = candidate
            break
    if date_col is None:
        raise HTTPException(status_code=422,
            detail=f"No date column found. Columns: {list(raw.columns)}")

    # ── Extract factor columns ────────────────────────────────────────────────
    factors_found = [c for c in EXT_FACTOR_COLS if c in raw.columns and raw[c].notna().any()]
    if not factors_found:
        raise HTTPException(status_code=422,
            detail=f"No factor columns found. Expected: {EXT_FACTOR_COLS}.")

    # ── Build tidy DataFrame: one row per month ───────────────────────────────
    keep = [date_col] + factors_found
    df = raw[keep].copy()
    df["date"] = _parse_date_series(df[date_col])
    df = df.dropna(subset=["date"])
    df["date"] = df["date"].dt.to_period("M").dt.to_timestamp()
    df = df.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)

    for factor in factors_found:
        df[factor] = pd.to_numeric(
            df[factor].astype(str).str.replace(r"[$,%\s]", "", regex=True),
            errors="coerce",
        )

    if df.empty:
        raise HTTPException(status_code=422, detail="No valid rows after parsing.")

    # ── Compute correlation matrix ────────────────────────────────────────────
    corr = df[factors_found].corr()
    # Round to 4 decimal places, replace NaN with 0
    matrix = [
        [round(corr.loc[r, c], 4) if not pd.isna(corr.loc[r, c]) else 0.0
         for c in factors_found]
        for r in factors_found
    ]

    # ── Build monthly data rows ───────────────────────────────────────────────
    df["date"] = df["date"].astype(str)
    monthly_data = df.to_dict(orient="records")

    date_min = monthly_data[0]["date"] if monthly_data else None
    date_max = monthly_data[-1]["date"] if monthly_data else None

    return StorageHeatmapResponse(
        factors_found=factors_found,
        row_count=len(monthly_data),
        date_range_start=date_min,
        date_range_end=date_max,
        correlation=CorrelationMatrix(factors=factors_found, matrix=matrix),
        monthly_data=monthly_data,
        message=f"Data ready. {len(factors_found)} factors, {len(monthly_data)} months.",
    )
