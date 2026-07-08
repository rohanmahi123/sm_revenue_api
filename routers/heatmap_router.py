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
import os
from typing import Dict, List, Optional

import pandas as pd
import requests as http_requests
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from auth import get_current_user
from config import settings
from db.main_session import get_main_db
from db.models import FileUpload, IngestionBatch
from ml.ext_factors import EXT_FACTOR_COLS, EXT_FACTORS_FILENAME, load_ext_factors
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


# ── Full dataset heatmap: SL CSV + external factors ───────────────────────────

# SL columns included in the full heatmap (must match CSV column names)
_SL_NUMERIC_COLS = [
    "Total Revenue", "COGS", "SG&A",
    "Raw Material", "Direct Labor", "Freight",
    "Storage", "Packaging", "Indirect Labor",
    "Rent & Utility", "Overhead",
]


class FullHeatmapRequest(BaseModel):
    batch_id: int = Field(..., description="Batch ID whose SUBLEDGER CSV to use")
    # Optional ext factor overrides — only applied when value is non-None AND non-zero
    CCI:          Optional[float] = Field(None, description="Override CCI — leave empty to use stored data")
    CPI:          Optional[float] = Field(None, description="Override CPI — leave empty to use stored data")
    Oil:          Optional[float] = Field(None, description="Override Oil — leave empty to use stored data")
    GDP:          Optional[float] = Field(None, description="Override GDP — leave empty to use stored data")
    Unemployment: Optional[float] = Field(None, description="Override Unemployment — leave empty to use stored data")
    ROI:          Optional[float] = Field(None, description="Override ROI — leave empty to use stored data")


class FullHeatmapResponse(BaseModel):
    columns: List[str]                  # axis labels (same for X and Y)
    correlation: List[List[float]]      # 2D matrix — ready for Plotly
    row_count: int                      # number of SL rows used
    sl_columns_found: List[str]         # which SL numeric cols were present
    ext_columns_found: List[str]        # which ext factor cols were merged
    message: str


@router.post(
    "/full-dataset/{model_id}",
    response_model=FullHeatmapResponse,
    summary="Full feature correlation heatmap — SL CSV + external factors",
    description=(
        "Loads the SUBLEDGER CSV from Supabase Storage for the given batch, "
        "merges macro-economic external factors (CCI, CPI, Oil, GDP, Unemployment, ROI) "
        "from the backend-stored external_factors.csv by year-month, "
        "then computes the Pearson correlation matrix across ALL numeric columns "
        "(Total Revenue, COGS, SG&A, cost components, and external factors). "
        "Returns the matrix as JSON — frontend renders with Plotly. "
        "Pass optional ext factor override values to get an updated matrix "
        "without re-uploading the CSV (useful for live what-if analysis)."
    ),
)
def full_dataset_heatmap(
    model_id: int,
    payload: FullHeatmapRequest,
    main_db: Session = Depends(get_main_db),
    current_user=Depends(get_current_user),
):
    # ── Validate batch belongs to company ────────────────────────────────────
    batch = main_db.query(IngestionBatch).filter(
        IngestionBatch.id == payload.batch_id,
        IngestionBatch.company_id == current_user.company_id,
    ).first()
    if not batch:
        raise HTTPException(status_code=404, detail="Batch not found.")

    sl_upload = main_db.query(FileUpload).filter(
        FileUpload.batch_id == payload.batch_id,
        FileUpload.file_type == "SUBLEDGER",
    ).first()
    if not sl_upload or not sl_upload.file_path:
        raise HTTPException(status_code=404, detail="No SUBLEDGER file found for this batch.")

    # ── Load SL CSV from Supabase Storage ────────────────────────────────────
    file_path = sl_upload.file_path
    supabase_url = settings.SUPABASE_URL.rstrip("/")
    if file_path.startswith("http://") or file_path.startswith("https://"):
        full_url = file_path
    else:
        full_url = f"{supabase_url}/storage/v1/object/public/File%20Storage/{file_path}"

    try:
        resp = http_requests.get(full_url, timeout=60)
        resp.raise_for_status()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to read SUBLEDGER file: {exc}")

    try:
        df = pd.read_csv(io.BytesIO(resp.content))
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Cannot parse SUBLEDGER CSV: {exc}")

    if df.empty:
        raise HTTPException(status_code=422, detail="SUBLEDGER CSV is empty.")

    # ── Clean numeric columns ─────────────────────────────────────────────────
    df.columns = [str(c).strip() for c in df.columns]
    KEEP_AS_STR = {"Region", "Geo", "Country", "Item type", "Customer", "Order Date"}
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

    # ── Merge external factors by year-month ─────────────────────────────────
    ext_df = load_ext_factors(settings.MODEL_STORE_DIR)
    ext_cols_merged: List[str] = []

    if ext_df is not None:
        # Rename Order Date to order_date for merge helper
        date_col = next((c for c in df.columns if c.lower() in ("order date", "order_date")), None)
        if date_col:
            df["order_date"] = pd.to_datetime(df[date_col], errors="coerce")
            df["year_month"] = df["order_date"].dt.to_period("M")
            ext_df2 = ext_df.copy()
            merged = df.merge(ext_df2, on="year_month", how="left")
            merged.drop(columns=["year_month", "order_date"], inplace=True, errors="ignore")
            df = merged
            ext_cols_merged = [c for c in EXT_FACTOR_COLS if c in df.columns]

    # ── Apply user ext factor overrides (for live what-if refresh) ───────────
    overrides = {
        k: v for k, v in {
            "CCI": payload.CCI, "CPI": payload.CPI, "Oil": payload.Oil,
            "GDP": payload.GDP, "Unemployment": payload.Unemployment, "ROI": payload.ROI,
        }.items() if v is not None and v != 0
    }
    for factor, val in overrides.items():
        df[factor] = val  # overwrite entire column with the new value
        if factor not in ext_cols_merged:
            ext_cols_merged.append(factor)

    # ── Select columns for correlation ────────────────────────────────────────
    sl_cols_found = [c for c in _SL_NUMERIC_COLS if c in df.columns and df[c].notna().any()]
    all_cols = sl_cols_found + [c for c in ext_cols_merged if c not in sl_cols_found]

    if len(all_cols) < 2:
        raise HTTPException(
            status_code=422,
            detail=f"Need at least 2 numeric columns for correlation. Found: {all_cols}",
        )

    corr_df = df[all_cols].dropna(how="all")
    corr = corr_df.corr()

    matrix = [
        [round(corr.loc[r, c], 4) if not pd.isna(corr.loc[r, c]) else 0.0
         for c in all_cols]
        for r in all_cols
    ]

    return FullHeatmapResponse(
        columns=all_cols,
        correlation=matrix,
        row_count=len(corr_df),
        sl_columns_found=sl_cols_found,
        ext_columns_found=ext_cols_merged,
        message=(
            f"Correlation matrix computed. {len(sl_cols_found)} SL columns + "
            f"{len(ext_cols_merged)} ext factors = {len(all_cols)} total features. "
            + (f"Overrides applied: {list(overrides.keys())}." if overrides else "")
        ),
    )


# ── External factors vs SL columns cross-correlation heatmap ─────────────────
# Rows = SL business columns, Columns = external factors
# Result is a rectangular matrix (11 × 6) showing how macro factors
# correlate with each business metric in the CSV.

class ExtVsSlHeatmapRequest(BaseModel):
    batch_id: int = Field(..., description="Batch ID whose SUBLEDGER CSV to use")
    CCI:          Optional[float] = Field(None, description="Override CCI — leave empty to use stored data")
    CPI:          Optional[float] = Field(None, description="Override CPI — leave empty to use stored data")
    Oil:          Optional[float] = Field(None, description="Override Oil — leave empty to use stored data")
    GDP:          Optional[float] = Field(None, description="Override GDP — leave empty to use stored data")
    Unemployment: Optional[float] = Field(None, description="Override Unemployment — leave empty to use stored data")
    ROI:          Optional[float] = Field(None, description="Override ROI — leave empty to use stored data")


class ExtVsSlHeatmapResponse(BaseModel):
    sl_columns: List[str]           # Y axis — SL business columns
    ext_columns: List[str]          # X axis — external factors
    correlation: List[List[float]]  # rectangular matrix [sl_col][ext_col]
    row_count: int
    message: str


@router.post(
    "/external-factors/{model_id}",
    response_model=ExtVsSlHeatmapResponse,
    summary="Cross-correlation: external factors vs SL business columns",
    description=(
        "Loads the SUBLEDGER CSV from Supabase Storage for the given batch, "
        "merges external factors (CCI, CPI, Oil, GDP, Unemployment, ROI) by year-month, "
        "then computes the Pearson correlation between each external factor and each "
        "SL business column (Total Revenue, COGS, SG&A, Raw Material, etc.). "
        "Returns a rectangular matrix — Y axis is SL columns, X axis is external factors. "
        "Pass optional override values for live what-if analysis."
    ),
)
def external_vs_sl_heatmap(
    model_id: int,
    payload: ExtVsSlHeatmapRequest,
    main_db: Session = Depends(get_main_db),
    current_user=Depends(get_current_user),
):
    # ── Verify batch belongs to company ───────────────────────────────────────
    batch = main_db.query(IngestionBatch).filter(
        IngestionBatch.id == payload.batch_id,
        IngestionBatch.company_id == current_user.company_id,
    ).first()
    if not batch:
        raise HTTPException(status_code=404, detail="Batch not found.")

    sl_upload = main_db.query(FileUpload).filter(
        FileUpload.batch_id == payload.batch_id,
        FileUpload.file_type == "SUBLEDGER",
    ).first()
    if not sl_upload or not sl_upload.file_path:
        raise HTTPException(status_code=404, detail="No SUBLEDGER file found for this batch.")

    # ── Load SL CSV from Supabase Storage ────────────────────────────────────
    file_path = sl_upload.file_path
    supabase_url = settings.SUPABASE_URL.rstrip("/")
    full_url = (
        file_path if file_path.startswith("http")
        else f"{supabase_url}/storage/v1/object/public/File%20Storage/{file_path}"
    )
    try:
        resp = http_requests.get(full_url, timeout=60)
        resp.raise_for_status()
        df = pd.read_csv(io.BytesIO(resp.content))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to load SUBLEDGER CSV: {exc}")

    if df.empty:
        raise HTTPException(status_code=422, detail="SUBLEDGER CSV is empty.")

    # ── Clean numeric columns ─────────────────────────────────────────────────
    df.columns = [str(c).strip() for c in df.columns]
    KEEP_AS_STR = {"Region", "Geo", "Country", "Item type", "Customer", "Order Date"}
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

    # ── Merge external factors by year-month ─────────────────────────────────
    ext_df = load_ext_factors(settings.MODEL_STORE_DIR)
    ext_cols_merged: List[str] = []

    date_col = next((c for c in df.columns if c.lower() in ("order date", "order_date")), None)
    if ext_df is not None and date_col:
        df["_order_date"] = pd.to_datetime(df[date_col], errors="coerce")
        df["year_month"] = df["_order_date"].dt.to_period("M")
        merged = df.merge(ext_df, on="year_month", how="left")
        merged.drop(columns=["year_month", "_order_date"], inplace=True, errors="ignore")
        df = merged
        ext_cols_merged = [c for c in EXT_FACTOR_COLS if c in df.columns]

    # ── Apply user overrides ──────────────────────────────────────────────────
    overrides = {
        k: v for k, v in {
            "CCI": payload.CCI, "CPI": payload.CPI, "Oil": payload.Oil,
            "GDP": payload.GDP, "Unemployment": payload.Unemployment, "ROI": payload.ROI,
        }.items() if v is not None and v != 0
    }
    for factor, val in overrides.items():
        df[factor] = val
        if factor not in ext_cols_merged:
            ext_cols_merged.append(factor)

    # ── Pick SL columns and ext columns that actually exist ───────────────────
    sl_cols = [c for c in _SL_NUMERIC_COLS if c in df.columns and df[c].notna().any()]
    ext_cols = [c for c in ext_cols_merged if df[c].notna().any()]

    if not sl_cols:
        raise HTTPException(status_code=422, detail="No SL numeric columns found in CSV.")
    if not ext_cols:
        raise HTTPException(status_code=422, detail="No external factor columns found after merge.")

    # ── Compute rectangular cross-correlation matrix ──────────────────────────
    # corr_df has all columns; we slice rows=sl_cols, cols=ext_cols
    all_cols = list(dict.fromkeys(sl_cols + ext_cols))  # deduplicated, order preserved
    full_corr = df[all_cols].corr()

    # matrix[i][j] = correlation between sl_cols[i] and ext_cols[j]
    matrix = [
        [round(full_corr.loc[sl, ex], 4) if not pd.isna(full_corr.loc[sl, ex]) else 0.0
         for ex in ext_cols]
        for sl in sl_cols
    ]

    return ExtVsSlHeatmapResponse(
        sl_columns=sl_cols,
        ext_columns=ext_cols,
        correlation=matrix,
        row_count=len(df),
        message=(
            f"Cross-correlation: {len(sl_cols)} SL columns × {len(ext_cols)} external factors. "
            + (f"Overrides applied: {list(overrides.keys())}." if overrides else "")
        ),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Shared schemas and helpers for append / replace-latest endpoints
# ─────────────────────────────────────────────────────────────────────────────

import datetime as _dt


class ExtFactorRowInput(BaseModel):
    """Ext factor values sent by the user for append or replace-latest."""
    CCI:          Optional[float] = Field(None, description="Consumer Confidence Index")
    CPI:          Optional[float] = Field(None, description="Consumer Price Index")
    Oil:          Optional[float] = Field(None, description="Crude Oil price")
    GDP:          Optional[float] = Field(None, description="GDP value")
    Unemployment: Optional[float] = Field(None, description="Unemployment rate")
    ROI:          Optional[float] = Field(None, description="Return on Investment")


class BatchExtFactorRowInput(ExtFactorRowInput):
    """Same as above but also needs batch_id for SL-based endpoints."""
    batch_id: int = Field(..., description="Batch ID whose SUBLEDGER CSV to use")


def _load_ext_factors_df() -> pd.DataFrame:
    """Load and parse external_factors.csv into a clean monthly DataFrame."""
    path = os.path.join(settings.MODEL_STORE_DIR, EXT_FACTORS_FILENAME)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="external_factors.csv not found in model_store.")
    raw = pd.read_csv(path)
    date_col = next((c for c in raw.columns if c.lower() in ("order date", "order_date", "date")), None)
    if date_col is None:
        raise HTTPException(status_code=422, detail="No date column found in external_factors.csv.")
    factors_found = [c for c in EXT_FACTOR_COLS if c in raw.columns]
    df = raw[[date_col] + factors_found].copy()
    df["date"] = _parse_date_series(df[date_col])
    df = df.dropna(subset=["date"])
    df["date"] = df["date"].dt.to_period("M").dt.to_timestamp()
    df = df.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)
    for f in factors_found:
        df[f] = pd.to_numeric(df[f].astype(str).str.replace(r"[$,%\s]", "", regex=True), errors="coerce")
    return df


def _load_sl_df_from_batch(batch_id: int, company_id, main_db: Session) -> pd.DataFrame:
    """Load and clean the SUBLEDGER CSV from Supabase Storage for a given batch."""
    batch = main_db.query(IngestionBatch).filter(
        IngestionBatch.id == batch_id,
        IngestionBatch.company_id == company_id,
    ).first()
    if not batch:
        raise HTTPException(status_code=404, detail="Batch not found.")
    sl_upload = main_db.query(FileUpload).filter(
        FileUpload.batch_id == batch_id,
        FileUpload.file_type == "SUBLEDGER",
    ).first()
    if not sl_upload or not sl_upload.file_path:
        raise HTTPException(status_code=404, detail="No SUBLEDGER file found for this batch.")
    file_path = sl_upload.file_path
    supabase_url = settings.SUPABASE_URL.rstrip("/")
    full_url = (
        file_path if file_path.startswith("http")
        else f"{supabase_url}/storage/v1/object/public/File%20Storage/{file_path}"
    )
    try:
        resp = http_requests.get(full_url, timeout=60)
        resp.raise_for_status()
        df = pd.read_csv(io.BytesIO(resp.content))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to load SUBLEDGER CSV: {exc}")
    df.columns = [str(c).strip() for c in df.columns]
    KEEP_AS_STR = {"Region", "Geo", "Country", "Item type", "Customer", "Order Date"}
    for col in df.columns:
        if col in KEEP_AS_STR:
            continue
        df[col] = pd.to_numeric(
            df[col].astype(str).str.replace("$", "", regex=False)
            .str.replace(",", "", regex=False).str.strip(),
            errors="coerce",
        )
    return df


def _user_ext_values(payload: ExtFactorRowInput) -> dict:
    return {
        k: v for k, v in {
            "CCI": payload.CCI, "CPI": payload.CPI, "Oil": payload.Oil,
            "GDP": payload.GDP, "Unemployment": payload.Unemployment, "ROI": payload.ROI,
        }.items() if v is not None
    }


def _compute_6x6(df: pd.DataFrame, factors: List[str]) -> tuple:
    corr = df[factors].corr()
    matrix = [
        [round(corr.loc[r, c], 4) if not pd.isna(corr.loc[r, c]) else 0.0 for c in factors]
        for r in factors
    ]
    return factors, matrix


def _compute_rect(df: pd.DataFrame, sl_cols: List[str], ext_cols: List[str]) -> List[List[float]]:
    all_cols = list(dict.fromkeys(sl_cols + ext_cols))
    corr = df[all_cols].corr()
    return [
        [round(corr.loc[sl, ex], 4) if not pd.isna(corr.loc[sl, ex]) else 0.0 for ex in ext_cols]
        for sl in sl_cols
    ]


def _compute_full(df: pd.DataFrame, all_cols: List[str]) -> List[List[float]]:
    corr = df[all_cols].corr()
    return [
        [round(corr.loc[r, c], 4) if not pd.isna(corr.loc[r, c]) else 0.0 for c in all_cols]
        for r in all_cols
    ]


# ─────────────────────────────────────────────────────────────────────────────
# 6×6  —  Option A: append new row
# ─────────────────────────────────────────────────────────────────────────────

@router.post(
    "/from-storage/append",
    response_model=StorageHeatmapResponse,
    summary="6×6 ext factors — append new monthly row then recompute",
    description=(
        "Appends the user-provided ext factor values as a new row (current month) "
        "to the stored external_factors.csv dataset, then recomputes the 6×6 "
        "Pearson correlation matrix. The original file is NOT modified — the append "
        "is in-memory only for this request."
    ),
)
def ext_6x6_append(
    payload: ExtFactorRowInput,
    current_user=Depends(get_current_user),
):
    df = _load_ext_factors_df()
    factors = [c for c in EXT_FACTOR_COLS if c in df.columns]
    user_vals = _user_ext_values(payload)
    if not user_vals:
        raise HTTPException(status_code=422, detail="Provide at least one external factor value.")

    # Build new row — current month, fill missing factors with column median
    new_month = pd.Timestamp(_dt.date.today()).to_period("M").to_timestamp()
    new_row = {"date": new_month}
    for f in factors:
        new_row[f] = user_vals.get(f, df[f].median())

    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    factors_out, matrix = _compute_6x6(df, factors)
    df["date"] = df["date"].astype(str)
    monthly_data = df[["date"] + factors].to_dict(orient="records")

    return StorageHeatmapResponse(
        factors_found=factors_out,
        row_count=len(monthly_data),
        date_range_start=monthly_data[0]["date"] if monthly_data else None,
        date_range_end=monthly_data[-1]["date"] if monthly_data else None,
        correlation=CorrelationMatrix(factors=factors_out, matrix=matrix),
        monthly_data=monthly_data,
        message=f"New row appended (current month). {len(factors_out)} factors, {len(monthly_data)} months total.",
    )


# ─────────────────────────────────────────────────────────────────────────────
# 6×6  —  Option B: replace latest row
# ─────────────────────────────────────────────────────────────────────────────

@router.post(
    "/from-storage/replace-latest",
    response_model=StorageHeatmapResponse,
    summary="6×6 ext factors — replace latest monthly row then recompute",
    description=(
        "Replaces the most recent month's ext factor values with the user-provided "
        "values, then recomputes the 6×6 Pearson correlation matrix. "
        "In-memory only — the stored file is not modified."
    ),
)
def ext_6x6_replace_latest(
    payload: ExtFactorRowInput,
    current_user=Depends(get_current_user),
):
    df = _load_ext_factors_df()
    factors = [c for c in EXT_FACTOR_COLS if c in df.columns]
    user_vals = _user_ext_values(payload)
    if not user_vals:
        raise HTTPException(status_code=422, detail="Provide at least one external factor value.")

    # Replace only the values the user sent in the latest row
    for f, v in user_vals.items():
        if f in df.columns:
            df.loc[df.index[-1], f] = v

    factors_out, matrix = _compute_6x6(df, factors)
    df["date"] = df["date"].astype(str)
    monthly_data = df[["date"] + factors].to_dict(orient="records")

    return StorageHeatmapResponse(
        factors_found=factors_out,
        row_count=len(monthly_data),
        date_range_start=monthly_data[0]["date"] if monthly_data else None,
        date_range_end=monthly_data[-1]["date"] if monthly_data else None,
        correlation=CorrelationMatrix(factors=factors_out, matrix=matrix),
        monthly_data=monthly_data,
        message=f"Latest row updated. {len(factors_out)} factors, {len(monthly_data)} months total.",
    )


# ─────────────────────────────────────────────────────────────────────────────
# 11×6  —  Option A: append new row
# ─────────────────────────────────────────────────────────────────────────────

@router.post(
    "/external-factors/append/{model_id}",
    response_model=ExtVsSlHeatmapResponse,
    summary="11×6 cross-correlation — append new monthly row then recompute",
    description=(
        "Appends the user-provided ext factor values as a new row to the merged dataset. "
        "SL column values for the new row are filled using the median of historical data. "
        "Recomputes the 11×6 cross-correlation matrix. In-memory only."
    ),
)
def ext_11x6_append(
    model_id: int,
    payload: BatchExtFactorRowInput,
    main_db: Session = Depends(get_main_db),
    current_user=Depends(get_current_user),
):
    sl_df = _load_sl_df_from_batch(payload.batch_id, current_user.company_id, main_db)
    ext_df = _load_ext_factors_df()
    user_vals = _user_ext_values(payload)
    if not user_vals:
        raise HTTPException(status_code=422, detail="Provide at least one external factor value.")

    # Merge ext factors into SL data
    date_col = next((c for c in sl_df.columns if c.lower() in ("order date", "order_date")), None)
    ext_cols_merged: List[str] = []
    if date_col and ext_df is not None:
        sl_df["year_month"] = pd.to_datetime(sl_df[date_col], errors="coerce").dt.to_period("M")
        ext_df["year_month"] = ext_df["date"].dt.to_period("M")
        sl_df = sl_df.merge(ext_df.drop(columns=["date"]), on="year_month", how="left")
        sl_df.drop(columns=["year_month"], inplace=True, errors="ignore")
        ext_cols_merged = [c for c in EXT_FACTOR_COLS if c in sl_df.columns]

    sl_cols = [c for c in _SL_NUMERIC_COLS if c in sl_df.columns and sl_df[c].notna().any()]
    ext_cols = [c for c in ext_cols_merged if sl_df[c].notna().any()]

    # Build new row: median SL values + user ext values
    new_row = {}
    for c in sl_cols:
        new_row[c] = sl_df[c].median()
    for f in ext_cols:
        new_row[f] = user_vals.get(f, sl_df[f].median())

    all_cols = list(dict.fromkeys(sl_cols + ext_cols))
    sl_df = pd.concat([sl_df[all_cols], pd.DataFrame([new_row])], ignore_index=True)
    matrix = _compute_rect(sl_df, sl_cols, ext_cols)

    return ExtVsSlHeatmapResponse(
        sl_columns=sl_cols,
        ext_columns=ext_cols,
        correlation=matrix,
        row_count=len(sl_df),
        message=f"New row appended. Cross-correlation: {len(sl_cols)} SL × {len(ext_cols)} ext factors.",
    )


# ─────────────────────────────────────────────────────────────────────────────
# 11×6  —  Option B: replace latest row
# ─────────────────────────────────────────────────────────────────────────────

@router.post(
    "/external-factors/replace-latest/{model_id}",
    response_model=ExtVsSlHeatmapResponse,
    summary="11×6 cross-correlation — replace latest row ext values then recompute",
    description=(
        "Replaces the ext factor values of the most recent row in the merged dataset "
        "with the user-provided values, then recomputes the 11×6 cross-correlation. "
        "In-memory only."
    ),
)
def ext_11x6_replace_latest(
    model_id: int,
    payload: BatchExtFactorRowInput,
    main_db: Session = Depends(get_main_db),
    current_user=Depends(get_current_user),
):
    sl_df = _load_sl_df_from_batch(payload.batch_id, current_user.company_id, main_db)
    ext_df = _load_ext_factors_df()
    user_vals = _user_ext_values(payload)
    if not user_vals:
        raise HTTPException(status_code=422, detail="Provide at least one external factor value.")

    date_col = next((c for c in sl_df.columns if c.lower() in ("order date", "order_date")), None)
    ext_cols_merged: List[str] = []
    if date_col and ext_df is not None:
        sl_df["year_month"] = pd.to_datetime(sl_df[date_col], errors="coerce").dt.to_period("M")
        sl_df = sl_df.merge(ext_df, on="year_month", how="left")
        sl_df.drop(columns=["year_month"], inplace=True, errors="ignore")
        ext_cols_merged = [c for c in EXT_FACTOR_COLS if c in sl_df.columns]

    sl_cols = [c for c in _SL_NUMERIC_COLS if c in sl_df.columns and sl_df[c].notna().any()]
    ext_cols = [c for c in ext_cols_merged if sl_df[c].notna().any()]

    # Replace ext factor values in the last row only
    for f, v in user_vals.items():
        if f in sl_df.columns:
            sl_df.loc[sl_df.index[-1], f] = v

    matrix = _compute_rect(sl_df, sl_cols, ext_cols)

    return ExtVsSlHeatmapResponse(
        sl_columns=sl_cols,
        ext_columns=ext_cols,
        correlation=matrix,
        row_count=len(sl_df),
        message=f"Latest row updated. Cross-correlation: {len(sl_cols)} SL × {len(ext_cols)} ext factors.",
    )


# ─────────────────────────────────────────────────────────────────────────────
# 17×17  —  Option A: append new row
# ─────────────────────────────────────────────────────────────────────────────

@router.post(
    "/full-dataset/append/{model_id}",
    response_model=FullHeatmapResponse,
    summary="17×17 full heatmap — append new monthly row then recompute",
    description=(
        "Appends the user-provided ext factor values as a new row to the merged dataset. "
        "SL column values for the new row are filled with historical medians. "
        "Recomputes the full N×N correlation matrix. In-memory only."
    ),
)
def full_17x17_append(
    model_id: int,
    payload: BatchExtFactorRowInput,
    main_db: Session = Depends(get_main_db),
    current_user=Depends(get_current_user),
):
    sl_df = _load_sl_df_from_batch(payload.batch_id, current_user.company_id, main_db)
    ext_df = _load_ext_factors_df()
    user_vals = _user_ext_values(payload)
    if not user_vals:
        raise HTTPException(status_code=422, detail="Provide at least one external factor value.")

    date_col = next((c for c in sl_df.columns if c.lower() in ("order date", "order_date")), None)
    ext_cols_merged: List[str] = []
    if date_col and ext_df is not None:
        sl_df["year_month"] = pd.to_datetime(sl_df[date_col], errors="coerce").dt.to_period("M")
        sl_df = sl_df.merge(ext_df, on="year_month", how="left")
        sl_df.drop(columns=["year_month"], inplace=True, errors="ignore")
        ext_cols_merged = [c for c in EXT_FACTOR_COLS if c in sl_df.columns]

    sl_cols = [c for c in _SL_NUMERIC_COLS if c in sl_df.columns and sl_df[c].notna().any()]
    ext_cols = [c for c in ext_cols_merged if sl_df[c].notna().any()]
    all_cols = list(dict.fromkeys(sl_cols + ext_cols))

    # Build new row: median SL + user ext values
    new_row = {}
    for c in sl_cols:
        new_row[c] = sl_df[c].median()
    for f in ext_cols:
        new_row[f] = user_vals.get(f, sl_df[f].median())

    sl_df = pd.concat([sl_df[all_cols], pd.DataFrame([new_row])], ignore_index=True)
    matrix = _compute_full(sl_df, all_cols)

    return FullHeatmapResponse(
        columns=all_cols,
        correlation=matrix,
        row_count=len(sl_df),
        sl_columns_found=sl_cols,
        ext_columns_found=ext_cols,
        message=f"New row appended. Full matrix: {len(all_cols)} features.",
    )


# ─────────────────────────────────────────────────────────────────────────────
# 17×17  —  Option B: replace latest row
# ─────────────────────────────────────────────────────────────────────────────

@router.post(
    "/full-dataset/replace-latest/{model_id}",
    response_model=FullHeatmapResponse,
    summary="17×17 full heatmap — replace latest row ext values then recompute",
    description=(
        "Replaces the ext factor values of the most recent row in the merged dataset "
        "with the user-provided values, then recomputes the full N×N correlation matrix. "
        "In-memory only."
    ),
)
def full_17x17_replace_latest(
    model_id: int,
    payload: BatchExtFactorRowInput,
    main_db: Session = Depends(get_main_db),
    current_user=Depends(get_current_user),
):
    sl_df = _load_sl_df_from_batch(payload.batch_id, current_user.company_id, main_db)
    ext_df = _load_ext_factors_df()
    user_vals = _user_ext_values(payload)
    if not user_vals:
        raise HTTPException(status_code=422, detail="Provide at least one external factor value.")

    date_col = next((c for c in sl_df.columns if c.lower() in ("order date", "order_date")), None)
    ext_cols_merged: List[str] = []
    if date_col and ext_df is not None:
        sl_df["year_month"] = pd.to_datetime(sl_df[date_col], errors="coerce").dt.to_period("M")
        sl_df = sl_df.merge(ext_df, on="year_month", how="left")
        sl_df.drop(columns=["year_month"], inplace=True, errors="ignore")
        ext_cols_merged = [c for c in EXT_FACTOR_COLS if c in sl_df.columns]

    sl_cols = [c for c in _SL_NUMERIC_COLS if c in sl_df.columns and sl_df[c].notna().any()]
    ext_cols = [c for c in ext_cols_merged if sl_df[c].notna().any()]
    all_cols = list(dict.fromkeys(sl_cols + ext_cols))

    # Replace ext factor values in last row only
    for f, v in user_vals.items():
        if f in sl_df.columns:
            sl_df.loc[sl_df.index[-1], f] = v

    matrix = _compute_full(sl_df, all_cols)

    return FullHeatmapResponse(
        columns=all_cols,
        correlation=matrix,
        row_count=len(sl_df),
        sl_columns_found=sl_cols,
        ext_columns_found=ext_cols,
        message=f"Latest row updated. Full matrix: {len(all_cols)} features.",
    )
