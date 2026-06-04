

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from auth import get_current_user
from config import settings
from db.models import Dataset, TrainedModel
from db.session import get_db
from ml.trainer import train_model
from schemas import TrainResponse

router = APIRouter(prefix="/train", tags=["Training"])


def _hash_file(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


@router.post(
    "",
    response_model=TrainResponse,
    summary="Upload a CSV and train a new model",
    description=(
        "Upload the main financial CSV plus a JSON payload of external macro "
        "factors (CCI, CPI, Oil, GDP, Unemployment, ROI). "
        "The API hashes the CSV: if the same file was uploaded before it "
        "reuses the existing Dataset record so you can see all models "
        "trained on it via GET /datasets/{id}/models."
    ),
)
async def train(
    # ── CSV file ──────────────────────────────────────────────────────────────
    file: UploadFile = File(..., description="Main financial data (.csv or .xlsx)"),

    # ── Form fields ───────────────────────────────────────────────────────────
    model_name: str = Form(..., description="Unique name for this trained model"),
    description: Optional[str] = Form(None),
    sheet_name: str = Form("10 SL", description="Excel sheet (ignored for CSV)"),
    test_size: float = Form(0.25),
    random_state: int = Form(42),

    # ── External factors as a JSON string ─────────────────────────────────────
    external_factors_json: Optional[str] = Form(
        None,
        description=(
            'JSON string: {"rows": [{"date":"2023-01-01","CCI":101,"CPI":2.5,...}]}'
        ),
    ),

    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    user_id = str(current_user.id)
    # 1. Read uploaded file
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    file_hash = _hash_file(content)

    # 2. Parse external factors
    ext_rows = None
    if external_factors_json:
        try:
            payload = json.loads(external_factors_json)
            ext_rows = payload.get("rows", [])
        except json.JSONDecodeError as exc:
            raise HTTPException(
                status_code=422,
                detail=f"external_factors_json is not valid JSON: {exc}",
            )

    # 3. Check if model_name already taken
    existing_model = db.query(TrainedModel).filter_by(model_name=model_name).first()
    if existing_model:
        raise HTTPException(
            status_code=409,
            detail=f"A model named '{model_name}' already exists. Choose a different name.",
        )

    # 4. Lookup or create Dataset record
    dataset = (
        db.query(Dataset)
        .filter_by(user_id=user_id, file_hash=file_hash)
        .first()
    )
    is_new_dataset = dataset is None

    # 5. Save CSV to a temp file so sklearn can read it
    suffix = os.path.splitext(file.filename or "upload.csv")[1] or ".csv"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        # 6. Train
        model_file_path, result, duration = train_model(
            csv_path=tmp_path,
            external_factor_rows=ext_rows,
            model_store_dir=settings.MODEL_STORE_DIR,
            model_name=model_name,
            sheet_name=sheet_name,
            test_size=test_size,
            random_state=random_state,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Training failed: {exc}")
    finally:
        os.unlink(tmp_path)

    # 7. Persist Dataset (only once per unique CSV per user)
    if is_new_dataset:
        import pandas as pd

        # Re-read a tiny slice just for metadata
        try:
            meta_df = pd.read_csv(
                content.decode("utf-8", errors="ignore").splitlines().__iter__()
                if suffix == ".csv"
                else None
            ) if suffix == ".csv" else pd.DataFrame()
        except Exception:
            meta_df = pd.DataFrame()

        date_start = date_end = None
        if "Order Date" in meta_df.columns:
            from ml.preprocessor import safe_parse_dates
            dates = safe_parse_dates(meta_df["Order Date"])
            valid = dates.dropna()
            if not valid.empty:
                date_start = str(valid.min().date())
                date_end = str(valid.max().date())

        dataset = Dataset(
            user_id=user_id,
            original_filename=file.filename or "unknown",
            file_hash=file_hash,
            file_size_bytes=len(content),
            row_count=int(meta_df.shape[0]) if not meta_df.empty else None,
            column_names=list(meta_df.columns) if not meta_df.empty else None,
            date_range_start=date_start,
            date_range_end=date_end,
            target_columns=result["targets"],
            notes=description,
        )
        db.add(dataset)
        db.flush()   # get dataset.id without committing

    # 8. Persist TrainedModel
    tm = TrainedModel(
        dataset_id=dataset.id,
        user_id=user_id,
        model_name=model_name,
        description=description,
        model_file_path=model_file_path,
        targets=result["targets"],
        feature_columns=result["feature_columns"],
        external_factors_used=result["external_factors_used"],
        test_size=test_size,
        random_state=random_state,
        metrics=result["metrics"],
        training_duration_seconds=duration,
    )
    db.add(tm)
    db.commit()
    db.refresh(tm)

    return TrainResponse(
        model_id=tm.id,
        dataset_id=dataset.id,
        model_name=tm.model_name,
        is_new_dataset=is_new_dataset,
        training_duration_seconds=duration,
        metrics=result["metrics"],
        best_model_per_target=result["best_model_per_target"],
        message=(
            "Model trained successfully. "
            + ("New dataset registered." if is_new_dataset else "Existing dataset reused.")
        ),
    )
