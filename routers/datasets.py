"""
GET /datasets                        – list datasets for a user
GET /datasets/{dataset_id}           – get dataset metadata
GET /datasets/{dataset_id}/models    – list models trained on this dataset
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from db.models import Dataset, TrainedModel
from db.session import get_db
from schemas import DatasetResponse, TrainedModelResponse

router = APIRouter(prefix="/datasets", tags=["Datasets"])


@router.get(
    "",
    response_model=list[DatasetResponse],
    summary="List all CSV datasets uploaded by a user",
)
def list_datasets(
    user_id: str = Query(..., description="User/tenant ID"),
    db: Session = Depends(get_db),
):
    datasets = (
        db.query(Dataset)
        .filter_by(user_id=user_id)
        .order_by(Dataset.uploaded_at.desc())
        .all()
    )
    results = []
    for ds in datasets:
        d = DatasetResponse.model_validate(ds)
        d.model_count = len(ds.trained_models)
        results.append(d)
    return results


@router.get(
    "/{dataset_id}",
    response_model=DatasetResponse,
    summary="Get metadata for a single dataset",
)
def get_dataset(dataset_id: int, db: Session = Depends(get_db)):
    ds = db.query(Dataset).get(dataset_id)
    if not ds:
        raise HTTPException(status_code=404, detail="Dataset not found.")
    d = DatasetResponse.model_validate(ds)
    d.model_count = len(ds.trained_models)
    return d


@router.get(
    "/{dataset_id}/models",
    response_model=list[TrainedModelResponse],
    summary="List all trained models associated with a dataset",
    description=(
        "Use this to check if a model trained on the same CSV already exists "
        "before kicking off a new training run."
    ),
)
def list_dataset_models(dataset_id: int, db: Session = Depends(get_db)):
    ds = db.query(Dataset).get(dataset_id)
    if not ds:
        raise HTTPException(status_code=404, detail="Dataset not found.")
    return (
        db.query(TrainedModel)
        .filter_by(dataset_id=dataset_id)
        .order_by(TrainedModel.trained_at.desc())
        .all()
    )
