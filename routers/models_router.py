
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from auth import get_current_user
from db.models import TrainedModel
from db.session import get_db
from schemas import TrainedModelResponse

router = APIRouter(prefix="/models", tags=["Models"])


@router.get(
    "",
    response_model=list[TrainedModelResponse],
    summary="List all trained models for a user",
)
def list_models(
    dataset_id: int | None = Query(None, description="Filter by dataset"),
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    q = db.query(TrainedModel).filter_by(user_id=str(current_user.id))
    if dataset_id is not None:
        q = q.filter_by(dataset_id=dataset_id)
    return q.order_by(TrainedModel.trained_at.desc()).all()


@router.get(
    "/{model_id}",
    response_model=TrainedModelResponse,
    summary="Get a single trained model by ID",
)
def get_model(model_id: int, db: Session = Depends(get_db), current_user=Depends(get_current_user)):
    tm = db.query(TrainedModel).get(model_id)
    if not tm:
        raise HTTPException(status_code=404, detail="Model not found.")
    return tm


@router.delete(
    "/{model_id}",
    summary="Delete a trained model (also removes the .pkl file)",
)
def delete_model(model_id: int, db: Session = Depends(get_db), current_user=Depends(get_current_user)):
    tm = db.query(TrainedModel).get(model_id)
    if not tm:
        raise HTTPException(status_code=404, detail="Model not found.")

    pkl = Path(tm.model_file_path)
    if pkl.exists():
        pkl.unlink()

    db.delete(tm)
    db.commit()
    return {"detail": f"Model '{tm.model_name}' deleted."}
