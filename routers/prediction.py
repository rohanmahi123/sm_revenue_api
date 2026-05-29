"""
POST /predict/{model_id}
  – Run inference against a stored model using provided daily input rows.
  – Generates a correlation heatmap from the external factors in the input.
    Change any external factor value → different heatmap returned.
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from db.models import TrainedModel
from db.session import get_db
from ml.heatmap import generate_heatmap
from ml.predictor import predict
from schemas import PredictRequest, PredictResponse

router = APIRouter(prefix="/predict", tags=["Prediction"])

EXTERNAL_FACTORS = ["CCI", "CPI", "Oil", "GDP", "Unemployment", "ROI"]


@router.post(
    "/{model_id}",
    response_model=PredictResponse,
    summary="Run predictions + get external factors heatmap",
    description=(
        "Send an array of monthly/daily input rows (cost components + macro values). "
        "Returns predictions for Revenue, COGS, SGA **and** a correlation heatmap "
        "built from the external factor values you provided. "
        "Change CCI/Oil/GDP values → heatmap updates automatically."
    ),
)
def run_prediction(
    model_id: int,
    payload: PredictRequest,
    db: Session = Depends(get_db),
):
    tm = db.query(TrainedModel).get(model_id)
    if not tm:
        raise HTTPException(status_code=404, detail="Model not found.")

    if not payload.rows:
        raise HTTPException(status_code=422, detail="rows list must not be empty.")

    # Serialise Pydantic rows → plain dicts
    raw_rows = [r.model_dump(by_alias=True) for r in payload.rows]


    try:
        results = predict(tm.model_file_path, raw_rows)
    except FileNotFoundError:
        raise HTTPException(
            status_code=500,
            detail="Model file not found on disk. It may have been deleted manually.",
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction error: {exc}")


    alias_map = {
        "Item type": "Item_type",
        "Raw Material": "Raw_Material",
        "Direct Labor": "Direct_Labor",
        "Indirect Labor": "Indirect_Labor",
        "Rent & Utility": "Rent_Utility",
    }
    plain_rows = []
    for row in raw_rows:
        plain = {}
        for k, v in row.items():
            plain[alias_map.get(k, k)] = v
        plain_rows.append(plain)

    factors_present = [
        f for f in EXTERNAL_FACTORS
        if any(r.get(f) is not None for r in plain_rows)
    ]

    heatmap_b64 = generate_heatmap(plain_rows)

    if heatmap_b64:
        note = (
            f"Heatmap shows correlation between: {', '.join(factors_present)}. "
            "Change any of these values in your request and the heatmap updates."
        )
    else:
        note = (
            "Heatmap not generated — provide at least 2 external factor columns "
            "(CCI, CPI, Oil, GDP, Unemployment, ROI) with values across multiple rows."
        )

    return PredictResponse(
        model_id=tm.id,
        model_name=tm.model_name,
        predictions=results,
        heatmap_base64=heatmap_b64,
        heatmap_note=note,
    )
