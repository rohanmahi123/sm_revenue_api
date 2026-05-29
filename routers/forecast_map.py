"""
PATCH FILE — Sales Forecast Map
================================
Just add these 2 lines to main.py (nothing else changes):

    from routers.forecast_map import router as forecast_map_router
    app.include_router(forecast_map_router)

New endpoint added:
    POST /forecast-map/{model_id}

Same payload as POST /predict/{model_id} — no changes to input format.
Response includes predictions + a regional sales forecast map as base64 PNG.
"""

from __future__ import annotations

import base64
import io
from collections import defaultdict
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from db.models import TrainedModel
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
    if not region_summary:
        return None

    regions = [r.region for r in region_summary]
    revenues = [r.total_revenue for r in region_summary]
    cogs = [r.total_COGS for r in region_summary]
    sga = [r.total_SGA for r in region_summary]

    n = len(regions)
    x = np.arange(n)
    bar_w = 0.28

    # ── Colors ────────────────────────────────────────────────────────────────
    BG = "#0f0f1a"
    GRID = "#1e1e30"
    REV_COLOR = "#4f9cf9"
    COGS_COLOR = "#f97b4f"
    SGA_COLOR = "#7bf97b"
    TEXT = "#e0e0ff"

    fig, (ax_map, ax_bar) = plt.subplots(
        1, 2, figsize=(16, 7), gridspec_kw={"width_ratios": [1.2, 1.8]}
    )
    fig.patch.set_facecolor(BG)

    # ── LEFT: Bubble "map" (region bubbles sized by revenue) ──────────────────
    ax_map.set_facecolor(BG)
    ax_map.set_xlim(-1, 3)
    ax_map.set_ylim(-1, n + 1)
    ax_map.axis("off")
    ax_map.set_title("Revenue by Region", color=TEXT, fontsize=13, fontweight="bold", pad=12)

    max_rev = max(revenues) if revenues else 1
    cmap = plt.get_cmap("cool")

    for i, (reg, rev) in enumerate(zip(regions, revenues)):
        size = 1800 * (rev / max_rev) + 300
        color = cmap(rev / max_rev)
        y_pos = n - i - 0.5

        ax_map.scatter(1, y_pos, s=size, color=color, alpha=0.85, zorder=3)
        ax_map.text(
            1, y_pos, f"${rev:,.0f}",
            ha="center", va="center",
            fontsize=8, color="white", fontweight="bold", zorder=4,
        )
        ax_map.text(
            1.85, y_pos, reg,
            ha="left", va="center",
            fontsize=10, color=TEXT,
        )

    # ── RIGHT: Grouped bar chart ───────────────────────────────────────────────
    ax_bar.set_facecolor(BG)
    ax_bar.set_axisbelow(True)
    ax_bar.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax_bar.set_facecolor(BG)

    b1 = ax_bar.bar(x - bar_w, revenues, bar_w, label="Total Revenue", color=REV_COLOR, alpha=0.9)
    b2 = ax_bar.bar(x,         cogs,     bar_w, label="COGS",          color=COGS_COLOR, alpha=0.9)
    b3 = ax_bar.bar(x + bar_w, sga,      bar_w, label="SG&A",          color=SGA_COLOR,  alpha=0.9)

    # Value labels on bars
    for bars in (b1, b2, b3):
        for bar in bars:
            h = bar.get_height()
            ax_bar.text(
                bar.get_x() + bar.get_width() / 2,
                h * 1.01,
                f"${h/1000:.1f}k" if h >= 1000 else f"${h:.0f}",
                ha="center", va="bottom",
                fontsize=7.5, color=TEXT,
            )

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(regions, rotation=30, ha="right", color=TEXT, fontsize=10)
    ax_bar.tick_params(axis="y", colors=TEXT)
    ax_bar.spines[:].set_visible(False)
    ax_bar.set_title("Revenue vs COGS vs SG&A by Region", color=TEXT, fontsize=13, fontweight="bold", pad=12)

    legend = ax_bar.legend(
        handles=[
            mpatches.Patch(color=REV_COLOR, label="Total Revenue"),
            mpatches.Patch(color=COGS_COLOR, label="COGS"),
            mpatches.Patch(color=SGA_COLOR,  label="SG&A"),
        ],
        facecolor="#1a1a2e", labelcolor=TEXT, framealpha=0.8, fontsize=9,
    )

    fig.suptitle(
        f"Sales Forecast Map — {model_name}",
        color=TEXT, fontsize=15, fontweight="bold", y=1.01,
    )
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=120, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


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
