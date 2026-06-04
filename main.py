"""
SM Revenue Forecasting API
==========================
FastAPI application with SQLAlchemy ORM.

Startup
-------
    uvicorn main:app --reload

Swagger UI → http://localhost:8000/docs
ReDoc      → http://localhost:8000/redoc
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from fastapi.security import HTTPBearer

from config import settings
from db.session import create_tables
from routers import datasets, models_router, prediction, training


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Create DB tables on startup (safe to call on every start — idempotent)."""
    create_tables()
    yield


app = FastAPI(
    title=settings.APP_TITLE,
    version=settings.APP_VERSION,
    description="""
## SM Revenue Forecasting API

Train and serve machine-learning models that predict **Total Revenue**, **COGS**,
and **SG&A** from financial CSV uploads enriched with external macro factors.

### Workflow

1. **POST /train** – upload a CSV + macro factors JSON → a model is trained,
   serialised as a `.pkl`, and its metadata (metrics, feature list, …) is
   stored in the database alongside the CSV hash.
2. **GET /datasets** – list all unique CSVs a user has uploaded.
3. **GET /datasets/{id}/models** – list every model trained on that CSV.
   Use this to *reuse* a previously trained model instead of retraining.
4. **GET /models** – list / filter all trained models.
5. **POST /predict/{model_id}** – send daily input rows, get back predictions.
6. **DELETE /models/{model_id}** – remove a model from the DB and disk.

### Key design decisions

* **CSV deduplication** – files are SHA-256 hashed. The same CSV from the same
  user always maps to one `Dataset` record, so metadata (column list, date range,
  row count) is stored once and reused.
* **Model versioning** – multiple models can be trained on the same dataset
  (different external factors, test splits, etc.) and are all queryable together.
* **ORM layer** – SQLAlchemy models in `db/models.py`; swap the DB URL in
  `.env` to move from SQLite → PostgreSQL / MySQL without code changes.
""",
    lifespan=lifespan,
    debug=settings.DEBUG,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(training.router)
app.include_router(prediction.router)
app.include_router(datasets.router)
app.include_router(models_router.router)

from routers.forecast_map import router as forecast_map_router
app.include_router(forecast_map_router)

from routers.heatmap_router import router as heatmap_router
app.include_router(heatmap_router)

def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )
    schema["components"]["securitySchemes"] = {
        "BearerAuth": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT",
        }
    }
    for path in schema["paths"].values():
        for method in path.values():
            method["security"] = [{"BearerAuth": []}]
            # Remove the raw "authorization" header field shown per-endpoint
            if "parameters" in method:
                method["parameters"] = [
                    p for p in method["parameters"]
                    if not (p.get("in") == "header" and
                            p.get("name", "").lower() == "authorization")
                ]
    app.openapi_schema = schema
    return schema

app.openapi = custom_openapi


@app.get("/", tags=["Health"])
def root():
    return {
        "status": "ok",
        "app": settings.APP_TITLE,
        "version": settings.APP_VERSION,
        "docs": "/docs",
    }


@app.get("/health", tags=["Health"])
def health():
    return {"status": "healthy"}
