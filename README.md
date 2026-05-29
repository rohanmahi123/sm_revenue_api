# SM Revenue Forecasting API

A **FastAPI** service that wraps the `SM_Rev_SGA` notebook model in a production-ready REST API with:

- **SQLAlchemy ORM** (SQLite by default, swap to PostgreSQL/MySQL via `.env`)
- **CSV deduplication** via SHA-256 hash — same file → same `Dataset` record
- **Model versioning** — many models per dataset, all queryable
- **Persistent `.pkl` storage** — models survive server restarts

---

## Project layout

```
sm_revenue_api/
├── main.py                  ← FastAPI app + lifespan (table creation)
├── config.py                ← Pydantic-settings (reads .env)
├── schemas.py               ← Request / response Pydantic models
│
├── db/
│   ├── models.py            ← SQLAlchemy ORM tables (Dataset, TrainedModel)
│   └── session.py           ← Engine, SessionLocal, get_db dependency
│
├── ml/
│   ├── preprocessor.py      ← date helpers, CSV loader, feature engineering
│   ├── trainer.py           ← baseline + RidgeCV training pipeline
│   └── predictor.py         ← load .pkl and run inference
│
├── routers/
│   ├── training.py          ← POST /train
│   ├── prediction.py        ← POST /predict/{model_id}
│   ├── datasets.py          ← GET  /datasets, /datasets/{id}/models
│   └── models_router.py     ← GET/DELETE /models
│
├── requirements.txt
└── .env.example
```

---

## Quick start

```bash
# 1. Clone / copy the project
cd sm_revenue_api

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure (optional — defaults work out of the box with SQLite)
cp .env.example .env
# edit DATABASE_URL, MODEL_STORE_DIR as needed

# 4. Run
uvicorn main:app --reload

# 5. Open Swagger UI
open http://localhost:8000/docs
```

---

## API reference

### `POST /train`  *(multipart/form-data)*

| Field | Type | Required | Description |
|---|---|---|---|
| `file` | file | ✅ | Financial CSV (or .xlsx) |
| `user_id` | string | ✅ | Your user / tenant identifier |
| `model_name` | string | ✅ | Unique name for this model |
| `description` | string | | Human-readable note |
| `sheet_name` | string | | Excel sheet name (default `10 SL`) |
| `test_size` | float | | Train/test split (default `0.25`) |
| `random_state` | int | | Seed (default `42`) |
| `external_factors_json` | string | | JSON string — see below |

**`external_factors_json` format:**
```json
{
  "rows": [
    {"date": "2023-01-01", "CCI": 101.2, "CPI": 2.8, "Oil": 73.2,
     "GDP": 640.0, "Unemployment": 4.2, "ROI": 3.85},
    {"date": "2023-02-01", "CCI": 99.8, "CPI": 2.9, "Oil": 72.0,
     "GDP": 641.5, "Unemployment": 4.1, "ROI": 3.90}
  ]
}
```

**Response:**
```json
{
  "model_id": 1,
  "dataset_id": 1,
  "model_name": "q1_2024_ridge",
  "is_new_dataset": true,
  "training_duration_seconds": 2.341,
  "metrics": {
    "Total Revenue": {
      "baseline": {"MAE": 1234, "RMSE": 2345, "R2": 0.91, "MAPE": 4.2},
      "ridge":    {"MAE": 980,  "RMSE": 1800, "R2": 0.94, "MAPE": 3.1}
    },
    "COGS": { ... },
    "SG&A": { ... },
    "best_model_per_target": {"Total Revenue": "ridge", "COGS": "ridge", "SG&A": "baseline"}
  },
  "best_model_per_target": {"Total Revenue": "ridge", ...},
  "message": "Model trained successfully. New dataset registered."
}
```

> **Tip** – if `is_new_dataset` is `false`, the CSV hash matched a previous upload.
> Use `GET /datasets/{dataset_id}/models` to see all existing models for that CSV.

---

### `GET /datasets?user_id=alice`
List all unique CSVs uploaded by a user with metadata (columns, row count, date range, model count).

### `GET /datasets/{id}/models`
List all trained models for a given CSV — use to decide whether to reuse an existing model or retrain.

### `GET /models?user_id=alice`
List all trained models. Optional `dataset_id` filter.

### `GET /models/{model_id}`
Full model detail including metrics and feature list.

### `DELETE /models/{model_id}`
Remove the model from the database **and** delete the `.pkl` file from disk.

---

### `POST /predict/{model_id}`

```json
{
  "rows": [
    {
      "order_date": "2026-01-15",
      "Region": "Asia",
      "Geo": "APAC",
      "Country": "Australia",
      "Item type": "Office Supplies",
      "Customer": "Customer A",
      "Raw Material": 50000,
      "Direct Labor": 12000,
      "Freight": 4000,
      "Storage": 2500,
      "Packaging": 1500,
      "Indirect Labor": 8000,
      "Rent & Utility": 5000,
      "Overhead": 7000,
      "CCI": 101.2,
      "CPI": 2.8,
      "Oil": 73.2,
      "GDP": 640.0,
      "Unemployment": 4.2,
      "ROI": 3.85
    }
  ]
}
```

**Response:**
```json
{
  "model_id": 1,
  "model_name": "q1_2024_ridge",
  "predictions": [
    {
      "order_date": "2026-01-15",
      "predicted_total_revenue": 182450.23,
      "predicted_COGS": 112300.45,
      "predicted_SGA": 24600.11,
      "model_used_revenue": "ridge",
      "model_used_COGS": "ridge",
      "model_used_SGA": "baseline"
    }
  ]
}
```

---

## Database schema

### `datasets`
| Column | Type | Notes |
|---|---|---|
| `id` | INTEGER PK | |
| `user_id` | VARCHAR | tenant key |
| `original_filename` | VARCHAR | |
| `file_hash` | VARCHAR(64) | SHA-256, unique per user |
| `file_size_bytes` | INTEGER | |
| `row_count` | INTEGER | |
| `column_names` | JSON | list of columns |
| `date_range_start` | VARCHAR | |
| `date_range_end` | VARCHAR | |
| `target_columns` | JSON | `["Total Revenue","COGS","SG&A"]` |
| `uploaded_at` | DATETIME | |
| `notes` | TEXT | |

### `trained_models`
| Column | Type | Notes |
|---|---|---|
| `id` | INTEGER PK | |
| `dataset_id` | FK → datasets | |
| `user_id` | VARCHAR | |
| `model_name` | VARCHAR(256) | unique |
| `description` | TEXT | |
| `model_file_path` | VARCHAR | path to `.pkl` |
| `targets` | JSON | |
| `feature_columns` | JSON | |
| `external_factors_used` | JSON | `{"CCI": true, ...}` |
| `test_size` / `random_state` | FLOAT / INT | |
| `metrics` | JSON | nested per-target metrics |
| `trained_at` | DATETIME | |
| `training_duration_seconds` | FLOAT | |

---

## Switching to PostgreSQL

Edit `.env`:
```
DATABASE_URL=postgresql+psycopg2://user:password@localhost:5432/sm_revenue
```

Install the driver:
```bash
pip install psycopg2-binary
```

No code changes needed — SQLAlchemy handles the rest.
