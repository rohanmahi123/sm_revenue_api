# SM Revenue Forecasting API

A **FastAPI** ML service that trains revenue/COGS/SG&A models and serves predictions,
integrated with a Supabase processing pipeline (Part 1).

---

## Project Layout

```
sm_revenue_api/
├── main.py                        ← FastAPI app entry point
├── config.py                      ← Pydantic-settings (.env reader)
├── schemas.py                     ← All Pydantic request/response models
├── auth.py                        ← JWT auth (shared SECRET_KEY with Part 1)
├── security.py                    ← bcrypt password hashing
├── models.py                      ← User + Company ORM (mirrors Part 1 Supabase tables)
├── database.py                    ← Main DB engine (Supabase — auth/batches/files)
│
├── db/
│   ├── models.py                  ← ML ORM tables (Dataset, TrainedModel, FileUpload, IngestionBatch)
│   ├── session.py                 ← ML DB engine (SQLite — trained models)
│   └── main_session.py            ← Main DB engine (Supabase — batches/file uploads)
│
├── ml/
│   ├── preprocessor.py            ← CSV loader, date helpers, feature engineering
│   ├── trainer.py                 ← Baseline + RidgeCV training pipeline
│   ├── predictor.py               ← Load .pkl and run inference
│   └── heatmap.py                 ← Correlation heatmap generator + factor file parsers
│
├── routers/
│   ├── training.py                ← POST /train
│   ├── prediction.py              ← POST /predict/*
│   ├── forecast_map.py            ← POST /forecast-map/*
│   ├── heatmap_router.py          ← POST /heatmap/*
│   ├── datasets.py                ← GET /datasets
│   └── models_router.py           ← GET/DELETE /models
│
├── model_store/                   ← Trained .pkl files (persisted via Docker volume)
│   └── alice_check.pkl            ← Example trained model
│
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .env                           ← Your secrets (never commit this)
└── .env.example                   ← Template
```

---

## Environment Variables (`.env`)

```env
# Main DB (Supabase) — auth, users, file_uploads, ingestion_batches
DATABASE_URL=postgresql://postgres.xxx:password@aws-xxx.pooler.supabase.com:5432/postgres

# ML DB (local SQLite) — trained_models, datasets
ML_DATABASE_URL=sqlite:///./sm_revenue.db

# Supabase project URL (used to build Storage file URLs)
SUPABASE_URL=https://xxxx.supabase.co

# JWT secret — MUST match Part 1 exactly so tokens are shared
SECRET_KEY=your-shared-secret-key-from-part1

# ML model .pkl storage directory
MODEL_STORE_DIR=model_store

DEBUG=false
DB_ECHO=false
```

---

## Docker

```bash
# Build and run
docker-compose up --build

# Run in background
docker-compose up --build -d

# Stop
docker-compose down
```

---

## API Reference

> **Auth:** All endpoints require `Authorization: Bearer <token>` header.
> Get the token from Part 1's `POST /login` endpoint.

---

### Training

#### `POST /train` *(multipart/form-data)*

| Field | Type | Required | Description |
|---|---|---|---|
| `file` | file | ✅ | Financial CSV or .xlsx (SL data with external factors) |
| `model_name` | string | ✅ | Unique name for this model |
| `description` | string | | Optional note |
| `sheet_name` | string | | Excel sheet name (default `10 SL`) |
| `test_size` | float | | Train/test split 0.05–0.5 (default `0.25`) |
| `random_state` | int | | Seed (default `42`) |
| `external_factors_json` | string | | JSON string of macro factor time-series |

**`external_factors_json` format (optional if CSV already has factor columns):**
```json
{
  "rows": [
    {"date": "2021-01-01", "CCI": 107.0, "CPI": 1.1, "Oil": 52.16, "GDP": 533.9, "Unemployment": 6.3, "ROI": 0.1},
    {"date": "2021-02-01", "CCI": 109.0, "CPI": 1.2, "Oil": 58.20, "GDP": 536.0, "Unemployment": 6.1, "ROI": 0.1}
  ]
}
```

**Response:**
```json
{
  "model_id": 1,
  "dataset_id": 1,
  "model_name": "aus_revenue_v1",
  "is_new_dataset": true,
  "training_duration_seconds": 3.24,
  "metrics": {
    "Total Revenue": {
      "baseline": {"MAE": 12340, "RMSE": 23450, "R2": 0.91, "MAPE": 4.2},
      "ridge":    {"MAE": 9800,  "RMSE": 18000, "R2": 0.94, "MAPE": 3.1}
    },
    "COGS": {"baseline": {...}, "ridge": {...}},
    "SG&A": {"baseline": {...}, "ridge": {...}},
    "best_model_per_target": {"Total Revenue": "ridge", "COGS": "ridge", "SG&A": "baseline"}
  },
  "best_model_per_target": {"Total Revenue": "ridge", "COGS": "ridge", "SG&A": "baseline"},
  "message": "Model trained successfully. New dataset registered."
}
```

---

### Prediction

#### `POST /predict/{model_id}` — Date-range monthly forecast

**Path:** `model_id` (integer)

**Body:**
```json
{
  "start_date": "2026-01-01",
  "end_date": "2026-12-31",
  "Region": "Australia & Oceania",
  "Geo": "APAC",
  "Country": "Australia",
  "Item_type": "Office Supplies",
  "Customer": "ABC Infra",
  "Raw_Material": 150000,
  "Direct_Labor": 45000,
  "Freight": 6000,
  "Storage": 20000,
  "Packaging": 8000,
  "Indirect_Labor": 21000,
  "Rent_Utility": 19000,
  "Overhead": 11000,
  "CCI": 107.0,
  "CPI": 1.1,
  "Oil": 52.16,
  "GDP": 533.9,
  "Unemployment": 6.3,
  "ROI": 0.1
}
```

> Only `start_date` and `end_date` are required. All other fields are optional.

**Response:**
```json
{
  "model_id": 1,
  "model_name": "aus_revenue_v1",
  "start_date": "2026-01-01",
  "end_date": "2026-12-31",
  "monthly_predictions": [
    {
      "month": "2026-01",
      "date": "2026-01-01",
      "predicted_total_revenue": 342000.50,
      "predicted_COGS": 261000.20,
      "predicted_SGA": 34500.10,
      "model_used_revenue": "ridge",
      "model_used_COGS": "ridge",
      "model_used_SGA": "baseline"
    }
  ],
  "summary": {
    "total_revenue": 4104006.00,
    "total_COGS": 3132002.40,
    "total_SGA": 414001.20,
    "months_count": 12
  }
}
```

---

#### `POST /predict/from-batch/{model_id}` — Predict from uploaded SUBLEDGER

Reads the SUBLEDGER CSV already uploaded for the given `batch_id`.
If the CSV contains external factor columns (CCI, CPI, Oil, GDP, Unemployment, ROI)
they are used automatically — no manual input needed.

**Path:** `model_id`

**Body:**
```json
{
  "batch_id": 1
}
```

> All external factor fields are optional overrides. If provided, they replace the CSV values.

```json
{
  "batch_id": 1,
  "CCI": 110.0,
  "CPI": 2.5
}
```

**Response:**
```json
{
  "model_id": 1,
  "model_name": "aus_revenue_v1",
  "batch_id": 1,
  "sl_file_path": "1/1/SUBLEDGER/sl_with_external_factors.csv",
  "predictions": [
    {
      "order_date": "2021-01-09",
      "predicted_total_revenue": 328450.20,
      "predicted_COGS": 247100.50,
      "predicted_SGA": 31200.80,
      "model_used_revenue": "ridge",
      "model_used_COGS": "ridge",
      "model_used_SGA": "baseline"
    }
  ],
  "summary": {
    "total_revenue": 25400000.00,
    "total_COGS": 19800000.00,
    "total_SGA": 2900000.00,
    "row_count": 80
  },
  "external_factors_info": "All external factors used from CSV: ['CCI', 'CPI', 'Oil', 'GDP', 'Unemployment', 'ROI']."
}
```

---

### Forecast Map

#### `POST /forecast-map/{model_id}` — Predict + regional map (manual rows)

Send raw input rows → returns predictions + region summary + chart.

**Body:**
```json
{
  "rows": [
    {
      "order_date": "2026-01-15",
      "Region": "Australia & Oceania",
      "Geo": "APAC",
      "Country": "Australia",
      "Item type": "Office Supplies",
      "Customer": "ABC Infra",
      "CCI": 107.0,
      "CPI": 1.1,
      "Oil": 52.16,
      "GDP": 533.9,
      "Unemployment": 6.3,
      "ROI": 0.1
    }
  ]
}
```

**Response:**
```json
{
  "model_id": 1,
  "model_name": "aus_revenue_v1",
  "predictions": [...],
  "region_summary": [
    {
      "region": "Australia & Oceania",
      "total_revenue": 328450.20,
      "total_COGS": 247100.50,
      "total_SGA": 31200.80,
      "row_count": 1
    }
  ],
  "forecast_map_base64": "<base64 PNG string>",
  "map_note": "1 region(s) found: Australia & Oceania."
}
```

---

#### `POST /forecast-map/from-batch/{model_id}?batch_id={batch_id}` — Predict from batch + map

**One call** — reads SUBLEDGER CSV, runs prediction on every row, returns day-by-day detail
+ region summary + chart. External factors auto-detected from CSV.

**Path:** `model_id`
**Query param:** `batch_id` (integer)

**Response:**
```json
{
  "model_id": 1,
  "model_name": "aus_revenue_v1",
  "batch_id": 1,
  "row_count": 80,
  "daily_predictions": [
    {
      "order_date": "2021-01-09",
      "region": "Australia & Oceania",
      "geo": "APAC",
      "country": "Australia",
      "item_type": "Office Supplies",
      "customer": "ABC Infra",
      "predicted_total_revenue": 328450.20,
      "predicted_COGS": 247100.50,
      "predicted_SGA": 31200.80,
      "model_used_revenue": "ridge",
      "model_used_COGS": "ridge",
      "model_used_SGA": "baseline"
    }
  ],
  "region_summary": [
    {
      "region": "Australia & Oceania",
      "total_revenue": 25400000.00,
      "total_COGS": 19800000.00,
      "total_SGA": 2900000.00,
      "row_count": 80
    }
  ],
  "forecast_map_base64": "<base64 PNG string>",
  "map_note": "1 region(s): Australia & Oceania.",
  "external_factors_info": "All external factors from CSV: ['CCI', 'CPI', 'Oil', 'GDP', 'Unemployment', 'ROI']."
}
```

---

### Heatmap

#### `POST /heatmap/upload` — Upload external factor files

Upload 1–6 xlsx/csv files (one per factor). Factor is auto-detected from column headers.
File name does not matter — only column headers are used.

**Body:** `multipart/form-data`
| Field | Type | Description |
|---|---|---|
| `files` | file(s) | Up to 6 xlsx/csv files |

**Response:**
```json
{
  "factors_loaded": ["CCI", "CPI", "Oil", "GDP", "Unemployment", "ROI"],
  "row_count": 75,
  "date_range_start": "2020-01-01",
  "date_range_end": "2026-03-01",
  "heatmap_base64": "<base64 PNG string>",
  "message": "Loaded 6 factor(s): CCI, CPI, Oil, GDP, Unemployment, ROI."
}
```

---

#### `POST /heatmap/from-csv` — Upload full CSV with factor columns

Upload any CSV that has external factor columns (CCI, CPI, Oil, etc.) in it.

**Body:** `multipart/form-data`
| Field | Type | Description |
|---|---|---|
| `file` | file | CSV/xlsx containing external factor columns |

**Response:** Same as `/heatmap/upload`

---

#### `POST /heatmap/from-subledger/{batch_id}` — Extract factors from batch SUBLEDGER

Reads the SUBLEDGER file already uploaded in the batch, extracts external factor columns,
merges with any stored data, and returns updated heatmap.

**Path:** `batch_id`

**Response:**
```json
{
  "factors_found": ["CCI", "CPI", "Oil", "GDP", "Unemployment", "ROI"],
  "row_count": 80,
  "date_range_start": "2021-01-01",
  "date_range_end": "2025-12-01",
  "heatmap_base64": "<base64 PNG string>",
  "message": "Extracted 6 factor(s) from SUBLEDGER (batch 1): CCI, CPI, Oil, GDP, Unemployment, ROI."
}
```

---

#### `POST /heatmap/refresh` — Append new values + update heatmap live

Send new values for all 6 external factors → appended to stored data → updated heatmap.

**Body:**
```json
{
  "CCI": 110.5,
  "CPI": 3.2,
  "Oil": 91.0,
  "GDP": 545.0,
  "Unemployment": 3.8,
  "ROI": 0.35
}
```

> All fields optional. At least one must be provided.

**Response:**
```json
{
  "row_count": 76,
  "new_row": {
    "date": "2026-06-01",
    "CCI": 110.5,
    "CPI": 3.2,
    "Oil": 91.0,
    "GDP": 545.0,
    "Unemployment": 3.8,
    "ROI": 0.35
  },
  "heatmap_base64": "<base64 PNG string>",
  "message": "Heatmap updated with new values."
}
```

---

#### `GET /heatmap/data` — Get stored factor data as JSON

Returns all stored external factor rows for the current company.

**Response:**
```json
{
  "row_count": 75,
  "columns": ["date", "CCI", "CPI", "Oil", "GDP", "Unemployment", "ROI"],
  "rows": [
    {"date": "2020-01-01", "CCI": 93.4, "CPI": 2.2, "Oil": 51.58, "GDP": 506.5, "Unemployment": 5.2, "ROI": 0.75}
  ]
}
```

---

### Datasets & Models

#### `GET /datasets`
List all uploaded CSVs for the current user.

#### `GET /datasets/{id}/models`
List all models trained on a specific dataset.

#### `GET /models`
List all trained models. Optional `?dataset_id=1` filter.

#### `GET /models/{model_id}`
Full model detail — metrics, feature list, training config.

#### `DELETE /models/{model_id}`
Remove model from DB and delete `.pkl` from disk.
