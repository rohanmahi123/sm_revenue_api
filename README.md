# SM Revenue Forecasting API

This is the machine learning backend for the SM Revenue platform. It handles model training, revenue/COGS/SG&A predictions, and external factor correlation data — all secured with JWT authentication shared with the main platform.

Built with FastAPI, SQLAlchemy, and scikit-learn. Models are stored as `.pkl` files, metadata lives in a local SQLite database, and batch/user data is read from Supabase PostgreSQL.

---

## Authentication

Every endpoint requires a Bearer JWT token in the request header. The token is issued by the main platform (Part 1) and must use the same `SECRET_KEY`.

```
Authorization: <your_jwt_token>
```

You can use the **Authorize** button in Swagger UI (`/docs`) to set the token once for all endpoints.

---

## Base URL

```
http://localhost:8000          (local Docker run)
http://<your-azure-url>        (production)
```

Swagger UI → `/docs`

---

## Health

### `GET /`

Quick check that the server is alive.

**Response**
```json
{
  "status": "ok",
  "app": "SM Revenue Forecasting API",
  "version": "1.0.0",
  "docs": "/docs"
}
```

---

### `GET /health`

Lightweight health ping — used by Azure and load balancers.

**Response**
```json
{ "status": "healthy" }
```

---

## Training

### `POST /train`

Upload a financial CSV and train a new ML model. The model learns to predict **Total Revenue**, **COGS**, and **SG&A** from the data. The API SHA-256 hashes the uploaded file — if you upload the exact same CSV again, it reuses the existing dataset record so you don't end up with duplicates.

**Content-Type:** `multipart/form-data`

| Field | Type | Required | Description |
|---|---|---|---|
| `file` | File | Yes | Financial data CSV |
| `model_name` | string | Yes | A unique name for this model |
| `description` | string | No | Optional notes about this training run |
| `sheet_name` | string | No | Excel sheet name (default: `10 SL`, ignored for CSV) |
| `test_size` | float | No | Train/test split ratio (default: `0.25`) |
| `random_state` | int | No | Random seed for reproducibility (default: `42`) |
| `external_factors_json` | string | No | Historical macro factor data as a JSON string |

**`external_factors_json` format**
```json
{
  "rows": [
    { "date": "2023-01-01", "CCI": 101.0, "CPI": 2.5, "Oil": 78.3, "GDP": 580.0, "Unemployment": 3.8, "ROI": 4.1 },
    { "date": "2023-02-01", "CCI": 103.0, "CPI": 2.6, "Oil": 80.1, "GDP": 582.0, "Unemployment": 3.7, "ROI": 4.2 }
  ]
}
```

> If your CSV already has CCI, CPI, Oil, GDP, Unemployment, ROI columns in it, you can skip `external_factors_json` entirely — the trainer picks them up automatically.

**Response**
```json
{
  "model_id": 1,
  "dataset_id": 1,
  "model_name": "revenue_model_v1",
  "is_new_dataset": true,
  "training_duration_seconds": 12.4,
  "metrics": {
    "Total Revenue": { "r2": 0.94, "mae": 1203.4, "rmse": 1890.2 },
    "COGS":          { "r2": 0.91, "mae": 980.1,  "rmse": 1420.5 },
    "SGA":           { "r2": 0.88, "mae": 540.3,  "rmse": 820.7 }
  },
  "best_model_per_target": {
    "Total Revenue": "RidgeCV",
    "COGS": "LinearRegression",
    "SGA": "RidgeCV"
  },
  "message": "Model trained successfully. New dataset registered."
}
```

---

## Prediction

### `POST /predict/{model_id}`

Date-range forecast. Give a start and end date plus a single snapshot of external factor values — the API creates one input row per month across that range, applies the same factor values to each month, runs the model, and returns the full monthly breakdown.

**Path param:** `model_id`

**Request body**
```json
{
  "start_date": "2025-01-01",
  "end_date": "2025-12-01",
  "Region": "Australia & Oceania",
  "Geo": "APAC",
  "Country": "Australia",
  "Item_type": "Beverages",
  "Customer": "MNO Coffee",
  "Raw_Material": 280000,
  "Direct_Labor": 55000,
  "Freight": 2200,
  "Storage": 18000,
  "Packaging": 2300,
  "Indirect_Labor": 19000,
  "Rent_Utility": 12000,
  "Overhead": 4500,
  "CCI": 95.0,
  "CPI": 4.0,
  "Oil": 70.0,
  "GDP": 720.0,
  "Unemployment": 4.1,
  "ROI": 3.8
}
```

> Only `start_date` and `end_date` are truly required. Everything else is optional — missing cost fields default to NaN and the model fills them from training medians.

**Response**
```json
{
  "model_id": 1,
  "model_name": "revenue_model_v1",
  "start_date": "2025-01-01",
  "end_date": "2025-12-01",
  "monthly_predictions": [
    {
      "month": "2025-01",
      "date": "2025-01-01",
      "predicted_total_revenue": 510000.25,
      "predicted_COGS": 410000.10,
      "predicted_SGA": 52000.80,
      "model_used_revenue": "RidgeCV",
      "model_used_COGS": "LinearRegression",
      "model_used_SGA": "RidgeCV"
    }
  ],
  "summary": {
    "total_revenue": 6120003.0,
    "total_COGS": 4920001.2,
    "total_SGA": 624009.6,
    "months_count": 12
  }
}
```

---

### `POST /predict/from-batch/{model_id}` ⭐ Main production endpoint

This is the most powerful prediction endpoint in the whole API. Give it a model ID and a batch ID, and it does everything automatically:

1. Pulls the SUBLEDGER CSV from Supabase Storage for that batch
2. Merges macro-economic external factors (CCI, CPI, Oil, GDP, Unemployment, ROI) from the backend-stored `external_factors.csv` — matched by year-month, no manual input needed
3. Runs revenue, COGS, and SG&A predictions on every historical row from the CSV
4. If you pass `prediction_end` beyond the CSV's last date, it **generates future monthly rows** for every unique Region/Geo/Country/Customer/Item_type combination found in the data, using median historical cost features
5. Returns actual values alongside predicted values for every historical row, so you can see exactly how well the model tracks reality
6. Returns a summary with total actual revenue, total predicted revenue, total actual gross profit, and total predicted gross profit

#### Path parameter

| Param | Type | Description |
|-------|------|-------------|
| `model_id` | integer | ID of the trained model to use for prediction |

#### Query parameters (all optional)

| Param | Type | Description |
|-------|------|-------------|
| `country` | string | Filter CSV rows to this country only (case-insensitive, partial match) |
| `region` | string | Filter CSV rows to this region only (case-insensitive, partial match) |
| `geo` | string | Filter CSV rows to this geo only (case-insensitive, partial match) |
| `prediction_start` | string (YYYY-MM-DD) | Start of the **future forecast window** — does NOT filter historical rows |
| `prediction_end` | string (YYYY-MM-DD) | End of the future forecast window — if beyond the CSV's last date, future rows are generated |

> **Important:** `prediction_start` and `prediction_end` define only the future forecast window. Historical CSV rows are always included in full — these params never remove historical data from the response.

#### Request body

```json
{
  "batch_id": 5
}
```

That's it. The batch ID tells the API which SUBLEDGER file to load from Supabase Storage. External factors are merged automatically from the backend-stored file — no need to send them manually.

#### Full example request

```
POST /predict/from-batch/1?country=Australia&prediction_start=2026-01-01&prediction_end=2027-06-01
Content-Type: application/json
Authorization: Bearer <your_token>

{
  "batch_id": 5
}
```

#### Response

```json
{
  "model_id": 1,
  "model_name": "revenue_model_v1",
  "batch_id": 5,
  "sl_file_path": "1/5/SUBLEDGER/file.csv",
  "filters_applied": {
    "country": "Australia",
    "region": null,
    "geo": null,
    "prediction_start": "2026-01-01",
    "prediction_end": "2027-06-01"
  },
  "predictions": [
    {
      "order_date": "2025-01-09",
      "row_type": "historical",
      "region": "Australia & Oceania",
      "geo": "APAC",
      "country": "Australia",
      "item_type": "Office Supplies",
      "customer": "ABC Infra",
      "actual_total_revenue": 480000.0,
      "actual_COGS": 360000.0,
      "actual_gross_profit": 120000.0,
      "predicted_total_revenue": 451864.33,
      "predicted_COGS": 341323.88,
      "predicted_SGA": 41691.93,
      "predicted_gross_profit": 110540.45,
      "model_used_revenue": "RidgeCV",
      "model_used_COGS": "LinearRegression",
      "model_used_SGA": "RidgeCV"
    },
    {
      "order_date": "2026-01-01",
      "row_type": "future",
      "region": "Australia & Oceania",
      "geo": "APAC",
      "country": "Australia",
      "item_type": "Office Supplies",
      "customer": "ABC Infra",
      "actual_total_revenue": null,
      "actual_COGS": null,
      "actual_gross_profit": null,
      "predicted_total_revenue": 468200.0,
      "predicted_COGS": 352100.0,
      "predicted_SGA": 43800.0,
      "predicted_gross_profit": 116100.0,
      "model_used_revenue": "RidgeCV",
      "model_used_COGS": "LinearRegression",
      "model_used_SGA": "RidgeCV"
    }
  ],
  "summary": {
    "historical_row_count": 80,
    "future_row_count": 36,
    "total_row_count": 116,
    "total_actual_revenue": 12450000.0,
    "total_actual_gross_profit": 3112500.0,
    "total_predicted_revenue": 14820000.0,
    "total_predicted_gross_profit": 3705000.0
  },
  "external_factors_info": "External factors merged from stored file by month: ['CCI', 'CPI', 'Oil', 'GDP', 'Unemployment', 'ROI']."
}
```

#### Response field reference

| Field | What it means |
|-------|---------------|
| `row_type` | `"historical"` = row came from the CSV; `"future"` = synthetically generated beyond the CSV date range |
| `actual_total_revenue` | Real revenue value from the CSV. Always `null` for future rows |
| `actual_COGS` | Real COGS value from the CSV. Always `null` for future rows |
| `actual_gross_profit` | `actual_total_revenue - actual_COGS`. Always `null` for future rows |
| `predicted_total_revenue` | What the model says revenue should be for that row |
| `predicted_COGS` | What the model says COGS should be |
| `predicted_SGA` | What the model says SG&A should be |
| `predicted_gross_profit` | `predicted_total_revenue - predicted_COGS` — calculated on the backend |
| `model_used_*` | Which algorithm won during training for that target (e.g. `"RidgeCV"`, `"LinearRegression"`) |
| `summary.total_actual_revenue` | Sum of actual revenue across all historical rows (future rows contribute 0) |
| `summary.total_predicted_revenue` | Sum of predicted revenue across ALL rows — historical + future combined |
| `summary.total_actual_gross_profit` | Sum of actual gross profit across historical rows |
| `summary.total_predicted_gross_profit` | Sum of predicted gross profit across all rows |

#### How future row generation works

When `prediction_end` is a date beyond the last row in the CSV:

1. The API finds every unique combination of `(Region, Geo, Country, Customer, Item_type)` in the filtered historical data
2. It generates one row per unique combination per future month — from the month after the last CSV date (or `prediction_start` if that's even further in the future) up to `prediction_end`
3. Cost features (Raw Material, Direct Labor, Freight, Storage, Packaging, Indirect Labor, Rent & Utility, Overhead) are filled in using the **median** historical value for that combination
4. External factors (CCI, CPI, Oil, etc.) are looked up from `external_factors.csv` for that month. If the month isn't in the file, the most recent available values are carried forward
5. The model then predicts revenue, COGS, and SG&A for each generated row exactly the same way it does for historical rows

So if your CSV covers Jan 2021 – Dec 2025, and you ask for `prediction_end=2027-06-01`, you'll get 18 months of future forecasts × however many unique segment combinations exist in your filtered data.

> **Sales forecast = Revenue forecast.** When plotting a sales forecast chart, use `predicted_total_revenue` on the Y axis. These are the same number — just different business vocabulary.

> **Security note:** The `company_id` embedded in your JWT token is matched against the batch record in the database. You cannot access another company's batch data — the API will return a 403 if the IDs don't match.

---

## Forecast Map

### `POST /forecast-map/{model_id}`

Same as `POST /predict/{model_id}` but accepts multiple rows and also returns a regional breakdown grouped by the `Region` field. Useful when you want to compare predictions across different regions in one call.

**Path param:** `model_id`

**Request body**
```json
{
  "rows": [
    {
      "Region": "Australia & Oceania",
      "Country": "Australia",
      "Item_type": "Beverages",
      "Customer": "MNO Coffee",
      "order_date": "2025-01-01",
      "CCI": 92.0,
      "CPI": 4.0,
      "Oil": 72.53,
      "GDP": 699.8,
      "Unemployment": 4.1,
      "ROI": 4.35
    }
  ]
}
```

**Response**
```json
{
  "model_id": 1,
  "model_name": "revenue_model_v1",
  "predictions": [ { "...row-level predictions..." } ],
  "region_summary": [
    {
      "region": "Australia & Oceania",
      "total_revenue": 510000.0,
      "total_COGS": 410000.0,
      "total_SGA": 52000.0,
      "row_count": 1
    }
  ],
  "forecast_map_base64": null,
  "map_note": "1 region(s) found: Australia & Oceania."
}
```

---

### `POST /forecast-map/from-batch/{model_id}` ⭐ Main production endpoint

The most complete prediction endpoint. Reads the SUBLEDGER CSV from a Supabase batch, auto-merges external factors by month, runs predictions on every single row, and returns the full day-by-day breakdown alongside the regional summary. This is what the frontend dashboard should call to populate the forecast table and regional charts.

**Path param:** `model_id`
**Query param:** `batch_id` (integer)

Example: `POST /forecast-map/from-batch/1?batch_id=5`

No request body needed.

**Response**
```json
{
  "model_id": 1,
  "model_name": "revenue_model_v1",
  "batch_id": 5,
  "row_count": 80,
  "daily_predictions": [
    {
      "order_date": "2025-01-09",
      "region": "Australia & Oceania",
      "geo": "APAC",
      "country": "Australia",
      "item_type": "Office Supplies",
      "customer": "ABC Infra",
      "predicted_total_revenue": 451864.33,
      "predicted_COGS": 341323.88,
      "predicted_SGA": 41691.93,
      "model_used_revenue": "RidgeCV",
      "model_used_COGS": "LinearRegression",
      "model_used_SGA": "RidgeCV"
    }
  ],
  "region_summary": [
    {
      "region": "Australia & Oceania",
      "total_revenue": 12450000.0,
      "total_COGS": 9870000.0,
      "total_SGA": 1230000.0,
      "row_count": 80
    }
  ],
  "forecast_map_base64": null,
  "map_note": "1 region(s): Australia & Oceania.",
  "external_factors_info": "External factors merged from stored file by month: ['CCI', 'CPI', 'Oil', 'GDP', 'Unemployment', 'ROI']."
}
```

> **Security note:** Same company_id check as `/predict/from-batch`.

---

## Heatmap / External Factors

### `GET /heatmap/from-storage` ⭐ Main production endpoint

Returns everything the frontend needs to draw a correlation heatmap and time-series charts for the 6 macro-economic external factors (CCI, CPI, Oil, GDP, Unemployment, ROI). The backend reads `external_factors.csv` (baked into the Docker image at build time), computes Pearson correlation between all factor pairs, and returns the matrix plus the raw monthly values as JSON. No image is ever generated server-side — all chart rendering happens on the frontend with Plotly.js.

#### Authentication

Requires a valid JWT token in the `Authorization: Bearer <token>` header. No request body or query parameters needed.

```
GET /heatmap/from-storage
Authorization: Bearer <your_token>
```

#### Response

```json
{
  "factors_found": ["CCI", "CPI", "Oil", "GDP", "Unemployment", "ROI"],
  "row_count": 80,
  "date_range_start": "2021-01-01",
  "date_range_end": "2025-12-01",
  "correlation": {
    "factors": ["CCI", "CPI", "Oil", "GDP", "Unemployment", "ROI"],
    "matrix": [
      [1.0,    0.23,  -0.45,  0.67,  -0.12,  0.34],
      [0.23,   1.0,    0.56,  0.78,  -0.34,  0.45],
      [-0.45,  0.56,   1.0,   0.32,  -0.21,  0.18],
      [0.67,   0.78,   0.32,  1.0,   -0.54,  0.62],
      [-0.12, -0.34,  -0.21, -0.54,   1.0,  -0.29],
      [0.34,   0.45,   0.18,  0.62,  -0.29,  1.0 ]
    ]
  },
  "monthly_data": [
    {
      "date": "2021-01-01",
      "CCI": 107.0,
      "CPI": 1.1,
      "Oil": 52.16,
      "GDP": 533.9,
      "Unemployment": 6.3,
      "ROI": 0.1
    },
    {
      "date": "2021-02-01",
      "CCI": 109.5,
      "CPI": 1.3,
      "Oil": 55.72,
      "GDP": 533.9,
      "Unemployment": 6.0,
      "ROI": 0.1
    }
  ],
  "message": "Data ready. 6 factors, 80 months."
}
```

#### Response field reference

| Field | What it means |
|-------|---------------|
| `factors_found` | List of factor columns that were actually present in the CSV. Could be fewer than 6 if the file is missing some columns |
| `row_count` | Number of monthly rows in the stored file |
| `date_range_start` | Earliest month in the file |
| `date_range_end` | Latest month in the file |
| `correlation.factors` | The factor names in order — this is both the X axis and Y axis label list for the heatmap |
| `correlation.matrix` | 2D array of Pearson correlation values (−1 to +1). `matrix[i][j]` is the correlation between `factors[i]` and `factors[j]`. Diagonal is always 1.0 |
| `monthly_data` | One object per month with the raw factor values. Use this for time-series line charts |
| `message` | Human-readable summary string |

#### How to read the correlation matrix

The matrix is square. `factors[0]` is CCI, `factors[1]` is CPI, etc. So `matrix[0][2]` = correlation between CCI and Oil. A value close to +1 means they move together. Close to −1 means they move in opposite directions. Near 0 means no relationship.

#### Rendering on the frontend (Plotly.js)

**Correlation heatmap:**
```js
const res = await fetch('/heatmap/from-storage', {
  headers: { Authorization: `Bearer ${token}` }
}).then(r => r.json())

Plotly.newPlot('heatmap-div', [{
  type: 'heatmap',
  z: res.correlation.matrix,
  x: res.correlation.factors,
  y: res.correlation.factors,
  colorscale: 'RdBu',
  zmin: -1,
  zmax: 1,
  text: res.correlation.matrix.map(row =>
    row.map(v => v.toFixed(2))
  ),
  texttemplate: '%{text}',
  showscale: true
}], {
  title: 'External Factor Correlation',
  width: 600,
  height: 500
})
```

**Time-series line chart per factor:**
```js
const traces = res.factors_found.map(factor => ({
  type: 'scatter',
  mode: 'lines+markers',
  name: factor,
  x: res.monthly_data.map(row => row.date),
  y: res.monthly_data.map(row => row[factor])
}))

Plotly.newPlot('timeseries-div', traces, {
  title: 'External Factors Over Time',
  xaxis: { title: 'Month' },
  yaxis: { title: 'Value' }
})
```

#### Live refresh (no extra API call needed)

If the user enters a new monthly data point on the frontend, you do not need to call any API to refresh the heatmap. Just:

1. Append the new row to `monthly_data` in your local state
2. Recompute the Pearson correlation matrix in JavaScript (or use a library like `simple-statistics`)
3. Call `Plotly.react('heatmap-div', ...)` with the updated matrix

The backend `external_factors.csv` is updated only at Docker image build time. For real-time data entry and immediate visual feedback, compute locally on the frontend.

#### Error responses

| Status | When it happens |
|--------|-----------------|
| `404` | `external_factors.csv` not found in `model_store/` — the Docker image may be missing the file |
| `422` | The CSV exists but has no recognisable factor columns (CCI, CPI, Oil, GDP, Unemployment, ROI) |
| `401` | Missing or invalid JWT token |

---

### `POST /heatmap/upload` — *Future use / admin only*

Upload 1–6 CSV files, one per external factor. The factor name is auto-detected from the column headers — file name does not matter. Files are merged on date and stored per company. Only CSV is supported; Excel files are rejected.

**Request:** `multipart/form-data`, field name `files`

**Response**
```json
{
  "factors_loaded": ["CCI", "CPI"],
  "row_count": 75,
  "date_range_start": "2020-01-01",
  "date_range_end": "2025-12-01",
  "heatmap_base64": null,
  "message": "Loaded 2 factor(s): CCI, CPI."
}
```

---

### `POST /heatmap/from-csv` — *Future use / admin only*

Upload the full training or SL CSV. The backend finds whichever external factor columns exist in it, extracts them, merges with any data already stored for the company, and saves.

**Request:** `multipart/form-data`, field name `file`

**Response:** Same structure as `/heatmap/upload`

---

### `POST /heatmap/from-subledger/{batch_id}` — *Future use / admin only*

Reads the SUBLEDGER CSV already in Supabase Storage for the given batch, extracts external factor columns from it, and stores them. No file upload needed — it reads directly from storage.

**Path param:** `batch_id`

**Response:** Same structure as `/heatmap/upload`

---

### `POST /heatmap/refresh` — *Not needed, frontend handles this*

Originally built to let you send fresh monthly factor values and get an updated heatmap back. Since heatmap rendering was moved entirely to the frontend, this endpoint is no longer needed in the production flow. The frontend receives `monthly_data` from `/heatmap/from-storage`, appends the new row locally, recomputes correlation using a JS library, and re-renders the Plotly chart — no server round-trip required.

**Request body (if ever needed)**
```json
{
  "CCI": 102.5,
  "CPI": 2.1,
  "Oil": 78.3,
  "GDP": 580.2,
  "Unemployment": 3.8,
  "ROI": 4.1
}
```

---

### `GET /heatmap/data` — *Future use / debugging*

Returns whatever factor data is stored in the per-company pkl file. Only has data if `/heatmap/upload` or `/heatmap/from-csv` was called before. In the current production setup, use `GET /heatmap/from-storage` instead.

**Response**
```json
{
  "row_count": 75,
  "columns": ["date", "CCI", "CPI", "Oil", "GDP", "Unemployment", "ROI"],
  "rows": [
    { "date": "2021-01-01", "CCI": 107.0, "CPI": 1.1, "Oil": 52.16, "GDP": 533.9, "Unemployment": 6.3, "ROI": 0.1 }
  ]
}
```

---

## Datasets

### `GET /datasets`

List all CSV datasets uploaded by the logged-in user, newest first.

**Response**
```json
[
  {
    "id": 1,
    "original_filename": "sl_with_external_factors.csv",
    "row_count": 80,
    "date_range_start": "2021-01-09",
    "date_range_end": "2025-12-15",
    "uploaded_at": "2026-05-28T10:00:00",
    "model_count": 1
  }
]
```

---

### `GET /datasets/{dataset_id}`

Get metadata for one specific dataset.

---

### `GET /datasets/{dataset_id}/models`

List all models that were trained on this dataset. Call this before starting a new training run — if a model already exists for the same CSV, you might be able to reuse it instead of training again.

---

## Models

### `GET /models`

List all trained models for the logged-in user.

**Optional query param:** `dataset_id` — filter by dataset

**Response**
```json
[
  {
    "id": 1,
    "model_name": "revenue_model_v1",
    "description": null,
    "targets": ["Total Revenue", "COGS", "SG&A"],
    "metrics": { "...": "..." },
    "trained_at": "2026-05-28T10:00:00",
    "training_duration_seconds": 12.4
  }
]
```

---

### `GET /models/{model_id}`

Get the full detail for a single model — metrics, feature list, training config.

---

### `DELETE /models/{model_id}`

Removes the model from the database and permanently deletes the `.pkl` file from disk.

**Response**
```json
{ "detail": "Model 'revenue_model_v1' deleted." }
```

---

## Full API Summary

| Method | Endpoint | Description | Status |
|---|---|---|---|
| GET | `/` | Health check | Active |
| GET | `/health` | Health ping for load balancers | Active |
| POST | `/train` | Train a new ML model from a CSV | Active |
| POST | `/predict/{model_id}` | Monthly forecast across a date range | Active |
| POST | `/predict/from-batch/{model_id}` | Predict from Supabase batch SUBLEDGER | **Main production** |
| POST | `/forecast-map/{model_id}` | Predict + regional breakdown from manual rows | Active |
| POST | `/forecast-map/from-batch/{model_id}` | Day-by-day predictions + regional from batch | **Main production** |
| GET | `/heatmap/from-storage` | Correlation matrix + monthly data for frontend | **Main production** |
| POST | `/heatmap/upload` | Upload individual factor CSV files | Future / admin |
| POST | `/heatmap/from-csv` | Extract factors from an uploaded CSV | Future / admin |
| POST | `/heatmap/from-subledger/{batch_id}` | Extract factors from batch SUBLEDGER | Future / admin |
| POST | `/heatmap/refresh` | Append new factor row | Not needed — frontend handles |
| GET | `/heatmap/data` | Return stored pkl factor data as JSON | Future / debugging |
| GET | `/datasets` | List datasets for the logged-in user | Active |
| GET | `/datasets/{id}` | Get single dataset metadata | Active |
| GET | `/datasets/{id}/models` | List all models trained on a dataset | Active |
| GET | `/models` | List all models | Active |
| GET | `/models/{id}` | Get model detail | Active |
| DELETE | `/models/{id}` | Delete model and its .pkl file | Active |

---

## How It All Fits Together

**Two databases running side by side:**
- **Supabase PostgreSQL** — users, companies, ingestion batches, file uploads. Shared with the main platform (Part 1). This service only reads from it.
- **SQLite (`sm_revenue.db`)** — trained models and dataset metadata. Local to this ML service only.

**Security model:**
- Every request decodes the JWT token to get `user_id` and `company_id`
- Before any batch operation, the `company_id` from the token is checked against the batch record in Supabase
- A user from Company A can never access Company B's batches or predictions

**External factors flow in production:**
- `model_store/external_factors.csv` is baked into the Docker image at build time
- When a batch prediction is triggered, SL data is fetched from Supabase and external factors are merged automatically by year-month — no manual input needed
- To update the factors file, update the CSV and rebuild the Docker image

**Heatmap flow:**
- Backend computes the Pearson correlation matrix and returns raw monthly values as JSON
- Frontend renders the heatmap using Plotly.js
- Live updates (when user adds a new month's data) are computed entirely on the frontend — no extra API call
