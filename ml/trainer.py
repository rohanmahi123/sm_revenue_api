"""
Training logic ported from notebook §8–§11.
Returns a model package dict (same structure as the .pkl the notebook saves).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline

from ml.preprocessor import (
    TARGETS,
    build_external_df,
    build_preprocessor,
    create_daily_dataset,
    load_financial_csv,
)


# ── Model builders ────────────────────────────────────────────────────────────

def _build_baseline(preprocessor):
    return Pipeline([
        ("preprocess", preprocessor),
        ("regressor", MultiOutputRegressor(LinearRegression())),
    ])


def _build_ridge(preprocessor):
    return Pipeline([
        ("preprocess", preprocessor),
        ("regressor", MultiOutputRegressor(RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0]))),
    ])


# ── Metrics helper ────────────────────────────────────────────────────────────

def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, target: str) -> Dict:
    with np.errstate(divide="ignore", invalid="ignore"):
        mape = np.mean(
            np.abs((y_true - y_pred) / np.where(y_true == 0, np.nan, y_true))
        ) * 100
    return {
        "MAE": round(float(mean_absolute_error(y_true, y_pred)), 4),
        "RMSE": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4),
        "R2": round(float(r2_score(y_true, y_pred)), 6),
        "MAPE": round(float(np.nan_to_num(mape, nan=0.0)), 4),
    }


# ── Main training function ────────────────────────────────────────────────────

def train_model(
    csv_path: str,
    external_factor_rows: Optional[List[dict]],
    model_store_dir: str,
    model_name: str,
    sheet_name: str = "10 SL",
    test_size: float = 0.25,
    random_state: int = 42,
) -> Tuple[str, Dict[str, Any], float]:
    """
    Full training pipeline.

    Returns
    -------
    model_file_path  : str   – absolute path to saved .pkl
    result           : dict  – metrics, feature_cols, best_model_per_target, targets
    duration         : float – training wall-clock seconds
    """
    t0 = time.perf_counter()

    # 1. Load financial data
    df = load_financial_csv(csv_path, sheet_name=sheet_name)

    # 2. Build external factor DataFrame (may be empty)
    ext_df = build_external_df(external_factor_rows) if external_factor_rows else None

    # 3. Create daily dataset
    daily_df = create_daily_dataset(df, ext_df)

    # 4. Resolve available targets
    available_targets = [t for t in TARGETS if t in daily_df.columns]
    if not available_targets:
        raise ValueError(
            f"None of the expected target columns {TARGETS} were found in the data."
        )

    # 5. Build preprocessor + feature list
    preprocessor_base, feature_cols = build_preprocessor(daily_df)

    X = daily_df[feature_cols].copy()
    y = daily_df[available_targets].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, shuffle=True
    )

    # 6. Fit models (each needs its own preprocessor clone to avoid state sharing)
    from sklearn.compose import ColumnTransformer  # noqa: F401 – clone workaround
    import copy

    baseline = _build_baseline(copy.deepcopy(preprocessor_base))
    ridge = _build_ridge(copy.deepcopy(preprocessor_base))

    baseline.fit(X_train, y_train)
    ridge.fit(X_train, y_train)

    # 7. Evaluate
    baseline_preds = baseline.predict(X_test)
    ridge_preds = ridge.predict(X_test)

    metrics: Dict[str, Any] = {}
    best_model_per_target: Dict[str, str] = {}

    for i, target in enumerate(available_targets):
        m_base = _compute_metrics(y_test.iloc[:, i].values, baseline_preds[:, i], target)
        m_ridge = _compute_metrics(y_test.iloc[:, i].values, ridge_preds[:, i], target)
        metrics[target] = {"baseline": m_base, "ridge": m_ridge}
        # Pick the model with higher R²
        best_model_per_target[target] = "ridge" if m_ridge["R2"] >= m_base["R2"] else "baseline"

    metrics["best_model_per_target"] = best_model_per_target

    # 8. Serialize model package
    package = {
        "baseline_model": baseline,
        "ridge_model": ridge,
        "best_model_per_target": best_model_per_target,
        "features": feature_cols,
        "targets": available_targets,
    }

    store = Path(model_store_dir)
    store.mkdir(parents=True, exist_ok=True)
    model_file = store / f"{model_name}.pkl"
    joblib.dump(package, model_file)

    duration = round(time.perf_counter() - t0, 3)

    result = {
        "metrics": metrics,
        "best_model_per_target": best_model_per_target,
        "feature_columns": feature_cols,
        "targets": available_targets,
        "external_factors_used": {
            f: (f in (daily_df.columns if daily_df is not None else []))
            for f in ["CCI", "CPI", "Oil", "GDP", "Unemployment", "ROI"]
        },
    }

    return str(model_file.resolve()), result, duration
