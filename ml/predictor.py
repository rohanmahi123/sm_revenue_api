"""
Prediction helpers – loads a saved .pkl and runs inference.
Mirrors the notebook's predict_from_user_input() + prepare_user_daily_input().
"""

from __future__ import annotations

from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd

from ml.preprocessor import DATE_FEATURES, safe_parse_dates


def _add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Order Date"] = safe_parse_dates(df["Order Date"].astype(str))
    df["Year"] = df["Order Date"].dt.year
    df["Month"] = df["Order Date"].dt.month
    df["Day"] = df["Order Date"].dt.day
    df["DayOfWeek"] = df["Order Date"].dt.dayofweek
    df["Quarter"] = df["Order Date"].dt.quarter
    df["IsMonthEnd"] = df["Order Date"].dt.is_month_end.astype(int)
    df["IsMonthStart"] = df["Order Date"].dt.is_month_start.astype(int)
    df["Month_Sin"] = np.sin(2 * np.pi * df["Month"] / 12)
    df["Month_Cos"] = np.cos(2 * np.pi * df["Month"] / 12)
    df["DayOfWeek_Sin"] = np.sin(2 * np.pi * df["DayOfWeek"] / 7)
    df["DayOfWeek_Cos"] = np.cos(2 * np.pi * df["DayOfWeek"] / 7)
    return df


def predict(model_file_path: str, input_rows: List[dict]) -> List[Dict[str, Any]]:
    """
    Load a saved model package and run predictions.

    Parameters
    ----------
    model_file_path : str     – path to the .pkl saved by trainer.py
    input_rows      : list    – list of row dicts matching PredictRow schema

    Returns
    -------
    list of dicts with order_date + predicted values + model names used
    """
    package = joblib.load(model_file_path)

    baseline = package["baseline_model"]
    ridge = package["ridge_model"]
    best = package["best_model_per_target"]
    features = package["features"]
    targets = package["targets"]

    # Normalise column aliases from the Pydantic schema
    rename_map = {
        "order_date": "Order Date",
        "Item_type": "Item type",
        "Raw_Material": "Raw Material",
        "Direct_Labor": "Direct Labor",
        "Indirect_Labor": "Indirect Labor",
        "Rent_Utility": "Rent & Utility",
    }
    df = pd.DataFrame(input_rows).rename(columns=rename_map)

    df = _add_date_features(df)

    # Keep only feature columns the model knows about; fill missing with NaN
    for col in features:
        if col not in df.columns:
            df[col] = np.nan

    X = df[features]

    baseline_preds = baseline.predict(X)
    ridge_preds = ridge.predict(X)

    # Build adaptive predictions
    final_preds = np.zeros_like(baseline_preds)
    for i, target in enumerate(targets):
        if best.get(target) == "ridge":
            final_preds[:, i] = ridge_preds[:, i]
        else:
            final_preds[:, i] = baseline_preds[:, i]

    target_key_map = {
        "Total Revenue": "predicted_total_revenue",
        "COGS": "predicted_COGS",
        "SG&A": "predicted_SGA",
    }
    model_key_map = {
        "Total Revenue": "model_used_revenue",
        "COGS": "model_used_COGS",
        "SG&A": "model_used_SGA",
    }

    results = []
    for row_idx in range(len(df)):
        row_result: Dict[str, Any] = {
            "order_date": str(df["Order Date"].iloc[row_idx].date()),
        }
        for i, target in enumerate(targets):
            pk = target_key_map.get(target, f"predicted_{target}")
            mk = model_key_map.get(target, f"model_used_{target}")
            row_result[pk] = round(float(final_preds[row_idx, i]), 4)
            row_result[mk] = best.get(target, "unknown")

        # Fill keys not present in this model's targets
        for pk in target_key_map.values():
            row_result.setdefault(pk, None)
        for mk in model_key_map.values():
            row_result.setdefault(mk, None)

        results.append(row_result)

    return results
