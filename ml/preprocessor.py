"""
Preprocessing helpers ported directly from the SM_Rev_SGA notebook.
All Colab-specific calls (files.upload, etc.) have been removed.
"""

from __future__ import annotations

import warnings
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

# ── Date parsing ──────────────────────────────────────────────────────────────

_DATE_FORMATS = [
    "%b-%y", "%b-%Y", "%b %Y", "%B-%y", "%B %Y",
    "%m-%Y", "%m/%Y", "%Y-%m", "%Y-%m-%d",
    "%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y",
]


def safe_parse_dates(series: pd.Series) -> pd.Series:
    series = series.copy().astype(str).str.strip()
    try:
        result = pd.to_datetime(series, errors="coerce", format="mixed", dayfirst=False)
    except TypeError:
        result = pd.to_datetime(series, errors="coerce", infer_datetime_format=True)

    still_bad = result.isna() & (series != "nan") & (series != "")
    if still_bad.any():
        for fmt in _DATE_FORMATS:
            if not still_bad.any():
                break
            parsed = pd.to_datetime(series[still_bad], format=fmt, errors="coerce")
            result[still_bad] = result[still_bad].fillna(parsed)
            still_bad = result.isna() & (series != "nan") & (series != "")

    return result


# ── Constants ─────────────────────────────────────────────────────────────────

TARGETS = ["Total Revenue", "COGS", "SG&A"]

EXTERNAL_FACTORS = ["CCI", "CPI", "Oil", "GDP", "Unemployment", "ROI"]

BASE_COST_FEATURES = [
    "Raw Material", "Direct Labor", "Freight", "Storage",
    "Packaging", "Indirect Labor", "Rent & Utility", "Overhead",
]

CATEGORICAL_FEATURES = ["Region", "Geo", "Country", "Item type", "Customer"]

DATE_FEATURES = [
    "Year", "Month", "Day", "DayOfWeek", "Quarter",
    "IsMonthEnd", "IsMonthStart",
    "Month_Sin", "Month_Cos", "DayOfWeek_Sin", "DayOfWeek_Cos",
]


# ── Financial CSV loader ──────────────────────────────────────────────────────

def load_financial_csv(path: str, sheet_name: str = "10 SL") -> pd.DataFrame:
    """
    Load the main financial data.
    Accepts both .xlsx (with auto header detection) and .csv files.
    """
    lower = path.lower()

    if lower.endswith((".xlsx", ".xls")):
        raw = pd.read_excel(path, sheet_name=sheet_name, header=None)
        header_row_idx = None
        for i in range(min(15, len(raw))):
            vals = raw.iloc[i].astype(str).str.strip().tolist()
            if "Total Revenue" in vals and "COGS" in vals and "SG&A" in vals:
                header_row_idx = i
                break
        if header_row_idx is None:
            df = pd.read_excel(path, sheet_name=sheet_name, header=2)
        else:
            headers = raw.iloc[header_row_idx].tolist()
            df = raw.iloc[header_row_idx + 1:].copy()
            df.columns = headers
    else:
        # CSV path
        df = pd.read_csv(path)

    df = df.dropna(axis=1, how="all").dropna(axis=0, how="all")
    df = df.loc[:, ~pd.isna(df.columns)]
    df.columns = [str(c).strip() for c in df.columns]

    for col in df.columns:
        df[col] = (
            df[col].astype(str)
            .str.replace("$", "", regex=False)
            .str.replace(",", "", regex=False)
            .str.strip()
        )
        try:
            df[col] = pd.to_numeric(df[col])
        except Exception:
            pass

    if "Order Date" in df.columns:
        df["Order Date"] = safe_parse_dates(df["Order Date"])
    elif "Year" in df.columns and "Mo" in df.columns:
        df["Order Date"] = pd.to_datetime(
            dict(
                year=pd.to_numeric(df["Year"], errors="coerce"),
                month=pd.to_numeric(df["Mo"], errors="coerce"),
                day=1,
            ),
            errors="coerce",
        )

    for col in df.columns:
        if col not in ["Region", "Geo", "Country", "Item type", "Customer", "Order Date"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    available_targets = [t for t in TARGETS if t in df.columns]
    df = df.dropna(subset=["Order Date"] + available_targets).reset_index(drop=True)
    return df


# ── External factors merger ───────────────────────────────────────────────────

def build_external_df(rows: List[dict]) -> pd.DataFrame:
    """
    Convert the list of ExternalFactorRow dicts received from the API
    into a clean DataFrame with a 'Date' column (month-start timestamps).
    """
    df = pd.DataFrame(rows)
    df["Date"] = pd.to_datetime(df["date"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    df = df.drop(columns=["date"], errors="ignore")
    return df.dropna(subset=["Date"])


# ── Date feature engineering ──────────────────────────────────────────────────

def add_daily_date_features(df: pd.DataFrame) -> pd.DataFrame:
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


def create_daily_dataset(
    df: pd.DataFrame,
    ext_df: Optional[pd.DataFrame],
) -> pd.DataFrame:
    """
    Aggregate raw rows to daily level, merge macro factors (monthly resolution),
    forward-fill gaps — mirrors the notebook's create_daily_dataset().
    """
    df = add_daily_date_features(df)

    numeric_sum_cols = [
        c for c in (BASE_COST_FEATURES + TARGETS) if c in df.columns
    ]
    categorical_cols = [c for c in CATEGORICAL_FEATURES if c in df.columns]
    date_feature_cols = [c for c in DATE_FEATURES if c in df.columns]

    aggregation: dict = {}
    for col in numeric_sum_cols:
        aggregation[col] = "sum"
    for col in categorical_cols:
        aggregation[col] = lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[0]
    for col in date_feature_cols:
        aggregation[col] = "first"

    daily_df = (
        df.groupby("Order Date", as_index=False)
        .agg(aggregation)
        .sort_values("Order Date")
        .reset_index(drop=True)
    )

    if ext_df is not None and not ext_df.empty:
        daily_df["_month_key"] = (
            daily_df["Order Date"].dt.to_period("M").dt.to_timestamp()
        )
        ext_merged = ext_df.rename(columns={"Date": "_month_key"})
        ext_cols = [c for c in EXTERNAL_FACTORS if c in ext_merged.columns]
        if ext_cols:
            daily_df = daily_df.merge(
                ext_merged[["_month_key"] + ext_cols],
                on="_month_key",
                how="left",
            )
            daily_df[ext_cols] = daily_df[ext_cols].ffill().bfill()
        daily_df = daily_df.drop(columns=["_month_key"], errors="ignore")

    return daily_df


# ── Preprocessor builder ──────────────────────────────────────────────────────

def build_preprocessor(daily_df: pd.DataFrame):
    """Returns (ColumnTransformer, feature_cols) — mirrors notebook §7."""
    all_possible = CATEGORICAL_FEATURES + BASE_COST_FEATURES + DATE_FEATURES + EXTERNAL_FACTORS
    feature_cols = [c for c in all_possible if c in daily_df.columns]

    cat_cols = [c for c in feature_cols if c in CATEGORICAL_FEATURES]
    num_cols = [c for c in feature_cols if c not in cat_cols]

    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", encoder),
    ])

    preprocessor = ColumnTransformer([
        ("num", num_pipeline, num_cols),
        ("cat", cat_pipeline, cat_cols),
    ])

    return preprocessor, feature_cols
