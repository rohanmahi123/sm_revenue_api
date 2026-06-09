"""
External Factors Helper
=======================
Loads the external factors CSV stored in model_store/external_factors.csv
and merges it with SL data by year-month.

CSV expected columns: Order Date, CCI, CPI, Oil, GDP, Unemployment, ROI
(same format as sl_with_external_factors.csv)
"""

import os
import pandas as pd

EXT_FACTOR_COLS = ["CCI", "CPI", "Oil", "GDP", "Unemployment", "ROI"]
EXT_FACTORS_FILENAME = "external_factors.csv"


def load_ext_factors(model_store_dir: str) -> pd.DataFrame | None:
    """
    Load external_factors.csv from model_store directory.
    Returns a DataFrame with columns: [year_month, CCI, CPI, Oil, GDP, Unemployment, ROI]
    or None if file does not exist.
    """
    path = os.path.join(model_store_dir, EXT_FACTORS_FILENAME)
    if not os.path.exists(path):
        return None

    df = pd.read_csv(path)

    # Find date column — support "Order Date" or "order_date"
    date_col = None
    for candidate in ["Order Date", "order_date", "date", "Date"]:
        if candidate in df.columns:
            date_col = candidate
            break

    if date_col is None:
        return None

    # Parse to year-month period
    df["year_month"] = pd.to_datetime(df[date_col], errors="coerce").dt.to_period("M")

    # Keep only factor columns + year_month
    keep = ["year_month"] + [c for c in EXT_FACTOR_COLS if c in df.columns]
    df = df[keep].dropna(subset=["year_month"])

    # One row per month (take first occurrence if duplicates)
    df = df.drop_duplicates(subset=["year_month"]).reset_index(drop=True)

    return df


def merge_ext_factors(sl_df: pd.DataFrame, ext_df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """
    Merge SL DataFrame with external factors DataFrame by year-month.

    Parameters
    ----------
    sl_df  : SL data — must have 'order_date' column
    ext_df : from load_ext_factors() — has 'year_month' + factor columns

    Returns
    -------
    merged DataFrame, info message string
    """
    sl_df = sl_df.copy()
    sl_df["year_month"] = pd.to_datetime(
        sl_df["order_date"], errors="coerce"
    ).dt.to_period("M")

    merged = sl_df.merge(ext_df, on="year_month", how="left")
    merged.drop(columns=["year_month"], inplace=True)

    # Report which factors were matched
    factors_present = [c for c in EXT_FACTOR_COLS if c in merged.columns and merged[c].notna().any()]
    factors_missing = [c for c in EXT_FACTOR_COLS if c not in factors_present]

    if factors_present and not factors_missing:
        info = f"External factors merged from stored file by month: {factors_present}."
    elif factors_present:
        info = (f"External factors from stored file: {factors_present}. "
                f"Not found in file: {factors_missing}.")
    else:
        info = "External factors file found but no matching months — model used training medians."

    return merged, info
