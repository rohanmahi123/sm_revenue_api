"""
Heatmap utilities
=================
1. parse_factor_file  – reads one xlsx/csv, detects which external factor it
                        contains, and returns a tidy {date, <factor>} DataFrame.
2. merge_factor_files – combines multiple parsed DataFrames on date.
3. save_factor_data   – persists the merged DataFrame to disk (per company).
4. load_factor_data   – reloads it.
5. generate_heatmap   – correlation heatmap → base64 PNG (unchanged API).
"""

from __future__ import annotations

import io
import os
import pickle
import re
from typing import List, Optional

import pandas as pd

EXTERNAL_FACTORS = ["CCI", "CPI", "Oil", "GDP", "Unemployment", "ROI"]

# ── Column-detection signatures ───────────────────────────────────────────────
# Maps factor name → (date_column_candidates, value_column_candidates)
_SIGNATURES: dict[str, tuple[list[str], list[str]]] = {
    "CCI": (
        ["observation_date", "date", "month", "months"],
        ["consumer confidence (%)", "consumer confidence", "cci", "value"],
    ),
    "CPI": (
        ["months", "date", "month", "observation_date"],
        ["all groups cpi (%)", "all groups cpi", "cpi (%)", "cpi", "value"],
    ),
    "Oil": (
        ["months", "date", "month", "observation_date"],
        ["oil price (crude)", "oil price", "oil", "price", "value"],
    ),
    "GDP": (
        ["month", "months", "date", "observation_date"],
        ["gdp current price, aud bn", "gdp current price", "gdp", "value"],
    ),
    "Unemployment": (
        ["months", "date", "month", "observation_date"],
        ["unemployment rate (%)", "unemployment rate", "unemployment", "value"],
    ),
    "ROI": (
        ["months", "date", "month", "observation_date"],
        ["roi (%)", "roi", "return", "value"],
    ),
}


# ── Date parsers ──────────────────────────────────────────────────────────────

def _parse_date_series(s: pd.Series) -> pd.Series:
    """
    Parse a date column that may be:
      - already datetime64 (pandas read it fine)
      - string like "Jan-20", "Jan 2020", "2020-01-01", "2020-01-31" …
    Returns a Series of pd.Timestamp (NaT for unparseable rows).
    """
    if pd.api.types.is_datetime64_any_dtype(s):
        return pd.to_datetime(s, errors="coerce")

    s_str = s.astype(str).str.strip()

    # Detect "Mon-YY" pattern (e.g. "Jan-20") FIRST — pandas misreads these
    # as "January 20th year 0001", so handle them before the generic parse.
    mon_yy_mask = s_str.str.match(r"^[A-Za-z]{3}-\d{2}$")
    parsed = pd.Series(pd.NaT, index=s_str.index)

    if mon_yy_mask.any():
        fixed = s_str[mon_yy_mask].str.replace(
            r"^([A-Za-z]{3})-(\d{2})$", r"\1-20\2", regex=True
        )
        parsed[mon_yy_mask] = pd.to_datetime(fixed, format="%b-%Y", errors="coerce")

    # Parse the rest with standard ISO / pandas default
    rest_mask = ~mon_yy_mask
    if rest_mask.any():
        parsed[rest_mask] = pd.to_datetime(s_str[rest_mask], errors="coerce")

    return parsed


def _find_col(cols_lower: list[str], candidates: list[str]) -> Optional[str]:
    """Return the first candidate that appears in cols_lower (case-insensitive)."""
    for c in candidates:
        if c in cols_lower:
            return c
    return None


# ── Public: parse one file ────────────────────────────────────────────────────

def parse_factor_file(content: bytes, filename: str) -> pd.DataFrame:
    """
    Read one xlsx or csv file and return a two-column DataFrame:
        date (pd.Timestamp)  |  <FactorName> (float)

    Factor name is auto-detected from column headers.
    Raises ValueError if the factor cannot be identified.
    """
    suffix = os.path.splitext(filename)[-1].lower()
    if suffix in (".xlsx", ".xls"):
        raise ValueError(
            f"Excel files (.xlsx/.xls) are not supported. "
            f"Please convert '{filename}' to CSV and re-upload."
        )
    raw = pd.read_csv(io.BytesIO(content))

    cols_lower = [str(c).strip().lower() for c in raw.columns]
    col_map = {cl: orig for cl, orig in zip(cols_lower, raw.columns)}

    # Detect which factor this file belongs to
    detected_factor: Optional[str] = None
    date_col: Optional[str] = None
    value_col: Optional[str] = None

    for factor, (date_cands, value_cands) in _SIGNATURES.items():
        dc = _find_col(cols_lower, date_cands)
        vc = _find_col(cols_lower, value_cands)
        if dc and vc:
            detected_factor = factor
            date_col = col_map[dc]
            value_col = col_map[vc]
            break

    if not detected_factor:
        raise ValueError(
            f"Cannot detect external factor from file '{filename}'. "
            f"Columns found: {list(raw.columns)}"
        )

    df = raw[[date_col, value_col]].copy()
    df.columns = ["date", detected_factor]

    # Parse dates
    df["date"] = _parse_date_series(df["date"])

    # Normalise value — strip "$", "%", spaces → float
    df[detected_factor] = (
        df[detected_factor]
        .astype(str)
        .str.replace(r"[$,%\s]", "", regex=True)
        .pipe(pd.to_numeric, errors="coerce")
    )

    df = df.dropna(subset=["date", detected_factor])
    df["date"] = df["date"].dt.to_period("M").dt.to_timestamp()  # normalise to month-start
    df = df.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)

    return df


# ── Public: merge multiple parsed DataFrames ──────────────────────────────────

def merge_factor_files(dfs: list[pd.DataFrame]) -> pd.DataFrame:
    """
    Outer-join all parsed factor DataFrames on `date`.
    Returns a wide DataFrame: date | CCI | CPI | Oil | GDP | Unemployment | ROI
    (columns present only if that file was uploaded).
    """
    if not dfs:
        return pd.DataFrame(columns=["date"])

    merged = dfs[0]
    for df in dfs[1:]:
        merged = pd.merge(merged, df, on="date", how="outer")

    return merged.sort_values("date").reset_index(drop=True)


# ── Disk persistence ──────────────────────────────────────────────────────────

def _store_path(store_dir: str, company_id: int | str) -> str:
    os.makedirs(store_dir, exist_ok=True)
    return os.path.join(store_dir, f"heatmap_factors_{company_id}.pkl")


def save_factor_data(df: pd.DataFrame, store_dir: str, company_id: int | str) -> None:
    path = _store_path(store_dir, company_id)
    with open(path, "wb") as f:
        pickle.dump(df, f)


def load_factor_data(store_dir: str, company_id: int | str) -> Optional[pd.DataFrame]:
    path = _store_path(store_dir, company_id)
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


# ── Public: generate heatmap ──────────────────────────────────────────────────

def generate_heatmap(input_rows: List[dict]) -> str | None:
    """Heatmap image generation moved to frontend. Returns None."""
    return None


def generate_heatmap_from_df(df: pd.DataFrame) -> str | None:
    """Heatmap image generation moved to frontend. Returns None."""
    return None
