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

import base64
import io
import os
import pickle
import re
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
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
        raw = pd.read_excel(io.BytesIO(content))
    else:
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
    """
    Build a correlation heatmap from a list of dicts OR from a DataFrame
    passed as the only element (internal use).

    Parameters
    ----------
    input_rows : list of dicts with keys matching EXTERNAL_FACTORS

    Returns
    -------
    base64 PNG string — or None if insufficient data
    """
    df = pd.DataFrame(input_rows)

    available = [
        col for col in EXTERNAL_FACTORS
        if col in df.columns and df[col].notna().any()
    ]

    if len(available) < 2:
        return None

    ext_df = df[available].apply(pd.to_numeric, errors="coerce").dropna(how="all")

    if ext_df.shape[0] < 2:
        return None

    corr = ext_df.corr()

    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor("#1e1e2e")
    ax.set_facecolor("#1e1e2e")

    n = len(available)
    cmap = plt.get_cmap("coolwarm")
    im = ax.imshow(corr.values, cmap=cmap, vmin=-1, vmax=1, aspect="auto")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white", fontsize=9)
    cbar.set_label("Correlation", color="white", fontsize=10)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(available, rotation=45, ha="right", color="white", fontsize=11)
    ax.set_yticklabels(available, color="white", fontsize=11)

    for i in range(n):
        for j in range(n):
            val = corr.values[i, j]
            text_color = "white" if abs(val) < 0.7 else "black"
            ax.text(
                j, i, f"{val:.2f}",
                ha="center", va="center",
                fontsize=10, color=text_color, fontweight="bold",
            )

    ax.set_title(
        "External Factors Correlation Heatmap",
        color="white", fontsize=13, fontweight="bold", pad=14,
    )

    for x in np.arange(-0.5, n, 1):
        ax.axhline(x, color="#2a2a3e", linewidth=1.5)
        ax.axvline(x, color="#2a2a3e", linewidth=1.5)

    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=120, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)

    return base64.b64encode(buf.read()).decode("utf-8")


def generate_heatmap_from_df(df: pd.DataFrame) -> str | None:
    """Convenience wrapper — accepts a DataFrame directly."""
    rows = df[[c for c in EXTERNAL_FACTORS if c in df.columns]].to_dict(orient="records")
    return generate_heatmap(rows)
