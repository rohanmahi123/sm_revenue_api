"""
Generates a correlation heatmap from external factor values
provided in the prediction input rows.
Returns base64 encoded PNG string.
"""

from __future__ import annotations

import base64
import io
from typing import List

import matplotlib
matplotlib.use("Agg")  # no display needed — server side
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

EXTERNAL_FACTORS = ["CCI", "CPI", "Oil", "GDP", "Unemployment", "ROI"]


def generate_heatmap(input_rows: List[dict]) -> str | None:
    """
    Build a correlation heatmap from the external factor columns
    present in the prediction input rows.

    Parameters
    ----------
    input_rows : list of dicts (raw prediction input)

    Returns
    -------
    base64 PNG string — or None if no external factor columns found
    """
    df = pd.DataFrame(input_rows)

    # Keep only external factor columns that are actually present + have data
    available = [
        col for col in EXTERNAL_FACTORS
        if col in df.columns and df[col].notna().any()
    ]

    if len(available) < 2:
        # Need at least 2 factors to draw a meaningful heatmap
        return None

    ext_df = df[available].apply(pd.to_numeric, errors="coerce").dropna(how="all")

    if ext_df.shape[0] < 2:
        return None

    corr = ext_df.corr()

    # ── Draw ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor("#1e1e2e")
    ax.set_facecolor("#1e1e2e")

    n = len(available)
    cmap = plt.get_cmap("coolwarm")

    im = ax.imshow(corr.values, cmap=cmap, vmin=-1, vmax=1, aspect="auto")

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white", fontsize=9)
    cbar.set_label("Correlation", color="white", fontsize=10)

    # Axis labels
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(available, rotation=45, ha="right", color="white", fontsize=11)
    ax.set_yticklabels(available, color="white", fontsize=11)

    # Annotate cells
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

    # Grid lines between cells
    for x in np.arange(-0.5, n, 1):
        ax.axhline(x, color="#2a2a3e", linewidth=1.5)
        ax.axvline(x, color="#2a2a3e", linewidth=1.5)

    plt.tight_layout()

    # ── Encode to base64 ──────────────────────────────────────────────────────
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)

    return base64.b64encode(buf.read()).decode("utf-8")
