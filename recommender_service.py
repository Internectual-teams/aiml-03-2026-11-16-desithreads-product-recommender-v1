"""
Recommender Service Module (Mock API Endpoint)
---------------------------------------------
Ticket: AIML-0007-W1-04
Client: DesiThreads
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any
import pandas as pd

# Global internal lookup structure for fast O(1) retrieval (Criterion 3)
_RECOMMENDATIONS_CACHE: dict[int, list[int]] | None = None

DEFAULT_CSV_FILENAME = "top_n_recommendations.csv"


def load_recommendations(csv_path: str | Path | None = None) -> dict[int, list[int]]:
    """Loads top_n_recommendations.csv into an in-memory dictionary."""
    global _RECOMMENDATIONS_CACHE

    target_path = Path(csv_path) if csv_path else Path(DEFAULT_CSV_FILENAME)

    if not target_path.exists():
        alternates = [
            Path(DEFAULT_CSV_FILENAME),
            Path(__file__).resolve().parent / DEFAULT_CSV_FILENAME,
            Path("feature_extraction_and_similarity") / DEFAULT_CSV_FILENAME,
        ]
        for alt in alternates:
            if alt.exists():
                target_path = alt
                break

    if not target_path.exists():
        _RECOMMENDATIONS_CACHE = {}
        return _RECOMMENDATIONS_CACHE

    df = pd.read_csv(target_path)
    rec_cols = [c for c in df.columns if c != "product_id"]

    cache: dict[int, list[int]] = {}
    for _, row in df.iterrows():
        try:
            pid = int(row["product_id"])
        except (ValueError, TypeError):
            continue

        rec_ids: list[int] = []
        for col in rec_cols:
            val = row[col]
            if pd.notna(val):
                try:
                    rec_ids.append(int(val))
                except (ValueError, TypeError):
                    pass
        cache[pid] = rec_ids

    _RECOMMENDATIONS_CACHE = cache
    return _RECOMMENDATIONS_CACHE


def get_recommendations_for_product(product_id: Any) -> list[int]:
    """Retrieves list of recommended product IDs for a given product_id.
    Gracefully handles invalid or non-existent product IDs by returning an empty list [].
    """
    global _RECOMMENDATIONS_CACHE

    if _RECOMMENDATIONS_CACHE is None:
        load_recommendations()

    try:
        pid = int(product_id)
    except (ValueError, TypeError):
        return []

    return _RECOMMENDATIONS_CACHE.get(pid, [])


# Eager initialization on load
load_recommendations()
