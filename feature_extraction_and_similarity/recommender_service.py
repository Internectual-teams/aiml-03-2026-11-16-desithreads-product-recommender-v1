"""Recommender Service Module (Mock API Endpoint)
---------------------------------------------
Ticket: AIML-0007-W1-04: Implement Recommendation Lookup Service (Mock)
Client: DesiThreads · e-commerce fashion retailer
Module: Feature Extraction & Similarity

This module loads 'top_n_recommendations.csv' upon initialization into an efficient
in-memory dictionary for fast O(1) lookups and provides get_recommendations_for_product(product_id).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any
import pandas as pd

# Global internal lookup structure for fast O(1) retrieval (Hint 2 & Criterion 3)
_RECOMMENDATIONS_CACHE: dict[int, list[int]] | None = None

DEFAULT_CSV_FILENAME = "top_n_recommendations.csv"


def load_recommendations(csv_path: str | Path | None = None) -> dict[int, list[int]]:
    """Loads top_n_recommendations.csv into an in-memory dictionary.

    Keys are product_id (int) and values are lists of recommended product IDs (list[int]).
    """
    global _RECOMMENDATIONS_CACHE

    target_path = Path(csv_path) if csv_path else Path(DEFAULT_CSV_FILENAME)

    # Search common fallback locations if not found at immediate path
    if not target_path.exists():
        alternates = [
            Path(DEFAULT_CSV_FILENAME),
            Path(__file__).resolve().parent / DEFAULT_CSV_FILENAME,
            Path(__file__).resolve().parent.parent / DEFAULT_CSV_FILENAME,
            Path("feature_extraction_and_similarity") / DEFAULT_CSV_FILENAME,
        ]
        for alt in alternates:
            if alt.exists():
                target_path = alt
                break

    if not target_path.exists():
        _RECOMMENDATIONS_CACHE = {}
        return _RECOMMENDATIONS_CACHE

    # Load recommendations CSV (Hint 3)
    df = pd.read_csv(target_path)

    # Extract recommendation columns (recommended_product_id_1 .. N)
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

    Args:
        product_id: The identifier of the product to fetch recommendations for.

    Returns:
        List of recommended product IDs (ints), or empty list [] if invalid/non-existent.
    """
    global _RECOMMENDATIONS_CACHE

    # Ensure recommendations are loaded into the cache (Criterion 3)
    if _RECOMMENDATIONS_CACHE is None:
        load_recommendations()

    # Safely convert input to int; return [] if not convertible (Criterion 4)
    try:
        pid = int(product_id)
    except (ValueError, TypeError):
        return []

    # Fast O(1) dictionary lookup; returns [] if non-existent (Criterion 4)
    return _RECOMMENDATIONS_CACHE.get(pid, [])


# Eager initialization upon module import/initialization (Criterion 3)
load_recommendations()
