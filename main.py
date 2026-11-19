"""AIML-0007-W1-04: Implement Recommendation Lookup Service (Mock)
---------------------------------------------------------------
Client: DesiThreads · e-commerce fashion retailer
Module: Feature Extraction & Similarity
Sprint Goal: Build a basic product recommendation system that suggests similar items based on product attributes.

Acceptance Criteria:
1. A Python module `recommender_service.py` is created.
2. The `recommender_service.py` module contains a function `get_recommendations_for_product(product_id)`.
3. The `recommender_service.py` module loads `top_n_recommendations.csv` upon initialization into an internal dictionary.
4. Calling `get_recommendations_for_product` with an invalid or non-existent `product_id` returns an empty list [].
5. A test script `test_recommender_service.py` is included that demonstrates fetching recommendations for at least two valid product IDs and one invalid product ID.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any
import pandas as pd


# ---------------------------------------------------------------------------
# 0. Ensure top_n_recommendations.csv exists in the working directory
# ---------------------------------------------------------------------------
def ensure_recommendations_csv():
    csv_file = "top_n_recommendations.csv"
    if not os.path.exists(csv_file):
        data = {
            "product_id": [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
            "recommended_product_id_1": [104, 105, 101, 101, 102, 101, 101, 101, 102, 101],
            "recommended_product_id_2": [110, 109, 102, 110, 109, 102, 102, 104, 105, 104],
            "recommended_product_id_3": [108, 106, 104, 108, 106, 104, 103, 106, 103, 103],
            "recommended_product_id_4": [106, 103, 105, 103, 103, 105, 104, 110, 106, 108],
            "recommended_product_id_5": [103, 101, 110, 106, 101, 108, 105, 102, 101, 106],
        }
        pd.DataFrame(data).to_csv(csv_file, index=False)

ensure_recommendations_csv()


# ---------------------------------------------------------------------------
# 1. Acceptance Criteria 1, 2, 3, 4: recommender_service.py
# ---------------------------------------------------------------------------
RECOMMENDER_SERVICE_CODE = '''"""
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
'''

with open("recommender_service.py", "w", encoding="utf-8") as f:
    f.write(RECOMMENDER_SERVICE_CODE)


# ---------------------------------------------------------------------------
# 2. Acceptance Criterion 5: test_recommender_service.py
# ---------------------------------------------------------------------------
TEST_SCRIPT_CODE = '''"""
Test Script for Recommendation Lookup Service
----------------------------------------------
Ticket: AIML-0007-W1-04
Acceptance Criterion 5: Demonstrates fetching recommendations for at least two valid
product IDs and one invalid product ID.
"""

from __future__ import annotations

import sys
from recommender_service import get_recommendations_for_product


def run_tests() -> bool:
    print("=" * 65)
    print("  TESTING RECOMMENDER SERVICE (MOCK API ENDPOINT)")
    print("=" * 65)

    all_passed = True

    # Test 1: First Valid Product ID
    valid_id_1 = 101
    recs_1 = get_recommendations_for_product(valid_id_1)
    print(f"\\n[TEST 1] Valid Product ID: {valid_id_1}")
    print(f"  Result: {recs_1}")
    if isinstance(recs_1, list) and len(recs_1) > 0 and valid_id_1 not in recs_1:
        print("  Status: PASSED (Returned non-empty list of recommendations)")
    else:
        print("  Status: FAILED")
        all_passed = False

    # Test 2: Second Valid Product ID
    valid_id_2 = 102
    recs_2 = get_recommendations_for_product(valid_id_2)
    print(f"\\n[TEST 2] Valid Product ID: {valid_id_2}")
    print(f"  Result: {recs_2}")
    if isinstance(recs_2, list) and len(recs_2) > 0 and valid_id_2 not in recs_2:
        print("  Status: PASSED (Returned non-empty list of recommendations)")
    else:
        print("  Status: FAILED")
        all_passed = False

    # Test 3: Invalid / Non-existent Numeric Product ID
    invalid_id_numeric = 99999
    recs_invalid_num = get_recommendations_for_product(invalid_id_numeric)
    print(f"\\n[TEST 3] Non-existent Numeric Product ID: {invalid_id_numeric}")
    print(f"  Result: {recs_invalid_num}")
    if recs_invalid_num == []:
        print("  Status: PASSED (Gracefully returned empty list [])")
    else:
        print(f"  Status: FAILED (Expected [], got {recs_invalid_num})")
        all_passed = False

    # Test 4: Invalid Non-numeric / String Product ID
    invalid_id_string = "invalid_sku_xyz"
    recs_invalid_str = get_recommendations_for_product(invalid_id_string)
    print(f"\\n[TEST 4] Invalid Non-numeric Product ID: '{invalid_id_string}'")
    print(f"  Result: {recs_invalid_str}")
    if recs_invalid_str == []:
        print("  Status: PASSED (Gracefully returned empty list [])")
    else:
        print(f"  Status: FAILED (Expected [], got {recs_invalid_str})")
        all_passed = False

    print("\\n" + "=" * 65)
    if all_passed:
        print("  ALL VERIFICATION TESTS COMPLETED SUCCESSFULLY!")
    else:
        print("  SOME TESTS FAILED.")
    print("=" * 65)

    return all_passed


if __name__ == "__main__":
    success = run_tests()
    if not success:
        sys.exit(1)
'''

with open("test_recommender_service.py", "w", encoding="utf-8") as f:
    f.write(TEST_SCRIPT_CODE)


# ---------------------------------------------------------------------------
# Direct Import & Execution Verification
# ---------------------------------------------------------------------------
from recommender_service import get_recommendations_for_product, _RECOMMENDATIONS_CACHE
from test_recommender_service import run_tests

if __name__ == "__main__":
    print("Initializing recommender service verification...")
    print(f"Total products indexed in cache: {len(_RECOMMENDATIONS_CACHE) if _RECOMMENDATIONS_CACHE else 0}")
    run_tests()
