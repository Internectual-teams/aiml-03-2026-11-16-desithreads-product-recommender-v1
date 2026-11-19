"""
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
    print(f"\n[TEST 1] Valid Product ID: {valid_id_1}")
    print(f"  Result: {recs_1}")
    if isinstance(recs_1, list) and len(recs_1) > 0 and valid_id_1 not in recs_1:
        print("  Status: PASSED (Returned non-empty list of recommendations)")
    else:
        print("  Status: FAILED")
        all_passed = False

    # Test 2: Second Valid Product ID
    valid_id_2 = 102
    recs_2 = get_recommendations_for_product(valid_id_2)
    print(f"\n[TEST 2] Valid Product ID: {valid_id_2}")
    print(f"  Result: {recs_2}")
    if isinstance(recs_2, list) and len(recs_2) > 0 and valid_id_2 not in recs_2:
        print("  Status: PASSED (Returned non-empty list of recommendations)")
    else:
        print("  Status: FAILED")
        all_passed = False

    # Test 3: Invalid / Non-existent Numeric Product ID
    invalid_id_numeric = 99999
    recs_invalid_num = get_recommendations_for_product(invalid_id_numeric)
    print(f"\n[TEST 3] Non-existent Numeric Product ID: {invalid_id_numeric}")
    print(f"  Result: {recs_invalid_num}")
    if recs_invalid_num == []:
        print("  Status: PASSED (Gracefully returned empty list [])")
    else:
        print(f"  Status: FAILED (Expected [], got {recs_invalid_num})")
        all_passed = False

    # Test 4: Invalid Non-numeric / String Product ID
    invalid_id_string = "invalid_sku_xyz"
    recs_invalid_str = get_recommendations_for_product(invalid_id_string)
    print(f"\n[TEST 4] Invalid Non-numeric Product ID: '{invalid_id_string}'")
    print(f"  Result: {recs_invalid_str}")
    if recs_invalid_str == []:
        print("  Status: PASSED (Gracefully returned empty list [])")
    else:
        print(f"  Status: FAILED (Expected [], got {recs_invalid_str})")
        all_passed = False

    print("\n" + "=" * 65)
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
