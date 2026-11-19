"""Feature Extraction & Similarity Module
-------------------------------------
Ticket: AIML-0007-W1-01
Task: Implement Categorical Feature Encoding
Module: feature_extraction_and_similarity
Author: Sujith (Team AIML-03)

This script loads the preprocessed product dataset (cleaned_products.csv)
and encodes the categorical features ('category', 'color', 'material')
using One-Hot Encoding (pd.get_dummies). The resulting feature matrix
is saved as a NumPy binary file (product_features.npy).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

# Paths resolved relative to this script's directory
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_INPUT_CSV = PROJECT_ROOT / "data_preprocessing" / "cleaned_products.csv"
DEFAULT_OUTPUT_NPY = SCRIPT_DIR / "product_features.npy"

# Target categorical columns for one-hot encoding
CATEGORICAL_COLUMNS: list[str] = ["category", "color", "material"]

# Target data type for encoded numerical matrix
FEATURE_DTYPE = np.float32


def get_reference_sample_data() -> pd.DataFrame:
    """Returns the reference sample product dataset specified in Ticket AIML-0007-W1-01.

    This is used for verification when the upstream preprocessing dependency
    (cleaned_products.csv) is not yet available, ensuring full pipeline testability
    without creating unauthorized mock files in teammate Affan's data_preprocessing directory.
    """
    return pd.DataFrame(
        {
            "product_id": [101, 102, 103],
            "name": [
                "Chanderi Silk Saree",
                "Embroidered Kurta Set",
                "Cotton Saree",
            ],
            "category": ["Saree", "Kurta", "Saree"],
            "color": ["Red", "Blue", "Green"],
            "material": ["Silk", "Cotton", "Cotton"],
        }
    )


def encode_categorical_features(
    df: pd.DataFrame,
    categorical_columns: Sequence[str] = CATEGORICAL_COLUMNS,
    dtype: np.dtype = FEATURE_DTYPE,
) -> tuple[np.ndarray, list[str]]:
    """Encodes specified categorical columns of a DataFrame using One-Hot Encoding.

    Preserves original product row order and ensures each row represents exactly one product.
    Columns 'product_id' and 'name' are excluded from encoding.
    Missing categorical values are handled safely without producing NaNs.

    Args:
        df: Input pandas DataFrame containing product records.
        categorical_columns: Sequence of categorical column names to encode.
        dtype: Numerical numpy dtype for the resulting feature matrix (default: np.float32).

    Returns:
        tuple of (numpy.ndarray of shape (n_products, n_features), list of feature column names).

    Raises:
        KeyError: If any of the required categorical columns are missing in df.
        ValueError: If df is empty.
    """
    if df.empty:
        raise ValueError("Input DataFrame is empty. Cannot extract features.")

    # Verify all required categorical columns exist
    missing_cols = [col for col in categorical_columns if col not in df.columns]
    if missing_cols:
        raise KeyError(
            f"Missing required categorical column(s) in dataset: {missing_cols}. "
            f"Available columns: {list(df.columns)}"
        )

    # Extract only the specified categorical columns to ensure product_id, name,
    # and any other columns are strictly excluded from encoding
    subset_df = df[list(categorical_columns)].copy()

    # Safely handle missing categorical values: fill NaN with empty/safe string representation
    # to avoid unexpected NaN dummy values while maintaining numerical outputs
    for col in categorical_columns:
        subset_df[col] = (
            subset_df[col].fillna("Unknown").astype(str).str.strip()
        )

    # Apply One-Hot Encoding using pandas get_dummies
    # dummy_na=False avoids redundant null columns since missing values are safely handled
    encoded_df = pd.get_dummies(
        subset_df,
        columns=list(categorical_columns),
        prefix=list(categorical_columns),
        dtype=dtype,
    )

    # Convert encoded DataFrame into NumPy array
    feature_matrix = encoded_df.to_numpy(dtype=dtype)
    feature_names = list(encoded_df.columns)

    return feature_matrix, feature_names


def validate_feature_matrix(
    feature_matrix: np.ndarray,
    df: pd.DataFrame,
    output_path: Path | None = None,
) -> dict[str, object]:
    """Performs sanity and integrity checks on the generated feature matrix.

    Verifications:
    1. The number of rows matches the input DataFrame.
    2. The matrix is purely numerical.
    3. No unexpected NaN or infinite values exist.
    4. The output file exists and is non-empty (if output_path is provided).

    Args:
        feature_matrix: The extracted numpy feature array.
        df: The original DataFrame.
        output_path: Path to the saved .npy file (optional).

    Returns:
        Validation summary dictionary.

    Raises:
        AssertionError: If any validation check fails.
    """
    # 1. Row count match
    assert feature_matrix.shape[0] == len(df), (
        f"Row count mismatch: feature matrix has {feature_matrix.shape[0]} rows, "
        f"input DataFrame has {len(df)} rows."
    )

    # 2. Numerical dtype
    assert np.issubdtype(feature_matrix.dtype, np.number), (
        f"Feature matrix dtype is not numerical: {feature_matrix.dtype}"
    )

    # 3. No NaN or Inf
    assert not np.isnan(
        feature_matrix
    ).any(), "Feature matrix contains unexpected NaN values."
    assert not np.isinf(
        feature_matrix
    ).any(), "Feature matrix contains unexpected Inf values."

    # 4. Check file existence if path provided
    if output_path is not None:
        assert (
            output_path.exists()
        ), f"Output file does not exist at {output_path}"
        assert (
            output_path.stat().st_size > 0
        ), f"Output file is empty at {output_path}"

    return {
        "status": "PASSED",
        "n_products": feature_matrix.shape[0],
        "n_features": feature_matrix.shape[1],
        "dtype": str(feature_matrix.dtype),
        "has_nan": False,
        "has_inf": False,
    }


def extract_and_save_features(
    input_csv_path: Path | str | None = None,
    output_npy_path: Path | str | None = None,
    df: pd.DataFrame | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Orchestrates loading, encoding, saving, and validating product features.

    If input_csv_path is not found and df is not supplied, it safely explains the
    missing preprocessing dependency and processes the reference sample dataset
    so that the pipeline can be tested, validated, and the deliverable artifact
    generated without violating project boundaries.
    """
    if output_npy_path is None:
        output_npy_path = DEFAULT_OUTPUT_NPY
    else:
        output_npy_path = Path(output_npy_path)

    data_source_description = ""

    if df is not None:
        data_source_description = "Provided in-memory DataFrame"
    elif input_csv_path is not None:
        resolved_input = Path(input_csv_path)
        if not resolved_input.exists():
            raise FileNotFoundError(
                f"Specified input CSV file not found at: {resolved_input}"
            )
        print(f"[INFO] Reading product data from: {resolved_input}")
        df = pd.read_csv(resolved_input)
        data_source_description = f"CSV file ({resolved_input})"
    else:
        # Default behavior: check DEFAULT_INPUT_CSV
        if DEFAULT_INPUT_CSV.exists():
            print(
                f"[INFO] Reading preprocessed product data from: {DEFAULT_INPUT_CSV}"
            )
            df = pd.read_csv(DEFAULT_INPUT_CSV)
            data_source_description = (
                f"Preprocessed dataset ({DEFAULT_INPUT_CSV.name})"
            )
        else:
            print("=" * 70)
            print("[NOTICE] Upstream dependency status:")
            print(f"  Target path: {DEFAULT_INPUT_CSV}")
            print("  Status: File not found.")
            print(
                "  Reason: Teammate Affan's 'data_preprocessing/' module has not yet"
            )
            print("          delivered or generated 'cleaned_products.csv'.")
            print(
                "  Boundary compliance: In accordance with project boundaries, no"
            )
            print("                       unauthorized mock data was created in")
            print("                       'data_preprocessing/'.")
            print(
                "  Action: Running feature extraction on the reference sample"
            )
            print(
                "          dataset defined in Ticket AIML-0007-W1-01 specification"
            )
            print(
                "          to validate the encoding pipeline and generate the"
            )
            print("          deliverable 'product_features.npy'.")
            print("=" * 70)
            df = get_reference_sample_data()
            data_source_description = "Specification Reference Sample Dataset"

    print(
        f"[INFO] Processing {len(df)} products from: {data_source_description}"
    )
    print(f"[INFO] Categorical columns to encode: {CATEGORICAL_COLUMNS}")

    # One-Hot Encoding
    features, feature_names = encode_categorical_features(
        df, CATEGORICAL_COLUMNS, dtype=FEATURE_DTYPE
    )

    # Ensure output directory exists
    output_npy_path.parent.mkdir(parents=True, exist_ok=True)

    # Save NumPy array
    np.save(output_npy_path, features)
    print(f"[SUCCESS] Saved feature matrix to: {output_npy_path}")

    # Validate output
    validation = validate_feature_matrix(features, df, output_npy_path)

    # Print summary report
    print("\n" + "=" * 55)
    print("      CATEGORICAL FEATURE ENCODING REPORT")
    print("=" * 55)
    print(f"Number of Products Processed : {validation['n_products']}")
    print(f"Original Categorical Columns : {CATEGORICAL_COLUMNS}")
    print(f"Number of Generated Features : {validation['n_features']}")
    print(f"Feature Matrix Shape         : {features.shape}")
    print(f"Feature Matrix Data Type     : {features.dtype}")
    print(f"Unexpected NaN / Inf Values  : None (Validation Passed)")
    print(f"Output File Location         : {output_npy_path.resolve()}")
    print(f"Generated Feature Names      : {feature_names}")
    print("=" * 55 + "\n")

    return features, feature_names


def main() -> None:
    """CLI entry point for feature extraction."""
    parser = argparse.ArgumentParser(
        description="Encode categorical features for DesiThreads Product Recommender v1."
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default=None,
        help="Path to input cleaned_products.csv (defaults to ../data_preprocessing/cleaned_products.csv)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Path to output product_features.npy (defaults to ./product_features.npy)",
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Force execution on specification reference sample data for testing.",
    )

    args = parser.parse_args()

    sample_df = get_reference_sample_data() if args.sample else None

    try:
        extract_and_save_features(
            input_csv_path=args.input,
            output_npy_path=args.output,
            df=sample_df,
        )
    except Exception as exc:
        print(f"[ERROR] Feature extraction failed: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
