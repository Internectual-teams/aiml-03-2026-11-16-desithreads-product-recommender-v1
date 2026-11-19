"""Feature Extraction & Similarity Module
-------------------------------------
Task: Compute Pairwise Product Similarity Matrix
Module: feature_extraction_and_similarity
Author: Sujith (Team AIML-03)

This script loads the encoded product feature matrix (product_features.npy)
and computes pairwise product cosine similarity. The resulting symmetric
similarity matrix is saved as a NumPy binary file (similarity_matrix.npy).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Paths resolved relative to this script's directory
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_INPUT_FEATURES = SCRIPT_DIR / "product_features.npy"
DEFAULT_OUTPUT_SIMILARITY = SCRIPT_DIR / "similarity_matrix.npy"

# Precision dtype for similarity matrix
SIMILARITY_DTYPE = np.float32


def compute_cosine_similarity(
    feature_matrix: np.ndarray,
    dtype: np.dtype = SIMILARITY_DTYPE,
) -> np.ndarray:
    """Computes the pairwise cosine similarity matrix for product feature vectors.

    Given a 2D feature matrix X of shape (N, D), computes S of shape (N, N) where:
        S[i, j] = (X[i] . X[j]) / (||X[i]|| * ||X[j]||)

    Safely handles zero-norm vectors by setting their similarities to 0.0,
    avoiding division by zero or NaN values.

    Args:
        feature_matrix: 2D NumPy array of shape (n_products, n_features).
        dtype: Numerical numpy dtype for the similarity matrix (default: np.float32).

    Returns:
        Symmetric 2D NumPy array of shape (n_products, n_products) with values in [0, 1].

    Raises:
        ValueError: If feature_matrix is not 2D or is empty.
    """
    if feature_matrix.ndim != 2:
        raise ValueError(
            f"Expected a 2D feature matrix, got {feature_matrix.ndim}D array with shape {feature_matrix.shape}."
        )

    n_products, n_features = feature_matrix.shape
    if n_products == 0:
        raise ValueError("Feature matrix contains 0 products. Cannot compute similarity.")

    # Calculate L2 norms along feature dimension (axis=1)
    norms = np.linalg.norm(feature_matrix, axis=1, keepdims=True)

    # Identify zero-norm vectors to handle division by zero safely
    is_zero_vector = (norms.squeeze(-1) == 0.0)

    # Avoid zero division during normalization
    safe_norms = np.where(norms == 0.0, 1.0, norms)
    normalized_matrix = feature_matrix / safe_norms

    # Pairwise dot product between normalized feature vectors
    similarity_matrix = np.dot(normalized_matrix, normalized_matrix.T)

    # Enforce exact bounds and handle zero vectors
    if np.any(is_zero_vector):
        similarity_matrix[is_zero_vector, :] = 0.0
        similarity_matrix[:, is_zero_vector] = 0.0

    # Ensure self-similarity on the diagonal is exactly 1.0 for non-zero vectors
    diagonal_values = np.where(is_zero_vector, 0.0, 1.0)
    np.fill_diagonal(similarity_matrix, diagonal_values)

    # Clip values to [0.0, 1.0] to guard against floating-point inaccuracies
    similarity_matrix = np.clip(similarity_matrix, 0.0, 1.0)

    return similarity_matrix.astype(dtype)


def validate_similarity_matrix(
    similarity_matrix: np.ndarray,
    n_products: int,
    output_path: Path | None = None,
) -> dict[str, object]:
    """Performs sanity and integrity checks on the computed similarity matrix.

    Verifications:
    1. Shape is square (n_products, n_products).
    2. Numerical dtype.
    3. No NaN or infinite values.
    4. Matrix is symmetric (S == S.T within floating tolerance).
    5. All values fall within [0.0, 1.0].
    6. Output file exists and is non-empty (if output_path provided).

    Args:
        similarity_matrix: The computed similarity array.
        n_products: Expected number of products.
        output_path: Path to saved .npy file (optional).

    Returns:
        Validation summary dictionary.

    Raises:
        AssertionError: If any validation check fails.
    """
    # 1. Square shape match
    assert similarity_matrix.shape == (n_products, n_products), (
        f"Shape mismatch: expected ({n_products}, {n_products}), "
        f"got {similarity_matrix.shape}."
    )

    # 2. Numerical dtype
    assert np.issubdtype(similarity_matrix.dtype, np.number), (
        f"Similarity matrix dtype is not numerical: {similarity_matrix.dtype}"
    )

    # 3. No NaN or Inf
    assert not np.isnan(similarity_matrix).any(), (
        "Similarity matrix contains unexpected NaN values."
    )
    assert not np.isinf(similarity_matrix).any(), (
        "Similarity matrix contains unexpected Inf values."
    )

    # 4. Symmetry check: S == S.T
    assert np.allclose(similarity_matrix, similarity_matrix.T, atol=1e-5), (
        "Similarity matrix is not symmetric."
    )

    # 5. Range check: values in [0.0, 1.0]
    assert (similarity_matrix >= 0.0).all() and (similarity_matrix <= 1.0 + 1e-5).all(), (
        "Similarity matrix values out of bounds [0.0, 1.0]."
    )

    # 6. File verification
    if output_path is not None:
        assert output_path.exists(), f"Output file does not exist at {output_path}"
        assert output_path.stat().st_size > 0, f"Output file is empty at {output_path}"

    return {
        "status": "PASSED",
        "shape": similarity_matrix.shape,
        "dtype": str(similarity_matrix.dtype),
        "is_symmetric": True,
        "min_similarity": float(similarity_matrix.min()),
        "max_similarity": float(similarity_matrix.max()),
        "has_nan": False,
        "has_inf": False,
    }


def compute_and_save_similarity(
    input_features_path: Path | str | None = None,
    output_matrix_path: Path | str | None = None,
) -> np.ndarray:
    """Loads product features, computes similarity, saves the matrix, and validates it.

    Args:
        input_features_path: Path to product_features.npy.
        output_matrix_path: Path to save similarity_matrix.npy.

    Returns:
        The computed similarity matrix as a NumPy array.
    """
    resolved_input = Path(input_features_path or DEFAULT_INPUT_FEATURES)
    resolved_output = Path(output_matrix_path or DEFAULT_OUTPUT_SIMILARITY)

    if not resolved_input.exists():
        raise FileNotFoundError(
            f"Input features file not found at: {resolved_input.resolve()}.\n"
            "Please run 'python feature_extraction_and_similarity/extract_features.py' first."
        )

    print(f"[INFO] Loading product features from: {resolved_input.resolve()}")
    feature_matrix = np.load(resolved_input)

    n_products, n_features = feature_matrix.shape
    print(f"[INFO] Loaded feature matrix: {n_products} products, {n_features} features (dtype: {feature_matrix.dtype})")

    # Compute cosine similarity
    print("[INFO] Computing pairwise cosine similarity matrix...")
    similarity_matrix = compute_cosine_similarity(feature_matrix, dtype=SIMILARITY_DTYPE)

    # Ensure output directory exists
    resolved_output.parent.mkdir(parents=True, exist_ok=True)

    # Save similarity matrix
    np.save(resolved_output, similarity_matrix)
    print(f"[SUCCESS] Saved similarity matrix to: {resolved_output.resolve()}")

    # Validate the generated matrix
    validation = validate_similarity_matrix(similarity_matrix, n_products, resolved_output)

    # Print summary report
    print("\n" + "=" * 55)
    print("       PRODUCT SIMILARITY COMPUTATION REPORT")
    print("=" * 55)
    print(f"Products Processed          : {n_products}")
    print(f"Input Feature Matrix Shape  : ({n_products}, {n_features})")
    print(f"Similarity Matrix Shape     : {similarity_matrix.shape}")
    print(f"Similarity Matrix Data Type : {similarity_matrix.dtype}")
    print(f"Symmetry Verified (S == S.T): {validation['is_symmetric']}")
    print(f"Similarity Range            : [{validation['min_similarity']:.4f}, {validation['max_similarity']:.4f}]")
    print(f"Unexpected NaN / Inf Values : None (Validation Passed)")
    print(f"Output File Location        : {resolved_output.resolve()}")
    print("=" * 55)
    print("\nSample Similarity Matrix Values:")
    print(np.round(similarity_matrix, 4))
    print("=" * 55 + "\n")

    return similarity_matrix


def main() -> None:
    """CLI entry point for similarity computation."""
    parser = argparse.ArgumentParser(
        description="Compute pairwise product similarity matrix for DesiThreads Product Recommender v1."
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default=None,
        help="Path to input product_features.npy (defaults to ./product_features.npy)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Path to output similarity_matrix.npy (defaults to ./similarity_matrix.npy)",
    )

    args = parser.parse_args()

    try:
        compute_and_save_similarity(
            input_features_path=args.input,
            output_matrix_path=args.output,
        )
    except Exception as exc:
        print(f"[ERROR] Similarity computation failed: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
