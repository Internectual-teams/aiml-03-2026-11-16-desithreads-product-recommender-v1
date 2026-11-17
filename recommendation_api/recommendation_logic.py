from pathlib import Path
import csv

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PRODUCTS_CSV = PROJECT_ROOT / "data_preprocessing" / "cleaned_products.csv"


def get_recommendations(
    product_id: int,
    similarity_matrix: np.ndarray,
    num_recommendations: int = 5,
) -> list[int]:
    """Return the most similar product IDs, excluding the input product."""

    # The CSV order must match the matrix row and column order.
    with PRODUCTS_CSV.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)

        if not reader.fieldnames or "product_id" not in reader.fieldnames:
            raise ValueError("The CSV must contain a product_id column.")

        product_ids = [int(row["product_id"]) for row in reader]

    if len(product_ids) != len(set(product_ids)):
        raise ValueError("Product IDs must be unique.")

    expected_shape = (len(product_ids), len(product_ids))

    if similarity_matrix.shape != expected_shape:
        raise ValueError(
            f"Expected matrix shape {expected_shape}, "
            f"got {similarity_matrix.shape}."
        )

    if not np.isfinite(similarity_matrix).all():
        raise ValueError("Similarity scores must be finite numbers.")

    if product_id not in product_ids:
        raise ValueError(f"Unknown product ID: {product_id}")

    if isinstance(num_recommendations, bool) or not isinstance(
        num_recommendations, (int, np.integer)
    ):
        raise TypeError("num_recommendations must be an integer.")

    if not 0 <= num_recommendations <= len(product_ids) - 1:
        raise ValueError(
            f"num_recommendations must be between "
            f"0 and {len(product_ids) - 1}."
        )

    product_index = product_ids.index(product_id)
    scores = similarity_matrix[product_index]

    # Remove the input product before ranking.
    candidates = [
        index for index in range(len(product_ids))
        if index != product_index
    ]

    # Highest score first; equal scores use original matrix order.
    candidates.sort(key=lambda index: (-float(scores[index]), index))

    return [
        product_ids[index]
        for index in candidates[:num_recommendations]
    ]