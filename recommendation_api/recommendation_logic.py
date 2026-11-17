import csv
import os
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PRODUCTS_CSV = (
    PROJECT_ROOT / "data_preprocessing" / "cleaned_products.csv"
)


def load_product_ids() -> list[int]:
    """Load product IDs in the same order as the similarity matrix."""

    # Explicit test mode for your teammate's three-product sample.
    if os.getenv("DESITHREADS_SAMPLE_MODE") == "1":
        return [101, 102, 103]

    if not PRODUCTS_CSV.is_file():
        raise FileNotFoundError(
            f"Product data not found: {PRODUCTS_CSV}. "
            "Provide the real CSV or enable DESITHREADS_SAMPLE_MODE=1 "
            "when using the supplied sample arrays."
        )

    with PRODUCTS_CSV.open(
        mode="r",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        reader = csv.DictReader(file)

        if not reader.fieldnames or "product_id" not in reader.fieldnames:
            raise ValueError("CSV must contain a product_id column.")

        product_ids = []

        for row_number, row in enumerate(reader, start=2):
            try:
                product_ids.append(int(row["product_id"]))
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Invalid product_id in CSV row {row_number}."
                ) from error

    if not product_ids:
        raise ValueError("The product CSV is empty.")

    if len(product_ids) != len(set(product_ids)):
        raise ValueError("The product CSV contains duplicate product IDs.")

    return product_ids


def get_recommendations(
    product_id: int,
    similarity_matrix: np.ndarray,
    num_recommendations: int = 5,
) -> list[int]:
    """Return the highest-scoring product IDs, excluding the input ID."""

    if isinstance(product_id, bool) or not isinstance(
        product_id, (int, np.integer)
    ):
        raise TypeError("product_id must be an integer.")

    if isinstance(num_recommendations, bool) or not isinstance(
        num_recommendations, (int, np.integer)
    ):
        raise TypeError("num_recommendations must be an integer.")

    if not isinstance(similarity_matrix, np.ndarray):
        raise TypeError("similarity_matrix must be a NumPy array.")

    if similarity_matrix.dtype.kind not in "iuf":
        raise ValueError("Similarity scores must be real numbers.")

    product_ids = load_product_ids()
    number_of_products = len(product_ids)

    expected_shape = (number_of_products, number_of_products)

    if similarity_matrix.shape != expected_shape:
        raise ValueError(
            f"Expected similarity matrix shape {expected_shape}, "
            f"but received {similarity_matrix.shape}."
        )

    if product_id not in product_ids:
        raise ValueError(f"Unknown product ID: {product_id}")

    if not 0 <= num_recommendations <= number_of_products - 1:
        raise ValueError(
            "num_recommendations must be between "
            f"0 and {number_of_products - 1}."
        )

    product_index = product_ids.index(product_id)
    similarity_scores = similarity_matrix[product_index]

    if not np.isfinite(similarity_scores).all():
        raise ValueError("Similarity scores contain NaN or infinity.")

    candidate_indices = [
        index
        for index in range(number_of_products)
        if index != product_index
    ]

    # Highest similarity first; ties keep the original matrix order.
    candidate_indices.sort(
        key=lambda index: (-float(similarity_scores[index]), index)
    )

    return [
        product_ids[index]
        for index in candidate_indices[:num_recommendations]
    ]