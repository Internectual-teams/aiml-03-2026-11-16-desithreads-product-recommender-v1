"""Product recommendation logic for DesiThreads."""

import csv
import os
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PRODUCTS_CSV = (
    PROJECT_ROOT / "data_preprocessing" / "cleaned_products.csv"
)


def load_product_ids() -> list[int]:
    """Load unique integer IDs in the matrix's original product order."""

    # Optional path for testing with a separate sample CSV.
    csv_path = Path(
        os.getenv("DESITHREADS_PRODUCTS_CSV", str(PRODUCTS_CSV))
    )

    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Product CSV not found: {csv_path}. "
            "Provide cleaned_products.csv or set "
            "DESITHREADS_PRODUCTS_CSV to your sample CSV."
        )

    product_ids = []
    seen_ids = set()

    with csv_path.open(
        mode="r",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        reader = csv.DictReader(file)

        if not reader.fieldnames or "product_id" not in reader.fieldnames:
            raise ValueError("CSV must contain a product_id column.")

        for row_number, row in enumerate(reader, start=2):
            try:
                product_id = int(row["product_id"])
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Invalid product_id at CSV row {row_number}."
                ) from error

            if product_id in seen_ids:
                raise ValueError(
                    f"Duplicate product ID: {product_id}"
                )

            product_ids.append(product_id)
            seen_ids.add(product_id)

    if not product_ids:
        raise ValueError("The product CSV is empty.")

    return product_ids


def get_recommendations(
    product_id: int,
    similarity_matrix: np.ndarray,
    num_recommendations: int = 5,
) -> list[int]:
    """
    Return product IDs ranked by descending similarity.

    The CSV order must match the matrix row and column order.
    The input product is excluded.
    Ties retain the original CSV order.
    Zero-similarity products remain eligible.
    If fewer products exist than requested, return all available ones.
    A request for zero recommendations returns an empty list.
    """

    if isinstance(product_id, bool) or not isinstance(
        product_id, (int, np.integer)
    ):
        raise TypeError("product_id must be an integer.")

    if isinstance(num_recommendations, bool) or not isinstance(
        num_recommendations, (int, np.integer)
    ):
        raise TypeError("num_recommendations must be an integer.")

    if num_recommendations < 0:
        raise ValueError("num_recommendations cannot be negative.")

    if not isinstance(similarity_matrix, np.ndarray):
        raise TypeError("similarity_matrix must be a NumPy array.")

    if similarity_matrix.ndim != 2:
        raise ValueError("similarity_matrix must be two-dimensional.")

    if similarity_matrix.shape[0] != similarity_matrix.shape[1]:
        raise ValueError("similarity_matrix must be square.")

    if similarity_matrix.dtype.kind not in "iuf":
        raise ValueError("Similarity scores must be real numbers.")

    product_ids = load_product_ids()
    number_of_products = len(product_ids)

    if similarity_matrix.shape != (
        number_of_products,
        number_of_products,
    ):
        raise ValueError(
            "Similarity matrix dimensions do not match "
            "the number of product IDs."
        )

    id_to_index = {
        current_id: index
        for index, current_id in enumerate(product_ids)
    }

    if product_id not in id_to_index:
        raise ValueError(f"Unknown product ID: {product_id}")

    source_index = id_to_index[product_id]
    scores = similarity_matrix[source_index]

    if not np.isfinite(scores).all():
        raise ValueError("Similarity scores contain NaN or infinity.")

    # Exclude the requested product before sorting.
    candidate_indices = [
        index
        for index in range(number_of_products)
        if index != source_index
    ]

    # Sort without changing the supplied similarity matrix.
    candidate_indices.sort(
        key=lambda index: (-float(scores[index]), index)
    )

    # Slicing safely handles requests larger than the available count.
    selected_indices = candidate_indices[:int(num_recommendations)]

    return [
        int(product_ids[index])
        for index in selected_indices
    ]