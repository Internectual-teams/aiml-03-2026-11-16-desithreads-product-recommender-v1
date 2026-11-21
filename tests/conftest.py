import csv

import numpy as np
import pytest

from recommendation_api import recommendation_logic as logic
from recommendation_api.app import create_app


@pytest.fixture
def product_ids():
    # Deliberately different from matrix indices.
    return [101, 205, 309, 412]


@pytest.fixture
def similarity_matrix():
    return np.array(
        [
            [1.0, 0.2, 0.87654, 0.6],
            [0.2, 1.0, 0.4, 0.3],
            [0.87654, 0.4, 1.0, 0.5],
            [0.6, 0.3, 0.5, 1.0],
        ],
        dtype=float,
    )


@pytest.fixture
def configured_logic(monkeypatch, product_ids):
    # Unit tests do not read the real CSV or enable sample mode.
    monkeypatch.setattr(
        logic,
        "load_product_ids",
        lambda: product_ids.copy(),
    )
    return logic


@pytest.fixture
def client(tmp_path, product_ids, similarity_matrix):
    csv_path = tmp_path / "cleaned_products.csv"

    products = [
        {
            "product_id": product_id,
            "name": f"Product {product_id}",
            "category": "Saree",
            "color": "Blue",
        }
        for product_id in product_ids
    ]

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["product_id", "name", "category", "color"],
        )
        writer.writeheader()
        writer.writerows(products)

    # Small fixtures for exercising startup loading, not production data.
    np.save(
        tmp_path / "product_features.npy",
        np.eye(len(product_ids)),
    )
    np.save(
        tmp_path / "similarity_matrix.npy",
        similarity_matrix,
    )

    app = create_app(
        csv_path=csv_path,
        arrays_directory=tmp_path,
    )
    app.config["TESTING"] = True

    with app.test_client() as test_client:
        yield test_client