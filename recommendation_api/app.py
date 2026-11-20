"""
DesiThreads Recommendation API — AIML-0009-W1-03

Example request:
    GET /recommend/101?limit=2

Illustrative response:
    {
        "recommended_product_ids": [103],
        "recommendations": [
            {
                "product_id": 103,
                "name": "Cotton Saree",
                "category": "Saree",
                "color": "Green",
                "similarity_score": 0.333
            }
        ]
    }

Behavior:
- Default limit: 5.
- Valid limits: integers from 1 to 20.
- Unknown product: HTTP 404.
- Invalid limit: HTTP 400.
- Excludes the requested product.
- Returns fewer results when fewer products are available.
- Includes zero-similarity candidates; no minimum score is imposed.
- Ties are resolved using CSV row order.
- CSV rows must match the array row order.

Local commands:
    uv run python recommendation_api/app.py --create-sample
    uv run python recommendation_api/app.py --sample
    uv run python recommendation_api/app.py
"""

import argparse
import csv
import re
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, request


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_DIRECTORY = PROJECT_ROOT / "sample_data"
REQUIRED_COLUMNS = {"product_id", "name", "category", "color"}


def create_sample_data() -> None:
    """Create 20 fictional products and matching arrays for local testing."""

    categories = ["Saree", "Kurta", "Lehenga", "Dupatta"]
    colors = ["Red", "Blue", "Green", "Yellow", "Pink"]
    materials = ["Cotton", "Silk", "Linen"]

    products = []

    for category_index, category in enumerate(categories):
        for color_index, color in enumerate(colors):
            material = materials[
                (category_index + color_index) % len(materials)
            ]

            products.append(
                {
                    "product_id": 101 + len(products),
                    "name": f"{color} {material} {category}",
                    "category": category,
                    "color": color,
                    "material": material,
                }
            )

    output_paths = [
        SAMPLE_DIRECTORY / "cleaned_products.csv",
        SAMPLE_DIRECTORY / "product_features.npy",
        SAMPLE_DIRECTORY / "similarity_matrix.npy",
    ]

    if any(path.exists() for path in output_paths):
        raise FileExistsError(
            "Sample files already exist. Use --sample to run them. "
            "Move existing sample files elsewhere before regenerating."
        )

    # One-hot encode category, color and material.
    feature_keys = [
        (field, value)
        for field in ("category", "color", "material")
        for value in sorted({product[field] for product in products})
    ]

    features = np.array(
        [
            [
                float(product[field] == value)
                for field, value in feature_keys
            ]
            for product in products
        ],
        dtype=np.float64,
    )

    norms = np.linalg.norm(features, axis=1, keepdims=True)
    normalized_features = features / norms

    similarity = normalized_features @ normalized_features.T
    similarity = np.clip(similarity, 0.0, 1.0)
    np.fill_diagonal(similarity, 1.0)

    SAMPLE_DIRECTORY.mkdir(parents=True, exist_ok=True)

    with output_paths[0].open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "product_id", "name", "category", "color", "material"
            ],
        )
        writer.writeheader()
        writer.writerows(products)

    np.save(output_paths[1], features)
    np.save(output_paths[2], similarity)

    print(f"Created {len(products)} fictional sample products.")
    print(f"Files saved in: {SAMPLE_DIRECTORY}")
    print("Product IDs: 101–120")


def load_products(csv_path: Path) -> list[dict]:
    """Load product metadata, preserving the original CSV order."""

    products = []
    seen_ids = set()

    with csv_path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)

        missing = REQUIRED_COLUMNS - set(reader.fieldnames or [])

        if missing:
            raise ValueError(
                f"Missing CSV columns: {', '.join(sorted(missing))}"
            )

        for row_number, row in enumerate(reader, start=2):
            try:
                product_id = int(row["product_id"])
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Invalid product_id at CSV row {row_number}."
                ) from error

            if product_id in seen_ids:
                raise ValueError(f"Duplicate product_id: {product_id}")

            product = {"product_id": product_id}

            for field in ("name", "category", "color"):
                value = (row.get(field) or "").strip()

                if not value:
                    raise ValueError(
                        f"Missing {field} at CSV row {row_number}."
                    )

                product[field] = value

            products.append(product)
            seen_ids.add(product_id)

    if not products:
        raise ValueError("The product CSV is empty.")

    return products


def rank_product_indices(
    source_index: int,
    similarity_matrix: np.ndarray,
    limit: int,
) -> list[int]:
    """Rank alternatives without modifying the shared matrix."""

    scores = similarity_matrix[source_index]

    candidates = [
        index
        for index in range(len(scores))
        if index != source_index
    ]

    candidates.sort(
        key=lambda index: (-float(scores[index]), index)
    )

    return candidates[:limit]


def create_app(
    csv_path: Path | None = None,
    arrays_directory: Path | None = None,
) -> Flask:
    app = Flask(__name__)

    csv_path = Path(csv_path) if csv_path is not None else (
        PROJECT_ROOT / "data_preprocessing" / "cleaned_products.csv"
    )
    arrays_directory = (
        Path(arrays_directory)
        if arrays_directory is not None
        else PROJECT_ROOT / "feature_extraction_and_similarity"
    )

    # All disk reads happen at application startup.
    products = load_products(csv_path)

    features = np.load(
        arrays_directory / "product_features.npy",
        allow_pickle=False,
    )
    similarity = np.load(
        arrays_directory / "similarity_matrix.npy",
        allow_pickle=False,
    )

    count = len(products)

    if features.ndim != 2 or features.shape[0] != count:
        raise ValueError(
            "Product feature rows must match the CSV product count."
        )

    if similarity.shape != (count, count):
        raise ValueError(
            f"Expected similarity matrix shape {(count, count)}, "
            f"received {similarity.shape}."
        )

    for name, array in (
        ("Product features", features),
        ("Similarity matrix", similarity),
    ):
        if array.dtype.kind not in "iuf":
            raise ValueError(f"{name} must contain real numbers.")

        if not np.isfinite(array).all():
            raise ValueError(f"{name} contains NaN or infinity.")

    id_to_index = {
        product["product_id"]: index
        for index, product in enumerate(products)
    }

    app.config["PRODUCT_FEATURES"] = features
    app.config["SIMILARITY_MATRIX"] = similarity

    @app.get("/")
    def index():
        return jsonify(
            message="DesiThreads Recommendation API",
            example=f"/recommend/{products[0]['product_id']}?limit=5",
        )

    @app.get("/recommend/<product_id>")
    def recommend(product_id: str):
        # Validate limit explicitly: invalid input must not become default 5.
        limits = request.args.getlist("limit")

        if len(limits) > 1:
            return jsonify(
                error="Provide limit only once."
            ), 400

        raw_limit = limits[0] if limits else "5"

        if re.fullmatch(r"(?:[1-9]|1[0-9]|20)", raw_limit) is None:
            return jsonify(
                error="limit must be an integer between 1 and 20."
            ), 400

        limit = int(raw_limit)

        try:
            requested_id = int(product_id)
        except ValueError:
            return jsonify(error="Product not found"), 404

        if requested_id not in id_to_index:
            return jsonify(error="Product not found"), 404

        source_index = id_to_index[requested_id]

        selected_indices = rank_product_indices(
            source_index,
            similarity,
            limit,
        )

        recommendations = [
            {
                **products[index],
                "similarity_score": round(
                    float(similarity[source_index, index]), 3
                ),
            }
            for index in selected_indices
        ]

        return jsonify(
            recommended_product_ids=[
                item["product_id"] for item in recommendations
            ],
            recommendations=recommendations,
        ), 200

    @app.errorhandler(404)
    def endpoint_not_found(error):
        return jsonify(error="Endpoint not found"), 404

    return app


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the DesiThreads recommendation API."
    )

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--create-sample",
        action="store_true",
        help="Create a separate 20-product sample dataset.",
    )
    mode.add_argument(
        "--sample",
        action="store_true",
        help="Run using the generated sample dataset.",
    )

    args = parser.parse_args()

    if args.create_sample:
        create_sample_data()
        return

    if args.sample:
        application = create_app(
            csv_path=SAMPLE_DIRECTORY / "cleaned_products.csv",
            arrays_directory=SAMPLE_DIRECTORY,
        )
        print("Using fictional sample data.")
    else:
        application = create_app()

    application.run(
        host="127.0.0.1",
        port=5000,
        debug=False,
    )


if __name__ == "__main__":
    main()