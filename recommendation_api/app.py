import os
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, request

# Support both direct execution and package execution.
if __package__:
    from .recommendation_logic import (
        get_recommendations,
        load_product_ids,
    )
else:
    from recommendation_logic import (
        get_recommendations,
        load_product_ids,
    )


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIRECTORY = PROJECT_ROOT / "feature_extraction_and_similarity"


def create_app() -> Flask:
    app = Flask(__name__)

    # Load these files once during application startup.
    product_features = np.load(
        DATA_DIRECTORY / "product_features.npy",
        allow_pickle=False,
    )
    similarity_matrix = np.load(
        DATA_DIRECTORY / "similarity_matrix.npy",
        allow_pickle=False,
    )
    product_ids = load_product_ids()
    product_id_set = set(product_ids)

    number_of_products = len(product_ids)

    # Validate the inputs before accepting requests.
    if product_features.ndim != 2:
        raise ValueError("Product features must be a two-dimensional array.")

    if product_features.shape[0] != number_of_products:
        raise ValueError("Feature rows do not match the product IDs.")

    if similarity_matrix.shape != (
        number_of_products,
        number_of_products,
    ):
        raise ValueError("Similarity matrix does not match the product IDs.")

    if not np.isfinite(product_features).all():
        raise ValueError("Product features contain invalid values.")

    if not np.isfinite(similarity_matrix).all():
        raise ValueError("Similarity matrix contains invalid values.")

    # Retain the loaded arrays for the application's lifetime.
    app.config["PRODUCT_FEATURES"] = product_features
    app.config["SIMILARITY_MATRIX"] = similarity_matrix

    if os.getenv("DESITHREADS_SAMPLE_MODE") == "1":
        app.logger.warning(
            "Sample mode enabled: using product IDs 101, 102, 103."
        )

    @app.get("/recommend/<int:product_id>")
    def recommend(product_id: int):
        if product_id not in product_id_set:
            return jsonify(
                error=f"Unknown product ID: {product_id}"
            ), 404

        # For small datasets, recommend all available alternatives,
        # up to five. Explicit requests must fit the available count.
        default_count = min(5, number_of_products - 1)
        raw_count = request.args.get("num_recommendations")

        if raw_count is None:
            count = default_count
        else:
            try:
                count = int(raw_count)
            except ValueError:
                return jsonify(
                    error="num_recommendations must be an integer."
                ), 400

        if not 0 <= count < number_of_products:
            return jsonify(
                error=(
                    "num_recommendations must be between "
                    f"0 and {number_of_products - 1}."
                )
            ), 400

        recommendations = get_recommendations(
            product_id=product_id,
            similarity_matrix=similarity_matrix,
            num_recommendations=count,
        )

        return jsonify(
            recommended_product_ids=recommendations
        ), 200

    @app.errorhandler(404)
    def not_found(error):
        return jsonify(error="Endpoint not found."), 404

    return app


if __name__ == "__main__":
    application = create_app()
    application.run(
        host="127.0.0.1",
        port=5000,
        debug=False,
    )