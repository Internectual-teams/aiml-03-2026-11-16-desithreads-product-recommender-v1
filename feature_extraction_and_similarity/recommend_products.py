"""AIML-0007-W1-03: Generate Top-N Product Recommendations
-------------------------------------------------------
Client: DesiThreads (e-commerce ethnic wear)
Module: Feature Extraction & Similarity
Sprint Goal: Build a basic product recommendation system that suggests similar items based on product attributes.

Acceptance Criteria:
1. get_top_n_recommendations(product_id, n=5) returns a list of n product IDs.
2. all_product_recommendations is generated (keys = product IDs, values = top 5 recommended IDs).
3. top_n_recommendations.csv is saved with columns: product_id, recommended_product_id_1..N.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 1. Dataset & Similarity Matrix Preparation
# ---------------------------------------------------------------------------
def load_or_create_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Loads existing product catalog and similarity matrix if available,
    or generates a realistic DesiThreads catalog with computed cosine similarities.
    Ensures at least 6 products exist so top-5 recommendations can be made.
    """
    # 1. Locate and load product catalog
    possible_catalog_paths = [
        "products.csv",
        "cleaned_products.csv",
        "data_preprocessing/cleaned_products.csv",
        "datasets/products.csv",
    ]
    catalog_path = next((p for p in possible_catalog_paths if os.path.exists(p)), None)

    if catalog_path:
        products_df = pd.read_csv(catalog_path)
    else:
        # Authentic DesiThreads ethnic wear catalog (10 items)
        products_data = [
            {"product_id": 101, "name": "Chanderi Silk Saree", "category": "Saree", "color": "Red", "material": "Silk"},
            {"product_id": 102, "name": "Embroidered Kurta Set", "category": "Kurta", "color": "Blue", "material": "Cotton"},
            {"product_id": 103, "name": "Cotton Saree", "category": "Saree", "color": "Green", "material": "Cotton"},
            {"product_id": 104, "name": "Banarasi Silk Saree", "category": "Saree", "color": "Red", "material": "Silk"},
            {"product_id": 105, "name": "Printed Cotton Kurta", "category": "Kurta", "color": "Blue", "material": "Cotton"},
            {"product_id": 106, "name": "Silk Blend Kurta", "category": "Kurta", "color": "Yellow", "material": "Silk"},
            {"product_id": 107, "name": "Georgette Anarkali Suit", "category": "Suit", "color": "Pink", "material": "Georgette"},
            {"product_id": 108, "name": "Silk Lehenga Choli", "category": "Lehenga", "color": "Red", "material": "Silk"},
            {"product_id": 109, "name": "Cotton Casual Kurta", "category": "Kurta", "color": "White", "material": "Cotton"},
            {"product_id": 110, "name": "Kanjeevaram Silk Saree", "category": "Saree", "color": "Gold", "material": "Silk"},
        ]
        products_df = pd.DataFrame(products_data)

    product_ids = products_df["product_id"].tolist()

    # 2. Locate and load similarity matrix if compatible with catalog size
    possible_sim_paths = [
        "similarity_matrix.csv",
        "similarity_matrix.npy",
        "feature_extraction_and_similarity/similarity_matrix.npy",
    ]
    sim_path = next((p for p in possible_sim_paths if os.path.exists(p)), None)

    sim_df = None
    if sim_path and sim_path.endswith(".csv"):
        sim_raw = pd.read_csv(sim_path)
        if sim_raw.columns[0] in ["product_id", "Unnamed: 0"] or not np.issubdtype(sim_raw.dtypes.iloc[0], np.number):
            sim_df = pd.read_csv(sim_path, index_col=0)
        elif len(sim_raw) == len(product_ids):
            sim_df = sim_raw
            sim_df.index = product_ids
            sim_df.columns = product_ids
    elif sim_path and sim_path.endswith(".npy"):
        sim_arr = np.load(sim_path)
        if sim_arr.shape == (len(product_ids), len(product_ids)):
            sim_df = pd.DataFrame(sim_arr, index=product_ids, columns=product_ids)

    # If similarity matrix not found or incompatible in dimension, compute it
    if sim_df is None or sim_df.shape[0] < 6:
        cat_cols = [c for c in ["category", "color", "material"] if c in products_df.columns]
        encoded = pd.get_dummies(products_df[cat_cols], columns=cat_cols, dtype=np.float32)
        feat_matrix = encoded.to_numpy(dtype=np.float32)

        norms = np.linalg.norm(feat_matrix, axis=1, keepdims=True)
        safe_norms = np.where(norms == 0.0, 1.0, norms)
        normalized = feat_matrix / safe_norms
        sim_matrix = np.dot(normalized, normalized.T)
        sim_matrix = np.clip(sim_matrix, 0.0, 1.0)
        np.fill_diagonal(sim_matrix, 1.0)

        sim_df = pd.DataFrame(sim_matrix, index=product_ids, columns=product_ids)

    return products_df, sim_df


# Global catalog and similarity DataFrame (Hint 1)
products_df, similarity_df = load_or_create_inputs()


# ---------------------------------------------------------------------------
# Acceptance Criterion 1: get_top_n_recommendations(product_id, n=5)
# ---------------------------------------------------------------------------
def get_top_n_recommendations(
    product_id: Any,
    n: int = 5,
    sim_data: pd.DataFrame | None = None,
) -> list[Any]:
    """Finds top N most similar products for a given product_id.
    Excludes the product itself from its own recommendation list.

    Args:
        product_id: Target product ID to generate recommendations for.
        n: Number of recommendations to return (default: 5).
        sim_data: Optional similarity DataFrame (defaults to global similarity_df).

    Returns:
        List of n recommended product IDs sorted in descending order of similarity.

    Raises:
        KeyError: If product_id is not found in the similarity matrix.
    """
    df_sim = sim_data if sim_data is not None else similarity_df

    # Match product_id safely handling int/str variations
    target_id = product_id
    if target_id not in df_sim.index:
        for idx in df_sim.index:
            if str(idx) == str(product_id):
                target_id = idx
                break
        else:
            raise KeyError(f"Product ID '{product_id}' not found in similarity matrix.")

    # Hint 2: Identify corresponding row for the product
    product_scores = df_sim.loc[target_id]

    # Hint 4: Exclude the original product's own index
    other_product_scores = product_scores[product_scores.index != target_id]

    # Hint 3: Sort similarity scores in descending order
    sorted_scores = other_product_scores.sort_values(ascending=False)

    # Hint 5: Select top N product IDs
    top_n_product_ids = sorted_scores.head(n).index.tolist()

    return top_n_product_ids


# ---------------------------------------------------------------------------
# Acceptance Criterion 2: all_product_recommendations dictionary / DataFrame
# ---------------------------------------------------------------------------
all_product_recommendations: dict[Any, list[Any]] = {
    pid: get_top_n_recommendations(pid, n=5)
    for pid in similarity_df.index
}


# ---------------------------------------------------------------------------
# Acceptance Criterion 3: Save to top_n_recommendations.csv
# ---------------------------------------------------------------------------
def save_recommendations_csv(
    recommendations_dict: dict[Any, list[Any]],
    output_filename: str = "top_n_recommendations.csv",
    n: int = 5,
) -> pd.DataFrame:
    """Formats recommendations into structured columns and saves to CSV."""
    rows = []
    column_names = ["product_id"] + [f"recommended_product_id_{i}" for i in range(1, n + 1)]

    for pid, rec_list in recommendations_dict.items():
        row = {"product_id": pid}
        for i in range(1, n + 1):
            row[f"recommended_product_id_{i}"] = rec_list[i - 1] if i - 1 < len(rec_list) else None
        rows.append(row)

    out_df = pd.DataFrame(rows, columns=column_names)
    out_df.to_csv(output_filename, index=False)
    return out_df


# Generate and save top_n_recommendations.csv
recommendations_table = save_recommendations_csv(all_product_recommendations, "top_n_recommendations.csv", n=5)


# ---------------------------------------------------------------------------
# Demonstration & Verification Printout
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 65)
    print(" AIML-0007-W1-03: TOP-N RECOMMENDATIONS VERIFICATION")
    print("=" * 65)
    print(f"Total Products in Catalog : {len(similarity_df)}")
    print(f"Similarity Matrix Shape   : {similarity_df.shape}")

    # Test Criterion 1
    sample_pid = similarity_df.index[0]
    sample_recs = get_top_n_recommendations(sample_pid, n=5)
    print(f"\nCriterion 1 Check: get_top_n_recommendations({sample_pid}, n=5)")
    print(f"  Recommended Product IDs : {sample_recs}")
    print(f"  Returns list of n items : {isinstance(sample_recs, list) and len(sample_recs) == 5}")
    print(f"  Self-recommendation excluded: {sample_pid not in sample_recs}")

    # Test Criterion 2
    print(f"\nCriterion 2 Check: all_product_recommendations")
    print(f"  Generated Type          : {type(all_product_recommendations).__name__}")
    print(f"  Number of Products Mapped: {len(all_product_recommendations)}")
    for pid in list(all_product_recommendations.keys())[:3]:
        print(f"    Product {pid} -> Top 5: {all_product_recommendations[pid]}")

    # Test Criterion 3
    print(f"\nCriterion 3 Check: top_n_recommendations.csv")
    print(f"  File Exists             : {os.path.exists('top_n_recommendations.csv')}")
    print(f"  Columns                 : {list(recommendations_table.columns)}")
    print(f"  Table Shape             : {recommendations_table.shape}")
    print("\nPreview of top_n_recommendations.csv:")
    print(recommendations_table.head())
    print("=" * 65)
