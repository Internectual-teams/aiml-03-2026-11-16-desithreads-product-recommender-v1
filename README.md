# DesiThreads Product Recommender v1

**Team AIML-03** · Internectual Project Team · Sprint 2026-11-16 to 2026-11-22

## Client: DesiThreads

DesiThreads is a popular online retailer for Indian ethnic wear. They've noticed that customers often browse a few items and then leave without purchasing, potentially due to not finding enough appealing options. To improve user engagement and conversion, they want to implement a 'You might also like' section on their product pages, starting with a basic similarity-based recommender.

**Sprint goal:** To build a basic product recommendation system that suggests similar items to users based on product attributes.

## Team and parts

- **Affan Qureshi**: Data Preprocessing & Integration (integration owner this week)

- **rachagulla sujith**: Feature Extraction & Similarity

- **Ajithkumar**: Recommendation API

## Contract between the parts

```markdown
# Repository Structure

```
internectual-aiml-team-03-sprint-1/
├── data_preprocessing/
│   ├── raw_products.csv (Input provided)
│   └── preprocess_data.py
│   └── cleaned_products.csv (Output)
├── feature_extraction_and_similarity/
│   ├── extract_features.py
│   ├── compute_similarity.py
│   ├── product_features.npy (Output)
│   └── similarity_matrix.npy (Output)
├── recommendation_api/
│   ├── recommendation_logic.py
│   ├── app.py
│   └── requirements.txt
├── README.md
└── run_all.sh
```

# Interfaces

## 1. Data Preprocessing to Feature Extraction

*   **Input to `data_preprocessing/preprocess_data.py`**: `raw_products.csv`
    ```csv
    product_id,name,category,color,material,price,description
    101,"Chanderi Silk Saree",Saree,Red,Silk,2500,"Elegant red saree"
    102,"Embroidered Kurta Set",Kurta,Blue,Cotton,1800,"Comfortable blue kurta"
    # ... more products
    ```
*   **Output from `data_preprocessing/preprocess_data.py`**: `cleaned_products.csv`
    ```csv
    product_id,name,category,color,material
    101,"Chanderi Silk Saree",Saree,Red,Silk
    102,"Embroidered Kurta Set",Kurta,Blue,Cotton
    # ... more products
    ```

## 2. Feature Extraction to Similarity Computation

*   **Input to `feature_extraction_and_similarity/extract_features.py`**: `cleaned_products.csv`
*   **Output from `feature_extraction_and_similarity/extract_features.py`**: `product_features.npy`
    *   A NumPy array where each row represents the numerical feature vector for a product, in the same order as `cleaned_products.csv`. 
    *   Shape: `(num_products, num_features)`.

## 3. Similarity Computation to Recommendation API

*   **Input to `feature_extraction_and_similarity/compute_similarity.py`**: `product_features.npy`
*   **Output from `feature_extraction_and_similarity/compute_similarity.py`**: `similarity_matrix.npy`
    *   A NumPy array representing the pairwise similarity between all products.
    *   Shape: `(num_products, num_products)`. `similarity_matrix[i, j]` is the similarity between product `i` and product `j`.

## 4. Recommendation API

*   **Input to `recommendation_api/recommendation_logic.py`**: Loads `product_features.npy` and `similarity_matrix.npy` internally.
*   **Function Signature**: 
    ```python
    def get_recommendations(product_id: int, num_recommendations: int = 5) -> list[int]:
        # Returns a list of recommended product IDs
        pass
    ```
*   **API Endpoint (Flask/FastAPI in `recommendation_api/app.py`)**:
    *   `GET /recommend/{product_id}`
    *   **Request Example (path parameter)**: `/recommend/101`
    *   **Response Example**: `HTTP 200 OK`
        ```json
        {
          "recommended_product_ids": [105, 103, 108, 112, 107]
        }
        ```

# How to Run the Product

1.  **Set up virtual environment and install dependencies**:
    *   `python -m venv venv`
    *   `source venv/bin/activate`
    *   `pip install -r recommendation_api/requirements.txt`
2.  **Generate `cleaned_products.csv`**:
    *   `python data_preprocessing/preprocess_data.py`
3.  **Generate `product_features.npy`**:
    *   `python feature_extraction_and_similarity/extract_features.py`
4.  **Generate `similarity_matrix.npy`**:
    *   `python feature_extraction_and_similarity/compute_similarity.py`
5.  **Start the Recommendation API**:
    *   `python recommendation_api/app.py` (or `flask run` if using Flask)
6.  **Test the API**:
    *   Open your browser or use `curl` to visit `http://127.0.0.1:5000/recommend/101` (assuming Flask default port 5000).
```

## How we work

1. Create a branch named after your ticket key (for example `aiml-07-w5-02`). Never push directly to `main`.

2. Build your ticket in your part's folder. Commit often with clear messages.

3. Open a pull request into `main` and submit the PR link on your ticket in the Internectual workspace.

4. Review one teammate's pull request every week (comment and approve).

5. The integration owner merges approved pull requests, makes everything run together, updates this README with run steps and the live demo link.



_This repository is private during the sprint and becomes public afterwards so you can show it in your portfolio._