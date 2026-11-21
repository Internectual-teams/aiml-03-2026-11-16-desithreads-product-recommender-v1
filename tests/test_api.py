import pytest


def test_success_returns_ids_and_product_details(client):
    response = client.get("/recommend/101?limit=2")

    assert response.status_code == 200
    assert response.is_json

    data = response.get_json()

    assert data["recommended_product_ids"] == [309, 412]
    assert data["recommendations"] == [
        {
            "product_id": 309,
            "name": "Product 309",
            "category": "Saree",
            "color": "Blue",
            "similarity_score": 0.877,
        },
        {
            "product_id": 412,
            "name": "Product 412",
            "category": "Saree",
            "color": "Blue",
            "similarity_score": 0.6,
        },
    ]


def test_unknown_product_returns_404(client):
    response = client.get("/recommend/999")

    assert response.status_code == 404
    assert response.get_json() == {"error": "Product not found"}


@pytest.mark.parametrize(
    "invalid_limit",
    ["abc", "0", "-1", "21", "2.5", ""],
)
def test_invalid_limit_returns_400(client, invalid_limit):
    response = client.get(
        "/recommend/101",
        query_string={"limit": invalid_limit},
    )

    assert response.status_code == 400
    assert "error" in response.get_json()


def test_returns_fewer_products_without_error(client):
    response = client.get("/recommend/101?limit=20")

    assert response.status_code == 200

    data = response.get_json()
    assert data["recommended_product_ids"] == [309, 412, 205]
    assert len(data["recommendations"]) == 3


def test_missing_limit_uses_default_without_error(client):
    response = client.get("/recommend/101")

    assert response.status_code == 200
    # Default is five, but only three alternatives are available.
    assert response.get_json()["recommended_product_ids"] == [
        309, 412, 205
    ]


def test_limit_one_returns_only_best_match(client):
    response = client.get("/recommend/101?limit=1")

    assert response.status_code == 200
    assert response.get_json()["recommended_product_ids"] == [309]


def test_duplicate_limits_return_400(client):
    response = client.get("/recommend/101?limit=1&limit=2")

    assert response.status_code == 400