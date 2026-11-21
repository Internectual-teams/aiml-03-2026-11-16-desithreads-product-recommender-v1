import numpy as np
import pytest


def test_excludes_input_product(configured_logic, similarity_matrix):
    result = configured_logic.get_recommendations(
        101, similarity_matrix, 3
    )

    assert 101 not in result
    assert len(result) == 3


def test_results_follow_descending_similarity(
    configured_logic, similarity_matrix
):
    result = configured_logic.get_recommendations(
        101, similarity_matrix, 3
    )

    # Scores: 309 -> 0.87654, 412 -> 0.6, 205 -> 0.2
    assert result == [309, 412, 205]


def test_respects_requested_limit(configured_logic, similarity_matrix):
    result = configured_logic.get_recommendations(
        101, similarity_matrix, 2
    )

    assert result == [309, 412]


def test_returns_available_when_request_is_too_large(
    configured_logic, similarity_matrix
):
    result = configured_logic.get_recommendations(
        101, similarity_matrix, 10
    )

    assert result == [309, 412, 205]


def test_returns_unique_python_integer_ids(
    configured_logic, similarity_matrix
):
    result = configured_logic.get_recommendations(
        101, similarity_matrix, 3
    )

    assert isinstance(result, list)
    assert len(result) == len(set(result))
    assert all(type(product_id) is int for product_id in result)


def test_ties_keep_original_product_order(
    configured_logic, similarity_matrix
):
    matrix = similarity_matrix.copy()
    matrix[0, 1:] = 0.5

    result = configured_logic.get_recommendations(101, matrix, 3)

    assert result == [205, 309, 412]


def test_unknown_product_raises_error(
    configured_logic, similarity_matrix
):
    with pytest.raises(ValueError, match="Unknown product"):
        configured_logic.get_recommendations(
            999, similarity_matrix, 2
        )


def test_single_product_returns_empty_list(
    configured_logic, monkeypatch
):
    monkeypatch.setattr(
        configured_logic,
        "load_product_ids",
        lambda: [101],
    )

    result = configured_logic.get_recommendations(
        101, np.array([[1.0]]), 5
    )

    assert result == []


def test_does_not_modify_similarity_matrix(
    configured_logic, similarity_matrix
):
    original = similarity_matrix.copy()

    configured_logic.get_recommendations(
        101, similarity_matrix, 2
    )

    np.testing.assert_array_equal(similarity_matrix, original)