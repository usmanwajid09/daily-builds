"""Tests for the from-scratch KNN regressor."""
import numpy as np
import pytest

from knn_regressor import KNNRegressor


@pytest.fixture
def small():
    # 1D points 0..4 with targets 0, 10, 20, 30, 40.
    x = np.array([[0.0], [1.0], [2.0], [3.0], [4.0]])
    y = np.array([0.0, 10.0, 20.0, 30.0, 40.0])
    return x, y


def test_k1_interpolates_training_points(small):
    x, y = small
    model = KNNRegressor(k=1).fit(x, y)
    np.testing.assert_allclose(model.predict(x), y)


def test_uniform_prediction_is_mean_of_k_nearest(small):
    x, y = small
    model = KNNRegressor(k=3, weights="uniform").fit(x, y)
    # Query 1.1: nearest are x=1 (10), x=2 (20), x=0 (0) -> mean 10.
    np.testing.assert_allclose(model.predict([[1.1]]), [10.0])
    # Query 2.9: nearest are x=3 (30), x=2 (20), x=4 (40) -> mean 30.
    np.testing.assert_allclose(model.predict([[2.9]]), [30.0])


def test_distance_weighting_matches_manual_computation(small):
    x, y = small
    model = KNNRegressor(k=3, weights="distance").fit(x, y)
    # Query 1.1: neighbours x=1 (d=0.1, y=10), x=2 (d=0.9, y=20),
    # x=0 (d=1.1, y=0). Weighted mean = sum(w*y)/sum(w).
    weights = np.array([1 / 0.1, 1 / 0.9, 1 / 1.1])
    targets = np.array([10.0, 20.0, 0.0])
    expected = float(np.sum(weights * targets) / np.sum(weights))
    got = float(model.predict([[1.1]])[0])
    assert got == pytest.approx(expected)
    # The nearer point (y=10) must dominate: result is much closer to 10
    # than to the uniform mean of 10.
    assert abs(got - 10.0) < abs(got - 20.0)
    assert abs(got - 10.0) < abs(got - 0.0)


def test_predict_shape_1d_and_2d_input(small):
    x, y = small
    model = KNNRegressor(k=2).fit(x, y)
    single = model.predict([1.5])
    assert single.shape == (1,)
    assert single.dtype == float
    batch = model.predict([[1.5], [2.5], [3.5]])
    assert batch.shape == (3,)


def test_score_is_one_on_perfect_fit(small):
    x, y = small
    model = KNNRegressor(k=1).fit(x, y)
    assert model.score(x, y) == pytest.approx(1.0)


def test_score_constant_y_exact_predictions_is_one():
    x = np.array([[0.0], [1.0], [2.0]])
    y = np.array([5.0, 5.0, 5.0])
    model = KNNRegressor(k=1).fit(x, y)
    assert model.score(x, y) == pytest.approx(1.0)


def test_predict_before_fit_raises():
    with pytest.raises(RuntimeError):
        KNNRegressor().predict([[1.0]])


def test_rejects_k_zero():
    with pytest.raises(ValueError):
        KNNRegressor(k=0)


def test_rejects_k_larger_than_n_samples(small):
    x, y = small
    with pytest.raises(ValueError):
        KNNRegressor(k=6).fit(x, y)


def test_rejects_bad_weights():
    with pytest.raises(ValueError):
        KNNRegressor(weights="gaussian")


def test_rejects_mismatched_lengths():
    x = np.array([[0.0], [1.0], [2.0]])
    y = np.array([1.0, 2.0])
    with pytest.raises(ValueError):
        KNNRegressor().fit(x, y)


def test_rejects_nan_in_x_or_y():
    x = np.array([[0.0], [np.nan], [2.0]])
    y = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        KNNRegressor().fit(x, y)
    x_ok = np.array([[0.0], [1.0], [2.0]])
    y_nan = np.array([1.0, np.nan, 3.0])
    with pytest.raises(ValueError):
        KNNRegressor().fit(x_ok, y_nan)


def test_rejects_wrong_feature_count_at_predict(small):
    x, y = small
    model = KNNRegressor(k=2).fit(x, y)
    with pytest.raises(ValueError):
        model.predict([[1.0, 2.0]])


def test_exact_match_distance_zero_returns_that_target(small):
    x, y = small
    model = KNNRegressor(k=3, weights="distance").fit(x, y)
    # Query exactly at x=2 (target 20); the zero-distance target wins.
    np.testing.assert_allclose(model.predict([[2.0]]), [20.0])
    # Uniform weighting is unaffected by exact matches (still a mean).
    uniform = KNNRegressor(k=3, weights="uniform").fit(x, y)
    np.testing.assert_allclose(uniform.predict([[2.0]]), [20.0])
