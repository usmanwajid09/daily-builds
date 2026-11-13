"""Tests for ridge_regression.model.RidgeRegression."""

import numpy as np
import pytest

from ridge_regression.model import RidgeRegression


def make_linear(n=50, seed=0, noise=0.0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 2))
    y = 3.0 * x[:, 0] - 2.0 * x[:, 1] + 5.0 + rng.normal(scale=noise, size=n)
    return x, y


def test_recovery_of_true_coefficients():
    """With no noise, alpha=0 recovers the true coefficients exactly."""
    x, y = make_linear()
    model = RidgeRegression(alpha=0.0).fit(x, y)
    assert model.coef_ == pytest.approx([3.0, -2.0], abs=1e-9)
    assert model.intercept_ == pytest.approx(5.0, abs=1e-9)


def test_r2_is_one_for_perfect_fit():
    x, y = make_linear()
    model = RidgeRegression(alpha=0.0).fit(x, y)
    assert model.score(x, y) == pytest.approx(1.0, abs=1e-12)


def test_alpha_shrinks_coefficients():
    """Larger alpha produces smaller coefficient magnitudes."""
    x, y = make_linear(noise=0.5)
    small = RidgeRegression(alpha=1.0).fit(x, y)
    big = RidgeRegression(alpha=1000.0).fit(x, y)
    assert np.linalg.norm(big.coef_) < np.linalg.norm(small.coef_)
    assert np.linalg.norm(big.coef_) < np.linalg.norm([3.0, -2.0])


def test_predict_single_sample_and_shape():
    x, y = make_linear()
    model = RidgeRegression(alpha=0.0).fit(x, y)
    single = model.predict(x[0])
    assert single.shape == (1,)
    assert single[0] == pytest.approx(3.0 * x[0, 0] - 2.0 * x[0, 1] + 5.0, abs=1e-6)


def test_predict_before_fit_raises():
    with pytest.raises(RuntimeError, match="not fitted"):
        RidgeRegression().predict([[1.0, 2.0]])


def test_score_before_fit_raises():
    with pytest.raises(RuntimeError, match="not fitted"):
        RidgeRegression().score([[1.0, 2.0]], [1.0])


def test_negative_alpha_raises():
    with pytest.raises(ValueError, match="alpha must be >= 0"):
        RidgeRegression(alpha=-1.0)


def test_non_numeric_alpha_raises():
    with pytest.raises(ValueError, match="alpha must be a number"):
        RidgeRegression(alpha="strong")


def test_mismatched_samples_raises():
    x, y = make_linear()
    with pytest.raises(ValueError, match="same number of samples"):
        RidgeRegression().fit(x, y[:-1])


def test_1d_x_raises():
    with pytest.raises(ValueError, match="2D array"):
        RidgeRegression().fit([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])


def test_empty_data_raises():
    with pytest.raises(ValueError, match="empty"):
        RidgeRegression().fit(np.empty((0, 2)), np.empty(0))


def test_nan_input_raises():
    x, y = make_linear()
    x[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        RidgeRegression().fit(x, y)


def test_wrong_feature_count_in_predict_raises():
    x, y = make_linear()
    model = RidgeRegression().fit(x, y)
    with pytest.raises(ValueError, match="features"):
        model.predict([[1.0, 2.0, 3.0]])


def test_score_can_be_negative_for_bad_model():
    """A model fitted on one relationship scores < 0 on an unrelated set."""
    x, y = make_linear()
    model = RidgeRegression(alpha=0.0).fit(x, y)
    rng = np.random.default_rng(7)
    y_other = rng.permutation(y)  # same values, scrambled: predictions are way off
    assert model.score(x, y_other) < 0.0


def test_singular_design_matrix_still_fits():
    """Duplicate columns (singular X^T X) work via lstsq at alpha=0."""
    x = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]])
    y = np.array([2.0, 4.0, 6.0, 8.0])
    model = RidgeRegression(alpha=0.0).fit(x, y)
    assert model.predict(x) == pytest.approx(y, abs=1e-9)
