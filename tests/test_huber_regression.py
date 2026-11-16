"""Tests for huber_regression."""

import numpy as np
import pytest

from huber_regression import HuberRegression, huber_loss, ols_fit


def make_data(seed=0, n=100, slope=3.0, intercept=2.0, noise=0.5):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 10, n)
    y = intercept + slope * x + rng.normal(0, noise, n)
    return x.reshape(-1, 1), y, rng


def corrupt(y, rng, frac=0.15, size=15.0):
    y = y.copy()
    idx = rng.choice(len(y), size=int(frac * len(y)), replace=False)
    y[idx] += rng.choice([-size, size], size=len(idx))
    return y, idx


def test_recovers_true_params_on_clean_data():
    X, y, _ = make_data(seed=1)
    model = HuberRegression().fit(X, y)
    assert abs(model.coef_[0] - 3.0) < 0.2
    assert abs(model.intercept_ - 2.0) < 0.2


def test_beats_ols_with_gross_outliers():
    X, y, rng = make_data(seed=2)
    y_bad, _ = corrupt(y, rng, frac=0.15)
    huber = HuberRegression().fit(X, y_bad)
    ols_coef, ols_intercept = ols_fit(X, y_bad)
    huber_err = abs(huber.coef_[0] - 3.0)
    ols_err = abs(ols_coef[0] - 3.0)
    assert huber_err < ols_err


def test_outliers_get_lower_weights_than_inliers():
    X, y, rng = make_data(seed=3)
    y_bad, bad_idx = corrupt(y, rng, frac=0.15)
    model = HuberRegression().fit(X, y_bad)
    good_idx = np.setdiff1d(np.arange(len(y)), bad_idx)
    assert np.all(model.weights_[bad_idx] < 1.0)
    # Inliers keep ~1 weight on average (a few clean points with large
    # Gaussian noise may legitimately be down-weighted below 1).
    assert model.weights_[good_idx].mean() > 0.9
    assert model.weights_[bad_idx].mean() < model.weights_[good_idx].mean()


def test_predict_before_fit_raises():
    X, _, _ = make_data(seed=4)
    with pytest.raises(ValueError, match="not fitted"):
        HuberRegression().predict(X)


def test_invalid_delta_and_n_iter_raise():
    with pytest.raises(ValueError, match="delta"):
        HuberRegression(delta=0)
    with pytest.raises(ValueError, match="delta"):
        HuberRegression(delta=-1.35)
    with pytest.raises(ValueError, match="n_iter"):
        HuberRegression(n_iter=0)
    with pytest.raises(ValueError, match="n_iter"):
        HuberRegression(n_iter=2.5)


def test_mismatched_lengths_raise():
    X, y, _ = make_data(seed=5)
    with pytest.raises(ValueError, match="mismatched"):
        HuberRegression().fit(X, y[:-1])
    with pytest.raises(ValueError, match="mismatched"):
        ols_fit(X, y[:-1])


def test_predict_output_shape():
    X, y, _ = make_data(seed=6)
    model = HuberRegression().fit(X, y)
    preds = model.predict(X)
    assert preds.shape == (len(y),)
    assert np.isfinite(preds).all()


def test_huber_loss_quadratic_near_zero_linear_far():
    r_small = np.array([0.0, 0.5, 1.0])
    assert np.allclose(huber_loss(r_small, delta=1.35), 0.5 * r_small ** 2)
    r_large = np.array([5.0, 10.0, 20.0])
    expected = 1.35 * (np.abs(r_large) - 0.5 * 1.35)
    assert np.allclose(huber_loss(r_large, delta=1.35), expected)
    # Loss is continuous at the kink.
    assert abs(huber_loss(1.35, delta=1.35) - 0.5 * 1.35 ** 2) < 1e-12


def test_1d_x_raises():
    X, y, _ = make_data(seed=7)
    with pytest.raises(ValueError, match="2D"):
        HuberRegression().fit(X.ravel(), y)
