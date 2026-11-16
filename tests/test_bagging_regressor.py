"""Tests for the bagging regressor (decision stumps + bootstrap averaging)."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bagging_regressor.model import BaggingRegressor, DecisionStumpRegressor


def make_sine(seed=42, n=200, noise=0.3):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 2 * np.pi, size=n)
    y = np.sin(x) + rng.normal(0, noise, size=n)
    return x.reshape(-1, 1), y


def mse(y_true, y_pred):
    return float(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2))


def test_stump_fits_step_function():
    """A single stump should recover a clean step function almost exactly."""
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 10, size=(100, 1))
    y = np.where(X[:, 0] < 5, 0.0, 10.0)
    stump = DecisionStumpRegressor().fit(X, y)
    assert mse(y, stump.predict(X)) < 0.01
    assert stump.feature_index_ == 0
    assert abs(stump.threshold_ - 5.0) < 1.0
    assert abs(stump.left_value_ - 0.0) < 0.1
    assert abs(stump.right_value_ - 10.0) < 0.1


def test_bagging_beats_single_stump_on_noisy_sine():
    X, y = make_sine()
    split = int(0.8 * len(y))
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]
    stump_mse = mse(y_test, DecisionStumpRegressor().fit(X_train, y_train).predict(X_test))
    bagged_mse = mse(
        y_test,
        BaggingRegressor(n_estimators=50, random_state=42).fit(X_train, y_train).predict(X_test),
    )
    assert bagged_mse < stump_mse


def test_predict_output_shape():
    X, y = make_sine()
    model = BaggingRegressor(n_estimators=10, random_state=0).fit(X, y)
    preds = model.predict(X)
    assert preds.shape == (len(y),)
    stump = DecisionStumpRegressor().fit(X, y)
    assert stump.predict(X).shape == (len(y),)


def test_estimators_length_matches_n_estimators():
    X, y = make_sine()
    for n in (1, 5, 25):
        model = BaggingRegressor(n_estimators=n, random_state=0).fit(X, y)
        assert len(model.estimators_) == n
        assert model.n_estimators == n


def test_different_random_state_gives_different_predictions():
    X, y = make_sine()
    a = BaggingRegressor(n_estimators=10, random_state=1).fit(X, y).predict(X)
    b = BaggingRegressor(n_estimators=10, random_state=2).fit(X, y).predict(X)
    assert not np.allclose(a, b)  # different bootstrap draws -> different fit
    assert np.all(np.isfinite(a)) and np.all(np.isfinite(b))


def test_predict_before_fit_raises():
    X, _ = make_sine()
    with pytest.raises(ValueError):
        BaggingRegressor().predict(X)
    with pytest.raises(ValueError):
        DecisionStumpRegressor().predict(X)


def test_invalid_n_estimators_raises():
    with pytest.raises(ValueError):
        BaggingRegressor(n_estimators=0)
    with pytest.raises(ValueError):
        BaggingRegressor(n_estimators=-5)
    with pytest.raises(ValueError):
        BaggingRegressor(n_estimators=2.5)


def test_mismatched_xy_raises():
    X, y = make_sine()
    with pytest.raises(ValueError):
        BaggingRegressor().fit(X, y[:-1])
    with pytest.raises(ValueError):
        DecisionStumpRegressor().fit(X[:10], y)


def test_bagging_with_max_features_works():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(100, 4))
    y = X[:, 0] + 2 * X[:, 1] + rng.normal(0, 0.1, size=100)
    model = BaggingRegressor(n_estimators=20, random_state=0, max_features=1).fit(X, y)
    preds = model.predict(X)
    assert preds.shape == (100,)
    assert np.all(np.isfinite(preds))
    assert len(model.estimators_) == 20
    # A 1-feature stump ensemble should still beat predicting the mean.
    assert mse(y, preds) < float(np.var(y))
