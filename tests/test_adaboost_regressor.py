"""Tests for adaboost_regressor.model.AdaBoostRegressor."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from adaboost_regressor.model import AdaBoostRegressor, _weighted_median


def _friedman(seed=0, n=300):
    """Friedman #1 data: non-linear, 5 features, noisy target."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 1, size=(n, 5))
    y = (10 * np.sin(np.pi * x[:, 0] * x[:, 1])
         + 20 * (x[:, 2] - 0.5) ** 2
         + 10 * x[:, 3] + 5 * x[:, 4]
         + rng.normal(scale=0.5, size=n))
    return x, y


def _sine(seed=0, n=150):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 1, size=(n, 1))
    y = np.sin(2 * np.pi * x[:, 0]) + rng.normal(scale=0.05, size=n)
    return x, y


def _linear(seed=0, n=100):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 2))
    y = 3.0 * x[:, 0] - 2.0 * x[:, 1] + rng.normal(scale=0.1, size=n)
    return x, y


def test_r2_on_friedman_data():
    x, y = _friedman()
    model = AdaBoostRegressor(n_estimators=100, learning_rate=1.0).fit(x, y)
    assert model.score(x, y) >= 0.6


def test_r2_on_sine_wave():
    # A single-feature smooth target: one stump already captures the
    # coarse step, so the threshold is modest but the fit must be real.
    x, y = _sine()
    model = AdaBoostRegressor(n_estimators=100, learning_rate=1.0).fit(x, y)
    assert model.score(x, y) >= 0.7


def test_r2_on_linear_data():
    x, y = _linear()
    model = AdaBoostRegressor(n_estimators=100, learning_rate=1.0).fit(x, y)
    assert model.score(x, y) >= 0.7


def test_square_loss_runs():
    x, y = _friedman()
    model = AdaBoostRegressor(n_estimators=100, loss="square").fit(x, y)
    assert model.score(x, y) >= 0.6


def test_ensemble_beats_single_stump():
    x, y = _friedman()
    one = AdaBoostRegressor(n_estimators=1).fit(x, y)
    many = AdaBoostRegressor(n_estimators=100).fit(x, y)
    assert many.score(x, y) > one.score(x, y) + 0.1


def test_staged_predict_matches_final():
    x, y = _friedman()
    model = AdaBoostRegressor(n_estimators=20).fit(x, y)
    staged = list(model.staged_predict(x))
    assert len(staged) == len(model.estimators_)
    assert np.allclose(staged[-1], model.predict(x))


def test_staged_predict_improves():
    x, y = _friedman()
    model = AdaBoostRegressor(n_estimators=50).fit(x, y)
    staged = list(model.staged_predict(x))

    def r2(pred):
        ss_res = float(np.sum((y - pred) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        return 1 - ss_res / ss_tot

    assert r2(staged[-1]) > r2(staged[0]) + 0.1


def test_estimator_errors_below_one():
    x, y = _friedman()
    model = AdaBoostRegressor(n_estimators=30).fit(x, y)
    assert np.all(model.estimator_errors_ < 1.0)
    assert np.all(model.estimator_errors_ > 0)
    assert np.all(model.estimator_weights_ > 0)


def test_feature_importances_sum_to_one():
    x, y = _linear()
    model = AdaBoostRegressor(n_estimators=50).fit(x, y)
    assert model.feature_importances_.shape == (2,)
    assert model.feature_importances_.sum() == pytest.approx(1.0)
    assert np.all(model.feature_importances_ >= 0)


def test_weighted_median():
    assert _weighted_median([1.0, 2.0, 3.0], [1.0, 1.0, 1.0]) == 2.0
    assert _weighted_median([1.0, 2.0, 3.0], [0.1, 0.1, 0.8]) == 3.0
    assert _weighted_median([5.0], [1.0]) == 5.0
    # Non-positive total weight falls back to the plain median.
    assert _weighted_median([1.0, 2.0, 3.0], [-1.0, -1.0, 1.0]) == 2.0


def test_predict_before_fit_raises():
    x, _ = _friedman()
    with pytest.raises(ValueError, match="not fitted"):
        AdaBoostRegressor().predict(x)


def test_mismatched_lengths_raises():
    x, y = _friedman()
    with pytest.raises(ValueError, match="different lengths"):
        AdaBoostRegressor().fit(x, y[:-1])


def test_1d_x_raises():
    with pytest.raises(ValueError, match="2-dimensional"):
        AdaBoostRegressor().fit(np.zeros(10), np.zeros(10))


def test_bad_hyperparameters_raise():
    with pytest.raises(ValueError, match="n_estimators"):
        AdaBoostRegressor(n_estimators=0)
    with pytest.raises(ValueError, match="learning_rate"):
        AdaBoostRegressor(learning_rate=0.0)
    with pytest.raises(ValueError, match="loss"):
        AdaBoostRegressor(loss="exponential")


def test_perfect_fit_stops_early():
    x = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0.0, 0.0, 1.0, 1.0])
    model = AdaBoostRegressor(n_estimators=50).fit(x, y)
    assert len(model.estimators_) == 1
    assert np.array_equal(model.predict(x), y)


def test_deterministic():
    x, y = _friedman()
    a = AdaBoostRegressor(n_estimators=30).fit(x, y)
    b = AdaBoostRegressor(n_estimators=30).fit(x, y)
    assert np.array_equal(a.predict(x), b.predict(x))
