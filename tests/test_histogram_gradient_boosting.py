"""Tests for histogram_gradient_boosting.HistogramGradientBoostingRegressor."""
import numpy as np
import pytest

from histogram_gradient_boosting.model import HistogramGradientBoostingRegressor


def make_data(seed=0, n=300, noise=0.1):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-3, 3, size=(n, 2))
    y = np.sin(x[:, 0]) + 0.5 * x[:, 1] + rng.normal(0, noise, size=n)
    return x, y


def test_r2_above_0_9():
    x, y = make_data(seed=0)
    model = HistogramGradientBoostingRegressor(
        n_estimators=200, max_bins=32, min_samples_leaf=10,
        random_state=0).fit(x[:240], y[:240])
    assert model.score(x[240:], y[240:]) > 0.9


def test_train_mse_decreases():
    x, y = make_data(seed=1)
    model = HistogramGradientBoostingRegressor(
        n_estimators=50, max_bins=32, min_samples_leaf=10,
        random_state=0).fit(x, y)
    scores = model.train_score_
    assert len(scores) == 50
    assert scores[-1] < scores[0]
    assert all(scores[i + 1] <= scores[i] + 1e-12 for i in range(len(scores) - 1))


def test_predict_shape_and_finite():
    x, y = make_data(seed=2)
    model = HistogramGradientBoostingRegressor(
        n_estimators=20, max_bins=16, min_samples_leaf=5).fit(x, y)
    pred = model.predict(x[:5])
    assert pred.shape == (5,)
    assert np.all(np.isfinite(pred))


def test_few_bins_still_works():
    x, y = make_data(seed=3)
    model = HistogramGradientBoostingRegressor(
        n_estimators=100, max_bins=4, min_samples_leaf=10,
        random_state=0).fit(x[:240], y[:240])
    assert model.score(x[240:], y[240:]) > 0.8


def test_many_bins_matches_exactish():
    x, y = make_data(seed=4)
    few = HistogramGradientBoostingRegressor(
        n_estimators=100, max_bins=8, min_samples_leaf=10,
        random_state=0).fit(x[:240], y[:240])
    many = HistogramGradientBoostingRegressor(
        n_estimators=100, max_bins=255, min_samples_leaf=10,
        random_state=0).fit(x[:240], y[:240])
    assert many.score(x[240:], y[240:]) >= few.score(x[240:], y[240:]) - 0.05


def test_linear_data_recovered():
    rng = np.random.default_rng(5)
    x = rng.uniform(-2, 2, size=(200, 1))
    y = 3 * x[:, 0] - 1 + rng.normal(0, 0.1, size=200)
    model = HistogramGradientBoostingRegressor(
        n_estimators=150, max_bins=32, min_samples_leaf=5,
        random_state=0).fit(x[:160], y[:160])
    assert model.score(x[160:], y[160:]) > 0.95


def test_learning_rate_effect():
    x, y = make_data(seed=6)
    fast = HistogramGradientBoostingRegressor(
        n_estimators=10, learning_rate=1.0, max_bins=32,
        min_samples_leaf=10, random_state=0).fit(x, y)
    slow = HistogramGradientBoostingRegressor(
        n_estimators=10, learning_rate=0.01, max_bins=32,
        min_samples_leaf=10, random_state=0).fit(x, y)
    mse_fast = np.mean((y - fast.predict(x)) ** 2)
    mse_slow = np.mean((y - slow.predict(x)) ** 2)
    assert mse_fast < mse_slow


def test_subsample_mode():
    x, y = make_data(seed=7)
    model = HistogramGradientBoostingRegressor(
        n_estimators=50, max_bins=32, min_samples_leaf=10,
        subsample=0.7, random_state=0).fit(x, y)
    assert model.score(x, y) > 0.9


def test_reproducible_with_random_state():
    x, y = make_data(seed=8)
    m1 = HistogramGradientBoostingRegressor(
        n_estimators=20, max_bins=16, subsample=0.8,
        random_state=3).fit(x, y)
    m2 = HistogramGradientBoostingRegressor(
        n_estimators=20, max_bins=16, subsample=0.8,
        random_state=3).fit(x, y)
    assert np.allclose(m1.predict(x), m2.predict(x))


def test_min_split_gain_blocks_splits():
    x, y = make_data(seed=9)
    model = HistogramGradientBoostingRegressor(
        n_estimators=10, max_bins=32, min_split_gain=1e9,
        min_samples_leaf=5).fit(x, y)
    # No split can beat the huge gain threshold: every tree is a stump
    # leaf, so predictions are (near) the training mean.
    assert np.allclose(model.predict(x), y.mean(), atol=1e-6)


def test_constant_target():
    rng = np.random.default_rng(10)
    x = rng.normal(size=(60, 2))
    y = np.full(60, 2.0)
    model = HistogramGradientBoostingRegressor(
        n_estimators=10, max_bins=16, min_samples_leaf=5).fit(x, y)
    assert np.allclose(model.predict(x), 2.0)
    assert model.score(x, y) == 1.0


def test_single_feature():
    rng = np.random.default_rng(11)
    x = rng.uniform(-2, 2, size=(150, 1))
    y = np.sin(x[:, 0]) + rng.normal(0, 0.05, size=150)
    model = HistogramGradientBoostingRegressor(
        n_estimators=100, max_bins=32, min_samples_leaf=5,
        random_state=0).fit(x[:120], y[:120])
    assert model.score(x[120:], y[120:]) > 0.9


def test_predict_before_fit_raises():
    model = HistogramGradientBoostingRegressor()
    with pytest.raises(ValueError, match="not fitted"):
        model.predict(np.zeros((2, 2)))


def test_value_error_1d_X():
    with pytest.raises(ValueError, match="2D"):
        HistogramGradientBoostingRegressor().fit(np.zeros(10), np.zeros(10))


def test_value_error_mismatched_lengths():
    with pytest.raises(ValueError, match="inconsistent lengths"):
        HistogramGradientBoostingRegressor().fit(np.zeros((5, 2)), np.zeros(4))


def test_value_error_empty_input():
    with pytest.raises(ValueError, match="empty"):
        HistogramGradientBoostingRegressor().fit(np.zeros((0, 2)), np.zeros(0))


def test_value_error_nonfinite():
    x, y = make_data(seed=12)
    y[0] = np.inf
    with pytest.raises(ValueError, match="finite"):
        HistogramGradientBoostingRegressor().fit(x, y)


def test_value_error_bad_params():
    with pytest.raises(ValueError):
        HistogramGradientBoostingRegressor(n_estimators=0)
    with pytest.raises(ValueError):
        HistogramGradientBoostingRegressor(learning_rate=0)
    with pytest.raises(ValueError):
        HistogramGradientBoostingRegressor(max_bins=1)
    with pytest.raises(ValueError):
        HistogramGradientBoostingRegressor(min_samples_leaf=0)
    with pytest.raises(ValueError):
        HistogramGradientBoostingRegressor(reg_lambda=-1)
    with pytest.raises(ValueError):
        HistogramGradientBoostingRegressor(min_split_gain=-1)
    with pytest.raises(ValueError):
        HistogramGradientBoostingRegressor(subsample=0)
    with pytest.raises(ValueError):
        HistogramGradientBoostingRegressor(subsample=1.5)


def test_value_error_wrong_feature_count_at_predict():
    x, y = make_data(seed=13)
    model = HistogramGradientBoostingRegressor(
        n_estimators=5, max_bins=16, min_samples_leaf=5).fit(x, y)
    with pytest.raises(ValueError, match="features"):
        model.predict(np.zeros((3, 5)))
