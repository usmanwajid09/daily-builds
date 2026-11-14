import numpy as np
import pytest

from gradient_boosting.model import GradientBoostingRegressor


def _data(seed=42, n=200):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 2))
    y = 3.0 * x[:, 0] - 2.0 * x[:, 1] + rng.normal(scale=0.2, size=n)
    return x, y


def test_r2_on_linear_data():
    x, y = _data()
    model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1,
                                      max_depth=3).fit(x, y)
    assert model.score(x, y) >= 0.95


def test_predictions_shape_and_finite():
    x, y = _data()
    model = GradientBoostingRegressor(n_estimators=20).fit(x, y)
    preds = model.predict(x[:5])
    assert preds.shape == (5,)
    assert np.isfinite(preds).all()


def test_single_point_prediction():
    x, y = _data()
    model = GradientBoostingRegressor(n_estimators=50).fit(x, y)
    assert model.predict(np.array([0.0, 0.0])).shape == (1,)


def test_constant_target():
    x = np.random.default_rng(0).normal(size=(50, 2))
    y = np.full(50, 7.5)
    model = GradientBoostingRegressor(n_estimators=10).fit(x, y)
    assert np.allclose(model.predict(x), 7.5)
    assert model.score(x, y) == 1.0


def test_nonlinear_target_beats_linear_baseline():
    rng = np.random.default_rng(1)
    n = 200
    x = rng.normal(size=(n, 1))
    y = np.sin(3 * x[:, 0]) + rng.normal(scale=0.1, size=n)
    model = GradientBoostingRegressor(n_estimators=150, learning_rate=0.1,
                                      max_depth=3).fit(x, y)
    assert model.score(x, y) >= 0.9


def test_more_trees_do_not_hurt_much():
    x, y = _data()
    small = GradientBoostingRegressor(n_estimators=10).fit(x, y)
    big = GradientBoostingRegressor(n_estimators=100).fit(x, y)
    assert big.score(x, y) >= small.score(x, y) - 0.05


def test_predict_before_fit_raises():
    with pytest.raises(RuntimeError, match="not fitted"):
        GradientBoostingRegressor().predict(np.array([[0.0, 0.0]]))


def test_mismatched_lengths_raise():
    with pytest.raises(ValueError, match="same number of samples"):
        GradientBoostingRegressor().fit(np.zeros((3, 2)), np.zeros(2))


def test_nan_raises():
    with pytest.raises(ValueError, match="NaN"):
        GradientBoostingRegressor().fit(
            np.array([[0.0, np.nan], [1.0, 1.0]]), np.array([1.0, 2.0]))
    with pytest.raises(ValueError, match="NaN"):
        GradientBoostingRegressor().fit(
            np.array([[0.0, 0.0], [1.0, 1.0]]), np.array([1.0, np.nan]))


def test_empty_data_raises():
    with pytest.raises(ValueError, match="empty"):
        GradientBoostingRegressor().fit(np.zeros((0, 2)), np.zeros(0))


def test_bad_hyperparameters_raise():
    with pytest.raises(ValueError, match="n_estimators"):
        GradientBoostingRegressor(n_estimators=0)
    with pytest.raises(ValueError, match="learning_rate"):
        GradientBoostingRegressor(learning_rate=0)
    with pytest.raises(ValueError, match="max_depth"):
        GradientBoostingRegressor(max_depth=0)
    with pytest.raises(ValueError, match="min_samples_leaf"):
        GradientBoostingRegressor(min_samples_leaf=0)


def test_wrong_feature_count_raises():
    x, y = _data()
    model = GradientBoostingRegressor(n_estimators=5).fit(x, y)
    with pytest.raises(ValueError, match="features"):
        model.predict(np.zeros((2, 3)))
