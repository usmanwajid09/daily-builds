"""Tests for extra_trees_regressor.ExtraTreesRegressor."""
import numpy as np
import pytest

from extra_trees_regressor.model import ExtraTreesRegressor


def make_data(seed=0, n=200, noise=0.1):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-3, 3, size=(n, 2))
    y = np.sin(x[:, 0]) + 0.5 * x[:, 1] + rng.normal(0, noise, size=n)
    return x, y


def test_r2_above_0_9_on_sinusoidal_data():
    x, y = make_data(seed=0)
    model = ExtraTreesRegressor(n_estimators=50, random_state=0).fit(x[:160], y[:160])
    assert model.score(x[160:], y[160:]) > 0.9


def test_forest_beats_single_tree():
    x, y = make_data(seed=1)
    forest = ExtraTreesRegressor(n_estimators=50, random_state=0).fit(x[:160], y[:160])
    single = ExtraTreesRegressor(n_estimators=1, random_state=0).fit(x[:160], y[:160])
    assert forest.score(x[160:], y[160:]) >= single.score(x[160:], y[160:]) - 0.05


def test_predict_shape_and_finite():
    x, y = make_data(seed=2)
    model = ExtraTreesRegressor(n_estimators=10, random_state=0).fit(x, y)
    pred = model.predict(x[:5])
    assert pred.shape == (5,)
    assert np.all(np.isfinite(pred))


def test_feature_importances_sum_to_one():
    x, y = make_data(seed=3)
    model = ExtraTreesRegressor(n_estimators=20, random_state=0).fit(x, y)
    assert model.feature_importances_.shape == (2,)
    assert model.feature_importances_.sum() == pytest.approx(1.0)
    assert np.all(model.feature_importances_ >= 0)
    assert np.all(model.feature_importances_ > 0)


def test_reproducible_with_random_state():
    x, y = make_data(seed=4)
    m1 = ExtraTreesRegressor(n_estimators=10, random_state=7).fit(x, y)
    m2 = ExtraTreesRegressor(n_estimators=10, random_state=7).fit(x, y)
    assert np.allclose(m1.predict(x), m2.predict(x))


def test_different_seeds_can_differ():
    x, y = make_data(seed=5)
    rng = np.random.default_rng(99)
    x_new = rng.uniform(-3, 3, size=(50, 2))
    m1 = ExtraTreesRegressor(n_estimators=1, max_depth=2, random_state=1).fit(x, y)
    m2 = ExtraTreesRegressor(n_estimators=1, max_depth=2, random_state=2).fit(x, y)
    assert not np.allclose(m1.predict(x_new), m2.predict(x_new))


def test_bootstrap_mode():
    x, y = make_data(seed=6)
    model = ExtraTreesRegressor(n_estimators=20, bootstrap=True, random_state=0).fit(x, y)
    assert model.score(x, y) > 0.9


def test_max_depth_limits_tree():
    x, y = make_data(seed=7)
    deep = ExtraTreesRegressor(n_estimators=10, max_depth=None, random_state=0).fit(x, y)
    shallow = ExtraTreesRegressor(n_estimators=10, max_depth=1, random_state=0).fit(x, y)
    assert deep.score(x, y) > shallow.score(x, y)


def test_max_features_int_and_fraction():
    x, y = make_data(seed=8)
    for mf in (1, 0.5, None):
        model = ExtraTreesRegressor(n_estimators=5, max_features=mf, random_state=0).fit(x, y)
        assert model.predict(x[:3]).shape == (3,)


def test_constant_target_predicts_constant():
    rng = np.random.default_rng(9)
    x = rng.normal(size=(50, 2))
    y = np.full(50, 3.5)
    model = ExtraTreesRegressor(n_estimators=5, random_state=0).fit(x, y)
    assert np.allclose(model.predict(x), 3.5)
    assert model.score(x, y) == 1.0


def test_single_feature():
    rng = np.random.default_rng(10)
    x = rng.uniform(-2, 2, size=(100, 1))
    y = 2 * x[:, 0] + rng.normal(0, 0.1, size=100)
    model = ExtraTreesRegressor(n_estimators=20, random_state=0).fit(x[:80], y[:80])
    assert model.score(x[80:], y[80:]) > 0.9


def test_predict_before_fit_raises():
    model = ExtraTreesRegressor()
    with pytest.raises(ValueError, match="not fitted"):
        model.predict(np.zeros((2, 2)))


def test_value_error_1d_X():
    with pytest.raises(ValueError, match="2D"):
        ExtraTreesRegressor().fit(np.zeros(10), np.zeros(10))


def test_value_error_mismatched_lengths():
    with pytest.raises(ValueError, match="inconsistent lengths"):
        ExtraTreesRegressor().fit(np.zeros((5, 2)), np.zeros(4))


def test_value_error_empty_input():
    with pytest.raises(ValueError, match="empty"):
        ExtraTreesRegressor().fit(np.zeros((0, 2)), np.zeros(0))


def test_value_error_nonfinite():
    x, y = make_data(seed=11)
    x[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        ExtraTreesRegressor().fit(x, y)


def test_value_error_bad_params():
    with pytest.raises(ValueError):
        ExtraTreesRegressor(n_estimators=0)
    with pytest.raises(ValueError):
        ExtraTreesRegressor(min_samples_split=1)
    with pytest.raises(ValueError):
        ExtraTreesRegressor(min_samples_leaf=0)
    with pytest.raises(ValueError):
        ExtraTreesRegressor(max_depth=0)
    with pytest.raises(ValueError):
        ExtraTreesRegressor(max_features=1.5)
    with pytest.raises(ValueError):
        ExtraTreesRegressor(max_features=0)


def test_value_error_wrong_feature_count_at_predict():
    x, y = make_data(seed=12)
    model = ExtraTreesRegressor(n_estimators=5, random_state=0).fit(x, y)
    with pytest.raises(ValueError, match="features"):
        model.predict(np.zeros((3, 5)))


def test_score_perfect_on_train_for_deep_forest():
    x, y = make_data(seed=13, noise=0.0)
    model = ExtraTreesRegressor(n_estimators=30, random_state=0).fit(x, y)
    assert model.score(x, y) > 0.99
