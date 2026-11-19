"""Tests for random_forest_regressor."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from random_forest_regressor import RandomForestRegressor
from random_forest_regressor.demo import make_data


def small_data(seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-2, 2, size=(60, 2))
    y = 3.0 * x[:, 0] + 2.0 * x[:, 1] + rng.normal(0, 0.1, size=60)
    return x, y


def test_fit_predict_linear():
    x, y = small_data()
    model = RandomForestRegressor(n_estimators=10, random_state=0)
    model.fit(x, y)
    pred = model.predict(x)
    assert pred.shape == (60,)
    assert np.all(np.isfinite(pred))
    assert model.score(x, y) > 0.9


def test_predict_single_row():
    x, y = small_data()
    model = RandomForestRegressor(n_estimators=5, random_state=0).fit(x, y)
    pred = model.predict(x[0])
    assert pred.shape == (1,)


def test_oob_score_between_0_and_1():
    x, y = make_data()
    model = RandomForestRegressor(n_estimators=20, random_state=1).fit(x, y)
    assert model.oob_score_ is not None
    assert -1.0 <= model.oob_score_ <= 1.0
    assert model.oob_score_ > 0.7


def test_feature_importances_sum_to_one():
    x, y = make_data()
    model = RandomForestRegressor(n_estimators=10, max_features=None,
                                  random_state=0).fit(x, y)
    imp = model.feature_importances_
    assert imp.shape == (3,)
    assert imp.sum() == pytest.approx(1.0)
    assert imp[2] < imp[0] and imp[2] < imp[1]  # noise feature matters least


def test_constant_target_predicts_constant():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(20, 2))
    y = np.full(20, 7.5)
    model = RandomForestRegressor(n_estimators=5, random_state=0).fit(x, y)
    assert np.allclose(model.predict(x), 7.5)


def test_predict_before_fit_raises():
    model = RandomForestRegressor(n_estimators=5)
    with pytest.raises(RuntimeError, match="not fitted"):
        model.predict(np.zeros((2, 2)))


def test_fit_rejects_bad_shapes():
    x, y = small_data()
    with pytest.raises(ValueError, match="2D"):
        RandomForestRegressor().fit(np.zeros(5), y)
    with pytest.raises(ValueError, match="1D"):
        RandomForestRegressor().fit(x, np.zeros((60, 1)))
    with pytest.raises(ValueError, match="same number"):
        RandomForestRegressor().fit(x, np.zeros(10))
    with pytest.raises(ValueError, match="empty"):
        RandomForestRegressor().fit(np.zeros((0, 2)), np.zeros(0))


def test_fit_rejects_nan_and_inf():
    x, y = small_data()
    x_bad = x.copy()
    x_bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        RandomForestRegressor().fit(x_bad, y)
    y_bad = y.copy()
    y_bad[0] = np.inf
    with pytest.raises(ValueError, match="NaN"):
        RandomForestRegressor().fit(x, y_bad)


def test_bad_constructor_args():
    with pytest.raises(ValueError, match="n_estimators"):
        RandomForestRegressor(n_estimators=0)
    with pytest.raises(ValueError, match="max_depth"):
        RandomForestRegressor(max_depth=0)
    with pytest.raises(ValueError, match="min_samples_split"):
        RandomForestRegressor(min_samples_split=1)
    with pytest.raises(ValueError, match="max_features"):
        RandomForestRegressor(max_features=0)
    with pytest.raises(ValueError, match="max_features"):
        RandomForestRegressor(max_features=5).fit(*small_data())
    with pytest.raises(ValueError, match="random_state"):
        RandomForestRegressor(random_state=1.5)


def test_deterministic_with_seed():
    x, y = small_data()
    m1 = RandomForestRegressor(n_estimators=10, random_state=7).fit(x, y)
    m2 = RandomForestRegressor(n_estimators=10, random_state=7).fit(x, y)
    assert np.allclose(m1.predict(x), m2.predict(x))
    assert m1.oob_score_ == m2.oob_score_


def test_score_rejects_zero_variance_y():
    x, y = small_data()
    model = RandomForestRegressor(n_estimators=5, random_state=0).fit(x, y)
    with pytest.raises(ValueError, match="zero variance"):
        model.score(x, np.full(60, 1.0))


def test_predict_rejects_wrong_feature_count():
    x, y = small_data()
    model = RandomForestRegressor(n_estimators=5, random_state=0).fit(x, y)
    with pytest.raises(ValueError, match="features"):
        model.predict(np.zeros((2, 5)))
