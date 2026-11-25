import numpy as np
import pytest

from theil_sen_regressor import TheilSenRegressor


def ols_slope(x, y):
    return np.polyfit(np.asarray(x).ravel(), y, 1)[0]


def test_recovers_slope_on_clean_data():
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 10, 200)
    y = 2.0 * x + 1.0 + rng.normal(0, 0.3, 200)
    m = TheilSenRegressor().fit(x, y)
    assert abs(m.coef_[0] - 2.0) < 0.15


def test_robust_to_20_percent_outliers_beats_ols():
    rng = np.random.default_rng(1)
    x = rng.uniform(0, 10, 200)
    y = 2.0 * x + 1.0 + rng.normal(0, 0.3, 200)
    out = rng.choice(200, size=40, replace=False)
    y[out] += rng.choice([-1, 1], size=40) * rng.uniform(8, 15, 40)
    m = TheilSenRegressor().fit(x, y)
    ts_err = abs(m.coef_[0] - 2.0)
    assert ts_err < 0.3
    assert ts_err < abs(ols_slope(x, y) - 2.0)


def test_exact_fit_on_noiseless_line():
    x = np.arange(10, dtype=float)
    y = 3.0 * x - 2.0
    m = TheilSenRegressor().fit(x, y)
    assert m.coef_[0] == pytest.approx(3.0)
    assert m.intercept_ == pytest.approx(-2.0)


def test_intercept_correct():
    rng = np.random.default_rng(2)
    x = rng.uniform(0, 10, 200)
    y = -1.5 * x + 4.0 + rng.normal(0, 0.2, 200)
    m = TheilSenRegressor().fit(x, y)
    assert abs(m.intercept_ - 4.0) < 0.15


def test_multivariate_fit_shape():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(100, 3))
    y = 2 * X[:, 0] - X[:, 1] + 0.5 * X[:, 2] + 1 + rng.normal(0, 0.1, 100)
    m = TheilSenRegressor().fit(X, y)
    assert m.coef_.shape == (3,)
    assert isinstance(m.intercept_, float)
    assert np.allclose(m.coef_, [2, -1, 0.5], atol=0.2)


def test_predict_shape():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(50, 2))
    y = X[:, 0] + X[:, 1] + rng.normal(0, 0.1, 50)
    m = TheilSenRegressor().fit(X, y)
    assert m.predict(X).shape == (50,)
    assert m.predict(X[:5]).shape == (5,)


def test_score_near_one_on_clean_data():
    rng = np.random.default_rng(5)
    x = rng.uniform(0, 10, 100)
    y = 2.0 * x + 1.0 + rng.normal(0, 0.1, 100)
    m = TheilSenRegressor().fit(x, y)
    assert m.score(x, y) > 0.99


def test_empty_input_raises():
    with pytest.raises(ValueError):
        TheilSenRegressor().fit([], [])


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        TheilSenRegressor().fit([1, 2, 3], [1, 2])


def test_y_wrong_shape_raises():
    with pytest.raises(ValueError):
        TheilSenRegressor().fit([[1], [2], [3]], [[1], [2], [3]])


def test_nan_raises():
    with pytest.raises(ValueError):
        TheilSenRegressor().fit([1, 2, np.nan, 4], [1, 2, 3, 4])
    with pytest.raises(ValueError):
        TheilSenRegressor().fit([1, 2, 3, 4], [1, np.inf, 3, 4])


def test_constant_x_raises():
    with pytest.raises(ValueError):
        TheilSenRegressor().fit([5, 5, 5, 5], [1, 2, 3, 4])


def test_predict_before_fit_raises():
    with pytest.raises(RuntimeError):
        TheilSenRegressor().predict([1, 2, 3])


def test_max_subpopulation_deterministic():
    rng = np.random.default_rng(6)
    x = rng.uniform(0, 10, 500)
    y = 2.0 * x + 1.0 + rng.normal(0, 0.3, 500)
    a = TheilSenRegressor(max_subpopulation=100, random_state=7).fit(x, y)
    b = TheilSenRegressor(max_subpopulation=100, random_state=7).fit(x, y)
    assert a.coef_[0] == b.coef_[0]
    assert abs(a.coef_[0] - 2.0) < 0.3
