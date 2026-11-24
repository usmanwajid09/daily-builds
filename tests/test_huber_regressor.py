"""Tests for huber_regressor.model.HuberRegressor."""
import numpy as np
import pytest

from huber_regressor.model import HuberRegressor


def ols_fit(X, y):
    """Plain OLS via lstsq, returning (intercept, coef)."""
    X_aug = np.column_stack([np.ones(len(X)), X])
    params, *_ = np.linalg.lstsq(X_aug, y, rcond=None)
    return params[0], params[1:]


# 1. Coefficient recovery on clean noiseless data (single feature).
def test_clean_noiseless_single_feature():
    rng = np.random.default_rng(0)
    X = rng.uniform(-3, 3, size=(50, 1))
    y = 2 * X[:, 0] - 1
    model = HuberRegressor().fit(X, y)
    assert model.coef_[0] == pytest.approx(2.0, abs=1e-8)
    assert model.intercept_ == pytest.approx(-1.0, abs=1e-8)


# 2. Coefficient recovery on clean noiseless data (multi-feature).
def test_clean_noiseless_multi_feature():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(80, 3))
    true_coef = np.array([1.5, -2.0, 0.5])
    y = X @ true_coef + 3.0
    model = HuberRegressor().fit(X, y)
    assert model.coef_ == pytest.approx(true_coef, abs=1e-8)
    assert model.intercept_ == pytest.approx(3.0, abs=1e-8)


# 3. R^2 > 0.9 on noisy data.
def test_r2_high_on_noisy_data():
    rng = np.random.default_rng(2)
    X = rng.uniform(-3, 3, size=(100, 1))
    y = 2 * X[:, 0] - 1 + rng.normal(0, 0.5, size=100)
    model = HuberRegressor().fit(X, y)
    assert model.score(X, y) > 0.9


# 4. Beats OLS on outlier-contaminated data (smaller slope error).
def test_beats_ols_on_outlier_data():
    rng = np.random.default_rng(3)
    X = rng.uniform(-3, 3, size=(200, 1))
    y = 2 * X[:, 0] - 1 + rng.normal(0, 0.5, size=200)
    idx = rng.choice(200, size=20, replace=False)
    y[idx] += rng.uniform(8, 12, size=20)
    model = HuberRegressor().fit(X, y)
    ols_intercept, ols_coef = ols_fit(X, y)
    assert abs(model.coef_[0] - 2.0) < abs(ols_coef[0] - 2.0)
    assert abs(model.coef_[0] - 2.0) < 0.2  # Huber stays close to truth


# 5. Predict returns the right shape.
def test_predict_shape():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(40, 2))
    y = X @ np.array([1.0, -1.0]) + 0.5
    model = HuberRegressor().fit(X, y)
    preds = model.predict(rng.normal(size=(7, 2)))
    assert preds.shape == (7,)


# 6. Score lies in (-inf, 1].
def test_score_range():
    rng = np.random.default_rng(5)
    X = rng.uniform(-3, 3, size=(60, 1))
    y = 2 * X[:, 0] - 1 + rng.normal(0, 0.5, size=60)
    model = HuberRegressor().fit(X, y)
    r2 = model.score(X, y)
    assert r2 <= 1.0
    assert np.isfinite(r2)


# 7. fit_intercept=False forces a zero intercept.
def test_no_intercept():
    rng = np.random.default_rng(6)
    X = rng.uniform(-3, 3, size=(50, 1))
    y = 2 * X[:, 0]  # true line through the origin
    model = HuberRegressor(fit_intercept=False).fit(X, y)
    assert model.intercept_ == 0.0
    assert model.coef_[0] == pytest.approx(2.0, abs=1e-8)


# 8. ValueError for 1D X.
def test_1d_X_raises():
    with pytest.raises(ValueError):
        HuberRegressor().fit(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0]))


# 9. ValueError for mismatched lengths.
def test_mismatched_lengths_raises():
    X = np.zeros((5, 1))
    y = np.zeros(4)
    with pytest.raises(ValueError):
        HuberRegressor().fit(X, y)


# 10. ValueError for empty input.
def test_empty_input_raises():
    with pytest.raises(ValueError):
        HuberRegressor().fit(np.empty((0, 1)), np.empty(0))


# 11. ValueError for NaN in the data.
def test_nan_raises():
    X = np.array([[1.0], [np.nan], [3.0]])
    y = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        HuberRegressor().fit(X, y)
    X2 = np.array([[1.0], [2.0], [3.0]])
    y2 = np.array([1.0, np.nan, 3.0])
    with pytest.raises(ValueError):
        HuberRegressor().fit(X2, y2)


# 12. ValueError for epsilon <= 0.
def test_bad_epsilon_raises():
    with pytest.raises(ValueError):
        HuberRegressor(epsilon=0)
    with pytest.raises(ValueError):
        HuberRegressor(epsilon=-1.35)


# 13. Predict before fit raises.
def test_predict_before_fit_raises():
    with pytest.raises(ValueError):
        HuberRegressor().predict(np.array([[1.0]]))


# 14. Determinism: identical fits on identical data.
def test_determinism():
    rng = np.random.default_rng(7)
    X = rng.uniform(-3, 3, size=(100, 2))
    y = X @ np.array([2.0, -1.0]) + 1.0 + rng.normal(0, 0.3, size=100)
    y[::10] += 10.0  # outliers
    m1 = HuberRegressor().fit(X, y)
    m2 = HuberRegressor().fit(X, y)
    assert np.array_equal(m1.coef_, m2.coef_)
    assert m1.intercept_ == m2.intercept_
    assert m1.n_iter_ == m2.n_iter_


# 15. Attributes are exposed after fitting.
def test_fitted_attributes():
    rng = np.random.default_rng(8)
    X = rng.uniform(-3, 3, size=(30, 1))
    y = 2 * X[:, 0] - 1 + rng.normal(0, 0.5, size=30)
    model = HuberRegressor().fit(X, y)
    assert model.coef_.shape == (1,)
    assert isinstance(model.intercept_, float)
    assert 1 <= model.n_iter_ <= 100


# 16. Works when the data has no outliers (agrees with OLS).
def test_agrees_with_ols_when_clean():
    rng = np.random.default_rng(9)
    X = rng.uniform(-3, 3, size=(100, 1))
    y = 2 * X[:, 0] - 1 + rng.normal(0, 0.5, size=100)
    model = HuberRegressor().fit(X, y)
    _, ols_coef = ols_fit(X, y)
    assert abs(model.coef_[0] - ols_coef[0]) < 0.05
