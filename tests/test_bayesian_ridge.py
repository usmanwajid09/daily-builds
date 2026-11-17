"""Tests for bayesian_ridge.BayesianRidgeRegressor."""
import numpy as np
import pytest

from bayesian_ridge.model import BayesianRidgeRegressor


def make_data(seed=0, n=100, noise=0.0, slope=2.0, intercept=-1.0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-3, 3, size=(n, 1))
    y = slope * X[:, 0] + intercept + rng.normal(0, noise, size=n)
    return X, y


def test_recovers_true_coefficients_noiseless():
    X, y = make_data(seed=0, noise=0.0)
    model = BayesianRidgeRegressor(alpha=1e-6, beta=1.0).fit(X, y)
    assert model.coef_[0] == pytest.approx(2.0, abs=1e-3)
    assert model.intercept_ == pytest.approx(-1.0, abs=1e-3)


def test_r2_above_0_9_on_noisy_linear_data():
    X, y = make_data(seed=1, noise=0.5)
    X_train, X_test = X[:80], X[80:]
    y_train, y_test = y[:80], y[80:]
    model = BayesianRidgeRegressor().fit(X_train, y_train)
    assert model.score(X_test, y_test) > 0.9


def test_predictive_std_positive_and_grows_with_distance():
    X, y = make_data(seed=2, noise=0.5)
    model = BayesianRidgeRegressor().fit(X, y)
    std = model.predict_std(X)
    assert np.all(std > 0)
    near = model.predict_std(np.array([[X[:, 0].mean()]]))
    far = model.predict_std(np.array([[X[:, 0].mean() + 50.0]]))
    assert far[0] > near[0]


def test_posterior_covariance_symmetric_positive_definite():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(50, 3))
    y = X @ np.array([1.0, -2.0, 0.5]) + rng.normal(0, 0.3, size=50)
    model = BayesianRidgeRegressor().fit(X, y)
    S = model.posterior_cov_
    assert np.allclose(S, S.T)
    assert np.all(np.linalg.eigvalsh(S) > 0)


def test_predict_shape():
    X, y = make_data(seed=4)
    model = BayesianRidgeRegressor().fit(X, y)
    pred = model.predict(X[:5])
    assert pred.shape == (5,)
    assert model.predict_std(X[:5]).shape == (5,)


def test_value_error_1d_X():
    X, y = make_data(seed=5)
    with pytest.raises(ValueError):
        BayesianRidgeRegressor().fit(X[:, 0], y)


def test_value_error_mismatched_lengths():
    X, y = make_data(seed=6)
    with pytest.raises(ValueError):
        BayesianRidgeRegressor().fit(X, y[:-1])


def test_value_error_empty_input():
    with pytest.raises(ValueError):
        BayesianRidgeRegressor().fit(np.empty((0, 1)), np.empty(0))


def test_value_error_nan_input():
    X, y = make_data(seed=7)
    X_bad = X.copy()
    X_bad[0, 0] = np.nan
    with pytest.raises(ValueError):
        BayesianRidgeRegressor().fit(X_bad, y)
    y_bad = y.copy()
    y_bad[0] = np.nan
    with pytest.raises(ValueError):
        BayesianRidgeRegressor().fit(X, y_bad)


def test_value_error_alpha_le_zero():
    with pytest.raises(ValueError):
        BayesianRidgeRegressor(alpha=0.0)
    with pytest.raises(ValueError):
        BayesianRidgeRegressor(alpha=-1.0)


def test_value_error_beta_le_zero():
    with pytest.raises(ValueError):
        BayesianRidgeRegressor(beta=0.0)
    with pytest.raises(ValueError):
        BayesianRidgeRegressor(beta=-2.0)
