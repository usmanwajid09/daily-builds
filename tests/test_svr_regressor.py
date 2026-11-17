"""Pytest tests for svr_regressor.model.SVRRegressor."""

import numpy as np
import pytest

from svr_regressor.model import SVRRegressor


def _linear_data(n=100, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-2, 2, size=(n, 1))
    y = 3.0 * X.ravel() + 1.0
    return X, y


def _sine_data(n=120, seed=42, noise=0.05):
    rng = np.random.default_rng(seed)
    X = np.linspace(0, 1, n).reshape(-1, 1)
    y = np.sin(X).ravel() + rng.normal(0, noise, size=n)
    return X, y


def test_fits_linear_target_well():
    X, y = _linear_data()
    model = SVRRegressor(epsilon=0.01, C=10.0, n_epochs=500, random_state=42)
    model.fit(X, y)
    assert model.score(X, y) > 0.9


def test_fits_sine_wave_reasonably():
    X, y = _sine_data()
    model = SVRRegressor(epsilon=0.05, C=10.0, n_epochs=500, random_state=42)
    model.fit(X, y)
    assert model.score(X, y) > 0.7


def test_predict_shape():
    X, y = _linear_data()
    model = SVRRegressor(n_epochs=100, random_state=42).fit(X, y)
    preds = model.predict(X[:5])
    assert preds.shape == (5,)
    assert preds.ndim == 1


def test_score_returns_float_in_sensible_range():
    X, y = _linear_data()
    model = SVRRegressor(n_epochs=200, random_state=42).fit(X, y)
    r2 = model.score(X, y)
    assert isinstance(r2, float)
    assert -1.0 <= r2 <= 1.0


def test_attributes_after_fit():
    X, y = _linear_data()
    model = SVRRegressor(n_epochs=50, random_state=42).fit(X, y)
    assert model.coef_.shape == (1,)
    assert isinstance(model.intercept_, float)
    assert model.n_features_in_ == 1


def test_value_error_1d_X():
    X, y = _linear_data()
    with pytest.raises(ValueError):
        SVRRegressor().fit(X.ravel(), y)


def test_value_error_mismatched_lengths():
    X, y = _linear_data()
    with pytest.raises(ValueError):
        SVRRegressor().fit(X, y[:-1])


def test_value_error_empty_input():
    with pytest.raises(ValueError):
        SVRRegressor().fit(np.empty((0, 1)), np.empty(0))


def test_value_error_nan_input():
    X, y = _linear_data()
    X_bad = X.copy()
    X_bad[0, 0] = np.nan
    with pytest.raises(ValueError):
        SVRRegressor().fit(X_bad, y)


def test_value_error_negative_C():
    X, y = _linear_data()
    with pytest.raises(ValueError):
        SVRRegressor(C=-1.0).fit(X, y)


def test_value_error_negative_epsilon():
    X, y = _linear_data()
    with pytest.raises(ValueError):
        SVRRegressor(epsilon=-0.1).fit(X, y)


def test_deterministic_with_fixed_seed():
    X, y = _linear_data()
    m1 = SVRRegressor(n_epochs=100, random_state=42).fit(X, y)
    m2 = SVRRegressor(n_epochs=100, random_state=42).fit(X, y)
    np.testing.assert_allclose(m1.predict(X), m2.predict(X))
