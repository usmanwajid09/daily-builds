"""Tests for the stacking regressor (numpy only, no sklearn)."""

import numpy as np
import pytest

from stacking_regressor.model import (
    DecisionTreeRegressor,
    KNNRegressor,
    RidgeRegressor,
    StackingRegressor,
)


def _linear_data(seed=0, n=100):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    y = 2.0 * X[:, 0] - 1.5 * X[:, 1] + 0.5 * X[:, 2] + rng.normal(0, 0.05, n)
    return X, y


def _nonlinear_data(seed=0, n=200):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-2.0, 2.0, size=(n, 2))
    y = np.sin(3 * X[:, 0]) + X[:, 1] ** 2 + rng.normal(0, 0.1, size=n)
    return X, y


def _rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def test_predictions_shape_and_finite():
    X, y = _nonlinear_data()
    model = StackingRegressor(random_state=42).fit(X, y)
    preds = model.predict(X)
    assert preds.shape == (len(y),)
    assert np.all(np.isfinite(preds))


def test_fits_linear_target_with_low_rmse():
    X, y = _linear_data()
    model = StackingRegressor(random_state=42).fit(X, y)
    assert _rmse(model.predict(X), y) < 0.2


def test_beats_or_matches_ridge_on_nonlinear_target():
    X, y = _nonlinear_data(seed=42)
    split = 160
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    stacker = StackingRegressor(random_state=42).fit(X_train, y_train)
    baseline = RidgeRegressor(alpha=1.0).fit(X_train, y_train)
    stack_rmse = _rmse(stacker.predict(X_test), y_test)
    base_rmse = _rmse(baseline.predict(X_test), y_test)
    # Allow a small tolerance; in practice the stacker wins by a wide margin.
    assert stack_rmse <= base_rmse + 0.05
    assert stack_rmse < base_rmse  # it should genuinely beat the linear model


def test_predict_before_fit_raises():
    X, _ = _linear_data()
    with pytest.raises(RuntimeError):
        StackingRegressor().predict(X)


def test_n_folds_one_raises():
    with pytest.raises(ValueError, match="n_folds"):
        StackingRegressor(n_folds=1)


def test_n_folds_more_than_samples_raises():
    X, y = _linear_data(n=10)
    with pytest.raises(ValueError, match="n_folds"):
        StackingRegressor(n_folds=20).fit(X, y)


def test_n_folds_non_int_raises():
    with pytest.raises(ValueError, match="n_folds"):
        StackingRegressor(n_folds=2.5)
    with pytest.raises(ValueError, match="n_folds"):
        StackingRegressor(n_folds="5")


def test_alpha_non_positive_raises():
    with pytest.raises(ValueError, match="alpha"):
        StackingRegressor(alpha=0.0)
    with pytest.raises(ValueError, match="alpha"):
        StackingRegressor(alpha=-1.0)


def test_x_1d_raises():
    _, y = _linear_data()
    with pytest.raises(ValueError, match="2D"):
        StackingRegressor().fit(np.arange(10, dtype=float), y[:10])


def test_mismatched_lengths_raises():
    X, y = _linear_data()
    with pytest.raises(ValueError, match="mismatched"):
        StackingRegressor().fit(X, y[:-5])


def test_non_numeric_raises():
    X, y = _linear_data()
    X_bad = X.astype(object)
    X_bad[0, 0] = "not a number"
    with pytest.raises(ValueError):
        StackingRegressor().fit(X_bad, y)


def test_fit_returns_self():
    X, y = _linear_data()
    model = StackingRegressor()
    assert model.fit(X, y) is model


def test_meta_features_shape_and_weights_length():
    X, y = _linear_data(n=60)
    model = StackingRegressor(n_folds=5, random_state=42).fit(X, y)
    assert model.meta_features_.shape == (60, 3)
    # 3 base-learner weights + 1 intercept.
    assert len(model.meta_model_) == 4
    assert len(model.base_models_) == 3
    assert model.n_features_in_ == 3


def test_deterministic_with_same_random_state():
    X, y = _nonlinear_data(seed=7)
    a = StackingRegressor(random_state=42).fit(X, y).predict(X)
    b = StackingRegressor(random_state=42).fit(X, y).predict(X)
    np.testing.assert_array_equal(a, b)


def test_base_learners_fit_linear_target():
    """Sanity: each base learner on its own learns a linear target."""
    X, y = _linear_data()
    for learner in (
        RidgeRegressor(alpha=1.0),
        KNNRegressor(n_neighbors=5),
        DecisionTreeRegressor(max_depth=5),
    ):
        learner.fit(X, y)
        preds = learner.predict(X)
        assert preds.shape == (len(y),)
        assert np.all(np.isfinite(preds))
