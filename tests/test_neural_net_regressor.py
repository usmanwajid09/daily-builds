import numpy as np
import pytest

from neural_net_regressor.model import MLPRegressor


def _linear_data(seed=0, n=200):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-2, 2, size=(n, 2))
    y = 3 * X[:, 0] - 2 * X[:, 1] + 1 + 0.05 * rng.normal(size=n)
    return X, y


def _sin_data(seed=0, n=200):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-3, 3, size=n)
    y = np.sin(x) + 0.1 * rng.normal(size=n)
    return x.reshape(-1, 1), y


def test_fits_linear_function_well():
    X, y = _linear_data()
    model = MLPRegressor(hidden_units=16, epochs=300, random_state=42).fit(X, y)
    assert model.score(X, y) > 0.9


def test_fits_nonlinear_sin_reasonably():
    X, y = _sin_data()
    model = MLPRegressor(hidden_units=32, epochs=500, random_state=42).fit(X, y)
    assert model.score(X, y) > 0.8


def test_predict_shape_and_1d_input():
    X, y = _linear_data()
    model = MLPRegressor(hidden_units=16, epochs=100, random_state=42).fit(X, y)
    preds = model.predict(X)
    assert preds.shape == (len(y),)
    assert np.isfinite(preds).all()
    single = model.predict(np.array([0.5, -0.5]))
    assert single.shape == (1,)


def test_score_returns_float_in_sensible_range():
    X, y = _linear_data()
    model = MLPRegressor(hidden_units=16, epochs=200, random_state=42).fit(X, y)
    r2 = model.score(X, y)
    assert isinstance(r2, float)
    assert 0.9 < r2 <= 1.0


def test_reproducible_with_same_random_state():
    X, y = _sin_data()
    m1 = MLPRegressor(hidden_units=16, epochs=100, random_state=7).fit(X, y)
    m2 = MLPRegressor(hidden_units=16, epochs=100, random_state=7).fit(X, y)
    assert np.allclose(m1.predict(X), m2.predict(X))
    assert m1.history_ == m2.history_


def test_early_stopping_reduces_epochs_on_easy_data():
    X, y = _linear_data()
    model = MLPRegressor(
        hidden_units=16, epochs=500, patience=5, random_state=42
    ).fit(X, y)
    assert model.n_epochs_ < 500
    assert model.score(X, y) > 0.9


def test_patience_zero_disables_early_stopping():
    X, y = _linear_data()
    model = MLPRegressor(
        hidden_units=16, epochs=50, patience=0, random_state=42
    ).fit(X, y)
    assert model.n_epochs_ == 50


def test_sgd_optimizer_works():
    X, y = _linear_data()
    model = MLPRegressor(
        hidden_units=16, learning_rate=0.05, epochs=500,
        optimizer="sgd", random_state=42,
    ).fit(X, y)
    assert model.score(X, y) > 0.8


def test_fit_rejects_bad_X_shape():
    X, y = _linear_data()
    with pytest.raises(ValueError):
        MLPRegressor().fit(X.ravel(), y)  # 1D
    with pytest.raises(ValueError):
        MLPRegressor().fit(X.reshape(10, 10, 4), np.zeros(10))  # 3D


def test_fit_rejects_mismatched_lengths():
    X, y = _linear_data()
    with pytest.raises(ValueError):
        MLPRegressor().fit(X, y[:-1])


def test_fit_rejects_empty_data():
    with pytest.raises(ValueError):
        MLPRegressor().fit(np.empty((0, 2)), np.empty(0))


def test_fit_rejects_nan_input():
    X, y = _linear_data()
    X_bad = X.copy()
    X_bad[0, 0] = np.nan
    with pytest.raises(ValueError):
        MLPRegressor().fit(X_bad, y)
    y_bad = y.copy()
    y_bad[3] = np.nan
    with pytest.raises(ValueError):
        MLPRegressor().fit(X, y_bad)


def test_bad_optimizer_name():
    with pytest.raises(ValueError):
        MLPRegressor(optimizer="rmsprop")


def test_bad_hyperparams():
    with pytest.raises(ValueError):
        MLPRegressor(hidden_units=0)
    with pytest.raises(ValueError):
        MLPRegressor(hidden_units=2.5)
    with pytest.raises(ValueError):
        MLPRegressor(learning_rate=0)
    with pytest.raises(ValueError):
        MLPRegressor(learning_rate=-0.1)
    with pytest.raises(ValueError):
        MLPRegressor(epochs=0)
    with pytest.raises(ValueError):
        MLPRegressor(batch_size=0)
    with pytest.raises(ValueError):
        MLPRegressor(patience=-1)
    with pytest.raises(ValueError):
        MLPRegressor(l2=-0.1)


def test_predict_before_fit_raises():
    X, _ = _linear_data()
    with pytest.raises(RuntimeError):
        MLPRegressor().predict(X)
    with pytest.raises(RuntimeError):
        MLPRegressor().score(X, np.zeros(len(X)))


def test_predict_rejects_wrong_feature_count():
    X, y = _linear_data()
    model = MLPRegressor(hidden_units=16, epochs=20, random_state=42).fit(X, y)
    with pytest.raises(ValueError):
        model.predict(np.zeros((5, 3)))


def test_history_and_n_epochs_tracked():
    X, y = _linear_data()
    model = MLPRegressor(
        hidden_units=16, epochs=30, patience=0, random_state=42
    ).fit(X, y)
    assert model.n_epochs_ == 30
    assert len(model.history_) == 30
    assert all(np.isfinite(v) and v >= 0 for v in model.history_)
    assert model.history_[-1] < model.history_[0]  # loss goes down
