import numpy as np
import pytest

from lasso_regression.model import LassoRegression


def _data(seed=42, n=200):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 5))
    # Only the first two features matter; the rest are pure noise.
    y = 2.0 * x[:, 0] - 1.5 * x[:, 1] + rng.normal(scale=0.5, size=n)
    return x, y


def test_recovers_coefficients_with_small_alpha():
    x, y = _data()
    model = LassoRegression(alpha=0.1).fit(x, y)
    assert model.coef_[0] == pytest.approx(2.0, abs=0.15)
    assert model.coef_[1] == pytest.approx(-1.5, abs=0.15)


def test_large_alpha_zeroes_noise_features():
    x, y = _data()
    model = LassoRegression(alpha=50.0).fit(x, y)
    assert np.all(model.coef_[2:] == 0.0)
    assert model.coef_[0] > 1.0  # the real signal survives


def test_large_alpha_zeroes_everything():
    x, y = _data()
    model = LassoRegression(alpha=10000.0).fit(x, y)
    assert np.all(model.coef_ == 0.0)
    assert model.predict(x).std() == pytest.approx(0.0, abs=1e-12)


def test_alpha_zero_matches_ols():
    x, y = _data()
    model = LassoRegression(alpha=0.0, tol=1e-8).fit(x, y)
    ols, *_ = np.linalg.lstsq(np.c_[x, np.ones(len(x))], y, rcond=None)
    assert np.allclose(model.coef_, ols[:-1], atol=1e-2)
    assert model.intercept_ == pytest.approx(ols[-1], abs=1e-2)


def test_r2_is_high_on_signal_data():
    x, y = _data()
    model = LassoRegression(alpha=1.0).fit(x, y)
    assert model.score(x, y) >= 0.9


def test_intercept_recovers_offset():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(100, 1))
    y = 3.0 * x[:, 0] + 10.0
    model = LassoRegression(alpha=0.01).fit(x, y)
    assert model.intercept_ == pytest.approx(10.0, abs=0.1)
    assert model.coef_[0] == pytest.approx(3.0, abs=0.1)


def test_constant_feature_column_handled():
    rng = np.random.default_rng(4)
    x = np.column_stack([rng.normal(size=100), np.full(100, 5.0)])
    y = 2.0 * x[:, 0] + rng.normal(scale=0.1, size=100)
    model = LassoRegression(alpha=1.0).fit(x, y)
    assert model.coef_[1] == 0.0
    assert np.isfinite(model.predict(x)).all()


def test_single_feature():
    x = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = np.array([2.0, 4.0, 6.0, 8.0])
    model = LassoRegression(alpha=0.001).fit(x, y)
    assert model.coef_[0] == pytest.approx(2.0, abs=0.05)


def test_predict_before_fit_raises():
    with pytest.raises(RuntimeError, match="not fitted"):
        LassoRegression().predict(np.array([[0.0]]))


def test_negative_alpha_raises():
    with pytest.raises(ValueError, match="alpha"):
        LassoRegression(alpha=-1.0)


def test_mismatched_lengths_raise():
    with pytest.raises(ValueError, match="same number of samples"):
        LassoRegression().fit(np.zeros((3, 2)), np.zeros(2))


def test_nan_raises():
    with pytest.raises(ValueError, match="NaN"):
        LassoRegression().fit(np.array([[0.0, np.nan]]), np.array([1.0]))


def test_empty_data_raises():
    with pytest.raises(ValueError, match="empty"):
        LassoRegression().fit(np.zeros((0, 2)), np.zeros(0))
