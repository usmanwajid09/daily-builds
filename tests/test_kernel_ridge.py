import numpy as np
import pytest

from kernel_ridge.model import KernelRidgeRegressor


def _sine(seed=42, n=100):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-3.0, 3.0, size=(n, 1))
    y = np.sin(x[:, 0]) + rng.normal(scale=0.1, size=n)
    return x, y


def _linear(seed=42, n=100):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 2))
    y = 2.0 * x[:, 0] - x[:, 1] + rng.normal(scale=0.1, size=n)
    return x, y


def test_fits_sine_wave_well():
    x, y = _sine()
    model = KernelRidgeRegressor(alpha=0.1, gamma=0.5).fit(x, y)
    assert model.score(x, y) > 0.95


def test_fits_linear_data_well():
    x, y = _linear()
    model = KernelRidgeRegressor(alpha=0.1).fit(x, y)
    assert model.score(x, y) > 0.95


def test_interpolates_training_points_with_tiny_alpha():
    # Evenly spaced points: random draws can land nearly on top of each
    # other, which makes the kernel matrix ill-conditioned at tiny alpha.
    x = np.linspace(-3.0, 3.0, 30).reshape(-1, 1)
    y = np.sin(x[:, 0])
    model = KernelRidgeRegressor(alpha=1e-8, gamma=0.5).fit(x, y)
    assert np.allclose(model.predict(x), y, atol=1e-4)


def test_large_alpha_shrinks_predictions():
    x, y = _sine()
    small = KernelRidgeRegressor(alpha=0.1, gamma=0.5).fit(x, y)
    large = KernelRidgeRegressor(alpha=1000.0, gamma=0.5).fit(x, y)
    assert np.abs(large.predict(x)).mean() < np.abs(small.predict(x)).mean()


def test_default_gamma_is_one_over_features():
    x, y = _linear()
    model = KernelRidgeRegressor(alpha=0.1).fit(x, y)
    assert model.gamma_ == pytest.approx(0.5)


def test_dual_coef_shape():
    x, y = _sine()
    model = KernelRidgeRegressor().fit(x, y)
    assert model.dual_coef_.shape == (len(y),)
    assert model.predict(x[:5]).shape == (5,)


def test_predict_before_fit_raises():
    with pytest.raises(ValueError, match="not fitted"):
        KernelRidgeRegressor().predict(np.zeros((2, 1)))


def test_bad_alpha_raises():
    with pytest.raises(ValueError, match="alpha"):
        KernelRidgeRegressor(alpha=0.0)
    with pytest.raises(ValueError, match="alpha"):
        KernelRidgeRegressor(alpha=-1.0)


def test_bad_gamma_raises():
    with pytest.raises(ValueError, match="gamma"):
        KernelRidgeRegressor(gamma=0.0)
    with pytest.raises(ValueError, match="gamma"):
        KernelRidgeRegressor(gamma=-2.0)


def test_mismatched_shapes_raise():
    x, y = _sine()
    with pytest.raises(ValueError, match="same number of samples"):
        KernelRidgeRegressor().fit(x, y[:10])
    with pytest.raises(ValueError, match="2D"):
        KernelRidgeRegressor().fit(x.ravel(), y)


def test_empty_data_raises():
    with pytest.raises(ValueError, match="empty"):
        KernelRidgeRegressor().fit(np.zeros((0, 1)), np.zeros(0))


def test_nan_input_raises():
    x, y = _sine()
    x[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        KernelRidgeRegressor().fit(x, y)


def test_wrong_feature_count_on_predict_raises():
    x, y = _sine()
    model = KernelRidgeRegressor().fit(x, y)
    with pytest.raises(ValueError, match="features"):
        model.predict(np.zeros((3, 2)))


def test_fit_returns_self():
    x, y = _sine()
    model = KernelRidgeRegressor()
    assert model.fit(x, y) is model
