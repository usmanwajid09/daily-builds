import numpy as np
import pytest

from gaussian_process_regressor.model import GaussianProcessRegressor


def _sine(seed=42, n=40):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-3.0, 3.0, size=(n, 1))
    y = np.sin(x[:, 0]) + rng.normal(scale=0.1, size=n)
    return x, y


def test_fits_sine_wave_well():
    x, y = _sine()
    model = GaussianProcessRegressor().fit(x, y)
    assert model.score(x, y) > 0.95


def test_interpolates_training_points():
    # Evenly spaced, noise-free points: random draws can land nearly on
    # top of each other, which makes the kernel matrix ill-conditioned
    # at tiny sigma_n.
    x = np.linspace(-3.0, 3.0, 20).reshape(-1, 1)
    y = np.sin(x[:, 0])
    model = GaussianProcessRegressor(sigma_n=1e-6).fit(x, y)
    assert np.allclose(model.predict(x), y, atol=1e-3)


def test_predictive_std_small_near_data():
    x, y = _sine()
    model = GaussianProcessRegressor().fit(x, y)
    _, std = model.predict(x, return_std=True)
    assert np.all(std < 0.3)


def test_predictive_std_large_far_from_data():
    x, y = _sine()
    model = GaussianProcessRegressor().fit(x, y)
    _, std_near = model.predict(np.array([[0.0]]), return_std=True)
    _, std_far = model.predict(np.array([[10.0]]), return_std=True)
    assert std_far[0] > std_near[0]
    assert std_far[0] == pytest.approx(1.0, abs=0.05)  # prior std, sigma_f=1


def test_std_non_negative_and_shape():
    x, y = _sine()
    model = GaussianProcessRegressor().fit(x, y)
    mean, std = model.predict(x[:5], return_std=True)
    assert mean.shape == (5,) and std.shape == (5,)
    assert (std >= 0).all()


def test_log_marginal_likelihood_finite():
    x, y = _sine()
    model = GaussianProcessRegressor().fit(x, y)
    lml = model.log_marginal_likelihood()
    assert np.isfinite(lml)
    # A model with a well-matched length scale beats a badly mismatched one.
    bad = GaussianProcessRegressor(length_scale=0.05).fit(x, y)
    assert lml > bad.log_marginal_likelihood()


def test_longer_length_scale_smoother():
    x, y = _sine()
    short = GaussianProcessRegressor(length_scale=0.2).fit(x, y)
    long = GaussianProcessRegressor(length_scale=5.0).fit(x, y)
    grid = np.linspace(-3, 3, 50).reshape(-1, 1)
    # Long length scale -> predictions vary less between neighbours.
    assert np.abs(np.diff(long.predict(grid))).mean() < \
        np.abs(np.diff(short.predict(grid))).mean()


def test_predict_before_fit_raises():
    with pytest.raises(ValueError, match="not fitted"):
        GaussianProcessRegressor().predict(np.zeros((2, 1)))
    with pytest.raises(ValueError, match="not fitted"):
        GaussianProcessRegressor().log_marginal_likelihood()


def test_bad_hyperparams_raise():
    with pytest.raises(ValueError, match="length_scale"):
        GaussianProcessRegressor(length_scale=0.0)
    with pytest.raises(ValueError, match="sigma_f"):
        GaussianProcessRegressor(sigma_f=-1.0)
    with pytest.raises(ValueError, match="sigma_n"):
        GaussianProcessRegressor(sigma_n=0.0)
    with pytest.raises(ValueError, match="jitter"):
        GaussianProcessRegressor(jitter=-1e-8)


def test_mismatched_shapes_raise():
    x, y = _sine()
    with pytest.raises(ValueError, match="same number of samples"):
        GaussianProcessRegressor().fit(x, y[:10])
    with pytest.raises(ValueError, match="2D"):
        GaussianProcessRegressor().fit(x.ravel(), y)


def test_empty_data_raises():
    with pytest.raises(ValueError, match="empty"):
        GaussianProcessRegressor().fit(np.zeros((0, 1)), np.zeros(0))


def test_nan_input_raises():
    x, y = _sine()
    x[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        GaussianProcessRegressor().fit(x, y)


def test_wrong_feature_count_on_predict_raises():
    x, y = _sine()
    model = GaussianProcessRegressor().fit(x, y)
    with pytest.raises(ValueError, match="features"):
        model.predict(np.zeros((3, 2)))


def test_fit_returns_self():
    x, y = _sine()
    model = GaussianProcessRegressor()
    assert model.fit(x, y) is model
