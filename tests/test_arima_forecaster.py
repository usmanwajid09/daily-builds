"""Tests for arima_forecaster. All deterministic (fixed seeds)."""

import numpy as np
import pytest

from arima_forecaster import ARForecaster
from arima_forecaster.model import _normal_quantile


def make_ar_series(phi, n, seed, sigma=1.0, intercept=0.0, burn_in=200):
    """Generate an AR(p) series with the given coefficients (plus burn-in)."""
    rng = np.random.default_rng(seed)
    p = len(phi)
    total = n + burn_in
    y = np.zeros(total)
    y[:p] = rng.normal(scale=sigma, size=p)
    noise = rng.normal(scale=sigma, size=total)
    for t in range(p, total):
        y[t] = intercept + float(np.dot(phi, y[t - p:t][::-1])) + noise[t]
    return y[burn_in:]


# ---------------------------------------------------------------------------
# Fitting / coefficient recovery
# ---------------------------------------------------------------------------

def test_recovers_known_ar2_coefficients():
    y = make_ar_series([0.6, -0.2], n=2000, seed=42)
    model = ARForecaster(p=2).fit(y)
    np.testing.assert_allclose(model.coef_, [0.6, -0.2], atol=0.08)
    assert abs(model.intercept_) < 0.08


def test_recovers_ar1_coefficient():
    y = make_ar_series([0.8], n=2000, seed=1)
    model = ARForecaster(p=1).fit(y)
    np.testing.assert_allclose(model.coef_, [0.8], atol=0.05)


def test_fit_returns_self_and_sets_attributes():
    y = make_ar_series([0.6, -0.2], n=100, seed=3)
    model = ARForecaster(p=2)
    assert model.fitted is False
    assert model.fit(y) is model
    assert model.fitted is True
    assert model.coef_.shape == (2,)
    assert isinstance(model.intercept_, float)
    assert model.sigma2_ > 0
    # Residual variance should be close to the true noise variance (1.0).
    assert abs(model.sigma2_ - 1.0) < 0.35


def test_sigma2_scales_with_noise():
    y = make_ar_series([0.5], n=1000, seed=5, sigma=2.0)
    model = ARForecaster(p=1).fit(y)
    assert abs(model.sigma2_ - 4.0) < 0.8


# ---------------------------------------------------------------------------
# Forecasting
# ---------------------------------------------------------------------------

def test_predict_shape_and_length():
    y = make_ar_series([0.6, -0.2], n=100, seed=3)
    model = ARForecaster(p=2).fit(y)
    for steps in (1, 5, 10):
        out = model.predict(steps=steps)
        assert isinstance(out, np.ndarray)
        assert out.shape == (steps,)
        assert np.all(np.isfinite(out))


def test_predict_one_step_matches_manual_computation():
    y = make_ar_series([0.6, -0.2], n=100, seed=3)
    model = ARForecaster(p=2).fit(y)
    expected = model.intercept_ + model.coef_[0] * y[-1] + model.coef_[1] * y[-2]
    assert model.predict(steps=1)[0] == pytest.approx(expected)


def test_forecast_beats_naive_last_value_baseline():
    y = make_ar_series([0.6, -0.2], n=210, seed=7, sigma=0.5)
    train, truth = y[:200], y[200:]
    model = ARForecaster(p=2).fit(train)
    forecast = model.predict(steps=10)
    naive = np.full(10, train[-1])
    assert np.mean((forecast - truth) ** 2) < np.mean((naive - truth) ** 2)


def test_predict_is_deterministic():
    y = make_ar_series([0.6, -0.2], n=100, seed=3)
    model = ARForecaster(p=2).fit(y)
    np.testing.assert_array_equal(model.predict(steps=5), model.predict(steps=5))


# ---------------------------------------------------------------------------
# Prediction intervals
# ---------------------------------------------------------------------------

def test_interval_ordering_and_shapes():
    y = make_ar_series([0.6, -0.2], n=100, seed=3)
    model = ARForecaster(p=2).fit(y)
    mean, lower, upper = model.forecast_interval(steps=5)
    assert mean.shape == lower.shape == upper.shape == (5,)
    assert np.all(lower <= mean)
    assert np.all(mean <= upper)
    np.testing.assert_allclose(mean, model.predict(steps=5))


def test_interval_widens_with_horizon():
    y = make_ar_series([0.6, -0.2], n=500, seed=11)
    model = ARForecaster(p=2).fit(y)
    _, lower, upper = model.forecast_interval(steps=10)
    widths = upper - lower
    assert np.all(np.diff(widths) > 0)


def test_interval_contains_true_values_most_of_the_time():
    covered, total = 0, 0
    for seed in range(50):
        y = make_ar_series([0.6, -0.2], n=305, seed=100 + seed)
        model = ARForecaster(p=2).fit(y[:300])
        _, lower, upper = model.forecast_interval(steps=5, alpha=0.05)
        covered += int(np.sum((y[300:] >= lower) & (y[300:] <= upper)))
        total += 5
    assert covered / total >= 0.75  # nominal coverage is 0.95


def test_smaller_alpha_gives_wider_interval():
    y = make_ar_series([0.6, -0.2], n=100, seed=3)
    model = ARForecaster(p=2).fit(y)
    _, lo95, hi95 = model.forecast_interval(steps=5, alpha=0.05)
    _, lo99, hi99 = model.forecast_interval(steps=5, alpha=0.01)
    assert np.all((hi99 - lo99) > (hi95 - lo95))


def test_non_table_alpha_uses_inverse_normal():
    y = make_ar_series([0.6, -0.2], n=100, seed=3)
    model = ARForecaster(p=2).fit(y)
    _, lo, hi = model.forecast_interval(steps=1, alpha=0.02)
    # z for alpha=0.02 is ~2.326; half-width should be z * sigma.
    half_width = (hi[0] - lo[0]) / 2.0
    assert half_width == pytest.approx(2.3263478740408408 * np.sqrt(model.sigma2_),
                                       rel=1e-3)


def test_normal_quantile_matches_known_values():
    assert _normal_quantile(0.975) == pytest.approx(1.959964, rel=1e-4)
    assert _normal_quantile(0.995) == pytest.approx(2.575829, rel=1e-4)
    assert _normal_quantile(0.5) == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def test_empty_series_raises():
    with pytest.raises(ValueError, match="empty"):
        ARForecaster(p=2).fit([])


def test_too_short_series_raises():
    with pytest.raises(ValueError, match="greater than p"):
        ARForecaster(p=3).fit([1.0, 2.0, 3.0])


def test_non_1d_input_raises():
    with pytest.raises(ValueError, match="1D"):
        ARForecaster(p=2).fit([[1.0, 2.0], [3.0, 4.0]])


def test_nan_input_raises():
    with pytest.raises(ValueError, match="finite"):
        ARForecaster(p=2).fit([1.0, np.nan, 3.0, 4.0, 5.0])


def test_inf_input_raises():
    with pytest.raises(ValueError, match="finite"):
        ARForecaster(p=2).fit([1.0, 2.0, np.inf, 4.0, 5.0])


def test_p_less_than_one_raises():
    with pytest.raises(ValueError, match="positive integer"):
        ARForecaster(p=0)
    with pytest.raises(ValueError, match="positive integer"):
        ARForecaster(p=-2)


def test_p_not_int_raises():
    with pytest.raises(ValueError, match="positive integer"):
        ARForecaster(p=2.5)
    with pytest.raises(ValueError, match="positive integer"):
        ARForecaster(p="2")


def test_predict_before_fit_raises():
    with pytest.raises(RuntimeError, match="not fitted"):
        ARForecaster(p=2).predict(steps=3)


def test_forecast_interval_before_fit_raises():
    with pytest.raises(RuntimeError, match="not fitted"):
        ARForecaster(p=2).forecast_interval(steps=3)


def test_steps_non_positive_raises():
    y = make_ar_series([0.6, -0.2], n=100, seed=3)
    model = ARForecaster(p=2).fit(y)
    with pytest.raises(ValueError, match="positive integer"):
        model.predict(steps=0)
    with pytest.raises(ValueError, match="positive integer"):
        model.predict(steps=-3)
    with pytest.raises(ValueError, match="positive integer"):
        model.forecast_interval(steps=0)
    with pytest.raises(ValueError, match="positive integer"):
        model.predict(steps=2.5)


def test_bad_alpha_raises():
    y = make_ar_series([0.6, -0.2], n=100, seed=3)
    model = ARForecaster(p=2).fit(y)
    for bad in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError, match="alpha"):
            model.forecast_interval(steps=3, alpha=bad)
