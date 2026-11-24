"""Tests for quantile_regressor.model.QuantileRegressor and pinball_loss."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from quantile_regressor.model import QuantileRegressor, pinball_loss


def _linear(seed=0, n=200, slope=3.0, intercept=1.0, noise=0.1):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 1))
    y = slope * x[:, 0] + intercept + rng.normal(scale=noise, size=n)
    return x, y


def _heteroscedastic(seed=0, n=300):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 10, size=(n, 1))
    y = 2.0 * x[:, 0] + rng.normal(scale=0.2 + 0.3 * x[:, 0])
    return x, y


# --- pinball_loss ---------------------------------------------------------

def test_pinball_loss_exact_underprediction():
    # y - pred = 2 > 0, tau = 0.5: loss = 0.5 * 2 = 1.0
    assert pinball_loss([3.0], [1.0], 0.5) == pytest.approx(1.0)


def test_pinball_loss_exact_overprediction():
    # y - pred = -2 < 0, tau = 0.5: loss = (0.5 - 1) * (-2) = 1.0
    assert pinball_loss([1.0], [3.0], 0.5) == pytest.approx(1.0)


def test_pinball_loss_asymmetric_quantile():
    # Under-prediction at tau=0.9 costs 0.9/unit; over-prediction 0.1/unit.
    assert pinball_loss([3.0], [1.0], 0.9) == pytest.approx(1.8)
    assert pinball_loss([1.0], [3.0], 0.9) == pytest.approx(0.2)
    # Mean over a mixed batch, hand-computed:
    #   u = [2, -2] -> losses [1.8, 0.2] at tau=0.9 -> mean 1.0
    assert pinball_loss([3.0, 1.0], [1.0, 3.0], 0.9) == pytest.approx(1.0)


def test_pinball_loss_zero_when_prediction_equals_truth():
    assert pinball_loss([1.0, 2.0, -3.0], [1.0, 2.0, -3.0], 0.5) == 0.0
    assert pinball_loss([1.0, 2.0], [1.0, 2.0], 0.9) == 0.0


def test_pinball_loss_never_negative():
    rng = np.random.default_rng(1)
    y_true = rng.normal(size=100)
    y_pred = rng.normal(size=100)
    for q in (0.1, 0.25, 0.5, 0.9):
        assert pinball_loss(y_true, y_pred, q) >= 0.0


def test_pinball_loss_rejects_bad_quantile():
    with pytest.raises(ValueError):
        pinball_loss([1.0], [1.0], 0.0)
    with pytest.raises(ValueError):
        pinball_loss([1.0], [1.0], 1.0)
    with pytest.raises(ValueError):
        pinball_loss([1.0], [1.0], 1.5)


# --- fitting --------------------------------------------------------------

def test_median_recovers_slope_on_clean_data():
    x, y = _linear()
    model = QuantileRegressor(quantile=0.5, max_iter=2000,
                              learning_rate=0.05, random_state=0)
    model.fit(x, y)
    assert model.coef_[0] == pytest.approx(3.0, abs=0.15)
    assert model.intercept_ == pytest.approx(1.0, abs=0.15)
    assert model.n_iter_ > 0


def test_median_robust_to_outliers():
    x, y = _linear(noise=0.1)
    y_out = y.copy()
    y_out[:10] += 50.0  # gross outliers on 5% of the points
    clean = QuantileRegressor(quantile=0.5, max_iter=2000,
                              learning_rate=0.05, random_state=0).fit(x, y)
    dirty = QuantileRegressor(quantile=0.5, max_iter=2000,
                              learning_rate=0.05, random_state=0).fit(x, y_out)
    # The median fit should barely move when outliers are added.
    assert abs(dirty.coef_[0] - clean.coef_[0]) < 0.3
    assert abs(dirty.intercept_ - clean.intercept_) < 0.5


def test_q90_predictions_mostly_above_data():
    x, y = _heteroscedastic()
    model = QuantileRegressor(quantile=0.9, max_iter=2000,
                              learning_rate=0.05, random_state=0)
    model.fit(x, y)
    coverage = float(np.mean(y <= model.predict(x)))
    assert 0.80 <= coverage <= 0.98


def test_quantile_ordering():
    x, y = _heteroscedastic()
    lo = QuantileRegressor(quantile=0.1, max_iter=2000,
                           learning_rate=0.05, random_state=0).fit(x, y)
    hi = QuantileRegressor(quantile=0.9, max_iter=2000,
                           learning_rate=0.05, random_state=0).fit(x, y)
    assert np.all(hi.predict(x) >= lo.predict(x))


def test_predict_shape():
    x, y = _linear()
    model = QuantileRegressor(random_state=0).fit(x, y)
    preds = model.predict(x)
    assert preds.shape == (len(y),)
    assert np.all(np.isfinite(preds))


def test_score_is_r2():
    x, y = _linear()
    model = QuantileRegressor(random_state=0).fit(x, y)
    assert model.score(x, y) == pytest.approx(0.99, abs=0.02)


def test_determinism_with_same_random_state():
    x, y = _linear()
    a = QuantileRegressor(random_state=7).fit(x, y)
    b = QuantileRegressor(random_state=7).fit(x, y)
    assert np.allclose(a.coef_, b.coef_)
    assert a.intercept_ == b.intercept_
    assert a.n_iter_ == b.n_iter_


def test_no_intercept_gives_zero_intercept():
    x, y = _linear()
    model = QuantileRegressor(fit_intercept=False, random_state=0).fit(x, y)
    assert model.intercept_ == 0.0


# --- validation -----------------------------------------------------------

def test_rejects_quantile_outside_zero_one():
    with pytest.raises(ValueError):
        QuantileRegressor(quantile=0.0)
    with pytest.raises(ValueError):
        QuantileRegressor(quantile=1.0)
    with pytest.raises(ValueError):
        QuantileRegressor(quantile=-0.2)
    with pytest.raises(ValueError):
        QuantileRegressor(quantile=1.2)


def test_rejects_nonpositive_learning_rate():
    with pytest.raises(ValueError):
        QuantileRegressor(learning_rate=0.0)
    with pytest.raises(ValueError):
        QuantileRegressor(learning_rate=-0.01)


def test_rejects_1d_x():
    x, y = _linear()
    with pytest.raises(ValueError):
        QuantileRegressor().fit(x[:, 0], y)


def test_rejects_mismatched_lengths():
    x, y = _linear()
    with pytest.raises(ValueError):
        QuantileRegressor().fit(x, y[:-5])
    with pytest.raises(ValueError):
        QuantileRegressor().fit(x[:-5], y)


def test_rejects_empty_input():
    with pytest.raises(ValueError):
        QuantileRegressor().fit(np.empty((0, 1)), np.empty(0))


def test_rejects_nan():
    x, y = _linear()
    x_bad = x.copy()
    x_bad[0, 0] = np.nan
    with pytest.raises(ValueError):
        QuantileRegressor().fit(x_bad, y)
    y_bad = y.copy()
    y_bad[0] = np.nan
    with pytest.raises(ValueError):
        QuantileRegressor().fit(x, y_bad)


def test_predict_before_fit_raises():
    x, y = _linear()
    with pytest.raises(ValueError):
        QuantileRegressor().predict(x)
    with pytest.raises(ValueError):
        QuantileRegressor().score(x, y)
