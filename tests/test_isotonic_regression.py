"""Tests for isotonic regression (PAVA)."""

import numpy as np
import pytest

from isotonic_regression.model import IsotonicRegression


def test_fitted_values_are_monotone_increasing():
    rng = np.random.default_rng(0)
    x = np.sort(rng.uniform(0, 10, size=50))
    y = rng.normal(size=50)
    model = IsotonicRegression().fit(x, y)
    assert np.all(np.diff(model.predict(x)) >= -1e-12)


def test_fitted_values_are_monotone_decreasing():
    rng = np.random.default_rng(1)
    x = np.sort(rng.uniform(0, 10, size=50))
    y = rng.normal(size=50)
    model = IsotonicRegression(increasing=False).fit(x, y)
    assert np.all(np.diff(model.predict(x)) <= 1e-12)


def test_exact_fit_on_already_monotone_data():
    x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    model = IsotonicRegression().fit(x, y)
    assert np.allclose(model.predict(x), y)


def test_pava_pools_a_single_violation():
    # y = [1, 3, 2, 4]: the 3 > 2 violation pools to (3+2)/2 = 2.5.
    x = np.array([0.0, 1.0, 2.0, 3.0])
    y = np.array([1.0, 3.0, 2.0, 4.0])
    model = IsotonicRegression().fit(x, y)
    assert np.allclose(model.predict(x), [1.0, 2.5, 2.5, 4.0])


def test_pava_pools_whole_sequence_when_decreasing():
    x = np.array([0.0, 1.0, 2.0, 3.0])
    y = np.array([4.0, 3.0, 2.0, 1.0])
    model = IsotonicRegression().fit(x, y)
    assert np.allclose(model.predict(x), [2.5, 2.5, 2.5, 2.5])


def test_unsorted_input_is_sorted_internally():
    x = np.array([3.0, 1.0, 2.0, 0.0])
    y = np.array([4.0, 2.0, 3.0, 1.0])
    model = IsotonicRegression().fit(x, y)
    # Sorted x is [0, 1, 2, 3] with y [1, 2, 3, 4]; already monotone.
    assert np.allclose(model.predict(np.array([0.0, 1.0, 2.0, 3.0])), [1.0, 2.0, 3.0, 4.0])


def test_interpolation_between_points():
    x = np.array([0.0, 2.0, 4.0])
    y = np.array([0.0, 2.0, 4.0])
    model = IsotonicRegression().fit(x, y)
    assert np.isclose(model.predict(np.array([1.0]))[0], 1.0)
    assert np.isclose(model.predict(np.array([3.0]))[0], 3.0)


def test_out_of_bounds_nan_by_default():
    x = np.array([0.0, 1.0, 2.0])
    y = np.array([0.0, 1.0, 2.0])
    model = IsotonicRegression().fit(x, y)
    preds = model.predict(np.array([-1.0, 3.0]))
    assert np.isnan(preds).all()


def test_out_of_bounds_clip():
    x = np.array([0.0, 1.0, 2.0])
    y = np.array([0.0, 1.0, 2.0])
    model = IsotonicRegression(out_of_bounds="clip").fit(x, y)
    assert np.allclose(model.predict(np.array([-1.0, 3.0])), [0.0, 2.0])


def test_out_of_bounds_raise():
    x = np.array([0.0, 1.0, 2.0])
    y = np.array([0.0, 1.0, 2.0])
    model = IsotonicRegression(out_of_bounds="raise").fit(x, y)
    with pytest.raises(ValueError, match="outside the training range"):
        model.predict(np.array([5.0]))


def test_invalid_out_of_bounds_raises():
    with pytest.raises(ValueError, match="out_of_bounds"):
        IsotonicRegression(out_of_bounds="extrapolate")


def test_sample_weights_change_fit():
    x = np.array([0.0, 1.0, 2.0])
    y = np.array([1.0, 3.0, 2.0])
    # Heavy weight on the last point pulls the pooled average toward 2:
    # (3*1 + 2*100) / 101 = 2.0099.
    model = IsotonicRegression().fit(x, y, sample_weight=[1.0, 1.0, 100.0])
    preds = model.predict(x)
    assert np.isclose(preds[1], preds[2])
    assert np.isclose(preds[1], 203 / 101, atol=1e-9)


def test_negative_sample_weight_raises():
    with pytest.raises(ValueError, match="must not be negative"):
        IsotonicRegression().fit(
            np.array([0.0, 1.0]), np.array([1.0, 2.0]),
            sample_weight=[1.0, -1.0],
        )


def test_fit_transform_returns_fitted_values():
    x = np.array([0.0, 1.0, 2.0, 3.0])
    y = np.array([1.0, 3.0, 2.0, 4.0])
    model = IsotonicRegression()
    fitted = model.fit_transform(x, y)
    assert np.allclose(fitted, [1.0, 2.5, 2.5, 4.0])


def test_score_is_high_on_monotone_data():
    x = np.linspace(0, 10, 50)
    y = np.sqrt(x) + np.random.default_rng(5).normal(scale=0.05, size=50)
    model = IsotonicRegression().fit(x, y)
    assert model.score(x, y) > 0.95


def test_predict_before_fit_raises():
    with pytest.raises(ValueError, match="not fitted"):
        IsotonicRegression().predict(np.array([1.0]))


def test_empty_data_raises():
    with pytest.raises(ValueError, match="empty"):
        IsotonicRegression().fit(np.array([]), np.array([]))


def test_mismatched_lengths_raises():
    with pytest.raises(ValueError, match="inconsistent lengths"):
        IsotonicRegression().fit(np.array([0.0, 1.0]), np.array([1.0]))


def test_2d_x_raises():
    with pytest.raises(ValueError, match="1D"):
        IsotonicRegression().fit(np.zeros((3, 2)), np.zeros(3))


def test_nan_input_raises():
    with pytest.raises(ValueError, match="NaN"):
        IsotonicRegression().fit(np.array([0.0, np.nan]), np.array([1.0, 2.0]))
