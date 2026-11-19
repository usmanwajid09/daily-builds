"""Tests for ransac_line_fitting."""

import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from ransac_line_fitting import least_squares_line, ransac_line
from ransac_line_fitting.demo import make_points


def test_ransac_recovers_true_line_with_outliers():
    points, true_mask = make_points()
    result = ransac_line(points, n_trials=200, threshold=1.0, seed=0)
    assert result.slope == pytest.approx(2.0, abs=0.1)
    assert result.intercept == pytest.approx(1.0, abs=0.3)
    # Most true inliers recovered, few false positives.
    assert int((result.inliers & true_mask).sum()) >= 55
    assert int((result.inliers & ~true_mask).sum()) <= 5


def test_ransac_beats_least_squares_with_outliers():
    points, _ = make_points()
    result = ransac_line(points, n_trials=200, threshold=1.0, seed=0)
    ls_slope, ls_intercept = least_squares_line(points)
    ransac_error = abs(result.slope - 2.0) + abs(result.intercept - 1.0)
    ls_error = abs(ls_slope - 2.0) + abs(ls_intercept - 1.0)
    assert ransac_error < ls_error
    assert ls_error > 0.5  # least squares really is dragged off


def test_ransac_clean_data_matches_least_squares():
    rng = np.random.default_rng(3)
    x = rng.uniform(0, 10, size=50)
    y = -0.5 * x + 4.0 + rng.normal(0, 0.1, size=50)
    points = np.column_stack([x, y])
    result = ransac_line(points, n_trials=100, threshold=0.5, seed=3)
    ls_slope, ls_intercept = least_squares_line(points)
    assert result.slope == pytest.approx(ls_slope, abs=0.05)
    assert result.intercept == pytest.approx(ls_intercept, abs=0.1)
    assert result.n_inliers >= 45


def test_ransac_exact_line_all_inliers():
    x = np.linspace(0, 5, 20)
    points = np.column_stack([x, 3.0 * x - 2.0])
    result = ransac_line(points, n_trials=50, threshold=0.1, seed=1)
    assert result.slope == pytest.approx(3.0)
    assert result.intercept == pytest.approx(-2.0)
    assert result.n_inliers == 20


def test_ransac_deterministic_with_seed():
    points, _ = make_points()
    first = ransac_line(points, n_trials=100, threshold=1.0, seed=42)
    second = ransac_line(points, n_trials=100, threshold=1.0, seed=42)
    assert first.slope == second.slope
    assert first.intercept == second.intercept
    assert np.array_equal(first.inliers, second.inliers)


def test_ransac_result_predict():
    points, _ = make_points()
    result = ransac_line(points, n_trials=200, threshold=1.0, seed=0)
    assert result.predict(0.0) == pytest.approx(result.intercept)
    assert result.predict(5.0) == pytest.approx(5 * result.slope + result.intercept)


def test_least_squares_exact():
    x = np.array([0.0, 1.0, 2.0, 3.0])
    points = np.column_stack([x, 2.0 * x + 1.0])
    slope, intercept = least_squares_line(points)
    assert slope == pytest.approx(2.0)
    assert intercept == pytest.approx(1.0)


def test_least_squares_vertical_raises():
    points = np.array([[1.0, 0.0], [1.0, 2.0], [1.0, 5.0]])
    with pytest.raises(ValueError, match="vertical"):
        least_squares_line(points)


def test_ransac_vertical_data_raises():
    points = np.column_stack([np.ones(10), np.arange(10, dtype=float)])
    with pytest.raises(ValueError, match="no line reached"):
        ransac_line(points, n_trials=50, threshold=1.0, seed=0)


def test_ransac_tiny_threshold_raises():
    rng = np.random.default_rng(5)
    x = rng.uniform(0, 10, size=30)
    points = np.column_stack([x, 2.0 * x + 1.0 + rng.normal(0, 0.5, size=30)])
    # min_inliers=10: no line can gather 10 points within 1e-9 of it.
    # (With the default min_inliers=2 the sampled pair itself would
    # always qualify, so the failure needs a higher bar.)
    with pytest.raises(ValueError, match="no line reached"):
        ransac_line(points, n_trials=50, threshold=1e-9, min_inliers=10, seed=0)


def test_too_few_points_raises():
    with pytest.raises(ValueError, match="at least 2 points"):
        ransac_line([[1.0, 2.0]])


def test_1d_points_raises():
    with pytest.raises(ValueError, match="shape"):
        least_squares_line([1.0, 2.0, 3.0])


def test_nan_points_raises():
    with pytest.raises(ValueError, match="finite"):
        ransac_line([[0.0, 0.0], [np.nan, 1.0], [1.0, 1.0]])


def test_bad_n_trials_raises():
    points, _ = make_points()
    with pytest.raises(ValueError, match="n_trials"):
        ransac_line(points, n_trials=0)


def test_bad_threshold_raises():
    points, _ = make_points()
    with pytest.raises(ValueError, match="threshold"):
        ransac_line(points, threshold=0.0)


def test_bad_min_inliers_raises():
    points, _ = make_points()
    with pytest.raises(ValueError, match="min_inliers"):
        ransac_line(points, min_inliers=1)
    with pytest.raises(ValueError, match="min_inliers"):
        ransac_line(points, min_inliers=1000)


def test_high_outlier_fraction_still_recovers():
    # 50% outliers: RANSAC should still find the line with enough trials.
    rng = np.random.default_rng(9)
    x_in = rng.uniform(0, 10, size=40)
    y_in = 1.5 * x_in - 1.0 + rng.normal(0, 0.2, size=40)
    x_out = rng.uniform(0, 10, size=40)
    y_out = rng.uniform(-5, 20, size=40)
    points = np.column_stack([
        np.concatenate([x_in, x_out]), np.concatenate([y_in, y_out])])
    result = ransac_line(points, n_trials=500, threshold=0.8, seed=9)
    assert result.slope == pytest.approx(1.5, abs=0.15)
    assert result.intercept == pytest.approx(-1.0, abs=0.4)
    assert result.n_inliers >= 35
