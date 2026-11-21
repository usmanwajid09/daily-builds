"""Tests for ransac_homography."""

import numpy as np
import pytest

from ransac_homography import (
    apply_homography,
    compute_homography,
    ransac_homography,
    reprojection_errors,
)
from ransac_homography.demo import make_data


def _known_homography():
    angle = np.deg2rad(20.0)
    c, s = np.cos(angle), np.sin(angle)
    return np.array(
        [[c, -s, 30.0], [s, c, -10.0], [0.0005, -0.0003, 1.0]]
    )


def _grid_points(n=25, span=200.0, seed=0):
    rng = np.random.default_rng(seed)
    return rng.uniform(0, span, size=(n, 2))


def test_dlt_recovers_identity():
    pts = _grid_points()
    h = compute_homography(pts, pts)
    assert np.allclose(h, np.eye(3), atol=1e-6)


def test_dlt_recovers_translation():
    pts = _grid_points()
    dst = pts + np.array([15.0, -25.0])
    h = compute_homography(pts, dst)
    assert np.allclose(h[:2, 2], [15.0, -25.0], atol=1e-6)
    assert np.allclose(h[2, :], [0, 0, 1], atol=1e-6)


def test_dlt_recovers_perspective_homography():
    true_h = _known_homography()
    src = _grid_points()
    dst = apply_homography(true_h, src)
    h = compute_homography(src, dst)
    assert np.allclose(h, true_h, atol=1e-6)


def test_dlt_with_exactly_four_points():
    true_h = _known_homography()
    src = np.array([[0.0, 0.0], [100.0, 0.0], [100.0, 100.0], [0.0, 100.0]])
    dst = apply_homography(true_h, src)
    h = compute_homography(src, dst)
    assert np.allclose(h, true_h, atol=1e-6)


def test_apply_homography_known_points():
    h = np.array([[2.0, 0.0, 5.0], [0.0, 2.0, -3.0], [0.0, 0.0, 1.0]])
    pts = np.array([[1.0, 2.0], [0.0, 0.0], [-4.0, 7.0]])
    out = apply_homography(h, pts)
    assert np.allclose(out, [[7.0, 1.0], [5.0, -3.0], [-3.0, 11.0]])


def test_reprojection_errors_zero_for_exact_fit():
    true_h = _known_homography()
    src = _grid_points()
    dst = apply_homography(true_h, src)
    errors = reprojection_errors(true_h, src, dst)
    assert errors.shape == (len(src),)
    assert np.allclose(errors, 0.0, atol=1e-8)


def test_reprojection_errors_measure_offset():
    src = np.array([[0.0, 0.0], [10.0, 0.0]])
    dst = np.array([[3.0, 4.0], [13.0, 4.0]])
    errors = reprojection_errors(np.eye(3), src, dst)
    assert np.allclose(errors, [5.0, 5.0])


def test_ransac_recovers_homography_with_outliers():
    true_h, src, dst, outliers = make_data()
    est_h, inliers, _ = ransac_homography(
        src, dst, threshold=3.0, max_iterations=1000, seed=42
    )
    # Noise (sigma 0.5 px) limits the fit; the translation terms carry
    # most of the residual, so allow a sub-pixel Frobenius error and
    # check the reprojection error directly.
    assert np.linalg.norm(est_h - true_h) < 1.0
    assert reprojection_errors(est_h, src[inliers], dst[inliers]).mean() < 1.0
    true_inliers = np.ones(len(src), dtype=bool)
    true_inliers[outliers] = False
    assert (inliers & true_inliers).sum() == true_inliers.sum()
    assert (~inliers & ~true_inliers).sum() >= 18  # most outliers rejected


def test_ransac_exact_without_noise_or_outliers():
    true_h = _known_homography()
    src = _grid_points(n=30)
    dst = apply_homography(true_h, src)
    est_h, inliers, iterations = ransac_homography(
        src, dst, threshold=1.0, max_iterations=100, seed=0
    )
    assert np.all(inliers)
    assert np.allclose(est_h, true_h, atol=1e-6)
    assert iterations == 1  # early stop: all points are inliers


def test_ransac_uses_seed_for_reproducibility():
    _, src, dst, _ = make_data()
    first = ransac_homography(src, dst, threshold=3.0, seed=7)
    second = ransac_homography(src, dst, threshold=3.0, seed=7)
    assert np.allclose(first[0], second[0])
    assert np.array_equal(first[1], second[1])
    assert first[2] == second[2]


def test_ransac_raises_when_no_model_found():
    rng = np.random.default_rng(0)
    src = rng.uniform(0, 100, size=(20, 2))
    dst = rng.uniform(0, 100, size=(20, 2))  # unrelated points
    with pytest.raises(RuntimeError):
        ransac_homography(src, dst, threshold=0.001, max_iterations=50, seed=0)


def test_too_few_points_raise():
    pts = _grid_points(n=3)
    with pytest.raises(ValueError):
        compute_homography(pts, pts)
    with pytest.raises(ValueError):
        ransac_homography(pts, pts)


def test_mismatched_counts_raise():
    src = _grid_points(n=10)
    dst = _grid_points(n=9)
    with pytest.raises(ValueError):
        compute_homography(src, dst)
    with pytest.raises(ValueError):
        ransac_homography(src, dst)


def test_collinear_points_raise():
    src = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    dst = src + 1.0
    with pytest.raises(ValueError):
        compute_homography(src, dst)


def test_invalid_shapes_raise():
    pts = _grid_points()
    with pytest.raises(ValueError):
        compute_homography(pts.ravel(), pts)
    with pytest.raises(ValueError):
        compute_homography(pts, np.zeros((10, 3)))
    with pytest.raises(ValueError):
        apply_homography(np.eye(2), pts)
    with pytest.raises(ValueError):
        apply_homography(np.eye(3), np.array([[np.nan, 0.0]]))


def test_invalid_ransac_params_raise():
    _, src, dst, _ = make_data()
    with pytest.raises(ValueError):
        ransac_homography(src, dst, threshold=0)
    with pytest.raises(ValueError):
        ransac_homography(src, dst, threshold=-1)
    with pytest.raises(ValueError):
        ransac_homography(src, dst, max_iterations=0)
    with pytest.raises(ValueError):
        ransac_homography(src, dst, min_inliers=3)
    with pytest.raises(ValueError):
        ransac_homography(src, dst, min_inliers=61)
