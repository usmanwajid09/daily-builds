"""Tests for perspective_warp."""

import numpy as np
import pytest

from perspective_warp import (
    apply_homography,
    compute_homography,
    warp_perspective,
)
from perspective_warp.demo import make_image


def _corners(height=40, width=40):
    return np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
                    dtype=float)


def test_homography_maps_src_to_dst():
    src = _corners()
    dst = np.array([[5, 0], [34, 0], [39, 39], [0, 39]], dtype=float)
    h = compute_homography(src, dst)
    assert h.shape == (3, 3)
    assert h[2, 2] == pytest.approx(1.0)
    assert np.allclose(apply_homography(h, src), dst, atol=1e-6)


def test_identity_homography():
    src = _corners()
    h = compute_homography(src, src.copy())
    assert np.allclose(h, np.eye(3), atol=1e-9)
    image = make_image()
    warped = warp_perspective(image, h, (40, 40))
    assert warped.shape == image.shape
    assert np.allclose(warped, image, atol=1e-9)


def test_translation_warp_moves_bright_pixel():
    image = np.zeros((20, 20))
    image[5, 5] = 100.0
    # Shift the image right by 3 and down by 2: src point (x, y) lands
    # at output (x + 3, y + 2).
    src = np.array([[0, 0], [19, 0], [19, 19], [0, 19]], dtype=float)
    dst = src + np.array([3, 2])
    h = compute_homography(src, dst)
    warped = warp_perspective(image, h, (20, 20))
    assert warped[7, 8] == pytest.approx(100.0, abs=1e-6)
    assert warped[5, 5] == pytest.approx(0.0, abs=1e-6)


def test_scale_warp_of_linear_gradient_is_exact():
    # A linear gradient is reproduced exactly by bilinear interpolation
    # under an affine map, so a 2x scale should match the source gradient.
    yy, xx = np.mgrid[0:20, 0:20]
    image = 5.0 * xx + 2.0 * yy
    src = np.array([[0, 0], [19, 0], [19, 19], [0, 19]], dtype=float)
    dst = src * 2.0  # scale by 2
    h = compute_homography(src, dst)
    warped = warp_perspective(image, h, (40, 40))
    expected = 5.0 * (np.mgrid[0:40, 0:40][1] / 2.0) + 2.0 * (np.mgrid[0:40, 0:40][0] / 2.0)
    assert np.allclose(warped[:38, :38], expected[:38, :38], atol=1e-6)


def test_round_trip_warp_is_close():
    image = make_image()
    src = _corners()
    dst = np.array([[5, 0], [34, 0], [39, 39], [0, 39]], dtype=float)
    h = compute_homography(src, dst)
    warped = warp_perspective(image, h, (40, 40))
    back = warp_perspective(warped, np.linalg.inv(h), (40, 40))
    interior = np.abs(back[8:32, 8:32] - image[8:32, 8:32])
    assert interior.mean() < 5.0


def test_colour_image_warp():
    image = np.zeros((20, 20, 3))
    image[5, 5] = [10.0, 20.0, 30.0]
    src = np.array([[0, 0], [19, 0], [19, 19], [0, 19]], dtype=float)
    dst = src + np.array([2, 1])
    h = compute_homography(src, dst)
    warped = warp_perspective(image, h, (20, 20))
    assert warped.shape == (20, 20, 3)
    assert np.allclose(warped[6, 7], [10.0, 20.0, 30.0], atol=1e-6)


def test_point_count_validation():
    with pytest.raises(ValueError):
        compute_homography(np.zeros((3, 2)), np.zeros((4, 2)))
    with pytest.raises(ValueError):
        compute_homography(np.zeros((5, 2)), np.zeros((5, 2)))
    with pytest.raises(ValueError):
        compute_homography(np.zeros((4, 3)), np.zeros((4, 2)))


def test_collinear_points_raise():
    collinear = np.array([[0, 0], [1, 1], [2, 2], [3, 3]], dtype=float)
    good = _corners()
    with pytest.raises(ValueError):
        compute_homography(collinear, good)
    with pytest.raises(ValueError):
        compute_homography(good, collinear)


def test_non_finite_points_raise():
    bad = _corners()
    bad[0, 0] = np.inf
    with pytest.raises(ValueError):
        compute_homography(bad, _corners())
    with pytest.raises(ValueError):
        apply_homography(np.eye(3), np.array([[np.nan, 0.0]]))


def test_singular_homography_raises():
    image = make_image()
    singular = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=float)
    with pytest.raises(ValueError):
        warp_perspective(image, singular, (40, 40))
    with pytest.raises(ValueError):
        apply_homography(singular, [[10, 10]])
    with pytest.raises(ValueError):
        warp_perspective(image, np.zeros((2, 2)), (40, 40))


def test_output_shape_validation():
    image = make_image()
    with pytest.raises(ValueError):
        warp_perspective(image, np.eye(3), (0, 40))
    with pytest.raises(ValueError):
        warp_perspective(image, np.eye(3), (40,))
    with pytest.raises(ValueError):
        warp_perspective(np.zeros((5, 5, 3, 2)), np.eye(3), (5, 5))
    with pytest.raises(ValueError):
        warp_perspective(np.zeros((0, 5)), np.eye(3), (5, 5))


def test_apply_homography_validation():
    with pytest.raises(ValueError):
        apply_homography(np.eye(3), np.zeros((4, 3)))
    with pytest.raises(ValueError):
        apply_homography(np.eye(3), np.zeros((0, 2)))
    # A pure translation moves every point by the same offset.
    h = np.array([[1, 0, 4], [0, 1, -2], [0, 0, 1]], dtype=float)
    assert np.allclose(apply_homography(h, [[1, 2], [10, 10]]), [[5, 0], [14, 8]])
