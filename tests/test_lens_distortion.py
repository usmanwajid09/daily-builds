"""Tests for lens_distortion."""

import numpy as np
import pytest

from lens_distortion import (
    correct_image,
    distort_image,
    distort_points,
    undistort_points,
)
from lens_distortion.demo import make_grid

SHAPE = (120, 160)


def _ring_points(n=24, radius=55.0):
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    centre = np.array([(SHAPE[1] - 1) / 2.0, (SHAPE[0] - 1) / 2.0])
    return centre + radius * np.column_stack([np.cos(angles), np.sin(angles)])


def test_k_zero_is_identity_points():
    pts = _ring_points()
    out = distort_points(pts, 0.0, 0.0, image_shape=SHAPE)
    assert np.allclose(out, pts)


def test_k_zero_is_identity_image():
    grid = make_grid()
    assert np.array_equal(distort_image(grid, 0.0), grid)
    assert np.array_equal(correct_image(grid, 0.0), grid)


def test_barrel_pushes_points_outward():
    pts = _ring_points()
    centre = np.array([(SHAPE[1] - 1) / 2.0, (SHAPE[0] - 1) / 2.0])
    warped = distort_points(pts, 0.3, image_shape=SHAPE)
    r_before = np.linalg.norm(pts - centre, axis=1)
    r_after = np.linalg.norm(warped - centre, axis=1)
    assert np.all(r_after > r_before)


def test_pincushion_pulls_points_inward():
    pts = _ring_points()
    centre = np.array([(SHAPE[1] - 1) / 2.0, (SHAPE[0] - 1) / 2.0])
    warped = distort_points(pts, -0.3, image_shape=SHAPE)
    r_before = np.linalg.norm(pts - centre, axis=1)
    r_after = np.linalg.norm(warped - centre, axis=1)
    assert np.all(r_after < r_before)


def test_point_round_trip_barrel():
    pts = _ring_points()
    for k1, k2 in ((0.3, 0.0), (-0.25, 0.05), (0.2, 0.1)):
        warped = distort_points(pts, k1, k2, image_shape=SHAPE)
        back = undistort_points(warped, k1, k2, image_shape=SHAPE)
        assert np.allclose(back, pts, atol=1e-3)


def test_undistort_points_is_inverse_of_distort():
    rng = np.random.default_rng(0)
    pts = rng.uniform([0, 0], [SHAPE[1], SHAPE[0]], size=(50, 2))
    warped = distort_points(pts, 0.4, -0.05, image_shape=SHAPE)
    back = undistort_points(warped, 0.4, -0.05, image_shape=SHAPE)
    assert np.max(np.linalg.norm(back - pts, axis=1)) < 1e-3


def test_centre_point_does_not_move():
    centre = np.array([(SHAPE[1] - 1) / 2.0, (SHAPE[0] - 1) / 2.0])
    out = distort_points(centre[None, :], 0.5, 0.2, image_shape=SHAPE)
    assert np.allclose(out, centre[None, :])


def test_image_round_trip_low_error():
    grid = make_grid()
    distorted = distort_image(grid, 0.3, 0.05)
    corrected = correct_image(distorted, 0.3, 0.05)
    # Interior only: borders lose information to the black fill.
    inner = corrected[10:-10, 10:-10].astype(np.float64)
    target = grid[10:-10, 10:-10].astype(np.float64)
    mse = float(np.mean((inner - target) ** 2))
    assert mse < 1500.0


def test_distort_changes_image():
    grid = make_grid()
    distorted = distort_image(grid, 0.3)
    assert not np.array_equal(distorted, grid)
    assert distorted.shape == grid.shape
    assert distorted.dtype == np.uint8


def test_rgb_image_round_trip():
    rng = np.random.default_rng(2)
    img = rng.integers(0, 256, size=(60, 80, 3), dtype=np.uint8)
    distorted = distort_image(img, 0.2)
    assert distorted.shape == img.shape
    assert distorted.dtype == np.uint8
    corrected = correct_image(distorted, 0.2)
    inner = corrected[8:-8, 8:-8].astype(np.float64)
    target = img[8:-8, 8:-8].astype(np.float64)
    assert float(np.mean((inner - target) ** 2)) < 3000.0


def test_float_image_preserved():
    grid = make_grid().astype(np.float64)
    out = distort_image(grid, 0.2)
    assert out.dtype == np.float64
    assert out.shape == grid.shape


def test_custom_centre():
    pts = np.array([[50.0, 50.0], [60.0, 50.0]])
    centre = np.array([50.0, 50.0])
    out = distort_points(pts, 0.5, centre=centre, image_shape=SHAPE)
    assert np.allclose(out[0], [50.0, 50.0])
    assert out[1, 0] > 60.0  # pushed away from the custom centre


def test_k2_only_distortion():
    pts = _ring_points()
    warped = distort_points(pts, 0.0, 0.5, image_shape=SHAPE)
    centre = np.array([(SHAPE[1] - 1) / 2.0, (SHAPE[0] - 1) / 2.0])
    r_before = np.linalg.norm(pts - centre, axis=1)
    r_after = np.linalg.norm(warped - centre, axis=1)
    assert np.all(r_after > r_before)


def test_invalid_inputs_raise():
    grid = make_grid()
    with pytest.raises(ValueError):
        distort_image(grid.ravel(), 0.2)
    with pytest.raises(ValueError):
        distort_image(np.zeros((0, 0), dtype=np.uint8), 0.2)
    with pytest.raises(ValueError):
        distort_image(grid, "0.2")
    with pytest.raises(ValueError):
        distort_image(grid, np.nan)
    with pytest.raises(ValueError):
        distort_points(np.zeros((5, 3)), 0.2, image_shape=SHAPE)
    with pytest.raises(ValueError):
        distort_points(np.zeros((0, 2)), 0.2, image_shape=SHAPE)
    with pytest.raises(ValueError):
        distort_points(_ring_points(), 0.2)  # no centre, no image_shape
    with pytest.raises(ValueError):
        undistort_points(_ring_points(), 0.2, image_shape=SHAPE,
                         iterations=0)
    with pytest.raises(ValueError):
        distort_points(_ring_points(), 0.2, centre=[1, 2, 3],
                       image_shape=SHAPE)
