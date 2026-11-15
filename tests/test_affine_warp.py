"""Tests for affine_warp."""

import numpy as np
import pytest

from affine_warp import (
    build_affine_matrix,
    rotation_matrix,
    warp_affine,
    warp_rotate,
    warp_scale,
    warp_translate,
)
from affine_warp.demo import make_test_image


def test_identity_warp_is_unchanged():
    image = np.random.default_rng(0).random((10, 12)) * 255.0
    out = warp_affine(image, np.eye(3))
    assert np.allclose(out, image)


def test_translation_moves_bright_pixel():
    image = np.zeros((15, 15))
    image[5, 5] = 100.0
    out = warp_translate(image, dx=2, dy=3)
    assert out[8, 7] == pytest.approx(100.0)
    assert out[5, 5] == pytest.approx(0.0)


def test_half_pixel_translation_splits_intensity():
    image = np.zeros((15, 15))
    image[5, 5] = 100.0
    out = warp_translate(image, dx=0.5, dy=0.0)
    assert out[5, 5] == pytest.approx(50.0)
    assert out[5, 6] == pytest.approx(50.0)


def test_rotation_zero_is_identity():
    image = make_test_image()
    out = warp_rotate(image, 0.0)
    assert np.allclose(out, image)


def test_rotation_360_is_identity():
    image = make_test_image()
    out = warp_rotate(image, 360.0)
    assert np.allclose(out, image, atol=1e-6)


def test_rotation_preserves_total_brightness():
    image = make_test_image()
    out = warp_rotate(image, 30.0)
    assert out.sum() == pytest.approx(image.sum(), rel=0.05)


def test_scale_about_centre_keeps_centre():
    image = make_test_image()
    out = warp_scale(image, 2.0)
    assert out[30, 40] == pytest.approx(255.0)
    assert out.shape == image.shape


def test_rgb_warp_keeps_shape():
    image = np.random.default_rng(1).random((10, 12, 3)) * 255.0
    out = warp_translate(image, dx=1, dy=1)
    assert out.shape == image.shape


def test_output_shape_respected():
    image = make_test_image()
    out = warp_affine(image, np.eye(3), output_shape=(30, 40))
    assert out.shape == (30, 40)


def test_out_of_bounds_filled():
    image = np.ones((10, 10)) * 50.0
    out = warp_translate(image, dx=3, dy=0, fill_value=7.0)
    assert out[:, :3].min() == pytest.approx(7.0)
    assert out[:, :3].max() == pytest.approx(7.0)
    assert out[:, 3:].min() == pytest.approx(50.0)


def test_2x3_matrix_accepted():
    image = make_test_image()
    out = warp_affine(image, np.array([[1.0, 0.0, 2.0],
                                       [0.0, 1.0, 0.0]]))
    assert out.shape == image.shape
    assert out[20, 32] == pytest.approx(255.0)


def test_build_affine_matrix_translation():
    matrix = build_affine_matrix(translation=(4.0, -2.0))
    assert matrix[0, 2] == pytest.approx(4.0)
    assert matrix[1, 2] == pytest.approx(-2.0)


def test_rotation_matrix_about_point():
    matrix = rotation_matrix(90.0, center=(10.0, 10.0))
    point = matrix @ np.array([10.0, 10.0, 1.0])
    assert point[0] == pytest.approx(10.0)
    assert point[1] == pytest.approx(10.0)
    # (11, 10) rotated 90 deg CCW about (10, 10) -> (10, 11)
    point = matrix @ np.array([11.0, 10.0, 1.0])
    assert point[0] == pytest.approx(10.0)
    assert point[1] == pytest.approx(11.0)


def test_singular_matrix_raises():
    with pytest.raises(ValueError):
        warp_affine(np.zeros((5, 5)), np.zeros((3, 3)))


def test_bad_matrix_shape_raises():
    with pytest.raises(ValueError):
        warp_affine(np.zeros((5, 5)), np.eye(2))


def test_1d_image_raises():
    with pytest.raises(ValueError):
        warp_affine(np.zeros(10), np.eye(3))


def test_bad_output_shape_raises():
    with pytest.raises(ValueError):
        warp_affine(np.zeros((5, 5)), np.eye(3), output_shape=(0, 5))


def test_zero_scale_raises():
    with pytest.raises(ValueError):
        build_affine_matrix(scale=(0.0, 1.0))
    with pytest.raises(ValueError):
        warp_scale(np.zeros((5, 5)), 0.0)
