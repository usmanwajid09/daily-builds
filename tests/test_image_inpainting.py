"""Tests for image_inpainting."""

import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from image_inpainting import inpaint, inpaint_with_info
from image_inpainting.demo import make_gradient


def test_empty_mask_returns_copy_unchanged():
    image = np.arange(9, dtype=float).reshape(3, 3)
    filled, iterations, converged = inpaint_with_info(image, np.zeros((3, 3), dtype=bool))
    assert np.array_equal(filled, image)
    assert iterations == 0 and converged


def test_known_pixels_never_change():
    original, mask = make_gradient()
    filled = inpaint(original, mask)
    assert np.array_equal(filled[~mask], original[~mask])


def test_grayscale_hole_reconstructed_accurately():
    original, mask = make_gradient()
    filled, iterations, converged = inpaint_with_info(original, mask)
    assert converged
    assert iterations > 1
    # A linear gradient is the exact solution of the Laplace equation,
    # so diffusion should recover it almost exactly.
    assert np.abs(filled - original)[mask].mean() < 0.5


def test_constant_image_hole_fills_with_constant():
    image = np.full((10, 10), 77.0)
    mask = np.zeros((10, 10), dtype=bool)
    mask[4:6, 4:6] = True
    filled = inpaint(image, mask)
    assert np.allclose(filled[mask], 77.0)


def test_rgb_hole_fills_all_channels():
    original, mask = make_gradient()
    rgb = np.stack([original, original * 0.5, 255 - original], axis=2)
    filled = inpaint(rgb, mask)
    assert filled.shape == rgb.shape
    assert np.array_equal(filled[~mask], rgb[~mask])
    assert np.abs(filled - rgb)[mask].mean() < 1.0


def test_uint8_input_returns_uint8():
    image = np.full((8, 8), 200, dtype=np.uint8)
    mask = np.zeros((8, 8), dtype=bool)
    mask[3:5, 3:5] = True
    filled = inpaint(image, mask)
    assert filled.dtype == np.uint8
    assert np.all(filled[mask] == 200)


def test_single_pixel_hole_equals_neighbour_average():
    image = np.array([[1.0, 2.0, 3.0],
                      [4.0, 0.0, 6.0],
                      [7.0, 8.0, 9.0]])
    mask = np.zeros((3, 3), dtype=bool)
    mask[1, 1] = True
    filled = inpaint(image, mask, tolerance=1e-9)
    # Neighbours 2, 8, 4, 6 average to 5.0.
    assert filled[1, 1] == pytest.approx(5.0)


def test_full_mask_raises():
    with pytest.raises(ValueError, match="whole image"):
        inpaint(np.ones((4, 4)), np.ones((4, 4), dtype=bool))


def test_mask_shape_mismatch_raises():
    with pytest.raises(ValueError, match="mask must be 2-D"):
        inpaint(np.ones((4, 4)), np.zeros((4, 5), dtype=bool))


def test_non_bool_mask_raises():
    with pytest.raises(ValueError, match="boolean"):
        inpaint(np.ones((4, 4)), np.zeros((4, 4), dtype=int))


def test_1d_image_raises():
    with pytest.raises(ValueError, match="2-D"):
        inpaint(np.ones(9), np.zeros((3, 3), dtype=bool))


def test_bad_max_iterations_raises():
    with pytest.raises(ValueError, match="max_iterations"):
        inpaint(np.ones((4, 4)), np.zeros((4, 4), dtype=bool), max_iterations=0)


def test_bad_tolerance_raises():
    with pytest.raises(ValueError, match="tolerance"):
        inpaint(np.ones((4, 4)), np.zeros((4, 4), dtype=bool), tolerance=0.0)


def test_max_iterations_cap_is_respected():
    original, mask = make_gradient(size=40, hole=20)
    _, iterations, converged = inpaint_with_info(original, mask, max_iterations=3)
    assert iterations == 3
    assert not converged
