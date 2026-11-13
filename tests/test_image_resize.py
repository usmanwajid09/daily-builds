"""Tests for image_resize.resize."""

import numpy as np
import pytest

from image_resize.resize import resize, resize_bilinear, resize_nearest


def test_nearest_output_shape_grayscale():
    img = np.zeros((10, 20))
    out = resize_nearest(img, 5, 8)
    assert out.shape == (5, 8)


def test_nearest_output_shape_rgb():
    img = np.zeros((10, 20, 3))
    out = resize_nearest(img, 5, 8)
    assert out.shape == (5, 8, 3)


def test_bilinear_output_shape_rgb():
    img = np.zeros((10, 20, 3))
    out = resize_bilinear(img, 5, 8)
    assert out.shape == (5, 8, 3)


def test_same_size_returns_input():
    """Align-corners: resizing to the same size is the identity."""
    rng = np.random.default_rng(0)
    img = rng.random((7, 9))
    assert np.allclose(resize_nearest(img, 7, 9), img)
    assert np.allclose(resize_bilinear(img, 7, 9), img)


def test_nearest_integer_upscale_repeats_pixels():
    """2x nearest upscale: each pixel becomes a 2x2 block."""
    img = np.array([[1.0, 2.0], [3.0, 4.0]])
    out = resize_nearest(img, 4, 4)
    expected = np.array([[1.0, 1.0, 2.0, 2.0],
                         [1.0, 1.0, 2.0, 2.0],
                         [3.0, 3.0, 4.0, 4.0],
                         [3.0, 3.0, 4.0, 4.0]])
    assert np.array_equal(out, expected)


def test_bilinear_exact_on_linear_gradient():
    """Bilinear interpolation reproduces a linear ramp exactly."""
    img = np.linspace(0, 100, 11)[None, :].repeat(5, axis=0)
    out = resize_bilinear(img, 9, 21)
    expected = np.linspace(0, 100, 21)[None, :].repeat(9, axis=0)
    assert np.allclose(out, expected, atol=1e-9)


def test_bilinear_corners_match_input_corners():
    """Corner pixels of the output equal the input corners."""
    rng = np.random.default_rng(1)
    img = rng.random((6, 8))
    out = resize_bilinear(img, 12, 16)
    assert out[0, 0] == pytest.approx(img[0, 0])
    assert out[0, -1] == pytest.approx(img[0, -1])
    assert out[-1, 0] == pytest.approx(img[-1, 0])
    assert out[-1, -1] == pytest.approx(img[-1, -1])


def test_bilinear_constant_image_stays_constant():
    img = np.full((4, 5), 7.5)
    assert np.allclose(resize_bilinear(img, 9, 13), 7.5)


def test_nearest_downscale_picks_samples():
    """Downscaling a 4x4 to 2x2 picks the corner-ish samples."""
    img = np.arange(16.0).reshape(4, 4)
    out = resize_nearest(img, 2, 2)
    assert out.shape == (2, 2)
    # Align-corners: output (0,0) is input (0,0), (1,1) is input (3,3).
    assert out[0, 0] == 0.0
    assert out[1, 1] == 15.0


def test_resize_to_single_pixel_is_centre():
    img = np.arange(25.0).reshape(5, 5)
    out = resize_bilinear(img, 1, 1)
    assert out.shape == (1, 1)
    assert out[0, 0] == pytest.approx(12.0)  # centre of a 5x5 ramp
    out_n = resize_nearest(img, 1, 1)
    assert out_n[0, 0] == pytest.approx(12.0)


def test_rgb_channels_resized_independently():
    """Each channel keeps its own values after resizing."""
    img = np.zeros((2, 2, 3))
    img[:, :, 0] = 10.0
    img[:, :, 1] = 20.0
    img[:, :, 2] = 30.0
    out = resize_bilinear(img, 4, 4)
    assert np.allclose(out[:, :, 0], 10.0)
    assert np.allclose(out[:, :, 1], 20.0)
    assert np.allclose(out[:, :, 2], 30.0)


def test_unknown_method_raises():
    with pytest.raises(ValueError, match="unknown method"):
        resize(np.zeros((4, 4)), 2, 2, method="cubic")


def test_zero_target_size_raises():
    with pytest.raises(ValueError, match="at least 1"):
        resize_nearest(np.zeros((4, 4)), 0, 4)


def test_negative_target_size_raises():
    with pytest.raises(ValueError, match="at least 1"):
        resize_bilinear(np.zeros((4, 4)), 4, -3)


def test_non_integer_target_size_raises():
    with pytest.raises(ValueError, match="positive integer"):
        resize_nearest(np.zeros((4, 4)), 2.5, 4)


def test_empty_image_raises():
    with pytest.raises(ValueError, match="must not be empty"):
        resize_nearest(np.zeros((0, 4)), 2, 2)


def test_1d_image_raises():
    with pytest.raises(ValueError, match="2D grayscale or"):
        resize_nearest(np.zeros(8), 2, 2)


def test_5_channel_image_raises():
    with pytest.raises(ValueError, match="2D grayscale or"):
        resize_nearest(np.zeros((4, 4, 5)), 2, 2)


def test_nan_image_raises():
    img = np.zeros((4, 4))
    img[1, 1] = np.nan
    with pytest.raises(ValueError, match="finite"):
        resize_bilinear(img, 2, 2)


def test_integer_input_accepted():
    """uint8-style input works and is returned as float64."""
    img = np.arange(16, dtype=np.uint8).reshape(4, 4)
    out = resize_nearest(img, 2, 2)
    assert out.dtype == np.float64
    assert out[0, 0] == 0.0
