"""Tests for bilateral_filter."""

import numpy as np
import pytest

from bilateral_filter import bilateral_filter, gaussian_kernel_2d
from bilateral_filter.demo import make_test_image


def test_constant_image_unchanged():
    image = np.full((10, 12), 77.0)
    out = bilateral_filter(image)
    assert out.shape == image.shape
    assert np.allclose(out, 77.0)


def test_output_shape_and_dtype():
    image = np.random.default_rng(0).random((9, 11)) * 255.0
    out = bilateral_filter(image)
    assert out.shape == image.shape
    assert out.dtype == np.float64
    assert np.isfinite(out).all()


def test_impulse_does_not_bleed():
    # A bilateral filter treats an isolated spike as an edge, not as
    # noise: the spike itself is preserved but must not spread.
    image = np.zeros((11, 11))
    image[5, 5] = 255.0
    out = bilateral_filter(image, diameter=5, sigma_color=25.0,
                           sigma_space=2.0)
    assert out[5, 5] == pytest.approx(255.0)
    assert out[5, 6] < 5.0
    assert out[4, 5] < 5.0
    assert out[6, 6] < 5.0


def test_sharp_edge_preserved():
    image = make_test_image()
    out = bilateral_filter(image, diameter=5, sigma_color=25.0,
                           sigma_space=2.0)
    step_before = image[:, 35].mean() - image[:, 24].mean()
    step_after = out[:, 35].mean() - out[:, 24].mean()
    assert step_after > 0.9 * step_before


def test_denoises_flat_region():
    image = make_test_image()
    out = bilateral_filter(image, diameter=5, sigma_color=25.0,
                           sigma_space=2.0)
    assert out[:, :25].std() < image[:, :25].std()
    assert out[:, 35:].std() < image[:, 35:].std()


def test_rgb_image_works():
    rng = np.random.default_rng(1)
    image = rng.random((8, 9, 3)) * 255.0
    out = bilateral_filter(image)
    assert out.shape == image.shape
    assert np.isfinite(out).all()


def test_constant_rgb_unchanged():
    image = np.full((6, 7, 3), [10.0, 100.0, 200.0])
    out = bilateral_filter(image)
    assert np.allclose(out, image)


def test_diameter_one_is_identity():
    image = np.random.default_rng(2).random((7, 8)) * 255.0
    out = bilateral_filter(image, diameter=1)
    assert np.allclose(out, image)


def test_gaussian_kernel_sums_to_one():
    kernel = gaussian_kernel_2d(5, 1.0)
    assert kernel.shape == (5, 5)
    assert kernel.sum() == pytest.approx(1.0)
    assert kernel[2, 2] == kernel.max()


def test_even_diameter_raises():
    with pytest.raises(ValueError):
        bilateral_filter(np.zeros((5, 5)), diameter=4)


def test_zero_diameter_raises():
    with pytest.raises(ValueError):
        bilateral_filter(np.zeros((5, 5)), diameter=0)


def test_bad_sigma_color_raises():
    with pytest.raises(ValueError):
        bilateral_filter(np.zeros((5, 5)), sigma_color=0)


def test_bad_sigma_space_raises():
    with pytest.raises(ValueError):
        bilateral_filter(np.zeros((5, 5)), sigma_space=-1.0)


def test_1d_image_raises():
    with pytest.raises(ValueError):
        bilateral_filter(np.zeros(10))


def test_four_channel_image_raises():
    with pytest.raises(ValueError):
        bilateral_filter(np.zeros((5, 5, 4)))


def test_empty_image_raises():
    with pytest.raises(ValueError):
        bilateral_filter(np.zeros((0, 0)))
