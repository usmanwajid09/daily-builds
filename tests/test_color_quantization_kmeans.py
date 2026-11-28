"""Tests for color_quantization_kmeans."""

import numpy as np
import pytest

from color_quantization_kmeans import kmeans_quantize, quantization_mse
from color_quantization_kmeans.demo import make_test_image


def test_four_blocks_recover_four_colours():
    image, true_colors = make_test_image()
    quantized, palette = kmeans_quantize(image, n_colors=4, seed=0)
    assert quantized.shape == image.shape
    assert palette.shape == (4, 3)
    unique = np.unique(quantized.reshape(-1, 3), axis=0)
    assert len(unique) == 4
    for true in true_colors:
        dists = np.abs(palette - true).max(axis=1)
        assert dists.min() < 15.0


def test_mse_small_on_block_image():
    image, _ = make_test_image()
    quantized, _ = kmeans_quantize(image, n_colors=4, seed=0)
    mse = quantization_mse(image, quantized)
    assert mse < 300.0


def test_quantized_pixels_are_palette_colours():
    rng = np.random.default_rng(0)
    image = rng.random((10, 12, 3)) * 255.0
    quantized, palette = kmeans_quantize(image, n_colors=3, seed=1)
    for pixel in quantized.reshape(-1, 3):
        assert any(np.allclose(pixel, c) for c in palette)


def test_single_colour_returns_it():
    image = np.full((6, 6, 3), [10.0, 200.0, 90.0])
    quantized, palette = kmeans_quantize(image, n_colors=1, seed=0)
    assert np.allclose(quantized, [10.0, 200.0, 90.0])
    assert np.allclose(palette[0], [10.0, 200.0, 90.0])


def test_mse_zero_for_identical_images():
    image, _ = make_test_image()
    assert quantization_mse(image, image) == pytest.approx(0.0)


def test_invalid_n_colors_rejected():
    image, _ = make_test_image()
    with pytest.raises(ValueError, match="positive integer"):
        kmeans_quantize(image, n_colors=0)
    with pytest.raises(ValueError, match="exceeds"):
        kmeans_quantize(np.zeros((2, 2, 3)), n_colors=5)


def test_non_rgb_rejected():
    with pytest.raises(ValueError, match="RGB"):
        kmeans_quantize(np.zeros((5, 5)))
    with pytest.raises(ValueError, match="RGB"):
        kmeans_quantize(np.zeros((5, 5, 4)))


def test_deterministic_with_seed():
    image, _ = make_test_image()
    _, p1 = kmeans_quantize(image, n_colors=4, seed=7)
    _, p2 = kmeans_quantize(image, n_colors=4, seed=7)
    assert np.allclose(p1, p2)
