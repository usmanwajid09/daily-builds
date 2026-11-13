"""Tests for median_cut_quantizer.quantize."""

import numpy as np
import pytest

from median_cut_quantizer.quantize import quantize, unique_color_count


def make_two_colour_image():
    img = np.zeros((4, 4, 3), dtype=np.uint8)
    img[:2] = [255, 0, 0]   # red top half
    img[2:] = [0, 0, 255]   # blue bottom half
    return img


def test_two_colours_recovered_exactly():
    """With n_colors=2 the palette is exactly the two input colours."""
    img = make_two_colour_image()
    quantized, palette = quantize(img, n_colors=2)
    assert palette.shape == (2, 3)
    assert {tuple(c) for c in palette.tolist()} == {(255, 0, 0), (0, 0, 255)}
    assert np.array_equal(quantized, img)


def test_palette_size_capped_at_n_colors():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, size=(20, 20, 3), dtype=np.uint8)
    quantized, palette = quantize(img, n_colors=8)
    assert len(palette) <= 8
    assert unique_color_count(quantized) <= 8


def test_quantized_uses_only_palette_colours():
    rng = np.random.default_rng(1)
    img = rng.integers(0, 256, size=(16, 16, 3), dtype=np.uint8)
    quantized, palette = quantize(img, n_colors=4)
    used = {tuple(c) for c in np.unique(quantized.reshape(-1, 3), axis=0).tolist()}
    assert used <= {tuple(c) for c in palette.tolist()}


def test_single_colour_image_returns_one_palette_entry():
    img = np.full((5, 5, 3), [10, 200, 90], dtype=np.uint8)
    quantized, palette = quantize(img, n_colors=16)
    assert len(palette) == 1
    assert palette[0].tolist() == [10, 200, 90]
    assert np.array_equal(quantized, img)


def test_n_colors_one_gives_image_mean():
    img = make_two_colour_image()
    quantized, palette = quantize(img, n_colors=1)
    assert len(palette) == 1
    # Mean of 8 red + 8 blue pixels, rounded: (128, 0, 128).
    assert palette[0].tolist() == [128, 0, 128]
    assert np.all(quantized == np.array([128, 0, 128], dtype=np.uint8))


def test_output_shape_and_dtype():
    rng = np.random.default_rng(2)
    img = rng.integers(0, 256, size=(6, 9, 3), dtype=np.uint8)
    quantized, palette = quantize(img, n_colors=5)
    assert quantized.shape == img.shape
    assert quantized.dtype == np.uint8
    assert palette.dtype == np.uint8


def test_float_input_accepted():
    img = np.zeros((3, 3, 3), dtype=np.float64)
    img[:, :, 0] = 255.0
    quantized, palette = quantize(img, n_colors=2)
    assert palette[0].tolist() == [255, 0, 0]


def test_more_colours_than_unique_returns_unique_count():
    """Asking for more colours than exist returns only the real ones."""
    img = make_two_colour_image()
    quantized, palette = quantize(img, n_colors=64)
    assert len(palette) == 2


def test_unique_color_count():
    img = make_two_colour_image()
    assert unique_color_count(img) == 2
    rng = np.random.default_rng(3)
    img2 = rng.integers(0, 256, size=(10, 10, 3), dtype=np.uint8)
    assert unique_color_count(img2) > 2


def test_grayscale_image_raises():
    with pytest.raises(ValueError, match="\\(H, W, 3\\) RGB"):
        quantize(np.zeros((4, 4)), n_colors=4)


def test_rgba_image_raises():
    with pytest.raises(ValueError, match="\\(H, W, 3\\) RGB"):
        quantize(np.zeros((4, 4, 4), dtype=np.uint8), n_colors=4)


def test_n_colors_zero_raises():
    with pytest.raises(ValueError, match="at least 1"):
        quantize(make_two_colour_image(), n_colors=0)


def test_n_colors_non_integer_raises():
    with pytest.raises(ValueError, match="positive integer"):
        quantize(make_two_colour_image(), n_colors=4.5)


def test_empty_image_raises():
    with pytest.raises(ValueError, match="must not be empty"):
        quantize(np.zeros((0, 0, 3), dtype=np.uint8), n_colors=4)


def test_out_of_range_values_raise():
    img = np.zeros((2, 2, 3))
    img[0, 0, 0] = 300.0
    with pytest.raises(ValueError, match="range \\[0, 255\\]"):
        quantize(img, n_colors=2)


def test_nan_values_raise():
    img = np.zeros((2, 2, 3))
    img[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        quantize(img, n_colors=2)


def test_deterministic():
    """Same input gives the same palette every time."""
    rng = np.random.default_rng(4)
    img = rng.integers(0, 256, size=(12, 12, 3), dtype=np.uint8)
    q1, p1 = quantize(img, n_colors=6)
    q2, p2 = quantize(img, n_colors=6)
    assert np.array_equal(p1, p2)
    assert np.array_equal(q1, q2)
