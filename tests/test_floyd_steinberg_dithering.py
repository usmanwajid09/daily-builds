"""Tests for floyd_steinberg_dithering."""
import numpy as np
import pytest

from floyd_steinberg_dithering import dither_color, dither_grayscale
from floyd_steinberg_dithering.demo import build_gradient


def test_output_binary_values_only():
    gray = build_gradient()
    out = dither_grayscale(gray)
    assert out.dtype == np.uint8
    assert set(np.unique(out)) <= {0, 255}


def test_output_shape_matches():
    gray = build_gradient()
    assert dither_grayscale(gray).shape == gray.shape


def test_mean_brightness_preserved():
    gray = build_gradient()
    out = dither_grayscale(gray)
    # Error diffusion preserves the local (hence global) mean closely.
    assert abs(out.mean() - gray.mean()) < 5


def test_mid_grey_is_about_half_white():
    gray = np.full((32, 32), 128, dtype=np.uint8)
    out = dither_grayscale(gray)
    assert 0.4 < float((out == 255).mean()) < 0.6


def test_pure_black_and_white_unchanged():
    black = np.zeros((8, 8), dtype=np.uint8)
    white = np.full((8, 8), 255, dtype=np.uint8)
    assert (dither_grayscale(black) == 0).all()
    assert (dither_grayscale(white) == 255).all()


def test_serpentine_differs_from_plain():
    gray = build_gradient()
    a = dither_grayscale(gray, serpentine=True)
    b = dither_grayscale(gray, serpentine=False)
    assert not np.array_equal(a, b)
    # Both still preserve the mean.
    assert abs(a.mean() - gray.mean()) < 5
    assert abs(b.mean() - gray.mean()) < 5


def test_gradient_has_no_hard_bands():
    # A smooth gradient should dither into a mix, not a single hard step.
    gray = np.tile(np.linspace(0, 255, 64), (32, 1))
    out = dither_grayscale(gray)
    # Each column band should contain both black and white pixels.
    for x in (16, 32, 48):
        col = out[:, x - 4:x + 4]
        assert (col == 0).any() and (col == 255).any()


def test_color_dither_levels2_max_8_colours():
    rng = np.random.default_rng(0)
    rgb = rng.integers(0, 256, size=(24, 24, 3), dtype=np.uint8)
    out = dither_color(rgb, levels=2)
    assert out.shape == rgb.shape
    assert out.dtype == np.uint8
    assert len(np.unique(out.reshape(-1, 3), axis=0)) <= 8
    assert set(np.unique(out)) <= {0, 255}


def test_color_dither_levels4_palette_values():
    rgb = np.full((16, 16, 3), 100, dtype=np.uint8)
    out = dither_color(rgb, levels=4)
    assert set(np.unique(out)) <= {0, 85, 170, 255}


def test_color_dither_grey_input_matches_grayscale():
    gray = np.clip(np.round(build_gradient()), 0, 255).astype(np.uint8)
    rgb = np.stack([gray] * 3, axis=-1)
    colour = dither_color(rgb, levels=2)
    mono = dither_grayscale(gray)
    assert np.array_equal(colour[:, :, 0], mono)
    assert np.array_equal(colour[:, :, 1], mono)
    assert np.array_equal(colour[:, :, 2], mono)


def test_invalid_levels_raises():
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        dither_color(rgb, levels=1)
    with pytest.raises(ValueError):
        dither_color(rgb, levels=0)


def test_non_array_and_empty_raise():
    with pytest.raises(TypeError):
        dither_grayscale([[1, 2], [3, 4]])
    with pytest.raises(ValueError):
        dither_grayscale(np.array([], dtype=np.uint8))
    with pytest.raises(ValueError):
        dither_grayscale(np.zeros((8, 8, 3), dtype=np.uint8))


def test_color_requires_rgb():
    with pytest.raises(ValueError):
        dither_color(np.zeros((8, 8), dtype=np.uint8))


def test_float_input_accepted():
    gray = build_gradient().astype(np.float64)
    out = dither_grayscale(gray)
    assert set(np.unique(out)) <= {0, 255}


def test_dither_deterministic():
    gray = build_gradient()
    assert np.array_equal(dither_grayscale(gray), dither_grayscale(gray))
