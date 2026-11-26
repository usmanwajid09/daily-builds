"""Tests for the cartoonizer (numpy-only cartoon / posterize effect)."""
import numpy as np
import pytest

from cartoonizer import (
    cartoonize,
    edge_mask,
    edge_preserving_smooth,
    quantize_colors,
)
from cartoonizer.demo import build_scene, count_colours


def flat_noisy(seed=0, value=120, shape=(30, 40, 3), std=8.0):
    rng = np.random.default_rng(seed)
    img = rng.normal(value, std, size=shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def step_image():
    """Left half 50, right half 200 — a sharp vertical step edge."""
    img = np.full((20, 40, 3), 50, dtype=np.uint8)
    img[:, 20:] = 200
    return img


def test_output_shape_dtype():
    img = flat_noisy()
    out = cartoonize(img)
    assert out.shape == img.shape
    assert out.dtype == np.uint8


def test_distinct_colours_after_le_before():
    img = flat_noisy()
    before = len(np.unique(img.reshape(-1, 3), axis=0))
    after = len(np.unique(cartoonize(img).reshape(-1, 3), axis=0))
    assert after <= before


def test_quantize_n2_gives_few_colours():
    img = flat_noisy()
    out = quantize_colors(img, n_colors=2)
    assert out.dtype == np.uint8
    assert len(np.unique(out.reshape(-1, 3), axis=0)) <= 2 ** 3
    # Every channel value is one of the two bin centres.
    assert set(np.unique(out[:, :, 0])) <= {64, 192}


def test_edge_mask_finds_step_edge():
    mask = edge_mask(step_image(), threshold=30.0)
    assert mask.dtype == bool
    # Edge columns around the step are flagged.
    assert mask[:, 19].any() and mask[:, 20].any()
    # Far from the step there is nothing.
    assert not mask[:, 5].any() and not mask[:, 35].any()


def test_edge_mask_empty_on_flat_image():
    flat = np.full((20, 20, 3), 128, dtype=np.uint8)
    assert not edge_mask(flat, threshold=30.0).any()


def test_smoothing_reduces_noise_but_keeps_step():
    noisy = flat_noisy()
    smoothed = edge_preserving_smooth(noisy, iterations=2)
    assert smoothed.std() < noisy.std()

    step = step_image().astype(np.float64)
    rng = np.random.default_rng(1)
    step_noisy = np.clip(step + rng.normal(0, 4, size=step.shape), 0, 255)
    smoothed_step = edge_preserving_smooth(step_noisy, iterations=1)
    # The step across the middle must survive: left vs right means far apart.
    left = smoothed_step[:, :10].mean()
    right = smoothed_step[:, 30:].mean()
    assert abs(right - left) > 100


def test_invalid_n_colors_raises():
    img = flat_noisy()
    with pytest.raises(ValueError):
        quantize_colors(img, n_colors=1)
    with pytest.raises(ValueError):
        cartoonize(img, n_colors=0)
    with pytest.raises(TypeError):
        quantize_colors(img, n_colors=2.5)


def test_non_image_input_raises():
    with pytest.raises(TypeError):
        cartoonize([[1, 2], [3, 4]])          # plain list, not ndarray
    with pytest.raises(TypeError):
        cartoonize("not an image")
    with pytest.raises(ValueError):
        cartoonize(np.array([], dtype=np.uint8))  # empty


def test_grayscale_input_returns_rgb():
    gray = np.full((20, 20), 150, dtype=np.uint8)
    out = cartoonize(gray)
    assert out.shape == (20, 20, 3)
    assert out.dtype == np.uint8


def test_float_input_accepted():
    img = flat_noisy().astype(np.float64)
    out = cartoonize(img)
    assert out.dtype == np.uint8 and out.shape == img.shape


def test_demo_scene_fewer_colours_after():
    scene = build_scene()
    cartoon = cartoonize(scene)
    assert count_colours(cartoon) < count_colours(scene)


def test_edge_pixels_are_black_in_cartoon():
    img = step_image()
    out = cartoonize(img)
    mask = edge_mask(img, threshold=30.0)
    assert (out[mask] == 0).all()
