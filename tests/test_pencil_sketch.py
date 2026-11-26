"""Tests for the pencil_sketch program."""

import numpy as np
import pytest

from pencil_sketch.model import (
    color_dodge_blend,
    gaussian_blur,
    pencil_sketch,
    rgb_to_grayscale,
)


def make_step_image(height=40, width=60):
    """Dark on the left, bright on the right, sharp vertical step."""
    image = np.full((height, width), 50.0)
    image[:, width // 2:] = 220.0
    return image


def test_output_shape_dtype_and_range():
    image = make_step_image()
    sketch = pencil_sketch(image)
    assert sketch.shape == image.shape
    assert sketch.dtype == np.uint8
    assert sketch.min() >= 0 and sketch.max() <= 255


def test_flat_white_stays_white():
    sketch = pencil_sketch(np.full((20, 20), 255, dtype=np.uint8))
    assert np.all(sketch == 255)


def test_flat_black_stays_black():
    # Colour-dodge of base 0 is 0 for any blend value, so a flat black
    # input gives a flat black sketch (not white).
    sketch = pencil_sketch(np.zeros((20, 20), dtype=np.uint8))
    assert np.all(sketch == 0)


def test_edge_darker_than_flat_region_on_step():
    sketch = pencil_sketch(make_step_image())
    flat = sketch[5:15, 40:55].mean()      # flat bright region
    # The dark line sits on the dark side of the step: the two columns
    # immediately left of it (the step is at column 30).
    edge = sketch[5:35, 28:30].mean()
    assert flat > 200
    assert edge < flat - 50


def test_grayscale_and_rgb_inputs_work():
    gray = make_step_image()
    rgb = np.stack([gray, gray, gray], axis=2)
    sketch_gray = pencil_sketch(gray)
    sketch_rgb = pencil_sketch(rgb)
    assert sketch_rgb.shape == gray.shape
    assert sketch_rgb.dtype == np.uint8
    np.testing.assert_array_equal(sketch_gray, sketch_rgb)


def test_rgb_to_grayscale_weights():
    rgb = np.array([[[255, 0, 0], [0, 255, 0], [0, 0, 255]]], dtype=np.float64)
    gray = rgb_to_grayscale(rgb)
    assert gray.shape == (1, 3)
    np.testing.assert_allclose(
        gray[0], [0.299 * 255, 0.587 * 255, 0.114 * 255], rtol=1e-9
    )
    # 2D input passes through unchanged.
    two_d = np.array([[10.0, 20.0]])
    np.testing.assert_array_equal(rgb_to_grayscale(two_d), two_d)


def test_gaussian_blur_reduces_noise_and_preserves_shape():
    rng = np.random.default_rng(0)
    noisy = np.full((30, 30), 128.0) + rng.normal(0, 20, size=(30, 30))
    blurred = gaussian_blur(noisy, sigma=1.0)
    assert blurred.shape == noisy.shape
    assert blurred.std() < noisy.std()
    # A flat image is unchanged by the blur (replicate padding).
    flat = np.full((10, 10), 100.0)
    np.testing.assert_allclose(gaussian_blur(flat), flat, atol=1e-9)


def test_color_dodge_blend_known_values():
    # blend == 255 -> 255 (divide-by-zero case).
    out = color_dodge_blend(np.array([[0.0, 100.0, 255.0]]), np.array([[255.0, 255.0, 255.0]]))
    np.testing.assert_array_equal(out, [[255.0, 255.0, 255.0]])
    # base == 0 -> 0 for any blend below 255.
    out = color_dodge_blend(np.array([[0.0]]), np.array([[200.0]]))
    np.testing.assert_array_equal(out, [[0.0]])
    # Known value: 100 / (255 - 128) * 255.
    out = color_dodge_blend(np.array([[100.0]]), np.array([[128.0]]))
    np.testing.assert_allclose(out, [[100.0 / 127.0 * 255.0]], rtol=1e-9)
    # Result is clipped to [0, 255].
    out = color_dodge_blend(np.array([[250.0]]), np.array([[200.0]]))
    assert out[0, 0] == 255.0


def test_invalid_sigma_raises():
    image = make_step_image()
    for bad in (0, -1.0, float("nan"), float("inf"), "1.0"):
        with pytest.raises((ValueError, TypeError)):
            pencil_sketch(image, sigma=bad)
    with pytest.raises((ValueError, TypeError)):
        gaussian_blur(image, sigma=0)


def test_non_array_input_raises():
    with pytest.raises(TypeError):
        pencil_sketch([[1, 2], [3, 4]])
    with pytest.raises(TypeError):
        pencil_sketch("not an image")
    with pytest.raises((ValueError, TypeError)):
        pencil_sketch(np.array([]))
    with pytest.raises(ValueError):
        pencil_sketch(np.zeros((4, 4, 4)))  # 4 channels, not RGB


def test_demo_behaviour_flat_near_white_edge_dark():
    """Same scene as demo.py: flat region near white, disc edge dark."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pencil_sketch"))
    from demo import make_image

    sketch = pencil_sketch(make_image())
    flat = sketch[2:10, 2:10].mean()
    y, x = np.mgrid[0:60, 0:80]
    r = np.sqrt((x - 25) ** 2 + (y - 30) ** 2)
    edge = sketch[(r >= 8) & (r <= 10)].mean()
    assert flat > 200
    assert edge < flat - 50
