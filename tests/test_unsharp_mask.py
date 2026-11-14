"""Tests for unsharp_mask (unsharp masking / image sharpening)."""

import numpy as np
import pytest

from unsharp_mask import gaussian_blur, unsharp_mask


def _step_edge(height=32, width=32):
    """Mid-grey step edge (100 -> 155) — room to overshoot without clipping."""
    image = np.full((height, width), 100.0, dtype=np.float64)
    image[:, width // 2:] = 155.0
    return image


def _edge_gradient_magnitude(image):
    """Max |horizontal gradient| across the middle row."""
    return np.abs(np.diff(image[image.shape[0] // 2, :])).max()


# --- gaussian_blur -------------------------------------------------------


def test_blur_constant_image_unchanged():
    image = np.full((16, 16), 123.0)
    out = gaussian_blur(image, sigma=1.5)
    np.testing.assert_allclose(out, image, rtol=0, atol=1e-12)


def test_blur_constant_rgb_image_unchanged():
    image = np.full((8, 8, 3), 77.0)
    out = gaussian_blur(image, sigma=2.0)
    np.testing.assert_allclose(out, image, rtol=0, atol=1e-12)


def test_blur_reduces_variance_of_noisy_image():
    rng = np.random.default_rng(0)
    image = rng.normal(128.0, 40.0, size=(32, 32))
    assert gaussian_blur(image, sigma=1.0).var() < image.var()


def test_blur_output_in_range_and_same_shape():
    rng = np.random.default_rng(1)
    image = rng.uniform(0, 255, size=(24, 24))
    out = gaussian_blur(image, sigma=1.0)
    assert out.shape == image.shape
    assert out.dtype == np.float64
    assert out.min() >= 0.0 and out.max() <= 255.0


def test_blur_step_edge_is_smoothed():
    image = _step_edge()
    out = gaussian_blur(image, sigma=1.0)
    assert out.shape == image.shape
    # The blurred edge is a ramp, not a jump: its max step is below 55.
    assert _edge_gradient_magnitude(out) < 55.0
    assert _edge_gradient_magnitude(out) > 0.0


# --- unsharp_mask --------------------------------------------------------


def test_unsharp_amount_zero_returns_image_unchanged():
    rng = np.random.default_rng(2)
    image = rng.uniform(0, 255, size=(20, 20))
    np.testing.assert_array_equal(unsharp_mask(image, amount=0.0), image)


def test_unsharp_increases_contrast_at_step_edge():
    image = _step_edge()
    sharpened = unsharp_mask(image, amount=1.0, radius=1.0)
    assert _edge_gradient_magnitude(sharpened) > _edge_gradient_magnitude(image)


def test_unsharp_high_threshold_leaves_near_flat_image_unchanged():
    rng = np.random.default_rng(3)
    # Near-flat: constant 100 plus tiny noise, detail well below threshold.
    image = 100.0 + rng.normal(0.0, 0.5, size=(24, 24))
    out = unsharp_mask(image, amount=2.0, radius=1.0, threshold=10.0)
    np.testing.assert_array_equal(out, image)


def test_unsharp_output_clipped_and_same_shape():
    image = np.zeros((32, 32), dtype=np.float64)
    image[:, 16:] = 255.0
    out = unsharp_mask(image, amount=5.0, radius=1.0)
    assert out.shape == image.shape
    assert out.dtype == np.float64
    assert out.min() >= 0.0 and out.max() <= 255.0


def test_unsharp_rgb_preserves_shape_and_processes_channels():
    rng = np.random.default_rng(4)
    image = rng.uniform(0, 255, size=(16, 16, 3))
    out = unsharp_mask(image, amount=1.0, radius=1.0)
    assert out.shape == image.shape
    assert out.dtype == np.float64
    assert out.min() >= 0.0 and out.max() <= 255.0
    # Channels are processed independently: a red-only edge stays red-only.
    rgb_edge = np.zeros((8, 8, 3))
    rgb_edge[:, 4:, 0] = 255.0
    out_edge = unsharp_mask(rgb_edge, amount=1.0, radius=1.0)
    np.testing.assert_array_equal(out_edge[:, :, 1], rgb_edge[:, :, 1])
    np.testing.assert_array_equal(out_edge[:, :, 2], rgb_edge[:, :, 2])


def test_unsharp_matches_manual_formula():
    image = np.zeros((32, 32), dtype=np.float64)
    image[:, 16:] = 255.0
    expected = np.clip(
        image + 1.5 * (image - gaussian_blur(image, sigma=2.0)), 0, 255
    )
    np.testing.assert_allclose(
        unsharp_mask(image, amount=1.5, radius=2.0), expected, rtol=0, atol=1e-12
    )


# --- validation ----------------------------------------------------------


def test_rejects_sigma_zero():
    with pytest.raises(ValueError):
        gaussian_blur(np.zeros((4, 4)), sigma=0)


def test_rejects_negative_sigma():
    with pytest.raises(ValueError):
        gaussian_blur(np.zeros((4, 4)), sigma=-1.0)
    with pytest.raises(ValueError):
        unsharp_mask(np.zeros((4, 4)), radius=-1.0)


def test_rejects_negative_amount():
    with pytest.raises(ValueError):
        unsharp_mask(np.zeros((4, 4)), amount=-0.5)


def test_rejects_negative_threshold():
    with pytest.raises(ValueError):
        unsharp_mask(np.zeros((4, 4)), threshold=-1.0)


def test_rejects_non_numeric_image():
    with pytest.raises(ValueError):
        unsharp_mask(np.array([["a", "b"], ["c", "d"]]))


def test_rejects_empty_image():
    with pytest.raises(ValueError):
        gaussian_blur(np.zeros((0, 0)))


def test_rejects_1d_image():
    with pytest.raises(ValueError):
        unsharp_mask(np.zeros(8))
