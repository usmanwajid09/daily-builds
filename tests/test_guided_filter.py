"""Tests for guided_filter (guided filter edge-preserving smoothing)."""

import numpy as np
import pytest

from guided_filter.guided_filter import edge_preserving_smooth, guided_filter


def _step_edge(height=20, width=20, left=0.0, right=100.0):
    image = np.full((height, width), left, dtype=np.float64)
    image[:, width // 2:] = right
    return image


def test_constant_image_unchanged():
    image = np.full((10, 10), 77.0)
    np.testing.assert_allclose(edge_preserving_smooth(image), image, atol=1e-9)


def test_radius_zero_returns_input():
    rng = np.random.default_rng(0)
    image = rng.normal(100.0, 20.0, size=(12, 12))
    np.testing.assert_allclose(
        edge_preserving_smooth(image, radius=0), image, atol=1e-9
    )


def test_step_edge_height_preserved():
    image = _step_edge()
    smoothed = edge_preserving_smooth(image, radius=2, eps=0.01)
    step_before = image[:, 15:].mean() - image[:, :5].mean()
    step_after = smoothed[:, 15:].mean() - smoothed[:, :5].mean()
    assert step_after > 0.95 * step_before


def test_flat_region_noise_reduced():
    rng = np.random.default_rng(0)
    image = np.full((30, 30), 100.0) + rng.normal(0.0, 10.0, size=(30, 30))
    # eps must exceed the noise variance (~100) for the filter to
    # treat the noise as flat and average it away.
    smoothed = edge_preserving_smooth(image, radius=4, eps=1000.0)
    assert smoothed.std() < 0.5 * image.std()


def test_larger_eps_smooths_more():
    rng = np.random.default_rng(1)
    image = np.full((30, 30), 100.0) + rng.normal(0.0, 10.0, size=(30, 30))
    weak = edge_preserving_smooth(image, radius=4, eps=0.01)
    strong = edge_preserving_smooth(image, radius=4, eps=100.0)
    assert strong.std() < weak.std()


def test_output_shape_and_dtype():
    image = _step_edge(height=12, width=18)
    out = edge_preserving_smooth(image)
    assert out.shape == image.shape
    assert out.dtype == np.float64


def test_guide_transfers_structure():
    # Noisy input, clean guide: the output should follow the guide's
    # step edge and lose most of the noise.
    rng = np.random.default_rng(2)
    guide = _step_edge()
    noisy = guide + rng.normal(0.0, 10.0, size=guide.shape)
    out = guided_filter(guide, noisy, radius=4, eps=0.1)
    assert out[:, 15:].mean() - out[:, :5].mean() > 90.0
    assert out.std() < noisy.std()


def test_uint8_input_accepted():
    image = np.zeros((10, 10), dtype=np.uint8)
    image[:, 5:] = 200
    out = edge_preserving_smooth(image)
    assert out.shape == image.shape
    assert out[:, 8:].mean() > 150.0


def test_rejects_1d_image():
    with pytest.raises(ValueError, match="2D"):
        edge_preserving_smooth(np.zeros(10))


def test_rejects_rgb_image():
    with pytest.raises(ValueError, match="2D"):
        edge_preserving_smooth(np.zeros((4, 4, 3)))


def test_rejects_empty_image():
    with pytest.raises(ValueError, match="empty"):
        edge_preserving_smooth(np.zeros((0, 0)))


def test_rejects_nan_image():
    image = np.zeros((5, 5))
    image[2, 2] = np.nan
    with pytest.raises(ValueError, match="finite"):
        edge_preserving_smooth(image)


def test_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="same shape"):
        guided_filter(np.zeros((5, 5)), np.zeros((6, 5)))


def test_rejects_negative_radius():
    with pytest.raises(ValueError, match="non-negative"):
        edge_preserving_smooth(np.zeros((5, 5)), radius=-1)


def test_rejects_non_integer_radius():
    with pytest.raises(ValueError, match="integer"):
        edge_preserving_smooth(np.zeros((5, 5)), radius=1.5)


def test_rejects_non_positive_eps():
    with pytest.raises(ValueError, match="positive"):
        edge_preserving_smooth(np.zeros((5, 5)), eps=0.0)
    with pytest.raises(ValueError, match="positive"):
        edge_preserving_smooth(np.zeros((5, 5)), eps=-1.0)


def test_rejects_list_input():
    with pytest.raises(ValueError, match="numpy array"):
        edge_preserving_smooth([[0, 1], [1, 0]])
