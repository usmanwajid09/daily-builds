"""Tests for scharr_edge_detector (Scharr edge detection)."""

import numpy as np
import pytest

from scharr_edge_detector.scharr import (
    edge_pixel_count,
    scharr_direction,
    scharr_edges,
    scharr_gradients,
    scharr_magnitude,
    threshold_edges,
)


def _step_edge(height=20, width=20, left=0.0, right=200.0):
    image = np.full((height, width), left, dtype=np.float64)
    image[:, width // 2:] = right
    return image


def test_constant_image_has_zero_magnitude():
    image = np.full((10, 10), 77.0)
    np.testing.assert_allclose(scharr_magnitude(image), 0.0, atol=1e-12)


def test_vertical_step_edge_gives_strong_horizontal_gradient():
    gx, gy = scharr_gradients(_step_edge())
    # Gradient points across the edge (x), not along it (y).
    assert gx[:, 10].max() > 1000.0
    np.testing.assert_allclose(gy, 0.0, atol=1e-12)


def test_horizontal_step_edge_gives_strong_vertical_gradient():
    image = np.zeros((20, 20))
    image[10:, :] = 200.0
    gx, gy = scharr_gradients(image)
    assert gy[10, :].max() > 1000.0
    np.testing.assert_allclose(gx, 0.0, atol=1e-12)


def test_magnitude_matches_gradient_hypot():
    image = _step_edge()
    gx, gy = scharr_gradients(image)
    np.testing.assert_allclose(scharr_magnitude(image), np.hypot(gx, gy))


def test_edges_returns_matching_magnitude_and_direction():
    magnitude, direction = scharr_edges(_step_edge())
    np.testing.assert_allclose(magnitude, scharr_magnitude(_step_edge()))
    np.testing.assert_allclose(direction, scharr_direction(_step_edge()))


def test_direction_vertical_edge_is_zero_and_horizontal_is_half_pi():
    direction = scharr_direction(_step_edge())
    assert abs(direction[10, 10]) < 1e-9  # gradient points right
    image = np.zeros((20, 20))
    image[10:, :] = 200.0
    direction = scharr_direction(image)
    assert abs(direction[10, 10] - np.pi / 2) < 1e-9  # gradient points down


def test_direction_range_is_within_pi():
    rng = np.random.default_rng(0)
    direction = scharr_direction(rng.normal(100.0, 30.0, size=(16, 16)))
    assert direction.min() >= -np.pi and direction.max() <= np.pi


def test_output_shape_matches_input():
    image = _step_edge(height=12, width=18)
    assert scharr_magnitude(image).shape == image.shape
    assert scharr_edges(image)[0].shape == image.shape


def test_threshold_edges_masks_strong_pixels_only():
    magnitude = np.array([[0.0, 50.0], [200.0, 500.0]])
    mask = threshold_edges(magnitude, 150.0)
    assert mask.tolist() == [[False, False], [True, True]]
    assert mask.dtype == bool


def test_edge_pixel_count_counts_thresholded_pixels():
    magnitude = np.array([[0.0, 50.0], [200.0, 500.0]])
    assert edge_pixel_count(magnitude, 150.0) == 2
    assert edge_pixel_count(magnitude, 0.0) == 4
    assert edge_pixel_count(magnitude, 1000.0) == 0


def test_uint8_input_accepted():
    image = np.zeros((8, 8), dtype=np.uint8)
    image[:, 4:] = 200
    assert scharr_magnitude(image).max() > 0


def test_flat_border_uses_replicate_padding():
    # A step at the very edge of the image must still be detected.
    image = np.zeros((10, 10))
    image[:, 9] = 200.0
    assert scharr_magnitude(image)[:, 9].max() > 0


def test_rejects_1d_image():
    with pytest.raises(ValueError, match="2D"):
        scharr_magnitude(np.zeros(10))


def test_rejects_3d_image():
    with pytest.raises(ValueError, match="2D"):
        scharr_magnitude(np.zeros((4, 4, 3)))


def test_rejects_empty_image():
    with pytest.raises(ValueError, match="empty"):
        scharr_magnitude(np.zeros((0, 0)))


def test_rejects_nan_image():
    image = np.zeros((5, 5))
    image[2, 2] = np.nan
    with pytest.raises(ValueError, match="finite"):
        scharr_magnitude(image)


def test_threshold_rejects_negative_threshold():
    with pytest.raises(ValueError, match="non-negative"):
        threshold_edges(np.zeros((3, 3)), -1.0)


def test_threshold_rejects_negative_magnitude():
    with pytest.raises(ValueError, match="negative"):
        threshold_edges(np.array([[-1.0]]), 0.0)
