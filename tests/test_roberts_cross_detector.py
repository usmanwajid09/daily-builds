"""Tests for roberts_cross_detector (Roberts cross edge detection)."""

import numpy as np
import pytest

from roberts_cross_detector.roberts import (
    edge_pixel_count,
    roberts_direction,
    roberts_edges,
    roberts_gradients,
    roberts_magnitude,
    threshold_edges,
)


def _step_edge(height=20, width=20, left=0.0, right=200.0):
    image = np.full((height, width), left, dtype=np.float64)
    image[:, width // 2:] = right
    return image


def test_constant_image_has_zero_magnitude():
    image = np.full((10, 10), 77.0)
    np.testing.assert_allclose(roberts_magnitude(image), 0.0, atol=1e-12)


def test_output_is_one_row_and_column_smaller():
    image = _step_edge(height=20, width=30)
    assert roberts_magnitude(image).shape == (19, 29)
    assert roberts_edges(image)[0].shape == (19, 29)


def test_vertical_step_edge_detected():
    # A 200 step across the diagonal: gx = -200, gy = +200 at the edge.
    magnitude = roberts_magnitude(_step_edge())
    assert magnitude[:, 9].max() == pytest.approx(200.0 * np.sqrt(2))
    assert magnitude[:, 0].max() == 0.0  # flat area, no response


def test_horizontal_step_edge_detected():
    image = np.zeros((20, 20))
    image[10:, :] = 200.0
    magnitude = roberts_magnitude(image)
    assert magnitude[9, :].max() == pytest.approx(200.0 * np.sqrt(2))


def test_diagonal_step_edge_gives_max_response():
    # Image split along the main diagonal: maximal diagonal gradient.
    image = np.zeros((10, 10))
    yy, xx = np.mgrid[0:10, 0:10]
    image[xx > yy] = 200.0
    magnitude = roberts_magnitude(image)
    # Only gy responds along the main diagonal, so the max is 200, not 200*sqrt(2).
    assert magnitude.max() == pytest.approx(200.0)


def test_magnitude_matches_gradient_hypot():
    image = _step_edge()
    gx, gy = roberts_gradients(image)
    np.testing.assert_allclose(roberts_magnitude(image), np.hypot(gx, gy))


def test_edges_returns_matching_magnitude_and_direction():
    magnitude, direction = roberts_edges(_step_edge())
    np.testing.assert_allclose(magnitude, roberts_magnitude(_step_edge()))
    np.testing.assert_allclose(direction, roberts_direction(_step_edge()))


def test_direction_range_is_within_pi():
    rng = np.random.default_rng(0)
    direction = roberts_direction(rng.normal(100.0, 30.0, size=(16, 16)))
    assert direction.min() >= -np.pi and direction.max() <= np.pi


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
    assert roberts_magnitude(image).max() > 0


def test_minimum_2x2_image_works():
    image = np.array([[100.0, 0.0], [0.0, 0.0]])
    magnitude = roberts_magnitude(image)
    assert magnitude.shape == (1, 1)
    assert magnitude[0, 0] == pytest.approx(100.0)


def test_rejects_1x1_image():
    with pytest.raises(ValueError, match="2x2"):
        roberts_magnitude(np.zeros((1, 1)))


def test_rejects_1d_image():
    with pytest.raises(ValueError, match="2D"):
        roberts_magnitude(np.zeros(10))


def test_rejects_empty_image():
    with pytest.raises(ValueError, match="empty"):
        roberts_magnitude(np.zeros((0, 0)))


def test_rejects_nan_image():
    image = np.zeros((5, 5))
    image[2, 2] = np.nan
    with pytest.raises(ValueError, match="finite"):
        roberts_magnitude(image)


def test_threshold_rejects_negative_threshold():
    with pytest.raises(ValueError, match="non-negative"):
        threshold_edges(np.zeros((3, 3)), -1.0)


def test_threshold_rejects_negative_magnitude():
    with pytest.raises(ValueError, match="negative"):
        threshold_edges(np.array([[-1.0]]), 0.0)
