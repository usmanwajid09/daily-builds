"""Tests for laplacian_edge_detector (Laplacian edge detection)."""

import numpy as np
import pytest

from laplacian_edge_detector.laplacian_edge_detector import (
    edge_mask,
    edge_pixel_count,
    laplacian,
    laplacian_magnitude,
    zero_crossings,
)


def _step_edge(height=20, width=20, left=0.0, right=200.0):
    image = np.full((height, width), left, dtype=np.float64)
    image[:, width // 2:] = right
    return image


def test_constant_image_has_zero_response():
    image = np.full((10, 10), 77.0)
    np.testing.assert_allclose(laplacian(image), 0.0, atol=1e-12)


def test_step_edge_response_sign_pattern():
    # 4-neighbour kernel: left of the step +200, right of it -200.
    response = laplacian(_step_edge())
    assert response[:, 9].max() == pytest.approx(200.0)
    assert response[:, 10].min() == pytest.approx(-200.0)
    np.testing.assert_allclose(response[:, :9], 0.0, atol=1e-12)
    np.testing.assert_allclose(response[:, 11:], 0.0, atol=1e-12)


def test_8_neighbour_kernel_response():
    # 8-neighbour kernel: three rows contribute, so 3x the 4-neighbour value.
    response = laplacian(_step_edge(), kernel="8-neighbour")
    assert response[:, 9].max() == pytest.approx(600.0)
    assert response[:, 10].min() == pytest.approx(-600.0)


def test_output_shape_matches_input():
    image = _step_edge(height=12, width=18)
    assert laplacian(image).shape == image.shape
    assert laplacian_magnitude(image).shape == image.shape


def test_rgb_applies_per_channel():
    image = np.zeros((8, 8, 3), dtype=np.float64)
    image[:, 4:, 0] = 200.0  # step only in the red channel
    response = laplacian(image)
    assert response.shape == image.shape
    # Step boundary is between columns 3 and 4 in this 8-wide image.
    assert response[:, 3, 0].max() == pytest.approx(200.0)
    np.testing.assert_allclose(response[:, :, 1], 0.0, atol=1e-12)
    np.testing.assert_allclose(response[:, :, 2], 0.0, atol=1e-12)


def test_uint8_input_accepted():
    image = np.zeros((8, 8), dtype=np.uint8)
    image[:, 4:] = 200
    assert laplacian(image).max() == pytest.approx(200.0)


def test_magnitude_is_absolute_response():
    image = _step_edge()
    np.testing.assert_allclose(laplacian_magnitude(image), np.abs(laplacian(image)))


def test_zero_crossings_on_step_edge():
    # Sign flips between columns 9 and 10, on all 20 rows -> 40 crossings.
    crossings = zero_crossings(laplacian(_step_edge()))
    assert crossings.sum() == 40
    assert crossings[:, 9].all()
    assert crossings[:, 10].all()
    assert not crossings[:, :9].any()
    assert not crossings[:, 11:].any()


def test_zero_crossings_constant_image_is_empty():
    assert not zero_crossings(laplacian(np.full((10, 10), 50.0))).any()


def test_zero_crossings_threshold_suppresses_weak_edges():
    response = laplacian(_step_edge())
    assert zero_crossings(response, threshold=500.0).sum() == 0
    assert zero_crossings(response, threshold=100.0).sum() == 40


def test_edge_mask_thresholds_magnitude():
    mask = edge_mask(_step_edge(), threshold=150.0)
    assert mask.dtype == bool
    assert mask[:, 9].all() and mask[:, 10].all()
    assert mask.sum() == 40


def test_edge_pixel_count_matches_mask():
    image = _step_edge()
    assert edge_pixel_count(image, 150.0) == int(edge_mask(image, 150.0).sum())
    assert edge_pixel_count(image, 1000.0) == 0


def test_rejects_unknown_kernel():
    with pytest.raises(ValueError, match="unknown kernel"):
        laplacian(np.zeros((5, 5)), kernel="diagonal")


def test_rejects_1d_image():
    with pytest.raises(ValueError, match="2D"):
        laplacian(np.zeros(10))


def test_rejects_4d_image():
    with pytest.raises(ValueError, match="2D"):
        laplacian(np.zeros((4, 4, 3, 3)))


def test_rejects_rgb_with_wrong_channel_count():
    with pytest.raises(ValueError, match="3 channels"):
        laplacian(np.zeros((4, 4, 4)))


def test_rejects_empty_image():
    with pytest.raises(ValueError, match="empty"):
        laplacian(np.zeros((0, 5)))


def test_rejects_nan_image():
    image = np.zeros((5, 5))
    image[2, 2] = np.nan
    with pytest.raises(ValueError, match="finite"):
        laplacian(image)


def test_rejects_list_input():
    with pytest.raises(ValueError, match="numpy array"):
        laplacian([[0, 1], [1, 0]])


def test_zero_crossings_rejects_negative_threshold():
    with pytest.raises(ValueError, match="non-negative"):
        zero_crossings(np.zeros((3, 3)), threshold=-1.0)


def test_edge_mask_rejects_negative_threshold():
    with pytest.raises(ValueError, match="non-negative"):
        edge_mask(np.zeros((3, 3)), threshold=-1.0)
