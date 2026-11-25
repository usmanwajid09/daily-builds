"""Tests for shi_tomasi_corners (Shi-Tomasi corner detection)."""

import numpy as np
import pytest

from shi_tomasi_corners.shi_tomasi_corners import corner_response, detect_corners


def _square_image(height=30, width=30, top=10, left=10, size=10, value=200.0):
    image = np.zeros((height, width), dtype=np.float64)
    image[top:top + size, left:left + size] = value
    return image


def test_constant_image_has_zero_response():
    np.testing.assert_allclose(corner_response(np.full((10, 10), 77.0)), 0.0)


def test_constant_image_has_no_corners():
    assert detect_corners(np.full((10, 10), 77.0)) == []


def test_response_shape_matches_input():
    image = _square_image(height=12, width=18)
    assert corner_response(image).shape == image.shape


def test_response_is_non_negative():
    rng = np.random.default_rng(0)
    image = rng.normal(100.0, 20.0, size=(16, 16))
    assert (corner_response(image) >= 0.0).all()


def test_square_image_detects_four_corners():
    corners = detect_corners(_square_image(), max_corners=10, min_distance=5)
    assert len(corners) == 4
    positions = {(row, col) for row, col, _ in corners}
    assert positions == {(10, 10), (10, 19), (19, 10), (19, 19)}


def test_corners_sorted_by_descending_strength():
    corners = detect_corners(_square_image(), max_corners=10, min_distance=5)
    strengths = [s for _, _, s in corners]
    assert strengths == sorted(strengths, reverse=True)


def test_straight_edge_is_not_a_corner():
    # A vertical step edge: strong gradient in one direction only,
    # so the smaller eigenvalue stays ~0 and no corner is found.
    image = np.zeros((20, 20))
    image[:, 10:] = 200.0
    assert detect_corners(image) == []


def test_max_corners_limits_results():
    corners = detect_corners(_square_image(), max_corners=2, min_distance=5)
    assert len(corners) == 2


def test_min_distance_suppresses_nearby_corners():
    # min_distance=3 keeps all four corners (the strong pixels next
    # to a corner lie within 3 px of it); a huge min_distance (larger
    # than the square) keeps only the strongest one.
    assert len(detect_corners(_square_image(), min_distance=3)) == 4
    assert len(detect_corners(_square_image(), min_distance=30)) == 1


def test_quality_level_filters_weak_corners():
    # quality_level=1.0 keeps only pixels at the maximum response.
    corners = detect_corners(_square_image(), quality_level=1.0, min_distance=5)
    assert len(corners) >= 1
    assert len(corners) <= 4
    strongest = corner_response(_square_image()).max()
    for _, _, strength in corners:
        assert strength == pytest.approx(strongest)


def test_uint8_input_accepted():
    image = np.zeros((30, 30), dtype=np.uint8)
    image[10:20, 10:20] = 200
    assert len(detect_corners(image, min_distance=5)) == 4


def test_larger_window_still_finds_square_corners():
    corners = detect_corners(_square_image(), window_size=5, min_distance=5)
    assert len(corners) == 4


def test_rejects_1d_image():
    with pytest.raises(ValueError, match="2D"):
        corner_response(np.zeros(10))


def test_rejects_rgb_image():
    with pytest.raises(ValueError, match="2D"):
        corner_response(np.zeros((4, 4, 3)))


def test_rejects_empty_image():
    with pytest.raises(ValueError, match="empty"):
        corner_response(np.zeros((0, 0)))


def test_rejects_nan_image():
    image = np.zeros((5, 5))
    image[2, 2] = np.nan
    with pytest.raises(ValueError, match="finite"):
        corner_response(image)


def test_rejects_even_window_size():
    with pytest.raises(ValueError, match="odd"):
        corner_response(np.zeros((5, 5)), window_size=4)


def test_rejects_zero_window_size():
    with pytest.raises(ValueError, match="odd"):
        corner_response(np.zeros((5, 5)), window_size=0)


def test_rejects_quality_level_out_of_range():
    with pytest.raises(ValueError, match="quality_level"):
        detect_corners(np.zeros((5, 5)), quality_level=0.0)
    with pytest.raises(ValueError, match="quality_level"):
        detect_corners(np.zeros((5, 5)), quality_level=1.5)


def test_rejects_zero_max_corners():
    with pytest.raises(ValueError, match="max_corners"):
        detect_corners(np.zeros((5, 5)), max_corners=0)


def test_rejects_zero_min_distance():
    with pytest.raises(ValueError, match="min_distance"):
        detect_corners(np.zeros((5, 5)), min_distance=0)
