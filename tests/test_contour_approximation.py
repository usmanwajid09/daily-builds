"""Tests for contour_approximation."""

import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from contour_approximation import (
    approximate_contour,
    douglas_peucker,
    polygon_area,
    polygon_perimeter,
    trace_contour,
)
from contour_approximation.demo import make_binary_image


def _rectangle(height=50, width=70, r0=10, r1=20, c0=20, c1=40):
    binary = np.zeros((height, width), dtype=bool)
    binary[r0:r1, c0:c1] = True
    return binary


def test_trace_rectangle_visits_boundary_once():
    binary = _rectangle()
    points = trace_contour(binary)
    # Perimeter pixels of a 10x20 rectangle: 2*(10+20) - 4 = 56.
    assert len(points) == 56
    assert len(np.unique(points, axis=0)) == 56
    # All points lie on the rectangle's edge rows/columns.
    xs, ys = points[:, 0], points[:, 1]
    assert np.all(((ys == 10) | (ys == 19) | (xs == 20) | (xs == 39)))


def test_trace_rectangle_corners_present():
    points = trace_contour(_rectangle())
    corners = {(20, 10), (39, 10), (39, 19), (20, 19)}
    assert corners <= {tuple(p) for p in points}


def test_trace_empty_image():
    points = trace_contour(np.zeros((5, 5), dtype=bool))
    assert points.shape == (0, 2)


def test_trace_single_pixel():
    binary = np.zeros((5, 5), dtype=bool)
    binary[2, 2] = True
    points = trace_contour(binary)
    assert len(points) == 1
    assert tuple(points[0]) == (2, 2)


def test_trace_disc_is_closed_loop():
    yy, xx = np.mgrid[0:50, 0:70]
    binary = (xx - 35) ** 2 + (yy - 25) ** 2 <= 10 ** 2
    points = trace_contour(binary)
    assert len(points) > 40
    # Consecutive points (wrapping around) are 8-neighbours.
    steps = np.diff(np.vstack([points, points[:1]]), axis=0)
    assert np.all(np.abs(steps).max(axis=1) <= 1)


def test_douglas_peucker_straight_line_collapses():
    points = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    simplified = douglas_peucker(points, epsilon=0.1)
    assert len(simplified) == 2
    assert np.array_equal(simplified[0], [0.0, 0.0])
    assert np.array_equal(simplified[-1], [3.0, 0.0])


def test_douglas_peucker_keeps_far_point():
    points = np.array([[0.0, 0.0], [1.0, 5.0], [2.0, 0.0]])
    simplified = douglas_peucker(points, epsilon=1.0)
    assert len(simplified) == 3


def test_douglas_peucker_zero_epsilon_keeps_all():
    points = np.array([[0.0, 0.0], [1.0, 2.0], [2.0, 0.0], [3.0, 1.0]])
    simplified = douglas_peucker(points, epsilon=0.0)
    assert len(simplified) == 4


def test_douglas_peucker_closed_square_gives_four_corners():
    square = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0],
                       [0.0, 10.0], [0.0, 0.0]])
    simplified = douglas_peucker(square, epsilon=0.5)
    assert len(simplified) == 5  # 4 corners + closing repeat
    assert np.array_equal(simplified[0], simplified[-1])


def test_polygon_area_unit_square():
    assert polygon_area([[0, 0], [1, 0], [1, 1], [0, 1]]) == pytest.approx(1.0)


def test_polygon_area_degenerate():
    assert polygon_area([[0, 0], [1, 1]]) == 0.0


def test_polygon_perimeter_unit_square():
    assert polygon_perimeter([[0, 0], [1, 0], [1, 1], [0, 1]]) == pytest.approx(4.0)


def test_approximate_rectangle_four_vertices():
    result = approximate_contour(_rectangle(), epsilon=0.5)
    vertices = result["vertices"]
    assert len(vertices) == 5  # 4 corners + closing repeat
    assert np.array_equal(vertices[0], vertices[-1])
    # Pixel-centre rectangle spans 19 x 9.
    assert result["area"] == pytest.approx(19 * 9)
    assert result["perimeter"] == pytest.approx(2 * (19 + 9))
    assert len(result["raw_points"]) == 56


def test_approximate_disc_few_vertices_small_area_error():
    yy, xx = np.mgrid[0:50, 0:70]
    binary = (xx - 35) ** 2 + (yy - 25) ** 2 <= 10 ** 2
    result = approximate_contour(binary, epsilon=1.0)
    assert 4 < len(result["vertices"]) < 30
    assert result["area"] == pytest.approx(np.pi * 10 ** 2, rel=0.15)


def test_approximate_larger_epsilon_fewer_vertices():
    binary = make_binary_image()
    small = approximate_contour(binary, epsilon=0.5)
    large = approximate_contour(binary, epsilon=3.0)
    assert len(large["vertices"]) <= len(small["vertices"])


def test_approximate_empty_image():
    result = approximate_contour(np.zeros((5, 5), dtype=bool))
    assert len(result["vertices"]) == 0
    assert result["area"] == 0.0 and result["perimeter"] == 0.0


def test_non_bool_image_raises():
    with pytest.raises(ValueError, match="boolean"):
        trace_contour(np.zeros((5, 5), dtype=int))


def test_1d_image_raises():
    with pytest.raises(ValueError, match="2-D"):
        trace_contour(np.zeros(9, dtype=bool))


def test_negative_epsilon_raises():
    with pytest.raises(ValueError, match="epsilon"):
        approximate_contour(_rectangle(), epsilon=-1.0)
    with pytest.raises(ValueError, match="epsilon"):
        douglas_peucker([[0, 0], [1, 1]], epsilon=-0.5)


def test_douglas_peucker_too_few_points_raises():
    with pytest.raises(ValueError, match="at least 2 points"):
        douglas_peucker([[0, 0]], epsilon=1.0)
