"""Tests for convex_hull."""

import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from convex_hull import graham_scan, jarvis_march, point_in_hull, polygon_area
from convex_hull.demo import make_points


def _square_with_interior():
    return np.array([
        [0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0],
        [5.0, 5.0], [3.0, 4.0], [7.0, 2.0], [5.0, 0.0], [0.0, 5.0],
    ])


def test_graham_square_drops_interior_and_edge_points():
    hull = graham_scan(_square_with_interior())
    assert len(hull) == 4
    assert set(map(tuple, hull)) == {(0.0, 0.0), (10.0, 0.0),
                                     (10.0, 10.0), (0.0, 10.0)}


def test_graham_hull_is_counter_clockwise():
    hull = graham_scan(_square_with_interior())
    # Signed area positive means counter-clockwise.
    x, y = hull[:, 0], hull[:, 1]
    signed = 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    assert signed > 0


def test_jarvis_matches_graham_on_random_points():
    rng = np.random.default_rng(7)
    points = rng.uniform(-5, 5, size=(50, 2))
    assert set(map(tuple, jarvis_march(points))) == set(map(tuple, graham_scan(points)))


def test_jarvis_matches_graham_on_square():
    hull_g = graham_scan(_square_with_interior())
    hull_j = jarvis_march(_square_with_interior())
    assert set(map(tuple, hull_j)) == set(map(tuple, hull_g))


def test_both_agree_on_grid_with_collinear_edges():
    xx, yy = np.meshgrid(np.arange(5), np.arange(4))
    points = np.column_stack([xx.ravel(), yy.ravel()]).astype(float)
    expected = {(0.0, 0.0), (4.0, 0.0), (4.0, 3.0), (0.0, 3.0)}
    assert set(map(tuple, graham_scan(points))) == expected
    assert set(map(tuple, jarvis_march(points))) == expected


def test_triangle_hull():
    points = np.array([[0.0, 0.0], [4.0, 0.0], [2.0, 3.0], [2.0, 1.0]])
    hull = graham_scan(points)
    assert len(hull) == 3
    assert polygon_area(hull) == pytest.approx(6.0)


def test_polygon_area_unit_square():
    hull = graham_scan(_square_with_interior())
    assert polygon_area(hull) == pytest.approx(100.0)


def test_collinear_points_return_endpoints():
    points = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    expected = {(0.0, 0.0), (3.0, 3.0)}
    assert set(map(tuple, graham_scan(points))) == expected
    assert set(map(tuple, jarvis_march(points))) == expected


def test_duplicate_points_ignored():
    points = np.array([[0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [1.0, 0.0],
                       [0.0, 1.0], [0.0, 1.0], [1.0, 1.0], [0.5, 0.5]])
    hull = graham_scan(points)
    assert len(hull) == 4


def test_single_point():
    assert np.array_equal(graham_scan([[3.0, 4.0]]), [[3.0, 4.0]])
    assert np.array_equal(jarvis_march([[3.0, 4.0]]), [[3.0, 4.0]])


def test_two_points():
    points = np.array([[0.0, 0.0], [2.0, 2.0]])
    assert len(graham_scan(points)) == 2
    assert len(jarvis_march(points)) == 2


def test_point_in_hull_interior_and_exterior():
    hull = graham_scan(_square_with_interior())
    assert point_in_hull((5.0, 5.0), hull)
    assert point_in_hull((5.0, 9.9), hull)
    assert not point_in_hull((20.0, 5.0), hull)
    assert not point_in_hull((-1.0, 5.0), hull)


def test_point_in_hull_on_boundary():
    hull = graham_scan(_square_with_interior())
    assert point_in_hull((5.0, 0.0), hull)   # on an edge
    assert point_in_hull((0.0, 0.0), hull)   # on a vertex


def test_point_in_hull_degenerate():
    assert point_in_hull((1.0, 1.0), [[1.0, 1.0]])
    assert not point_in_hull((1.0, 2.0), [[1.0, 1.0]])
    assert point_in_hull((1.0, 1.0), [[0.0, 0.0], [2.0, 2.0]])
    assert not point_in_hull((1.0, 2.0), [[0.0, 0.0], [2.0, 2.0]])


def test_demo_points_hull_area_sane():
    points = make_points()
    hull = graham_scan(points)
    assert 50.0 < polygon_area(hull) < 100.0
    assert point_in_hull(points.mean(axis=0), hull)
    assert not point_in_hull((20.0, 20.0), hull)


def test_empty_points_raises():
    with pytest.raises(ValueError, match="empty"):
        graham_scan(np.empty((0, 2)))


def test_1d_points_raises():
    with pytest.raises(ValueError, match="shape"):
        graham_scan([1.0, 2.0, 3.0])


def test_nan_points_raises():
    with pytest.raises(ValueError, match="finite"):
        jarvis_march([[0.0, 0.0], [np.nan, 1.0], [1.0, 1.0]])


def test_bad_point_shape_raises():
    hull = graham_scan(_square_with_interior())
    with pytest.raises(ValueError, match="shape"):
        point_in_hull((1.0, 2.0, 3.0), hull)
