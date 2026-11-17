"""Tests for hough_circles."""

import numpy as np
import pytest

from hough_circles import (
    circle_edge_map,
    detect_circles,
    hough_circle_transform,
    sobel_edges,
)
from hough_circles.demo import make_edge_image


def test_accumulator_shape_and_dtype():
    edges = make_edge_image()
    acc, radii = hough_circle_transform(edges, radii=[10, 14])
    assert acc.shape == (60, 80, 2)
    assert acc.dtype == np.int64
    assert radii == [10, 14]
    assert acc.max() > 0


def test_detects_single_circle():
    edges = circle_edge_map((60, 80), [(25, 20, 10)])
    circles = detect_circles(edges, radii=[10])
    assert len(circles) == 1
    circle = circles[0]
    assert abs(circle.x - 25) <= 1
    assert abs(circle.y - 20) <= 1
    assert circle.radius == 10
    # circumference of r=10 is ~63 pixels, so the peak should be large
    assert circle.votes >= 40


def test_detects_two_circles():
    edges = make_edge_image()
    circles = detect_circles(edges, radii=[8, 10, 12, 14, 16])
    assert len(circles) == 2
    found = sorted((c.x, c.y, c.radius) for c in circles)
    assert found[0][2] == 10
    assert abs(found[0][0] - 25) <= 1 and abs(found[0][1] - 20) <= 1
    assert found[1][2] == 14
    assert abs(found[1][0] - 55) <= 1 and abs(found[1][1] - 40) <= 1


def test_empty_edge_map_returns_no_circles():
    edges = np.zeros((40, 50), dtype=bool)
    assert detect_circles(edges, radii=[10]) == []
    acc, _ = hough_circle_transform(edges, radii=[10])
    assert acc.max() == 0


def test_high_threshold_returns_no_circles():
    edges = make_edge_image()
    assert detect_circles(edges, radii=[10, 14], threshold=10000) == []


def test_nms_merges_neighbouring_radii():
    # One circle, but both radius 10 and 11 are searched: report it once.
    edges = circle_edge_map((60, 80), [(25, 20, 10)])
    circles = detect_circles(edges, radii=[10, 11], min_distance=5)
    assert len(circles) == 1
    assert circles[0].radius in (10, 11)


def test_radii_validation():
    edges = make_edge_image()
    with pytest.raises(ValueError):
        detect_circles(edges, radii=[])
    with pytest.raises(ValueError):
        detect_circles(edges, radii=[0])
    with pytest.raises(ValueError):
        detect_circles(edges, radii=[-5])
    with pytest.raises(ValueError):
        detect_circles(edges, radii=[10.5])
    with pytest.raises(ValueError):
        detect_circles(edges, radii=10)


def test_edge_map_validation():
    with pytest.raises(ValueError):
        detect_circles(np.zeros(10), radii=[10])
    with pytest.raises(ValueError):
        detect_circles(np.zeros((5, 5, 3)), radii=[10])
    with pytest.raises(ValueError):
        detect_circles(np.zeros((0, 5), dtype=bool), radii=[10])
    with pytest.raises(ValueError):
        detect_circles(make_edge_image(), radii=[10], threshold=0)
    with pytest.raises(ValueError):
        detect_circles(make_edge_image(), radii=[10], min_distance=0)
    with pytest.raises(ValueError):
        hough_circle_transform(make_edge_image(), radii=[10], angle_step=0)


def test_sobel_edges_finds_disc_outline():
    yy, xx = np.mgrid[0:60, 0:80]
    disc = np.where((xx - 40) ** 2 + (yy - 30) ** 2 <= 12 ** 2, 200.0, 0.0)
    edges = sobel_edges(disc, threshold=50.0)
    assert edges.dtype == bool
    # The outline is a ~2-3px thick ring around the r=12 disc.
    assert 150 <= int(edges.sum()) <= 250
    circles = detect_circles(edges, radii=[12])
    assert len(circles) == 1
    assert abs(circles[0].x - 40) <= 1
    assert abs(circles[0].y - 30) <= 1


def test_sobel_edges_validation():
    with pytest.raises(ValueError):
        sobel_edges(np.zeros((5, 5, 3)))
    with pytest.raises(ValueError):
        sobel_edges(np.zeros((5, 5)), threshold=-1)


def test_circle_edge_map_validation():
    with pytest.raises(ValueError):
        circle_edge_map((0, 10), [(5, 5, 3)])
    with pytest.raises(ValueError):
        circle_edge_map((10, 10), [(5, 5, 0)])
    with pytest.raises(ValueError):
        circle_edge_map((10,), [(5, 5, 3)])
    edges = circle_edge_map((20, 20), [(10, 10, 5)])
    assert edges.shape == (20, 20)
    assert edges.dtype == bool
    assert int(edges.sum()) > 20
