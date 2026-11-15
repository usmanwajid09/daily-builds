"""Tests for hough_lines."""

import numpy as np
import pytest

from hough_lines import detect_lines, hough_transform, line_endpoints
from hough_lines.demo import make_edge_image


def test_accumulator_shape_and_peak_for_diagonal():
    edges = make_edge_image()
    acc, thetas, rhos = hough_transform(edges)
    assert acc.shape == (len(rhos), len(thetas))
    assert acc.dtype == np.int64
    assert acc.max() == 80  # the horizontal line has 80 edge pixels


def test_detects_diagonal_line():
    edges = make_edge_image()
    lines = detect_lines(edges, threshold=15)
    diagonal = [line for line in lines if abs(line[1] + 45) < 2]
    assert len(diagonal) == 1
    assert abs(diagonal[0][0]) < 2.0  # rho ~ 0
    assert diagonal[0][2] == 41


def test_detects_horizontal_line():
    edges = make_edge_image()
    lines = detect_lines(edges, threshold=15)
    horizontal = [line for line in lines if abs(abs(line[1]) - 90) < 2]
    assert len(horizontal) == 1
    assert abs(horizontal[0][0] + 20) < 2.0  # rho ~ -20
    assert horizontal[0][2] == 80


def test_lines_sorted_by_votes_descending():
    edges = make_edge_image()
    lines = detect_lines(edges, threshold=15)
    votes = [line[2] for line in lines]
    assert votes == sorted(votes, reverse=True)
    assert len(lines) == 2


def test_empty_edge_map_returns_no_lines():
    edges = np.zeros((20, 30), dtype=bool)
    acc, _, _ = hough_transform(edges)
    assert acc.max() == 0
    assert detect_lines(edges, threshold=1) == []


def test_single_pixel_below_threshold():
    edges = np.zeros((20, 20), dtype=bool)
    edges[10, 10] = True
    assert detect_lines(edges, threshold=10) == []


def test_non_binary_edge_map_counts_nonzero():
    edges = np.zeros((20, 20), dtype=np.uint8)
    edges[5, 2:18] = 255
    lines = detect_lines(edges, threshold=10)
    assert len(lines) == 1
    assert lines[0][2] == 16


def test_line_endpoints_horizontal_span_width():
    (x1, y1), (x2, y2) = line_endpoints(-20.0, -90.0, (80, 100))
    assert y1 == pytest.approx(20.0)
    assert y2 == pytest.approx(20.0)
    assert {x1, x2} == {0.0, 99.0}


def test_line_endpoints_inside_image():
    for rho, theta in [(0.0, -45.0), (-20.0, -90.0), (30.0, 0.0)]:
        (x1, y1), (x2, y2) = line_endpoints(rho, theta, (80, 100))
        for x, y in [(x1, y1), (x2, y2)]:
            assert 0 <= x <= 99
            assert 0 <= y <= 79


def test_line_endpoints_line_off_image_raises():
    with pytest.raises(ValueError):
        line_endpoints(500.0, 0.0, (80, 100))


def test_invalid_edge_map_ndim_raises():
    with pytest.raises(ValueError):
        hough_transform(np.zeros((5, 5, 3)))


def test_empty_edge_map_raises():
    with pytest.raises(ValueError):
        hough_transform(np.zeros((0, 0)))


def test_bad_threshold_raises():
    with pytest.raises(ValueError):
        detect_lines(np.zeros((10, 10)), threshold=0)


def test_bad_theta_step_raises():
    with pytest.raises(ValueError):
        hough_transform(np.zeros((10, 10)), theta_step=0)


def test_bad_shape_raises():
    with pytest.raises(ValueError):
        line_endpoints(0.0, 0.0, (10,))
