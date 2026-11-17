"""Tests for contour_detection."""

import numpy as np
import pytest

from contour_detection import (
    bounding_box,
    contour_area,
    contour_perimeter,
    find_contours,
)
from contour_detection.demo import make_binary_image


def _filled_rectangle(height=50, width=70, r0=10, r1=20, c0=20, c1=40):
    binary = np.zeros((height, width), dtype=bool)
    binary[r0:r1, c0:c1] = True
    return binary


def test_filled_rectangle_contour():
    binary = _filled_rectangle()
    contours = find_contours(binary)
    assert len(contours) == 1
    contour = contours[0]
    # Boundary pixels form a 19x9 rectangle in pixel-centre coordinates.
    assert contour.bbox == (20, 10, 39, 19)
    assert contour.area == pytest.approx(19 * 9)
    assert contour.perimeter == pytest.approx(2 * (19 + 9))
    # Perimeter pixels of a 10x20 rectangle: 2*(10+20) - 4 = 56, plus
    # the repeated start pixel that closes the loop.
    assert len(contour.points) == 57


def test_two_squares_give_two_contours():
    binary = np.zeros((50, 70), dtype=bool)
    binary[5:15, 5:15] = True
    binary[30:40, 50:60] = True
    contours = find_contours(binary)
    assert len(contours) == 2
    bboxes = sorted(c.bbox for c in contours)
    assert bboxes[0] == (5, 5, 14, 14)
    assert bboxes[1] == (50, 30, 59, 39)


def test_empty_image_gives_no_contours():
    assert find_contours(np.zeros((20, 30), dtype=bool)) == []


def test_single_pixel_contour():
    binary = np.zeros((10, 10), dtype=bool)
    binary[4, 6] = True
    contours = find_contours(binary)
    assert len(contours) == 1
    assert contours[0].points.tolist() == [[6, 4]]
    assert contours[0].area == 0.0
    assert contours[0].perimeter == 0.0
    assert contours[0].bbox == (6, 4, 6, 4)


def test_full_image_is_one_contour():
    binary = np.ones((8, 12), dtype=bool)
    contours = find_contours(binary)
    assert len(contours) == 1
    assert contours[0].bbox == (0, 0, 11, 7)
    assert contours[0].area == pytest.approx(11 * 7)


def test_ring_gives_only_outer_contour():
    yy, xx = np.mgrid[0:40, 0:40]
    binary = ((xx - 20) ** 2 + (yy - 20) ** 2 <= 12 ** 2) & (
        (xx - 20) ** 2 + (yy - 20) ** 2 >= 6 ** 2
    )
    contours = find_contours(binary)
    assert len(contours) == 1
    assert contours[0].bbox == (8, 8, 32, 32)


def test_diagonal_line_is_one_contour():
    binary = np.zeros((20, 20), dtype=bool)
    for i in range(5, 15):
        binary[i, i] = True
    contours = find_contours(binary)
    assert len(contours) == 1
    assert contours[0].area == 0.0


def test_contours_sorted_by_area_descending():
    binary = np.zeros((60, 70), dtype=bool)
    binary[5:10, 5:10] = True   # small square
    binary[20:40, 20:50] = True  # big rectangle
    contours = find_contours(binary)
    assert len(contours) == 2
    assert contours[0].area > contours[1].area
    assert contours[0].bbox == (20, 20, 49, 39)


def test_input_validation():
    with pytest.raises(ValueError):
        find_contours(np.zeros(10))
    with pytest.raises(ValueError):
        find_contours(np.zeros((5, 5, 3)))
    with pytest.raises(ValueError):
        find_contours(np.zeros((0, 5), dtype=bool))


def test_geometry_helpers():
    square = np.array([[0, 0], [4, 0], [4, 3], [0, 3]])
    assert contour_area(square) == pytest.approx(12.0)
    assert contour_perimeter(square) == pytest.approx(14.0)
    assert bounding_box(square) == (0, 0, 4, 3)
    assert contour_area(np.array([[0, 0], [1, 1]])) == 0.0
    assert contour_perimeter(np.array([[0, 0]])) == 0.0
    with pytest.raises(ValueError):
        bounding_box(np.zeros((0, 2), dtype=int))
    with pytest.raises(ValueError):
        contour_area(np.zeros((3, 3)))


def test_demo_image_contours():
    binary = make_binary_image()
    contours = find_contours(binary)
    assert len(contours) == 2
    # Rectangle area 19*9 = 171; disc area is close to pi*8^2 ~ 201
    # but measured through boundary pixel centres, so allow slack.
    assert contours[0].area == pytest.approx(171.0, abs=40)
    assert contours[1].area == pytest.approx(171.0, abs=40)
