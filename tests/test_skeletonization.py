"""Tests for skeletonization."""

import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from skeletonization import (
    count_branch_points,
    count_endpoints,
    neighbour_count,
    zhang_suen_thin,
)
from skeletonization.demo import make_bar, make_plus


def _bar():
    image = np.zeros((11, 21), dtype=bool)
    image[4:7, 3:18] = True  # 3 rows x 15 cols
    return image


def test_thin_bar_becomes_single_line():
    skeleton, _ = zhang_suen_thin(_bar())
    # One centred row. Zhang-Suen also trims ~1 pixel off each end
    # of a short stroke, so the 15-column bar leaves 12 pixels.
    assert int(skeleton.sum()) == 12
    assert np.all(skeleton[5, 4:16])
    assert not np.any(skeleton[[4, 6], :])
    assert np.array_equal(np.where(skeleton.any(axis=1))[0], [5])


def test_thin_bar_endpoints_and_no_branches():
    skeleton, _ = zhang_suen_thin(_bar())
    assert count_endpoints(skeleton) == 2
    assert count_branch_points(skeleton) == 0


def test_thin_plus_has_four_endpoints_one_branch():
    skeleton, _ = zhang_suen_thin(make_plus())
    assert count_endpoints(skeleton) == 4
    # The junction thins to a small cluster, not one pixel: under
    # 8-connectivity the centre and its four diagonal neighbours all
    # have >= 3 neighbours, so 5 branch points, not 1.
    assert count_branch_points(skeleton) == 5


def test_thin_square_ring_stays_a_ring():
    image = np.zeros((15, 15), dtype=bool)
    image[3:12, 3:12] = True
    image[5:10, 5:10] = False  # hollow centre -> ring
    skeleton, _ = zhang_suen_thin(image)
    # A ring's skeleton is a closed loop: no endpoints at all.
    assert count_endpoints(skeleton) == 0
    assert int(skeleton.sum()) > 0


def test_thin_single_pixel_unchanged():
    image = np.zeros((5, 5), dtype=bool)
    image[2, 2] = True
    skeleton, _ = zhang_suen_thin(image)
    assert np.array_equal(skeleton, image)


def test_thin_empty_image():
    skeleton, iterations = zhang_suen_thin(np.zeros((5, 5), dtype=bool))
    assert not np.any(skeleton)
    assert iterations == 1


def test_thin_result_is_subset_of_input():
    binary = make_plus()
    skeleton, _ = zhang_suen_thin(binary)
    assert np.all(~skeleton | binary)


def test_thin_one_pixel_wide_line_unchanged():
    image = np.zeros((5, 9), dtype=bool)
    image[2, 1:8] = True
    skeleton, _ = zhang_suen_thin(image)
    assert np.array_equal(skeleton, image)


def test_thin_is_idempotent():
    skeleton, _ = zhang_suen_thin(make_plus())
    again, _ = zhang_suen_thin(skeleton)
    assert np.array_equal(again, skeleton)


def test_thin_bar_demo_counts():
    skeleton, _ = zhang_suen_thin(make_bar())
    assert count_endpoints(skeleton) == 2
    assert count_branch_points(skeleton) == 0


def test_neighbour_count_centre():
    image = np.zeros((3, 3), dtype=bool)
    image[1, 1] = True
    image[0, 1] = True
    image[1, 0] = True
    counts = neighbour_count(image)
    assert counts[1, 1] == 2
    # (0, 1) touches (1, 1) and, diagonally, (1, 0): 2 neighbours.
    assert counts[0, 1] == 2


def test_endpoints_of_handmade_cross():
    skeleton = np.zeros((5, 5), dtype=bool)
    skeleton[2, :] = True
    skeleton[:, 2] = True
    assert count_endpoints(skeleton) == 4
    # Under 8-connectivity the centre's four diagonal neighbours also
    # see 4 neighbours each, so the junction counts as 5 branch points.
    assert count_branch_points(skeleton) == 5


def test_branch_points_of_handmade_t():
    skeleton = np.zeros((5, 5), dtype=bool)
    skeleton[0, 2] = True
    skeleton[1, 2] = True
    skeleton[2, :] = True
    assert count_endpoints(skeleton) == 3
    # Centre, its two horizontal neighbours and the pixel above the
    # centre all have >= 3 neighbours under 8-connectivity.
    assert count_branch_points(skeleton) == 4


def test_max_iterations_cap_respected():
    _, iterations = zhang_suen_thin(make_plus(), max_iterations=1)
    assert iterations == 1


def test_non_bool_image_raises():
    with pytest.raises(ValueError, match="boolean"):
        zhang_suen_thin(np.zeros((5, 5), dtype=int))


def test_1d_image_raises():
    with pytest.raises(ValueError, match="2-D"):
        zhang_suen_thin(np.zeros(9, dtype=bool))


def test_bad_max_iterations_raises():
    with pytest.raises(ValueError, match="max_iterations"):
        zhang_suen_thin(np.zeros((5, 5), dtype=bool), max_iterations=0)


def test_counts_reject_non_bool():
    with pytest.raises(ValueError, match="boolean"):
        count_endpoints(np.zeros((3, 3), dtype=int))
