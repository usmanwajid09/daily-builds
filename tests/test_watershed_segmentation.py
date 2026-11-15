"""Tests for watershed_segmentation."""

import numpy as np
import pytest

from watershed_segmentation import (
    WATERSHED,
    distance_transform,
    find_markers,
    segment_count,
    watershed,
    watershed_lines,
)
from watershed_segmentation.demo import make_mask


def _disc_mask(height=40, width=50, centre=(20, 25), radius=10):
    rows, cols = np.mgrid[0:height, 0:width]
    return (rows - centre[0]) ** 2 + (cols - centre[1]) ** 2 <= radius ** 2


def test_single_disc_is_one_segment():
    labels = watershed(_disc_mask())
    assert segment_count(labels) == 1
    assert watershed_lines(labels).sum() == 0
    assert (labels[_disc_mask()] > 0).all()


def test_two_overlapping_discs_split_in_two():
    labels = watershed(make_mask())
    assert segment_count(labels) == 2
    assert watershed_lines(labels).sum() > 0


def test_two_separated_discs_no_watershed_line():
    mask = _disc_mask(centre=(15, 12), radius=8) | \
        _disc_mask(centre=(15, 38), radius=8)
    labels = watershed(mask)
    assert segment_count(labels) == 2
    assert watershed_lines(labels).sum() == 0


def test_distance_transform_peak_at_centre():
    dist = distance_transform(_disc_mask(radius=10))
    assert dist.max() == pytest.approx(10.0, abs=1.0)
    assert dist[20, 25] == dist.max()
    assert dist[0, 0] == 0.0  # background
    assert dist[20, 15] < dist[20, 25]  # nearer the edge


def test_find_markers_single_disc():
    markers = find_markers(_disc_mask())
    assert segment_count(markers) == 1


def test_find_markers_two_discs():
    markers = find_markers(make_mask(), min_distance=5.0)
    assert segment_count(markers) == 2


def test_custom_markers_respected():
    mask = _disc_mask()
    markers = np.zeros(mask.shape, dtype=np.int64)
    markers[20, 25] = 7
    labels = watershed(mask, markers=markers)
    assert segment_count(labels) == 1
    assert set(np.unique(labels[mask])) == {7}


def test_background_stays_zero():
    labels = watershed(_disc_mask())
    assert (labels[~_disc_mask()] == 0).all()


def test_labels_cover_all_foreground():
    mask = make_mask()
    labels = watershed(mask)
    assert (labels[mask] != 0).all()


def test_empty_mask_raises():
    with pytest.raises(ValueError):
        watershed(np.zeros((10, 10), dtype=bool))


def test_1d_mask_raises():
    with pytest.raises(ValueError):
        watershed(np.zeros(10, dtype=bool))


def test_markers_wrong_shape_raises():
    with pytest.raises(ValueError):
        watershed(_disc_mask(), markers=np.zeros((5, 5), dtype=int))


def test_marker_outside_mask_raises():
    mask = _disc_mask()
    markers = np.zeros(mask.shape, dtype=int)
    markers[0, 0] = 1
    with pytest.raises(ValueError):
        watershed(mask, markers=markers)


def test_component_without_marker_raises():
    mask = _disc_mask(centre=(15, 12), radius=8) | \
        _disc_mask(centre=(15, 38), radius=8)
    markers = np.zeros(mask.shape, dtype=int)
    markers[15, 12] = 1  # only the left disc is seeded
    with pytest.raises(ValueError):
        watershed(mask, markers=markers)


def test_segment_count_validation():
    with pytest.raises(ValueError):
        segment_count(np.zeros((3, 3, 3)))
    assert segment_count(np.array([[1, 1, 2], [0, -1, 2]])) == 2


def test_watershed_lines_mask():
    labels = np.array([[1, WATERSHED, 2], [1, WATERSHED, 2]])
    lines = watershed_lines(labels)
    assert lines.sum() == 2
    assert lines.dtype == bool
