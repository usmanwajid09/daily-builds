"""Tests for distance_transform."""

import numpy as np
import pytest

from distance_transform import distance_transform, nearest_background
from distance_transform.demo import make_test_image


def single_background_centre(size=21, centre=(10, 10)):
    """All foreground except one background pixel at ``centre``."""
    binary = np.ones((size, size), dtype=np.uint8)
    binary[centre] = 0
    return binary


def test_output_shape_and_dtype():
    binary = make_test_image()
    dist = distance_transform(binary)
    assert dist.shape == binary.shape
    assert dist.dtype == np.float64
    assert np.isfinite(dist).all()


def test_background_pixels_are_zero():
    binary = make_test_image()
    dist = distance_transform(binary)
    assert np.all(dist[binary == 0] == 0.0)


def test_all_background_is_zero():
    binary = np.zeros((10, 10), dtype=np.uint8)
    assert np.all(distance_transform(binary) == 0.0)


def test_single_background_pixel_approx_euclidean():
    binary = single_background_centre()
    dist = distance_transform(binary, mask="chamfer34")
    y, x = np.mgrid[0:21, 0:21]
    euclidean = np.hypot(y - 10, x - 10)
    # Chamfer 3-4 is within ~1 px of Euclidean on this grid.
    assert np.abs(dist - euclidean).max() <= 1.0
    # Centre-adjacent pixels are ~1 away.
    assert dist[10, 11] == pytest.approx(1.0)
    assert dist[9, 10] == pytest.approx(1.0)


def test_chamfer5711_more_accurate_than_chamfer34():
    binary = single_background_centre()
    y, x = np.mgrid[0:21, 0:21]
    euclidean = np.hypot(y - 10, x - 10)
    err34 = np.abs(distance_transform(binary, mask="chamfer34")
                   - euclidean).max()
    err5711 = np.abs(distance_transform(binary, mask="chamfer5711")
                     - euclidean).max()
    assert err5711 <= err34
    assert err5711 <= 0.6


def test_disc_centre_is_radius():
    binary = make_test_image()  # disc of radius 12
    dist = distance_transform(binary, mask="chamfer5711")
    assert dist[20, 25] == pytest.approx(12.0, abs=0.5)
    assert dist.max() == pytest.approx(12.0, abs=0.5)


def test_scale_false_returns_raw_units():
    binary = single_background_centre()
    raw = distance_transform(binary, mask="chamfer34", scale=False)
    assert raw[10, 11] == pytest.approx(3.0)
    assert raw[9, 9] == pytest.approx(4.0)


def test_all_foreground_raises():
    with pytest.raises(ValueError):
        distance_transform(np.ones((5, 5), dtype=np.uint8))
    with pytest.raises(ValueError):
        nearest_background(np.ones((5, 5), dtype=np.uint8))


def test_invalid_mask_raises():
    binary = make_test_image()
    with pytest.raises(ValueError):
        distance_transform(binary, mask="euclidean")
    with pytest.raises(ValueError):
        nearest_background(binary, mask="euclidean")


def test_non_binary_values_raise():
    binary = np.zeros((5, 5), dtype=np.uint8)
    binary[2, 2] = 2
    with pytest.raises(ValueError):
        distance_transform(binary)


def test_rejects_bad_shapes():
    with pytest.raises(ValueError):
        distance_transform(np.zeros((5, 5, 3), dtype=np.uint8))
    with pytest.raises(ValueError):
        distance_transform(np.zeros((0, 0), dtype=np.uint8))


def test_nearest_background_self_mapping():
    binary = make_test_image()
    rows, cols = nearest_background(binary)
    assert rows.shape == binary.shape
    assert cols.shape == binary.shape
    # Every background pixel maps to itself.
    assert np.all(rows[binary == 0] == np.where(binary == 0)[0])
    assert np.all(cols[binary == 0] == np.where(binary == 0)[1])


def test_nearest_background_distances_agree():
    binary = single_background_centre()
    dist = distance_transform(binary, mask="chamfer5711")
    rows, cols = nearest_background(binary, mask="chamfer5711")
    y, x = np.mgrid[0:21, 0:21]
    nearest_dist = np.hypot(y - rows, x - cols)
    assert np.abs(nearest_dist - dist).max() <= 0.6


def test_bool_input_accepted():
    binary = make_test_image().astype(bool)
    dist = distance_transform(binary)
    assert dist.shape == binary.shape
    assert np.isfinite(dist).all()
