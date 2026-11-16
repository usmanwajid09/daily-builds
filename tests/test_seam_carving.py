"""Tests for seam_carving (Day 4 of the 60-day daily-builds streak)."""

import numpy as np
import pytest

from seam_carving.seam_carving import (
    energy_map,
    find_vertical_seam,
    remove_vertical_seams,
    shrink_width,
)


def _gray_image(h=20, w=30, seed=0):
    rng = np.random.default_rng(seed)
    return rng.random((h, w))


def _rgb_image(h=20, w=30, seed=0):
    rng = np.random.default_rng(seed)
    return rng.random((h, w, 3))


def test_width_reduced_by_exactly_n():
    img = _gray_image()
    out = remove_vertical_seams(img, 7)
    assert out.shape == (20, 30 - 7)


def test_height_unchanged():
    img = _rgb_image(h=15, w=25)
    out = remove_vertical_seams(img, 5)
    assert out.shape == (15, 20, 3)


def test_grayscale_works():
    img = _gray_image()
    out = remove_vertical_seams(img, 3)
    assert out.ndim == 2 and out.shape == (20, 27)


def test_rgb_works():
    img = _rgb_image()
    out = remove_vertical_seams(img, 3)
    assert out.ndim == 3 and out.shape == (20, 27, 3)


def test_seam_is_valid_connected_path():
    img = _gray_image(h=25, w=40)
    seam = find_vertical_seam(energy_map(img))
    assert seam.shape == (25,)
    assert np.all(seam >= 0) and np.all(seam < 40)
    assert np.all(np.abs(np.diff(seam)) <= 1)


def test_low_energy_seam_preferred():
    # Image with a perfectly uniform (zero-energy) column at index 10 and
    # random noise elsewhere: the optimal seam must pass through it.
    rng = np.random.default_rng(1)
    img = rng.random((20, 30))
    img[:, 10] = 0.5
    img[:, 9] = 0.5
    img[:, 11] = 0.5
    seam = find_vertical_seam(energy_map(img))
    energy = energy_map(img)
    seam_energy = energy[np.arange(20), seam].sum()
    middle_energy = energy[:, 15].sum()  # straight seam down the middle
    assert seam_energy <= middle_energy
    assert 10 in seam  # seam goes through the uniform column


def test_removing_zero_seams_returns_unchanged_copy():
    img = _gray_image()
    out = remove_vertical_seams(img, 0)
    assert out.shape == img.shape
    assert np.array_equal(out, img)
    assert out is not img


def test_invalid_n_raises():
    img = _gray_image()
    with pytest.raises(ValueError):
        remove_vertical_seams(img, -1)
    with pytest.raises(ValueError):
        remove_vertical_seams(img, 30)  # cannot remove all columns


def test_invalid_target_width_raises():
    img = _gray_image()
    with pytest.raises(ValueError):
        shrink_width(img, 30)   # not smaller than current width
    with pytest.raises(ValueError):
        shrink_width(img, 0)
    with pytest.raises(ValueError):
        shrink_width(img, -5)


def test_shrink_width_exact_target():
    img = _gray_image()
    out = shrink_width(img, 18)
    assert out.shape == (20, 18)


def test_invalid_input_raises():
    with pytest.raises(ValueError):
        remove_vertical_seams(np.zeros(10), 1)          # 1D
    with pytest.raises(ValueError):
        remove_vertical_seams(np.zeros((0, 5)), 1)      # empty
    with pytest.raises(ValueError):
        energy_map(np.zeros((3, 0)))                     # empty width


def test_removed_seam_pixels_actually_gone():
    # Random image: removing n seams must drop exactly n columns per row
    # and keep every remaining pixel value present in the original row.
    rng = np.random.default_rng(42)
    img = rng.integers(0, 256, size=(12, 20)).astype(float)
    out = remove_vertical_seams(img, 6)
    assert out.shape == (12, 14)
    for i in range(12):
        assert out[i].shape == (14,)
        # Every output pixel must come from the original row (values kept).
        assert np.all(np.isin(out[i], img[i]))
